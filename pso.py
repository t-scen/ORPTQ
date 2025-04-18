import torch
import torch.nn as nn
from quant import *
import transformers
import random

@torch.no_grad()
def get_psobounds(model,inps,attn_mask,wbits,dev):
    optfile = "../Logs/opt_bounds_pso_3bit.pt"
    #if os.path.exists(optfile):
    #    optbounds = torch.load(optfile)
    #    return optbounds
        
    layers = model.model.decoder.layers    
    def opt_hook(name):
        def tmp(_, inp, out):
            opt_hooks[name].opthook(inp[0].data, out.data)
        return tmp
    outs = torch.zeros_like(inps)

    optbounds = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        print("The ", i, " layers")

        handles = []
        opt_hooks = {}
        opt_bounds = {}
        for name,module in layer.named_modules():
            if isinstance(module,nn.Linear): # and not "gate" in name
                opt_hooks[name]=getABP(name,module,wbits)
                handles.append(module.register_forward_hook(opt_hook(name)))
        
        sam=set()
        while len(sam)<15:
            j = random.randint(0, 100)
            sam.add(j)
        
        for j in sam:
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attn_mask)[0]
        
        for name, opt in opt_hooks.items():
            opt_bounds[name] = {"a_bound":opt.a_bound,"b_bound":opt.b_bound}
            opt=None

        for h in handles:
            h.remove()   

        inps = outs
        optbounds[i] = opt_bounds
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
    torch.save(optbounds, optfile) 
    return optbounds 


class getABP:
    def __init__(self,name,module,wbits) -> None:
        self.layer = module
        self.name = name
        self.wbits = wbits
        self.dev = self.layer.weight.device
        self.weight = self.layer.weight.data.clone() # should be per-columns
        self.bias = self.layer.bias.data.clone()    
        self.Loss=nn.MSELoss(reduction='none')
        self.linear = torch.nn.functional.linear
        self.nsamples = 0

        self.a_bound = torch.ones((self.weight.shape[0],1), dtype=self.weight.dtype, device=self.dev)
        self.b_bound = torch.ones((self.weight.shape[0],1), dtype=self.weight.dtype, device=self.dev)

    def PAB(self, inp, out, nParti, wBits):
        def Fitness(partia,partib):
            def qLoss(a,b):
                wmax=self.weight.amax(-1,keepdim=True)*a  #[N,1]*[N,1]
                wmin=self.weight.amin(-1,keepdim=True)*b

                maxq = torch.tensor(2 ** wBits - 1)
                scale = (wmax-wmin)/maxq
                scale.clamp_(min=1e-5)
                zero = -wmin/scale
                qw = quantize_dequantize(self.weight, scale, zero, maxq)

                #print("inp.dtype=",inp.dtype)
                #print("qw.dtype=",qw.dtype)
                #print("self.Bias.dtype=",self.Bias.dtype)

                qout=self.linear(inp,qw,self.bias) #inp@qw.t()
                loss=self.Loss(out,qout)
                R=(loss.mean(0,keepdim=True)).t()
                return R  #col vector

            A=partia[:,0].unsqueeze(-1)
            B=partib[:,0].unsqueeze(-1) 
            Fits=qLoss(A,B)
            for i in range(1,nParti):
                A=partia[:,i].unsqueeze(-1)
                B=partib[:,i].unsqueeze(-1)
                Fitsi=qLoss(A,B)
                Fits=torch.cat([Fits,Fitsi],-1)
            return Fits

        omgi=0.9
        omge=0.4
        c1=2.0
        c2=2.0
        loops=8    

        Parts=torch.rand(self.weight.shape[0],nParti,dtype=self.weight.dtype,device=self.weight.device)*0.3+0.7  #origin pos
        Pones=torch.ones(self.weight.shape[0],nParti,dtype=self.weight.dtype,device=self.weight.device)*0.3+0.7

        Vels=torch.rand(self.weight.shape[0],nParti,dtype=self.weight.dtype,device=self.weight.device)/20 #origin velo
        pBest=Parts
        pBestV=Fitness(pBest,Pones)
    
        gBestV,gBesti=pBestV.min(-1,keepdim=True)
        gBest=torch.gather(pBest,-1,gBesti)

        for loop in range(loops):
            omg=(omgi-omge)*(loops-loop)/loops+omge
            Vels=omg*Vels+c1*torch.rand(self.weight.shape[0],nParti,dtype=self.weight.dtype,device=self.weight.device)/10*(pBest-Parts)+c1*torch.rand(self.weight.shape[0],nParti,dtype=self.weight.dtype,device=self.weight.device)/10*(gBest-Parts)
            Parts=((Parts+Vels-0.7)%0.3+0.7).clamp(0.7,1)
            PartsV=Fitness(Parts,Pones)
            mask=PartsV<pBestV
            pBest[mask]=Parts[mask]
            pBestV[mask]=PartsV[mask]
            gBestV,gBesti=pBestV.min(-1,keepdim=True)
            gBest=torch.gather(pBest,-1,gBesti)
        gBestA=gBest
    
    #for B;
        Parts=torch.rand(self.weight.shape[0],nParti,dtype=self.weight.dtype,device=self.weight.device)*0.3+0.7  #origin pos
        Pones=gBestA.repeat(1,nParti)
        Vels=torch.rand(self.weight.shape[0],nParti,dtype=self.weight.dtype,device=self.weight.device)/20 #origin velo
        pBest=Parts
        pBestV=Fitness(Pones,pBest)  #A=ones
    
        gBestV,gBesti=pBestV.min(-1,keepdim=True)
        gBest=torch.gather(pBest,-1,gBesti)

        for loop in range(loops):
            omg=(omgi-omge)*(loops-loop)/loops+omge
            Vels=omg*Vels+c1*torch.rand(self.weight.shape[0],nParti,dtype=self.weight.dtype,device=self.weight.device)/10*(pBest-Parts)+c1*torch.rand(self.weight.shape[0],nParti,dtype=self.weight.dtype,device=self.weight.device)/10*(gBest-Parts)
            Parts=((Parts+Vels-0.7)%0.3+0.7).clamp(0.7,1)
            PartsV=Fitness(Pones,Parts)
            mask=PartsV<pBestV
            pBest[mask]=Parts[mask]
            pBestV[mask]=PartsV[mask]
            gBestV,gBesti=pBestV.min(-1,keepdim=True)
            gBest=torch.gather(pBest,-1,gBesti)
        gBestB=gBest
    
        return gBestA,gBestB  # rows;

    def opthook(self,inp,out):	  
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            if len(out.shape) == 3:
                out = out.reshape((-1, out.shape[-1]))      
        ha,hb = self.PAB(inp,out,10,self.wbits)  #10 partis 
        self.a_bound = (self.a_bound*self.nsamples+ha)/(self.nsamples+tmp)
        self.b_bound = (self.b_bound*self.nsamples+hb)/(self.nsamples+tmp) 
        self.nsamples += tmp
        print("layer name:",self.name,", samples:",self.nsamples)
