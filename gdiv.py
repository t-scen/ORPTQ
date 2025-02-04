import torch
import torch.nn as nn
import torch.optim as optim
from quant import *
import transformers
import os
import logging
import random

@torch.no_grad()
def get_gdivbounds(model,inps,attn_mask,wbits,dev):
    optfile = "../Logs/opt_bounds_gdiv_4bit.pt"
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
                opt_hooks[name]=getABG(name,module,wbits)
                handles.append(module.register_forward_hook(opt_hook(name)))
        
        sam=set()
        while len(sam)<6:
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

#gold-div
class getABG:  
    def __init__(self,name,module,wbits) -> None:
        self.layer = module
        self.name = name
        self.dev = self.layer.weight.device
        self.weight = self.layer.weight.data.clone() # should be per-columns
        self.bias = self.layer.bias.data.clone()    
        self.Loss=nn.MSELoss(reduction='none')
        self.linear = torch.nn.functional.linear
        self.nsamples = 0
        self.wbits = wbits

        self.a_bound = torch.ones((self.weight.shape[0],1), dtype=self.weight.dtype, device=self.dev)
        self.b_bound = torch.ones((self.weight.shape[0],1), dtype=self.weight.dtype, device=self.dev)


    def GAB(self, inp, out, wBits):
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
        
        thres = 0.05
        Loops = 20

        A1=torch.ones((self.weight.shape[0],1),dtype=self.weight.dtype,device=self.weight.device)*0.5
        A2=torch.ones((self.weight.shape[0],1),dtype=self.weight.dtype,device=self.weight.device)*1.0
        B=torch.ones((self.weight.shape[0],1),dtype=self.weight.dtype,device=self.weight.device)*1.0

        IR=(torch.abs(A2-A1)<thres)
        U1=A1+0.382*(A2-A1)
        U2=A1+0.618*(A2-A1)
        loop=0
        while(IR.prod().item()!=1 or loop>Loops):           
            R1=qLoss(U1,B)
            R2=qLoss(U2,B)
            RA2=qLoss(A2,B)

            IR=(torch.abs(A2-A1)>=thres)&(R1<=R2)&(R2<=RA2)#1
            A2[IR]=U1[IR]
            U1[IR]=A1[IR]+0.382*(A2[IR]-A1[IR])
            U2[IR]=A1[IR]+0.618*(A2[IR]-A1[IR])

            IR=(torch.abs(A2-A1)>=thres)&(R1<=RA2)&(RA2<=R2)#2
            A2[IR]=U1[IR]
            U1[IR]=A1[IR]+0.382*(A2[IR]-A1[IR])
            U2[IR]=A1[IR]+0.618*(A2[IR]-A1[IR])

            IR=(torch.abs(A2-A1)>=thres)&(R2<=R1)&(R1<=RA2)#3
            A1[IR]=U1[IR]
            A2[IR]=U2[IR]
            U1[IR]=A1[IR]+0.382*(A2[IR]-A1[IR])
            U2[IR]=A1[IR]+0.618*(A2[IR]-A1[IR])

            IR=(torch.abs(A2-A1)>=thres)&(R2<=RA2)&(RA2<=R1)#4
            A1[IR]=U1[IR]
            A2[IR]=U2[IR]
            U1[IR]=A1[IR]+0.382*(A2[IR]-A1[IR])
            U2[IR]=A1[IR]+0.618*(A2[IR]-A1[IR])

            IR=(torch.abs(A2-A1)>=thres)&(RA2<=R2)&(R2<=R1)#5
            A1[IR]=U1[IR]
            U1[IR]=A1[IR]+0.382*(A2[IR]-A1[IR])
            U2[IR]=A1[IR]+0.618*(A2[IR]-A1[IR])

            IR=(torch.abs(A2-A1)>=thres)&(RA2<=R1)&(R1<=R2)#6
            A1[IR]=U1[IR]
            U1[IR]=A1[IR]+0.382*(A2[IR]-A1[IR])
            U2[IR]=A1[IR]+0.618*(A2[IR]-A1[IR])

            IR=(torch.abs(A2-A1)<thres)
            loop +=1
        A=(A1+A2)/2
        
        B1=torch.ones((self.weight.shape[0],1),dtype=self.weight.dtype,device=self.weight.device)*0.5
        B2=torch.ones((self.weight.shape[0],1),dtype=self.weight.dtype,device=self.weight.device)*1.0

        IR=(torch.abs(B2-B1)<thres)
        U1=B1+0.382*(B2-B1)
        U2=B1+0.618*(B2-B1)

        loop=0        
        while(IR.prod().item()!=1 or loop>Loops):         
            R1=qLoss(A,U1)
            R2=qLoss(A,U2)
            RB2=qLoss(A,B2)

            IR=(torch.abs(B2-B1)>=thres)&(R1<=R2)&(R2<=RB2)#1
            B2[IR]=U1[IR]
            U1[IR]=B1[IR]+0.382*(B2[IR]-B1[IR])
            U2[IR]=B1[IR]+0.618*(B2[IR]-B1[IR])

            IR=(torch.abs(B2-B1)>=thres)&(R1<=RB2)&(RB2<=R2)#2
            B2[IR]=U1[IR]
            U1[IR]=B1[IR]+0.382*(B2[IR]-B1[IR])
            U2[IR]=B1[IR]+0.618*(B2[IR]-B1[IR])

            IR=(torch.abs(B2-B1)>=thres)&(R2<=R1)&(R1<=RB2)#3
            B1[IR]=U1[IR]
            B2[IR]=U2[IR]
            U1[IR]=B1[IR]+0.382*(B2[IR]-B1[IR])
            U2[IR]=B1[IR]+0.618*(B2[IR]-B1[IR])

            IR=(torch.abs(B2-B1)>=thres)&(R2<=RB2)&(RB2<=R1)#4
            B1[IR]=U1[IR]
            B2[IR]=U2[IR]
            U1[IR]=B1[IR]+0.382*(B2[IR]-B1[IR])
            U2[IR]=B1[IR]+0.618*(B2[IR]-B1[IR])

            IR=(torch.abs(B2-B1)>=thres)&(RB2<=R2)&(R2<=R1)#5
            B1[IR]=U1[IR]
            U1[IR]=B1[IR]+0.382*(B2[IR]-B1[IR])
            U2[IR]=B1[IR]+0.618*(B2[IR]-B1[IR])

            IR=(torch.abs(B2-B1)>=thres)&(RB2<=R1)&(R1<=R2)#6
            B1[IR]=U1[IR]
            U1[IR]=B1[IR]+0.382*(B2[IR]-B1[IR])
            U2[IR]=B1[IR]+0.618*(B2[IR]-B1[IR])

            IR=(torch.abs(B2-B1)<thres)
            loop +=1
        B=(B1+B2)/2
        return A,B

    def opthook(self,inp,out):	  
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            if len(out.shape) == 3:
                out = out.reshape((-1, out.shape[-1]))      
        ha,hb = self.GAB(inp,out,self.wbits) 
        self.a_bound = (self.a_bound*self.nsamples+ha)/(self.nsamples+tmp)
        self.b_bound = (self.b_bound*self.nsamples+hb)/(self.nsamples+tmp) 
        self.nsamples += tmp
        print("layer name:",self.name,", samples:",self.nsamples)
