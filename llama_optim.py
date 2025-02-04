import torch
import torch.nn as nn
import torch.optim as optim
from quant import *
import transformers
import os
import logging



def get_optbounds(model,inps,attn_mask,position_ids,wbits,dev):
    bndfile = "../Logs/llama_bounds_3pd005.pt"
    #if os.path.exists(bndfile):
    #    qntbounds = torch.load(bndfile)
    #    return qntbounds
        
    layers = model.model.layers   # for llama
    def qnt_hook(name):
        def tmp(_, inp, out):
            qnt_hooks[name].qnthook(inp[0].data, out.data)
        return tmp
    outs = torch.zeros_like(inps)

    qntbounds = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        for param in layer.parameters():
            param.requires_grad_(False)

        handles = []
        qnt_hooks = {}
        qnt_bounds = {}
        for name,module in layer.named_modules():
            if isinstance(module,nn.Linear):  #and not "gate" in name
                qnt_hooks[name]=getAB(name,module,wbits)
                handles.append(module.register_forward_hook(qnt_hook(name)))
        
        for epochs in range(3):		
            for j in range(128):	
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attn_mask, position_ids=position_ids)[0]
                print("In get_bounds,mem allocated:", torch.cuda.memory_allocated())
        
        for name, bnd in qnt_hooks.items():
            qnt_bounds[name] = {"a_bound":bnd.sigmoid(bnd.a_bound),"b_bound":bnd.sigmoid(bnd.b_bound)}
            bnd=None

        for h in handles:
            h.remove()   

        inps = outs
        qntbounds[i] = qnt_bounds
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
    #torch.save(qntbounds, bndfile) 
    return qntbounds 

class getAB:
	def __init__(self,name,module,wbits) -> None:
		init_value = 4.
		dim1 = module.out_features
		self.dev=module.weight.device
		#self.dtype = module.weight.dtype
		self.wbits = wbits
            
		self.name = name
		self.epoch = 0

		self.a_bound = nn.Parameter(torch.ones((dim1,1), device=self.dev)*init_value)
		self.b_bound = nn.Parameter(torch.ones((dim1,1), device=self.dev)*init_value)

		self.linear = torch.nn.functional.linear

		self.weight = module.weight.data
		self.bias = None
		
		self.Loss=nn.MSELoss()
		self.sigmoid=nn.Sigmoid()
		self.optimizer = optim.AdamW([self.a_bound,self.b_bound], lr=0.005) 
		#self.loss_scaler = NativeScalerWithGradNormCount()


	def fakequant(self,w,bits):
		xmax = w.amax(-1,keepdim=True)*self.sigmoid(self.a_bound)
		xmin = w.amin(-1,keepdim=True)*self.sigmoid(self.b_bound)
		scale = (xmax-xmin)/(2**bits-1)
		scale.clamp_(min=1e-5)
		zero = -xmin/scale
		qw = quantize_dequantize(w, scale, zero, 2**bits-1)
		return qw

	def qnthook(self,inp,out):	
		#with traincast():
        
		print("hook-begin,mem allocated:", torch.cuda.memory_allocated())

		#self.weight = self.weight.float().to(self.dev)
		#self.bias = self.bias.float().to(self.dev)
		
		qweight=self.fakequant(self.weight,4)
		qout=self.linear(inp,qweight,self.bias)
        
		#qout=self.linear(inp.data.float(),qweight,self.bias)

		print("In hook,mem allocated:", torch.cuda.memory_allocated())
    
#		loss=self.Loss(out.data.float(),qout)
		
		loss=self.Loss(out,qout)				
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		#self.weight = self.weight.cpu()
		#self.bias = self.bias.cpu()	
		
		#qweight=None
		#qout=None
		self.epoch+=1
		print("hook end,mem allocated:", torch.cuda.memory_allocated())

		print(str(self.epoch)+":"+self.name)
		logging.info(f'{self.epoch}:{self.name},loss={loss}')
		#logger.info(str(cache["i"])+":"+self.name)
		#norm = self.loss_scaler(loss, self.optimizer,parameters=[self.a_bound,self.b_bound])


