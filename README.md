# ORPTQ

This repository contains the code for the ORPTQ. 
The purpose of the ORPTQ quantization method is to adjust the quantization range based on the second-order method, ensuring a better starting point. The second-order method was referred to GPTQ in ICLR 2023 paper [GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://arxiv.org/abs/2210.17323). 

The current release includes the following features:

* introducing parameters on the basis of the GPTQ algorithm in `gptq.py`
* 3 optimization approaches for opt and llama respectively.

command example:

python opt.py "opt-6.7b" "c4" --wbits 4 --abtype "pso" --quanttype "gptq"

model: the path of the opt model

dataset: "c4" or "wikitext"

wbits:4,3,2

abtype: "pso","optim","gdiv","mix","none"

quanttype "gptq","plain"
