import argparse
import json
import sys
import math
import torch
import torch.nn as nn
import esm
import quant  


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res

def load_quant(checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    layers = find_layers(model.esm.layers)
    new_layers = {"esm.layers.%s" % n : layers[n] for n in layers}
    quant.make_quant_linear(model, new_layers, wbits, groupsize)

    del layers, new_layers

    print('Loading model ...')
    model.load_state_dict(torch.load(checkpoint))

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)

    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    print('Done.')

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device_id', type=int, default=0, help='the id of gpu')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='bits to use for quantization')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--cs1', type=int, default=None, help='chunksize of tri_att_start and tri_att_end')
    parser.add_argument('--cs2', type=int, default=None, help='chunksize of mlp_seq')
    parser.add_argument('--cs3', type=int, default=None, help='chunksize of tri_mul_in and tri_mul_out')
    parser.add_argument('--cs4', type=int, default=None, help='chunksize of mlp_pair')

    
    args = parser.parse_args()

    torch.cuda.set_device(args.device_id)

    seq = ""

    with open(args.seq_path, "r") as f:
        seq_dict = json.load(f)

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    chunk_dict={"tri_mul_out": 0,
                "tri_mul_in": 0,
                "tri_att_start": 0,
                "tri_att_end": 0,
                "mlp_seq": 0,
                "mlp_pair": 0,   
                }
    for k in chunk_dict.keys():
        chunk_dict[k]=None
    
    chunk_dict["tri_att_start"] = args.cs1
    chunk_dict["tri_att_end"] = args.cs1
    chunk_dict["mlp_pair"] = args.cs2
    chunk_dict["tri_mul_in"] = args.cs3
    chunk_dict["tri_mul_out"] = args.cs3
    chunk_dict["mlp_pair"] = args.cs4

    model = load_quant(args.load, args.wbits, args.groupsize)
    model.set_device_id(args.device_id)
    model.set_chunk_size(chunk_dict)

    model = model.cuda()




  
