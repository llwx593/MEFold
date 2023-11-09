import argparse
import torch
import torch.nn as nn
import esm
import quant
from rtn_quant import Quantizer, quantize

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res

@torch.no_grad()
def esmfold_sequential(model):
    print('Starting ...')

    layers = model.esm.layers

    torch.cuda.empty_cache()
    
    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        layer = layers[i]

        full = find_layers(layer)
        sequential = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'], ['self_attn.out_proj'], ['fc1'], ["fc2"]]
        for names in sequential:
            subset = {n: full[n] for n in names}
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)
                quantizers['esm.layers.%d.%s' % (i, name)] = (quantizer, quantizer.scale, quantizer.zero, args.wbits)
                error = torch.mean((W - subset[name].weight.data)**2)
                print("name : ", name)
                print("error is ", error)

        del layer

        torch.cuda.empty_cache()

        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')

    return quantizers

def quant_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, _ = quantizers[name]
        quantizers[name], scale, zero = quantizers[name].cpu(), scale.cpu(), zero.cpu()
        layers[name] = layers[name].cpu()
        qlayers[name].pack(layers[name], scale, zero)
    print('Done.')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device_id', type=int, default=0, help='the id of gpu')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='bits to use for quantization')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--save', type=str, default='esmfold_rtn_quant.pt', help='Save quantized checkpoint under this name.')

    args = parser.parse_args()

    if args.device_id != -1:
        torch.cuda.set_device(args.device_id)
        device_str = "cuda:" + str(args.device_id)
        DEV = torch.device(device_str)

    model = esm.pretrained.esmfold_v1()
    model.eval()

    chunk_dict={"tri_mul_out": 0,
                "tri_mul_in": 0,
                "tri_att_start": 0,
                "tri_att_end": 0,
                "mlp_seq": 0,
                "mlp_pair": 0,   
                }
    for k in chunk_dict.keys():
        chunk_dict[k]=None
    
    model.set_chunk_size(chunk_dict)
    model.set_device_id(args.device_id)

    quantizers = esmfold_sequential(model)
    quant_pack(model, quantizers, args.wbits, args.groupsize)
    torch.save(model.state_dict(), args.save)
