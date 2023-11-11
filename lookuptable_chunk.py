import argparse
import math
import os
import json

def find_chunk_config_seed(log_path,s_len):
    peak_m=[]
    with open(log_path, "r") as f:
        data=f.readlines()
        for d in data:
            if d.startswith("trunk peak"):
                pm=d.strip().split(" ")[-1]
                peak_m.append(float(pm))
    module_len=int(len(peak_m)/4)

    peak_m_tri_att_start = peak_m[:module_len]
    peak_m_tri_mlp_seq = peak_m[module_len:module_len*2]
    peak_m_tri_mul_in = peak_m[module_len*2:module_len*3]
    peak_m_tri_mlp_pair = peak_m[module_len*3:]
    
    peak_m=[]
   
    peak_m.append(peak_m_tri_att_start)
    peak_m.append(peak_m_tri_mlp_pair)
    peak_m.append(peak_m_tri_mul_in)
    peak_m.append(peak_m_tri_mlp_seq)

    chunk_config = get_chunk_config(s_len,peak_m)
    peak_dict = {}
    peak_dict["tri_att_start"] = peak_m_tri_att_start
    peak_dict["tri_mlp_seq"] = peak_m_tri_mlp_seq
    peak_dict["tri_mul_in"] = peak_m_tri_mul_in
    peak_dict["tri_mlp_pair"] = peak_m_tri_mlp_pair

    return chunk_config, peak_dict
       
def get_chunk_config(s_len,peak_m):
    chunk_config={}

    cs=[]
    cs.append(None)
    while True:
        s_len=math.ceil(s_len/2)
        cs.append(s_len)
        if s_len==1:
            break
    chunksize = cs
    tmp_max_memory=peak_m[0][0]
    tmp_module=0
    tmp_chunk_index=[0 for i in range(4)]
    tmp_chunk_size=[chunksize[s] for s in tmp_chunk_index]
    chunk_config[tmp_max_memory]=tmp_chunk_size

    while True:
        if tmp_max_memory==peak_m[0][-1]:
            break
        tmp_chunk_index[tmp_module]+=1
        tmp_memory=[]
        for i in range(4):
            tmp_memory.append(peak_m[i][tmp_chunk_index[i]])
        tmp_max_memory=max(tmp_memory)
        tmp_module = tmp_memory.index(max(tmp_memory))
            
        tmp_chunk_size=[chunksize[s] for s in tmp_chunk_index]
        chunk_config[tmp_max_memory]=tmp_chunk_size

    for key in chunk_config.keys():
        print(key,":",chunk_config[key])

    return chunk_config
        
def get_real_chunkdic(ref_peak_m,real_len,ref_len):
    
    low_memo=ref_peak_m[-1][-1]
    low_memo_real =  72*((real_len/100)**2.12) + 6142.5
    print("real low memo:",low_memo_real)
    real_peak_m=ref_peak_m.copy()
    for i in range(len(ref_peak_m[0])-1,-1,-1):
        
        real_peak_m[0][i]=(ref_peak_m[0][i]-low_memo)*((real_len/ref_len)**3) + low_memo_real if ref_peak_m[0][i]!=low_memo else 72*((real_len/100)**2.12) + 6142.5 
        real_peak_m[1][i]=(ref_peak_m[1][i]-low_memo)*((real_len/ref_len)**2) + low_memo_real if ref_peak_m[1][i]!=low_memo else 72*((real_len/100)**2.12) + 6142.5 
        real_peak_m[2][i]=(ref_peak_m[2][i]-low_memo)*((real_len/ref_len)**2) + low_memo_real if ref_peak_m[2][i]!=low_memo else 72*((real_len/100)**2.12) + 6142.5
        real_peak_m[3][i] = low_memo_real
    for i in (real_peak_m):
        print(i)
    # get_chunk_config(real_len,real_peak_m)
    return real_peak_m
    # print(real_peak_m)   
    
def find_ref_peak(input_squence_len, peak_config):
    seed_len=[255,384,463,514,588, 682,718,779,863,949]

    with open(peak_config, "r") as f:
        seed_peak_dict = json.load(f)
    
    ref_len=0
    for i, s_len in enumerate(seed_len):
        if s_len>=input_squence_len:
            ref_len=i
            break
    ref_seed = seed_peak_dict[str(seed_len[ref_len])]
    seed_peak_m = []
    for key in ref_seed.keys():
        seed_peak_m.append(ref_seed[key])
    return seed_peak_m, seed_len[ref_len]
  
def func_sequence_get_chunkconfig(input_squence_len, peak_config):
    ref_peak_m, ref_seqence_len = find_ref_peak(input_squence_len, peak_config)
    print("Ref squence len : ",ref_seqence_len)
    real_peak_m = get_real_chunkdic(ref_peak_m,input_squence_len,ref_seqence_len)
    real_chunk_config = get_chunk_config(input_squence_len,real_peak_m)
    return real_chunk_config

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default="", help='the type of task')
    parser.add_argument('--log_path', type=str, default='', help='path of seed sequence')
    parser.add_argument('--chunk_config', type=str, default='', help='path of chunk config')
    parser.add_argument('--peak_config', type=str, default='', help='path of peak config')
    parser.add_argument('--input_seqlen', type=int, default=0, help='length of input sequence')

    args = parser.parse_args()

    seqlen_list = [255, 384, 463, 514, 588, 682, 718, 779, 863, 949]
    if args.task == "seed_task":
        log_list = os.listdir(args.log_path)
        chunk_config = {}
        peak_dict = {}
        for i in range(len(log_list)):
            file_path = os.path.join(args.log_path, log_list[i])
            seq_len = seqlen_list[i]
            per_config, per_dict = find_chunk_config_seed(file_path, seq_len)
            chunk_config[str(seq_len)] = per_config
            peak_dict[str(seq_len)] = per_dict
        cc_json = json.dumps(chunk_config, indent=4)
        with open(args.chunk_config, "w") as f:
            f.write(cc_json)
        pd_json = json.dumps(peak_dict, indent=4)
        with open(args.peak_config, "w") as f:
            f.write(pd_json)
    elif args.task == "real_inference":
        input_config = func_sequence_get_chunkconfig(args.input_seqlen, args.peak_config)
