# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from openfold.model.triangular_attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from torch import nn

from esm.esmfold.v1.misc import (
    Attention,
    Dropout,
    PairToSequence,
    ResidueMLP,
    SequenceToPair,
)
import time
import GPUtil
import logging
import threading
logging.getLogger().setLevel(logging.INFO)

# class MemoryIntimeLogger:
#     def __init__(self, device_id: int):
#         gpu = GPUtil.getGPUs()[device_id]
#         logging.info(f'Initializing: Used memory = {gpu.memoryUsed}MiB, Free memory = {gpu.memoryFree}MiB, '
#                      f'Total memory = {gpu.memoryTotal}MiB')

#         self.device_id = device_id
#         self.currentMemoryUsed = gpu.memoryUsed
#         self.allocatedMemory = gpu.memoryUsed
#         self.maxMemory = 0
#         self.flag = True
    
#     def IntimeRecord(self):
#         # gpu = GPUtil.getGPUs()[self.device_id]
#         import pynvml
#         pynvml.nvmlInit()
#         while True:
#             if not self.flag:
#                 print(f"mlp pair peak memory {self.maxMemory}")
#                 break
#             # freeMem = gpu.memoryFree
#             # usedMem = gpu.memoryUsed
#             # print("!!!",usedMem - self.allocatedMemory)
#             handler = pynvml.nvmlDeviceGetHandleByIndex(0)
#             meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
#             total = round(meminfo.total / 1024 / 1024, 2)
#             used = round(meminfo.used / 1024 / 1024, 2)
#             free = round(meminfo.free / 1024 / 1024, 2)
#             if used - self.allocatedMemory > self.maxMemory:
#                 self.maxMemory = used - self.allocatedMemory
#                 # logging.info(f'{mark}: Allocate memory = {usedMem - self.allocatedMemory: .1f}MiB, '
#                 #             f'Free memory = {freeMem: .1f}MiB, Increased memory = {usedMem - self.currentMemoryUsed: .1f}')
#             self.currentMemoryUsed = used 
#             time.sleep(0.00000000005)
        
    
#     def IntimeRecordThreadStart(self):
#         self.recordthread = threading.Thread(target=self.IntimeRecord,)
#         self.recordthread.start()
#         # self.recordthread.daemon=True
        
        
#     def IntimeRecordThreadEnd(self):
#         self.flag = False
#         self.recordthread.join()
        
#     def log(self, mark: str):
#         # synchronize
#         gpu = GPUtil.getGPUs()[self.device_id]
#         freeMem = gpu.memoryFree
#         usedMem = gpu.memoryUsed
#         logging.info(f'{mark}: Allocate memory = {usedMem - self.allocatedMemory: .1f}MiB, '
#                      f'Free memory = {freeMem: .1f}MiB, Increased memory = {usedMem - self.currentMemoryUsed: .1f}')
#         self.currentMemoryUsed = usedMem
        
class MemoryLogger:
    def __init__(self, device_id: int):
        gpu = GPUtil.getGPUs()[device_id]
        logging.info(f'Initializing: Used memory = {gpu.memoryUsed}MiB, Free memory = {gpu.memoryFree}MiB, '
                     f'Total memory = {gpu.memoryTotal}MiB')

        self.device_id = device_id
        self.currentMemoryUsed = gpu.memoryUsed
        self.allocatedMemory = gpu.memoryUsed
        self.maxMemory = 0
        self.flag = True
        
    def IntimeRecord(self):
        # gpu = GPUtil.getGPUs()[self.device_id]
        import pynvml
        pynvml.nvmlInit()
        while True:
            if not self.flag:
                print(f"trunk peak memory {self.maxMemory}")
                break
            # freeMem = gpu.memoryFree
            # usedMem = gpu.memoryUsed
            # print("!!!",usedMem - self.allocatedMemory)
            handler = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
            total = round(meminfo.total / 1024 / 1024, 2)
            used = round(meminfo.used / 1024 / 1024, 2)
            free = round(meminfo.free / 1024 / 1024, 2)
            # if used - self.allocatedMemory > self.maxMemory:
            #     self.maxMemory = used - self.allocatedMemory             
            # self.currentMemoryUsed = used 
            if used > self.maxMemory:
                self.maxMemory = used
            time.sleep(0.00000000005)
        
    
    def IntimeRecordThreadStart(self):
        self.recordthread = threading.Thread(target=self.IntimeRecord,)
        self.recordthread.start()
        # self.recordthread.daemon=True
        
    def IntimeRecordThreadEnd(self):
        self.flag = False
        self.recordthread.join()
        
    def log(self, mark: str):
        # synchronize
        gpu = GPUtil.getGPUs()[self.device_id]
        freeMem = gpu.memoryFree
        usedMem = gpu.memoryUsed
        logging.info(f'{mark}: Allocate memory = {usedMem - self.allocatedMemory: .1f}MiB, '
                     f'Free memory = {freeMem: .1f}MiB, Increased memory = {usedMem - self.currentMemoryUsed: .1f}')
        self.currentMemoryUsed = usedMem

# from memory_profiler import profile
class TriangularSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        sequence_state_dim,
        pairwise_state_dim,
        sequence_head_width,
        pairwise_head_width,
        dropout=0,
        **__kwargs,
    ):
        super().__init__()

        assert sequence_state_dim % sequence_head_width == 0
        assert pairwise_state_dim % pairwise_head_width == 0
        sequence_num_heads = sequence_state_dim // sequence_head_width
        pairwise_num_heads = pairwise_state_dim // pairwise_head_width
        assert sequence_state_dim == sequence_num_heads * sequence_head_width
        assert pairwise_state_dim == pairwise_num_heads * pairwise_head_width
        assert pairwise_state_dim % 2 == 0
        self.init_time=0
        self.sequence_state_dim = sequence_state_dim
        self.pairwise_state_dim = pairwise_state_dim

        self.layernorm_1 = nn.LayerNorm(sequence_state_dim)

        self.sequence_to_pair = SequenceToPair(
            sequence_state_dim, pairwise_state_dim // 2, pairwise_state_dim
        )
        self.pair_to_sequence = PairToSequence(pairwise_state_dim, sequence_num_heads)

        self.seq_attention = Attention(
            sequence_state_dim, sequence_num_heads, sequence_head_width, gated=True
        )
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            pairwise_state_dim,
            pairwise_state_dim,
            
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            pairwise_state_dim,
            pairwise_state_dim,
        )
        self.tri_att_start = TriangleAttentionStartingNode(
            pairwise_state_dim,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )  # type: ignore
        self.tri_att_end = TriangleAttentionEndingNode(
            pairwise_state_dim,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )  # type: ignore
    
        self.mlp_seq = ResidueMLP(sequence_state_dim, 4 * sequence_state_dim, dropout=dropout)
        self.mlp_pair = ResidueMLP(pairwise_state_dim, 4 * pairwise_state_dim, dropout=dropout)

        assert dropout < 0.4
        self.drop = nn.Dropout(dropout)
        self.row_drop = Dropout(dropout * 2, 2)
        self.col_drop = Dropout(dropout * 2, 1)

        torch.nn.init.zeros_(self.tri_mul_in.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_in.linear_z.bias)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.bias)
        torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.weight)
        torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.bias)
        torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.weight)
        torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.bias)

        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.weight)
        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.bias)
        torch.nn.init.zeros_(self.pair_to_sequence.linear.weight)
        torch.nn.init.zeros_(self.seq_attention.o_proj.weight)
        torch.nn.init.zeros_(self.seq_attention.o_proj.bias)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-2].weight)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-2].bias)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].weight)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].bias)

    def quantize(self):
        self.tri_mul_out.quantize()
        self.tri_mul_in.quantize()
        self.tri_att_start.quantize()
        self.tri_att_end.quantize()

    def freeze(self):
        self.tri_mul_out.freeze()
        self.tri_mul_in.freeze()
        self.tri_att_start.freeze()
        self.tri_att_end.freeze()

    def forward(self, sequence_state, pairwise_state, mask=None, chunk_size=None, memory_logger : MemoryLogger = None, **__kwargs):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim
          pairwise_state: B x L x L x pairwise_state_dim
          mask: B x L boolean tensor of valid positions

        Output:
          sequence_state: B x L x sequence_state_dim
          pairwise_state: B x L x L x pairwise_state_dim
        """
        assert len(sequence_state.shape) == 3
        assert len(pairwise_state.shape) == 4
        if mask is not None:
            assert len(mask.shape) == 2
        # sequence_state = sequence_state.to_mkldnn()
        # pairwise_state = pairwise_state.to_mkldnn()
        # print(f'!!!actual chunksize={chunk_size}')
        batch_dim, seq_dim, sequence_state_dim = sequence_state.shape
        pairwise_state_dim = pairwise_state.shape[3]
        assert sequence_state_dim == self.sequence_state_dim
        assert pairwise_state_dim == self.pairwise_state_dim
        assert batch_dim == pairwise_state.shape[0]
        assert seq_dim == pairwise_state.shape[1]
        assert seq_dim == pairwise_state.shape[2]
        record_time=[]
        # Update sequence state
        s1=time.time()
        bias = self.pair_to_sequence(pairwise_state)
        bias_time=time.time()-s1
        # print("=====Update sequence state time:",bias_time)

        # Self attention with bias + mlp.
        s1=time.time()
        y = self.layernorm_1(sequence_state)
        
        y, _ = self.seq_attention(y, mask=mask, bias=bias)
        
        sequence_state = sequence_state + self.drop(y)
        self_atten_time=time.time()-s1
        # print("=====Self attention time:",self_atten_time)

        s1=time.time()
        sequence_state = self.mlp_seq(sequence_state, chunk_size=chunk_size["mlp_seq"])
        transition1_time=time.time()-s1
        # memory_logger.log("mlp_seq")
        # print("=====mlp_seq time:",transition1_time)

        # Update pairwise state
        s1=time.time()
        pairwise_state = pairwise_state + self.sequence_to_pair(sequence_state)
        pairwise_time=time.time()-s1
        # print("=====pairwise time:",pairwise_time)

        # Axial attention with triangular bias.
        s0=time.time()
        tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
        s1=time.time()
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_mul_out(pairwise_state, mask=tri_mask, inplace_safe=True, _inplace_chunk_size=chunk_size["tri_mul_out"])
        )
        # memory_logger.log("tri_mul_out")
        tri_mul_out_time = time.time()-s1
        # print("--------tri_mul_out time:",time.time()-s1)

        s1=time.time()
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_mul_in(pairwise_state, mask=tri_mask, inplace_safe=True, _inplace_chunk_size=chunk_size["tri_mul_in"])
        )
        # memory_logger.log("tri_mul_in")
        tri_mul_in_time = time.time()-s1
        # print("--------tri_mul_in time:",time.time()-s1)

        s1=time.time()
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_att_start(pairwise_state, mask=tri_mask, chunk_size=chunk_size["tri_att_start"])
        )
        # memory_logger.log("tri_att_start")
        tri_att_start_time = time.time()-s1
        # print("--------tri_att_start time:",time.time()-s1)

        s1=time.time()
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_att_end(pairwise_state, mask=tri_mask, chunk_size=chunk_size["tri_att_end"])
        )
        # memory_logger.log("tri_att_end")
        tri_att_end_time = time.time()-s1
        # print("--------tri_att_end time:",time.time()-s1)
        triangular_time=time.time()-s0
        # print("======Axial attention time:",triangular_time)

        # MLP over pairis.
        s1=time.time()
        
        
        pairwise_state = self.mlp_pair(pairwise_state, chunk_size=chunk_size["mlp_pair"])
        # memory_logger.log("mlp_pair")
        transition2_time=time.time()-s1
        # print("======mlp_pair time:",transition2_time)
        record_time.append(bias_time)
        record_time.append(self_atten_time)
        record_time.append(transition1_time)
        record_time.append(pairwise_time)
        record_time.append(triangular_time)
        record_time.append(transition2_time)
        record_time.append(tri_mul_out_time)
        record_time.append(tri_mul_in_time)
        record_time.append(tri_att_start_time)
        record_time.append(tri_att_end_time)

        return sequence_state, pairwise_state, record_time


    def quantize_forward(self, sequence_state, pairwise_state, mask=None, chunk_size=None, **__kwargs):
        """
        Inputs:
        sequence_state: B x L x sequence_state_dim
        pairwise_state: B x L x L x pairwise_state_dim
        mask: B x L boolean tensor of valid positions

        Output:
        sequence_state: B x L x sequence_state_dim
        pairwise_state: B x L x L x pairwise_state_dim
        """
        assert len(sequence_state.shape) == 3
        assert len(pairwise_state.shape) == 4
        if mask is not None:
            assert len(mask.shape) == 2
        # sequence_state = sequence_state.to_mkldnn()
        # pairwise_state = pairwise_state.to_mkldnn()
        batch_dim, seq_dim, sequence_state_dim = sequence_state.shape
        pairwise_state_dim = pairwise_state.shape[3]
        assert sequence_state_dim == self.sequence_state_dim
        assert pairwise_state_dim == self.pairwise_state_dim
        assert batch_dim == pairwise_state.shape[0]
        assert seq_dim == pairwise_state.shape[1]
        assert seq_dim == pairwise_state.shape[2]
        record_time=[]
        # Update sequence state
        s1=time.time()
        bias = self.pair_to_sequence(pairwise_state)
        bias_time=time.time()-s1
        # print("=====Update sequence state time:",bias_time)

        # Self attention with bias + mlp.
        s1=time.time()
        y = self.layernorm_1(sequence_state)
        y, _ = self.seq_attention(y, mask=mask, bias=bias)
        sequence_state = sequence_state + self.drop(y)
        self_atten_time=time.time()-s1
        # print("=====Self attention time:",self_atten_time)

        s1=time.time()
        sequence_state = self.mlp_seq(sequence_state)
        transition1_time=time.time()-s1
        # print("=====transition time:",transition1_time)

        # Update pairwise state
        s1=time.time()
        pairwise_state = pairwise_state + self.sequence_to_pair(sequence_state)
        pairwise_time=time.time()-s1
        # print("=====pairwise time:",pairwise_time)

        # Axial attention with triangular bias.
        s0=time.time()
        tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
        s1=time.time()
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_mul_out.quantize_forward(pairwise_state, mask=tri_mask)
        )
        tri_mul_out_time = time.time()-s1
        # print("--------tri_mul_out time:",time.time()-s1)

        s1=time.time()
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_mul_in.quantize_forward(pairwise_state, mask=tri_mask)
        )
        tri_mul_in_time = time.time()-s1
        # print("--------tri_mul_in time:",time.time()-s1)

        s1=time.time()
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_att_start.quantize_forward(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )
        tri_att_start_time = time.time()-s1
        # print("--------tri_att_start time:",time.time()-s1)

        s1=time.time()
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_att_end.quantize_forward(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )
        tri_att_end_time = time.time()-s1
        # print("--------tri_att_end time:",time.time()-s1)
        triangular_time=time.time()-s0
        # print("======Axial attention time:",triangular_time)

        # MLP over pairis.
        s1=time.time()
        pairwise_state = self.mlp_pair(pairwise_state)
        transition2_time=time.time()-s1
        # print("======transition time:",transition2_time)
        record_time.append(bias_time)
        record_time.append(self_atten_time)
        record_time.append(transition1_time)
        record_time.append(pairwise_time)
        record_time.append(triangular_time)
        record_time.append(transition2_time)
        record_time.append(tri_mul_out_time)
        record_time.append(tri_mul_in_time)
        record_time.append(tri_att_start_time)
        record_time.append(tri_att_end_time)

        return sequence_state, pairwise_state, record_time

    def quantize_inference(self, sequence_state, pairwise_state, mask=None, chunk_size=None, **__kwargs):
        """
        Inputs:
        sequence_state: B x L x sequence_state_dim
        pairwise_state: B x L x L x pairwise_state_dim
        mask: B x L boolean tensor of valid positions

        Output:
        sequence_state: B x L x sequence_state_dim
        pairwise_state: B x L x L x pairwise_state_dim
        """
        assert len(sequence_state.shape) == 3
        assert len(pairwise_state.shape) == 4
        if mask is not None:
            assert len(mask.shape) == 2
        # sequence_state = sequence_state.to_mkldnn()
        # pairwise_state = pairwise_state.to_mkldnn()
        batch_dim, seq_dim, sequence_state_dim = sequence_state.shape
        pairwise_state_dim = pairwise_state.shape[3]
        assert sequence_state_dim == self.sequence_state_dim
        assert pairwise_state_dim == self.pairwise_state_dim
        assert batch_dim == pairwise_state.shape[0]
        assert seq_dim == pairwise_state.shape[1]
        assert seq_dim == pairwise_state.shape[2]
        record_time=[]
        # Update sequence state
        s1=time.time()
        bias = self.pair_to_sequence(pairwise_state)
        bias_time=time.time()-s1
        # print("=====Update sequence state time:",bias_time)

        # Self attention with bias + mlp.
        s1=time.time()
        y = self.layernorm_1(sequence_state)
        y, _ = self.seq_attention(y, mask=mask, bias=bias)
        sequence_state = sequence_state + self.drop(y)
        self_atten_time=time.time()-s1
        # print("=====Self attention time:",self_atten_time)

        s1=time.time()
        sequence_state = self.mlp_seq(sequence_state)
        transition1_time=time.time()-s1
        # print("=====transition time:",transition1_time)

        # Update pairwise state
        s1=time.time()
        pairwise_state = pairwise_state + self.sequence_to_pair(sequence_state)
        pairwise_time=time.time()-s1
        # print("=====pairwise time:",pairwise_time)

        # Axial attention with triangular bias.
        s0=time.time()
        tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
        s1=time.time()
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_mul_out.quantize_inference(pairwise_state, mask=tri_mask)
        )
        tri_mul_out_time = time.time()-s1
        # print("--------tri_mul_out time:",time.time()-s1)

        s1=time.time()
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_mul_in.quantize_inference(pairwise_state, mask=tri_mask)
        )
        tri_mul_in_time = time.time()-s1
        # print("--------tri_mul_in time:",time.time()-s1)

        s1=time.time()
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_att_start.quantize_inference(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )
        tri_att_start_time = time.time()-s1
        # print("--------tri_att_start time:",time.time()-s1)

        s1=time.time()
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_att_end.quantize_inference(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )
        tri_att_end_time = time.time()-s1
        # print("--------tri_att_end time:",time.time()-s1)
        triangular_time=time.time()-s0
        # print("======Axial attention time:",triangular_time)

        # MLP over pairis.
        s1=time.time()
        pairwise_state = self.mlp_pair(pairwise_state)
        transition2_time=time.time()-s1
        # print("======transition time:",transition2_time)
        record_time.append(bias_time)
        record_time.append(self_atten_time)
        record_time.append(transition1_time)
        record_time.append(pairwise_time)
        record_time.append(triangular_time)
        record_time.append(transition2_time)
        record_time.append(tri_mul_out_time)
        record_time.append(tri_mul_in_time)
        record_time.append(tri_att_start_time)
        record_time.append(tri_att_end_time)

        return sequence_state, pairwise_state, record_time