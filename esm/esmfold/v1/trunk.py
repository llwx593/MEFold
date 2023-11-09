# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as T
from contextlib import ExitStack
from dataclasses import dataclass

import torch
import torch.nn as nn
from openfold.model.structure_module import StructureModule

from esm.esmfold.v1.tri_self_attn_block import TriangularSelfAttentionBlock
import time

import GPUtil
import logging
logging.getLogger().setLevel(logging.INFO)
from openfold.utils.chunk_utils import chunk_layer
from functools import partialmethod, partial
import threading
class MemoryLogger:
    def __init__(self, device_id: int):
        gpu = GPUtil.getGPUs()[device_id]
        logging.info(f'Initializing for device :{device_id} ')
        logging.info(f'Initializing: Used memory = {gpu.memoryUsed}MiB, Free memory = {gpu.memoryFree}MiB, '
                     f'Total memory = {gpu.memoryTotal}MiB')

        self.device_id = device_id
        self.currentMemoryUsed = gpu.memoryUsed
        self.allocatedMemory = gpu.memoryUsed
        self.maxMemory = 0
        self.flag = True
        
    def IntimeRecord(self):
        import pynvml
        pynvml.nvmlInit()
        while True:
            if not self.flag:
                print(f"trunk peak memory {self.maxMemory}")
                break
            handler = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
            total = round(meminfo.total / 1024 / 1024, 2)
            used = round(meminfo.used / 1024 / 1024, 2)
            free = round(meminfo.free / 1024 / 1024, 2)
            if used > self.maxMemory:
                self.maxMemory = used
            time.sleep(0.00000000005)
        
    
    def IntimeRecordThreadStart(self):
        self.recordthread = threading.Thread(target=self.IntimeRecord,)
        self.recordthread.start()
        
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
        
@dataclass
class StructureModuleConfig:
    c_s: int = 384
    c_z: int = 128
    c_ipa: int = 16
    c_resnet: int = 128
    no_heads_ipa: int = 12
    no_qk_points: int = 4
    no_v_points: int = 8
    dropout_rate: float = 0.1
    no_blocks: int = 8
    no_transition_layers: int = 1
    no_resnet_blocks: int = 2
    no_angles: int = 7
    trans_scale_factor: int = 10
    epsilon: float = 1e-8
    inf: float = 1e5


@dataclass
class FoldingTrunkConfig:
    _name: str = "FoldingTrunkConfig"
    num_blocks: int = 48
    sequence_state_dim: int = 1024
    pairwise_state_dim: int = 128
    sequence_head_width: int = 32
    pairwise_head_width: int = 32
    position_bins: int = 32
    dropout: float = 0
    layer_drop: float = 0
    cpu_grad_checkpoint: bool = False

    max_recycles: int = 4
    chunk_size: T.Optional[int] = None

    structure_module: StructureModuleConfig = StructureModuleConfig()


def get_axial_mask(mask):
    """
    Helper to convert B x L mask of valid positions to axial mask used
    in row column attentions.

    Input:
      mask: B x L tensor of booleans

    Output:
      mask: B x L x L tensor of booleans
    """

    if mask is None:
        return None
    assert len(mask.shape) == 2
    batch_dim, seq_dim = mask.shape
    m = mask.unsqueeze(1).expand(batch_dim, seq_dim, seq_dim)
    m = m.reshape(batch_dim * seq_dim, seq_dim)
    return m


class RelativePosition(nn.Module):
    def __init__(self, bins, pairwise_state_dim):
        super().__init__()
        self.bins = bins

        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        self.embedding = torch.nn.Embedding(2 * bins + 2, pairwise_state_dim)

    def forward(self, residue_index, mask=None):
        """
        Input:
          residue_index: B x L tensor of indices (dytpe=torch.long)
          mask: B x L tensor of booleans

        Output:
          pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """

        assert residue_index.dtype == torch.long
        if mask is not None:
            assert residue_index.shape == mask.shape

        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # Add 1 to adjust for padding index.

        if mask is not None:
            mask = mask[:, None, :] * mask[:, :, None]
            diff[mask == False] = 0

        output = self.embedding(diff)
        return output


class FoldingTrunk(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = FoldingTrunkConfig(**kwargs)
        assert self.cfg.max_recycles > 0

        c_s = self.cfg.sequence_state_dim
        c_z = self.cfg.pairwise_state_dim

        assert c_s % self.cfg.sequence_head_width == 0
        assert c_z % self.cfg.pairwise_head_width == 0
        block = TriangularSelfAttentionBlock

        self.pairwise_positional_embedding = RelativePosition(self.cfg.position_bins, c_z)

        self.blocks = nn.ModuleList(
            [
                block(
                    sequence_state_dim=c_s,
                    pairwise_state_dim=c_z,
                    sequence_head_width=self.cfg.sequence_head_width,
                    pairwise_head_width=self.cfg.pairwise_head_width,
                    dropout=self.cfg.dropout,
                )
                for i in range(self.cfg.num_blocks)
            ]
        )

        self.recycle_bins = 15
        self.recycle_s_norm = nn.LayerNorm(c_s)
        self.recycle_z_norm = nn.LayerNorm(c_z)
        self.recycle_disto = nn.Embedding(self.recycle_bins, c_z)
        self.recycle_disto.weight[0].detach().zero_()

        self.structure_module = StructureModule(**self.cfg.structure_module)  # type: ignore
        self.trunk2sm_s = nn.Linear(c_s, self.structure_module.c_s)
        self.trunk2sm_z = nn.Linear(c_z, self.structure_module.c_z)

        self.chunk_size = self.cfg.chunk_size

    def set_chunk_size(self, chunk_size):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        self.chunk_size = chunk_size

    
    def forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles: T.Optional[int] = None, memory_logger : MemoryLogger = None) :
        """
        Inputs:
          seq_feats:     B x L x C            tensor of sequence features
          pair_feats:    B x L x L x C        tensor of pair features
          residx:        B x L                long tensor giving the position in the sequence
          mask:          B x L                boolean tensor indicating valid residues

        Output:
          predicted_structure: B x L x (num_atoms_per_residue * 3) tensor wrapped in a Coordinates object
        """
        print(f"!!! chunk_size={self.chunk_size}")
        device = seq_feats.device
        s_s_0 = seq_feats
        s_z_0 = pair_feats

        if no_recycles is None:
            no_recycles = self.cfg.max_recycles
        else:
            assert no_recycles >= 0, "Number of recycles must not be negative."
            no_recycles += 1  # First 'recycle' is just the standard forward pass through the model.

        def trunk_iter(s, z, residx, mask, memory_logger):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)
            record_time=[0 for i in range(10)]
            linear_init_time=0
            for block in self.blocks:
                s, z,r_t = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size, memory_logger=memory_logger)
                record_time = [record_time[i] + r_t[i] for i in range(10)]
                linear_init_time +=block.init_time
            return s, z, record_time

        s_s = s_s_0
        s_z = s_z_0
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)
        assert no_recycles > 0
        print("num_recycle:",no_recycles)
        memory_logger.IntimeRecordThreadStart()
        module_time=[0 for i in range(10)]
        for recycle_idx in range(no_recycles):
            with ExitStack() if recycle_idx == no_recycles - 1 else torch.no_grad():
                # === Recycling ===
                recycle_s = self.recycle_s_norm(recycle_s.detach())
                recycle_z = self.recycle_z_norm(recycle_z.detach())
                recycle_z += self.recycle_disto(recycle_bins.detach())
                s_s, s_z,r_t = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask, memory_logger)
                module_time=[module_time[i]+r_t[i] for i in range(10)]
                structure = self.structure_module(
                    {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                    true_aa,
                    mask.float(),
                )
                recycle_s = s_s
                recycle_z = s_z
                # Distogram needs the N, CA, C coordinates, and bin constants same as alphafold.
                recycle_bins = FoldingTrunk.distogram(
                    structure["positions"][-1][:, :, :3],
                    3.375,
                    21.375,
                    self.recycle_bins,
                )
        memory_logger.IntimeRecordThreadEnd()
        module_time_show=[]
        module_time_show.append(round(module_time[2],4))
        module_time_show.append(round(module_time[5],4))
        module_time_show.append(round(module_time[6],4))
        module_time_show.append(round(module_time[7],4))
        module_time_show.append(round(module_time[8],4))
        module_time_show.append(round(module_time[9],4))
        print(module_time_show)        
        assert isinstance(structure, dict)  # type: ignore
        structure["s_s"] = s_s
        structure["s_z"] = s_z

        return structure

    @staticmethod
    def distogram(coords, min_bin, max_bin, num_bins):
        # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
        boundaries = torch.linspace(
            min_bin,
            max_bin,
            num_bins - 1,
            device=coords.device,
        )
        boundaries = boundaries**2
        N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]
        # Infer CB coordinates.
        b = CA - N
        c = C - CA
        a = b.cross(c, dim=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        dists = (CB[..., None, :, :] - CB[..., :, None, :]).pow(2).sum(dim=-1, keepdims=True)
        bins = torch.sum(dists > boundaries, dim=-1)  # [..., L, L]
        return bins
    
    def _distogram(self, coords, min_bin, max_bin, num_bins):
        # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
        boundaries = torch.linspace(
            min_bin,
            max_bin,
            num_bins - 1,
            device=coords.device,
        )
        boundaries = boundaries**2
        N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]
        # Infer CB coordinates.
        b = CA - N
        c = C - CA
        a = b.cross(c, dim=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        dists = (CB[..., None, :, :] - CB[..., :, None, :]).pow(2).sum(dim=-1, keepdims=True)
        bins = torch.sum(dists > boundaries, dim=-1)  # [..., L, L]
        return bins
    
    @torch.jit.ignore
    def _distogram_chunk(self,
        x: torch.Tensor,
        chunk_size: int,
        min_bin: float, 
        max_bin: float,
        num_bins: int,
    ) -> torch.Tensor:
        return chunk_layer(
            partial(
                self._distogram,
                min_bin = min_bin,
                max_bin = max_bin,
                num_bins = num_bins
                
            ),
            {"coords": x,
             },
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]), 
        )

