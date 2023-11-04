import torch
import esm
import typing as T

from esm.esmfold.v1.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
    output_to_pdb,
)

def pdbstr2tensor(sequences: T.Union[str, T.List[str]],
        device_info,
        residx=None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = None,
        residue_index_offset: T.Optional[int] = 512,
        chain_linker: T.Optional[str] = "G" * 25,):
    if isinstance(sequences, str):
        sequences = [sequences]

    aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
        sequences, residue_index_offset, chain_linker
    )

    if residx is None:
        residx = _residx
    elif not isinstance(residx, torch.Tensor):
        residx = collate_dense_tensors(residx)

    aatype, mask, residx, linker_mask = map(
        lambda x: x.to(device_info), (aatype, mask, residx, linker_mask)
    )

    return aatype, mask, residx, linker_mask, chain_index

def out_to_pdb(output, linker_mask, chain_index):
    output["atom37_atom_exists"] = output["atom37_atom_exists"] * linker_mask.unsqueeze(2)

    output["mean_plddt"] = (output["plddt"] * output["atom37_atom_exists"]).sum(
        dim=(1, 2)
    ) / output["atom37_atom_exists"].sum(dim=(1, 2))
    output["chain_index"] = chain_index
    out_lst = output_to_pdb(output)
    return out_lst[0]