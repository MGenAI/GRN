from functools import partial
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def _length_to_offsets(lengths, device):
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets

def _offsets_to_doc_ids_tensor(offsets):
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    visual = torch.repeat_interleave(torch.arange(len(counts), device=device, dtype=torch.int32), counts)
    return visual

def _generate_overall_mask(offsets, querysid_refsid):
    document_id = _offsets_to_doc_ids_tensor(offsets) # to scale_ind
    def overall_mask(b, h, q_idx, kv_idx):
        querysid = document_id[q_idx]
        kv_sid = document_id[kv_idx]
        return querysid_refsid[querysid][kv_sid]
    return overall_mask

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def build_flex_attn_func(
        flex_attention,
        seq_l,
        prefix_lens,
        args,
        device,
        batch_size,
        heads,
        pad_seq_len,
        sequece_packing_scales,
        super_scale_lengths,
        super_querysid_super_refsid,
):
    """
    Build a flex attn function for a given scale schedule.
    Args:
        flex_attention: compiled flex attention
        seq_l: seq length
        prefix_lens: valid text prefix lens, [bs]
        args: arguments
        device: device
        batch_size: batch size
        heads: heads
        pad_seq_len: pad_seq_len
        sequece_packing_scales: list of scale schedule
        querysid_refsid: list of scale_pack_info
    Returns:
        attn_fn: flex attn function
    """
    assert sum(super_scale_lengths) == seq_l, f'{sum(super_scale_lengths)}!= {seq_l}'
    offsets = _length_to_offsets(super_scale_lengths, device=device)
    mask_mod = _generate_overall_mask(offsets, super_querysid_super_refsid)
    block_mask = create_block_mask(mask_mod, B = batch_size, H = heads, Q_LEN = seq_l, KV_LEN = seq_l, device = device, _compile = True)
    attn_fn = partial(flex_attention, block_mask=block_mask)
    return attn_fn
