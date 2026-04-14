import os
import json
import math
import bisect

import numpy as np
import torch
import torch.nn.functional as F

from grn.utils_t2iv.hbq_util_t2iv import multiclass_labels2onehot_input

def get_scale_pack_info(scale_schedule, first_full_spatial_size_scale_index, args):
    meta = {}
    sid2clipid_innsid = {}
    clipid_innsid2sid = {}
    scales_per_clip = first_full_spatial_size_scale_index + 1
    total_clips = len(scale_schedule) // scales_per_clip
    for si in range(len(scale_schedule)):
        clipid = si // scales_per_clip
        frame_ss, frame_ee = 0, scale_schedule[-1][0]
        sid2clipid_innsid[si] = (clipid, si % scales_per_clip)
        clipid_innsid2sid[(clipid, si % scales_per_clip)] = si
        meta[si] = {
            'clipid': clipid,
            'frame_ss': frame_ss,
            'frame_ee': frame_ee,
            'ref_sids': [],
        }
    return meta

def flatten_two_level_list(two_level_list):
    flatten_list = []
    for item in two_level_list:
        flatten_list.extend(item)
    return flatten_list

def shift_pt(pt, alpha):
    """shift pt (signal ratio) to lower one, recommand alpha=sqrt(height*width/256/256)"""
    if alpha > 1000:
        alpha = alpha - 1000
    noise_pt = 1 - pt
    noise_pt = alpha * noise_pt / (1+(alpha-1)*noise_pt) # shift noise_pt to higer one
    pt = 1 - noise_pt
    return pt

def video_encode(
    vae,
    inp_B3HW,
    vae_features=None,
    device='cuda',
    args=None,
    infer_mode=False,
    rope2d_freqs_grid=None,
    dynamic_resolution_h_w=None,
    tokens_remain=9999999,
    text_lens=[],
    caption_nums=[],
    rank_vary_generator=None,
    vis_verbose=False,
    meta_list=None,
    **kwargs,
):
    if rank_vary_generator is not None:
        numpy_generator = rank_vary_generator['numpy_generator']
        torch_cuda_generator = rank_vary_generator['torch_cuda_generator']
    else:
        numpy_generator = np.random.default_rng()
        torch_cuda_generator = torch.Generator(device='cuda')
    
    if vae_features is None:
        raw_features, _, _ = vae.encode_for_raw_features(inp_B3HW, scale_schedule=None, slice=True)
        raw_features_list = [raw_features]
        x_recon_raw = vae.decode(raw_features[-1], slice=True)
        x_recon_raw = torch.clamp(x_recon_raw, min=-1, max=1)
        print(f'raw_features[-1].shape: {raw_features[-1].shape}')
    else:
        raw_features_list = vae_features
    # raw_features_list: list of [1,d,t,h,w]:
    # import pdb; pdb.set_trace()
    gt_all_bit_indices = []
    pred_all_bit_indices = []
    var_input_list = []
    sequece_packing_scales = [] # with trunk
    h_div_w_template_list = np.array(list(dynamic_resolution_h_w.keys()))
    visual_rope_cache_list = []
    other_info_by_scale = []
    scale_pack_info_list = [] # per example, not per scale
    tokens_remain = tokens_remain-sum(text_lens)
    with torch.amp.autocast('cuda', enabled = False):
        for example_ind, raw_features in enumerate(raw_features_list):
            gt_all_bit_indices.append([])
            pred_all_bit_indices.append([])
            var_input_list.append([])
            visual_rope_cache_list.append([])
            other_info_by_scale.append([])
            B, C, T, H, W = raw_features[-1].shape
            h_div_w = H / W
            if not infer_mode and args.add_class_token > 0 and numpy_generator.random() > args.drop_condition_prob:
                class_token_id = int(meta_list[example_ind]['digit'])
            else:
                class_token_id = 1000
            mapped_h_div_w_template = h_div_w_template_list[np.argmin(np.abs(h_div_w-h_div_w_template_list))]
            min_t = min(dynamic_resolution_h_w[mapped_h_div_w_template][args.pn]['pt2scale_schedule'].keys())
            image_scale_schedule = dynamic_resolution_h_w[mapped_h_div_w_template][args.pn]['pt2scale_schedule'][min_t]
            scale_schedule = dynamic_resolution_h_w[mapped_h_div_w_template][args.pn]['pt2scale_schedule'][T]
            si = 0
            next_tokens_remain = tokens_remain - np.array(scale_schedule[si]).prod() - args.add_scale_token - args.add_class_token
            if next_tokens_remain < 0:
                break
            tokens_remain = next_tokens_remain
            vae_scale_schedule = scale_schedule
            first_full_spatial_size_scale_index = len(image_scale_schedule) - 1
            scale_pack_info = get_scale_pack_info(vae_scale_schedule, first_full_spatial_size_scale_index, args)
            scale_pack_info_list.append(scale_pack_info)
            preserve_scale_schedule = []
            assert len(vae_scale_schedule) == 1
            pt, ph, pw = vae_scale_schedule[0]
            preserve_scale_schedule.append((pt, ph, pw))
            assert len(scale_pack_info[si]['ref_sids']) == 0, f'current not support scale dependence because we pick scales for one example'

            target = raw_features[0]
            
            # sample noise proportion
            if not infer_mode and args.log_norm_sigma > 0:
                pt = torch.sigmoid(torch.randn(1, generator=torch_cuda_generator, device=target.device) * args.log_norm_sigma + args.log_norm_mean).item()
                pt = shift_pt(pt, args.alpha)
            else:
                pt = shift_pt(numpy_generator.random(), args.alpha)
            
            if args.refine_mode in ['ar_discrete_GRN_ind']:
                from grn.utils_t2iv.hbq_util_t2iv import raw_feature2index_label
                labels = raw_feature2index_label(target, hbq_round=args.hbq_round) # [B, hbq_round * d, t, h, w]
                classes = 2**args.hbq_round
            elif args.refine_mode in ['ar_discrete_GRN_bit']:
                from grn.utils_t2iv.hbq_util_t2iv import raw_feature2bit_label
                labels = raw_feature2bit_label(target, hbq_round=args.hbq_round) # [B, hbq_round * d, t, h, w]
                classes = 2
                
            random_labels = torch.randint(0, classes, size=labels.shape, generator=torch_cuda_generator, device=labels.device, dtype=labels.dtype) # random 0 or 1 labels
            random_mask = torch.rand(size=labels.shape, generator=torch_cuda_generator, device=labels.device, dtype=target.dtype) < pt
            mixed_xt = torch.where(random_mask, labels, random_labels) # [B, hbq_round * d, t, h, w]
            wandb_plot_index = min(9, int(pt / 0.1)) # 0~9
            this_scale_var_input = multiclass_labels2onehot_input(mixed_xt, classes) # [B,hbq_round * d * 2,t,h,w]
            scale_token_id = pt # 0~1, float
            indices = labels.type(torch.long).permute(0,2,3,4,1) # [B,d,t,h,w] -> [B,t,h,w,d]
            if not infer_mode:
                visual_rope_cache_list[-1].append(get_visual_rope_embeds(rope2d_freqs_grid, scale_schedule, si, 0, device, args, scale_pack_info, mapped_h_div_w_template))
            var_input_list[-1].append(this_scale_var_input)
            gt_all_bit_indices[-1].append(indices)
            other_info_by_scale[-1].append(
                {
                    'cur_scale': (pt, ph, pw), 
                    'largest_scale': scale_schedule[-1], 
                    'wandb_plot_index': wandb_plot_index, 
                    'is_semantic_scale': True,
                    'cur_bits': indices.shape[-1],
                    'cur_lvl': args.detail_num_lvl,
                    'scale_token_id': scale_token_id,
                    'class_token_id': class_token_id,
                }
            )
            sequece_packing_scales.append(preserve_scale_schedule)

    flatten_packing_scales = flatten_two_level_list(sequece_packing_scales)
    gt_all_bit_indices = flatten_two_level_list(gt_all_bit_indices)
    pred_all_bit_indices = flatten_two_level_list(pred_all_bit_indices)
    var_input_list = flatten_two_level_list(var_input_list)
    visual_rope_cache_list = flatten_two_level_list(visual_rope_cache_list)
    other_info_by_scale = flatten_two_level_list(other_info_by_scale)

    if infer_mode:
        return [labels, target], x_recon_raw, [target], None, None, scale_pack_info
    # set scale_lengths and querysid_refsid

    assert args.add_scale_token >= 0 and args.add_class_token >= 0, f'{args.add_scale_token=} {args.add_class_token=}'
    scale_lengths = [ pt * ph * pw + args.add_scale_token + args.add_class_token for pt,ph,pw in flatten_packing_scales]
    scale_lengths = scale_lengths + text_lens
    valid_scales = len(scale_lengths)
    pad_seq_len = args.train_max_token_len - np.sum(scale_lengths)
    assert pad_seq_len >= 0, f'pad_seq_len: {pad_seq_len} < 0, {scale_lengths=}'
    if pad_seq_len:
        scale_lengths = scale_lengths + [pad_seq_len]
    
    # update attention mask
    max_sid_nums = 2000
    querysid_refsid = torch.zeros((max_sid_nums, max_sid_nums), device=args.device, dtype=torch.bool) # Attention! this shape should be the same for different iterations !!!
    for i in range(valid_scales):
        querysid_refsid[i][i] = True
    base = 0
    for ind, scale_schedule in enumerate(sequece_packing_scales):
        scale_pack_info = scale_pack_info_list[ind]
        for local_querysid in range(len(scale_schedule)):
            global_querysid = local_querysid + base
            if args.add_class_token > 0: # class condition
                pass
            else: # text condition
                global_text_sid = len(flatten_packing_scales) + ind
                querysid_refsid[global_querysid][global_text_sid] = True
        base += len(scale_schedule)

    gt_ms_idx_Bl = []
    for item in gt_all_bit_indices:
        _, tt, hh, ww, dd = item.shape
        item = item.reshape(B, tt*hh*ww, dd)
        gt_ms_idx_Bl.append(item)
    gt_BLC = gt_ms_idx_Bl # torch.cat(gt_ms_idx_Bl, 1).contiguous().type(torch.long)
    for i in range(len(var_input_list)):
        # (B,d,t,H,W) -> (B,t,H,W,d)
        dim = var_input_list[i].shape[1]
        var_input_list[i] = var_input_list[i].permute(0,2,3,4,1)
        var_input_list[i] = var_input_list[i].reshape(B, -1, dim)
    x_BLC = var_input_list
    x_BLC_mask = None
    scale_or_time_ids = None
    return x_BLC, x_BLC_mask, scale_or_time_ids, gt_BLC, pred_all_bit_indices, visual_rope_cache_list, sequece_packing_scales, scale_lengths, querysid_refsid, other_info_by_scale, pad_seq_len

def video_decode(
    vae,
    all_indices,
    scale_schedule,
    label_type,
    args=None,
    noise_list=None,
    trunc_scales=-1,
    **kwargs,
):  
    if trunc_scales < 0:
        summed_codes = all_indices[-1]
    else:
        summed_codes = all_indices[trunc_scales-1]
    x_recon = vae.decode(summed_codes, slice=True)
    x_recon = torch.clamp(x_recon, min=-1, max=1)
    x_recon_256 = None
    return x_recon, x_recon_256

def get_visual_rope_embeds(rope2d_freqs_grid, scale_schedule, sid, pos_id, device=None, args=None, scale_pack_info=None, mapped_h_div_w_template=None):
    if args.rope_type == '3d':
        # freqs_frames: (2, max_frames, dim_div_2 / 3)
        rope2d_freqs_grid['freqs_frames'] = rope2d_freqs_grid['freqs_frames'].to(device)
        rope2d_freqs_grid['freqs_height'] = rope2d_freqs_grid['freqs_height'].to(device)
        rope2d_freqs_grid['freqs_width'] = rope2d_freqs_grid['freqs_width'].to(device)
        max_height = rope2d_freqs_grid['freqs_height'].shape[1]
        max_width = rope2d_freqs_grid['freqs_width'].shape[1]
        extreme_h_div_w = 3
        assert mapped_h_div_w_template <= extreme_h_div_w
        extreme_h = max_height
        extreme_w = extreme_h / extreme_h_div_w
        upw = np.sqrt(extreme_h * extreme_w / mapped_h_div_w_template)
        uph = mapped_h_div_w_template * upw
        uph, upw = int(uph), int(upw)
        pt, ph, pw = scale_schedule[sid]
        assert ph <= uph and pw <= upw
        frame_ss, frame_ee = scale_pack_info[sid]['frame_ss'], scale_pack_info[sid]['frame_ee']
        f_frames = rope2d_freqs_grid['freqs_frames'][:, frame_ss:frame_ee]
        f_height = rope2d_freqs_grid['freqs_height'][:, (torch.arange(ph) * (uph / ph)).round().int()]
        f_width = rope2d_freqs_grid['freqs_width'][:, (torch.arange(pw) * (upw / pw)).round().int()]
        rope_embeds = torch.cat([
            f_frames[   :,     :,  None,   None,   :].expand(-1, -1, ph, pw, -1),
            f_height[   :,  None,      :,  None,   :].expand(-1,  pt,-1, pw, -1),
            f_width[   :,  None,   None,      :,   :].expand(-1,  pt,ph, -1, -1),
        ], dim=-1)  # (2, pt, ph, pw, dim_div_2)
        rope_embeds = rope_embeds.reshape(2, 1, 1, 1, pt*ph*pw, -1)  # (2, 1, 1, 1, pt*ph*pw, dim_div_2)
    else:
        raise ValueError(f'rope_type {args.rope_type} not supported')
    return rope_embeds
