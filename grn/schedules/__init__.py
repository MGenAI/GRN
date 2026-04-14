def get_encode_decode_func(dynamic_scale_schedule):
    if 'GRN_vae_stride16' in dynamic_scale_schedule:
        from grn.schedules.global_refine import video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info
    else:
        raise NotImplementedError(f'{dynamic_scale_schedule} is unsupported')
    return video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info
