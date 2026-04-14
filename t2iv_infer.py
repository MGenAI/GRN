import os
import sys
import json
import random
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from grn.utils_t2iv.infer import *
from grn.utils_t2iv.load import load_visual_tokenizer
from grn.utils.video_decoder import EncodedVideoOpencv
from grn.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from grn.schedules import get_encode_decode_func
from grn.dataset.dataset_joint_vi import local_or_download


def get_default_config():
    """Builds and returns the global configuration namespace."""
    parser = argparse.ArgumentParser(description="T2IV Inference Script")
    
    # Core hyperparams
    parser.add_argument('--pn', type=str, default='1M', help='the total number of pixels per generated frame')
    parser.add_argument('--video_frames', type=int, default=209, help='Number of video frames')
    parser.add_argument('--model_path', type=str, default='/mnt/bn/foundation-ads/hanjian.thu123/GRN/weights/9a8a674133266e996d8d56e784a10d67.pth', help='Path to model checkpoint')
    parser.add_argument('--vae_path', type=str, default='/dev/shm/vae_e04839d1c0db284ae34b40811fc20ab4.ckpt', help='Path to VAE checkpoint')
    parser.add_argument('--text_encoder_ckpt', type=str, default='/dev/shm/umt5-xxl', help='Path to text encoder checkpoint')
    parser.add_argument('--test_training_prompt', type=str, default='1', help='Prompt type for inference (1, 4, 5, 6, vbench, geneval)')
    parser.add_argument('--cfg', type=int, default=1)

    # Other parameters
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_latent_dim', type=int, default=64)
    parser.add_argument('--hbq_round', type=int, default=4)
    parser.add_argument('--rope_type', type=str, default='3d')
    parser.add_argument('--num_lvl', type=int, default=2)
    parser.add_argument('--model', type=str, default='GRN2b')
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2)
    parser.add_argument('--sampling_per_bits', type=int, default=1)
    parser.add_argument('--text_channels', type=int, default=4096)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0)
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--cache_dir', type=str, default='/tmp')
    parser.add_argument('--checkpoint_type', type=str, default='torch')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', type=int, default=0)
    parser.add_argument('--dynamic_scale_schedule', type=str, default='GRN_vae_stride16')
    parser.add_argument('--train_h_div_w_list', type=str, default='[]')
    parser.add_argument('--max_infer_steps', type=int, default=50)
    parser.add_argument('--min_infer_steps', type=int, default=50)
    parser.add_argument('--video_caption_type', type=str, default='tarsier2_caption')
    parser.add_argument('--temporal_compress_rate', type=int, default=4)
    parser.add_argument('--cached_video_frames', type=int, default=81)
    parser.add_argument('--duration_resolution', type=float, default=0.25)
    parser.add_argument('--video_fps', type=int, default=16)
    parser.add_argument('--simple_text_proj', type=int, default=1)
    parser.add_argument('--min_duration', type=int, default=-1)
    parser.add_argument('--fsdp_save_flatten_model', type=int, default=1)
    parser.add_argument('--use_learnable_dim_proj', type=int, default=0)
    parser.add_argument('--use_fsq_cls_head', type=int, default=1)
    parser.add_argument('--use_feat_proj', type=int, default=0)
    parser.add_argument('--use_clipwise_caption', type=int, default=0)
    parser.add_argument('--use_ada_layer_norm', type=int, default=0)
    parser.add_argument('--cfg_type', type=str, default='cfg_interval_0.0')
    parser.add_argument('--add_scale_token', type=int, default=1)
    parser.add_argument('--vae_encoder_out_type', type=str, default='feature_tanh')
    parser.add_argument('--alpha', type=int, default=1004)
    parser.add_argument('--refine_mode', type=str, default='ar_discrete_GRN_bit')
    parser.add_argument('--add_class_token', type=int, default=0)
    parser.add_argument('--resample_rand_labels_per_step', type=int, default=0)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--cfg_val', type=float, default=3.0)
    parser.add_argument('--taui', type=float, default=1.0)
    parser.add_argument('--tauv', type=float, default=1.0)
    parser.add_argument('--scale_repetition', type=str, default='')
    parser.add_argument('--gt_leak', type=int, default=-1)
    parser.add_argument('--use_refined_prompt', type=str, default=None)
    parser.add_argument('--use_prompt_engineering', type=int, default=0)
    parser.add_argument('--quality_prompt', type=str, default='')
    parser.add_argument('--train_split_file', type=str, default='./data/infinity_toy_data/splits/1.000_000002500.jsonl')
    parser.add_argument('--n_sampes', type=int, default=1)
    parser.add_argument('--repeat_times', type=int, default=30)
    
    args = parser.parse_args()
    
    # Derived parameters
    args.max_duration = (args.video_frames - 1) / 4
    args.image_scale_repetition = json.dumps([args.repeat_times] * 1)
    args.video_scale_repetition = args.image_scale_repetition
    args.video_scale_probs = [1.0 for _ in json.loads(args.image_scale_repetition)]
    args.num_of_label_value = args.num_lvl
    args.semantic_num_lvl = args.num_lvl
    args.detail_num_lvl = args.num_lvl
    args.semantic_scale_dim = args.vae_latent_dim
    args.detail_scale_dim = args.vae_latent_dim
    args.infer_name = f""
    
    # Convert test_training_prompt to int if it's digit
    if args.test_training_prompt.isdigit():
        args.test_training_prompt = int(args.test_training_prompt)
        
    return args


def get_save_dir_root(args, test_training_prompt):
    """Determines the root directory for saving outputs."""
    if 'ema_weights' in args.model_path:
        base_dir = os.path.join('tmp_videos', '/'.join(args.model_path.rstrip('/').split('/')[-3:]))
    else:
        base_dir = os.path.join('tmp_videos', os.path.basename(os.path.dirname(args.model_path)), os.path.basename(args.model_path))

    if test_training_prompt == 1:
        synthetic = 1 if 'synthetic' in args.train_split_file else 0
        folder = f'train_{args.infer_name}_pn{args.pn}_fps{args.fps}_elegant_overfit100_synthetic_{synthetic}_rep_{args.scale_repetition}_vf{args.video_frames}_use_cfg_{args.cfg}_cfg{args.cfg_val}_taui{args.taui:.1f}_tauv{args.tauv:.1f}_gt_leak_{args.gt_leak}'
    elif test_training_prompt == 4:
        folder = f'val_{args.infer_name}_tarsier2_pn{args.pn}_fps{args.fps}_elegant_bugfix_rep_{args.scale_repetition}_vf{args.video_frames}_use_cfg_{args.cfg}_cfg{args.cfg_val}_taui{args.taui:.1f}_tauv{args.tauv:.1f}_gt_leak_{args.gt_leak}'
    elif test_training_prompt == 5:
        folder = f'training_fps{args.fps}_video_frames{args.video_frames}_high_motion_cfg{args.cfg_val}_taui{args.taui:.1f}_tauv{args.tauv:.1f}'
    elif test_training_prompt == 6:
        folder = f'training_fps{args.fps}_video_frames{args.video_frames}_movement_gt_leak_{args.gt_leak}_cfg{args.cfg_val}_taui{args.taui:.1f}_tauv{args.tauv:.1f}'
    else:
        folder = f'val_{args.infer_name}_cfg{args.cfg_val}_taui{args.taui:.1f}_tauv{args.tauv:.1f}_{args.use_refined_prompt=}'

    return os.path.join(base_dir, folder)


def load_dataset(test_training_prompt, args, save_dir_root):
    """Loads and formats the dataset based on the prompt type."""
    data = []
    
    if test_training_prompt == 'vbench':
        with open('./self_attn/vgpt/data/vbench_refine_prompt_v2.json', 'r') as f:
            full_info_list = json.load(f)
            for prompt_dict in full_info_list:
                prompt = prompt_dict['prompt_en']
                refined_prompt = prompt_dict['refined_prompt']
                if not refined_prompt:
                    continue
                for dimension in prompt_dict["dimension"]:
                    repeat = 25 if dimension == 'temporal_flickering' else 5
                    for index in range(repeat):
                        data.append({
                            'prompt': prompt,
                            'refined_prompt': refined_prompt,
                            'dimension': dimension,
                            'h_div_w': 0.5625,
                            'num_samples': 1,
                            'index': index,
                        })
        np.random.shuffle(data)
        
    elif test_training_prompt == 'geneval':
        prompt_cache_file = './self_attn/vgpt/evaluation/gen_eval/prompt_rewrite_cache_ursa.json' if args.use_refined_prompt == 'ursa' else './self_attn/vgpt/evaluation/gen_eval/prompt_rewrite_cache.json'
        with open(prompt_cache_file, 'r') as f:
            prompt_rewrite_cache = json.load(f)
        
        with open('./self_attn/vgpt/evaluation/gen_eval/prompts/evaluation_metadata.jsonl') as fp:
            for index, line in enumerate(fp):
                metadata = json.loads(line)
                outpath = os.path.join(save_dir_root, 'images', f"{index:0>5}")
                os.makedirs(outpath, exist_ok=True)
                
                prompt = metadata['prompt']
                refined_prompt = prompt_rewrite_cache[prompt]
                sample_path = os.path.join(outpath, "samples")
                os.makedirs(sample_path, exist_ok=True)
                
                with open(os.path.join(outpath, "metadata.jsonl"), "w") as mfp:
                    json.dump(metadata, mfp)
                    
                data.append({
                    'prompt': prompt,
                    'refined_prompt': refined_prompt,
                    'h_div_w': 1.0,
                    'num_samples': 4,
                    'save_dir': sample_path,
                })
        np.random.shuffle(data)
        
    elif isinstance(test_training_prompt, int):
        with open(args.train_split_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
                if len(data) >= 5000:
                    break
    return data


def setup_prompt(meta, args, test_training_prompt):
    """Extracts and refines the text prompt from the metadata."""
    if test_training_prompt == 'vbench':
        prompt = meta['refined_prompt']
    elif test_training_prompt == 'geneval':
        prompt = meta['refined_prompt'] if args.use_refined_prompt else meta['prompt']
    else:
        if args.video_caption_type in meta:
            prompt = meta[args.video_caption_type]
        elif meta.get('long_caption'):
            prompt = meta['long_caption']
        elif 'text' in meta:
            prompt = meta['text']
        else:
            prompt = ""

    if test_training_prompt == 1 and ('quality_prompt' in meta):
        prompt += ' ' + meta['quality_prompt']
    if len(args.quality_prompt):
        prompt += ' ' + args.quality_prompt

    if args.use_prompt_engineering:
        from tools.t2v_prompt_rewriter import PromptRewriter
        rewriter = PromptRewriter(system='', few_shot_history=[])
        rewrited_prompt = rewriter.rewrite(prompt).strip('.')
        prompt = f'{rewrited_prompt}. The quality is very high!'
        
    return prompt


def process_video_or_image(meta, args, test_training_prompt, h_div_w_templates, dynamic_resolution_h_w):
    """Loads video or image and determines the scale schedule."""
    video_frames = args.video_frames if test_training_prompt not in ['geneval'] else 1
    
    if test_training_prompt in [1, 3, 4, 5, 6]:
        if 'video_path' in meta:
            ext = '.mp4'
            begin_frame_id = meta['begin_frame_id']
            end_frame_id = meta['end_frame_id']
            duration = (end_frame_id - begin_frame_id) / meta.get('fps', 16)
            mapped_duration = int(np.round(duration / args.duration_resolution)) * args.duration_resolution
            
            if mapped_duration < args.min_duration or mapped_duration > args.max_duration:
                return None
                
            local_path = local_or_download(meta)
            try:
                video = EncodedVideoOpencv(local_path, os.path.basename(local_path), num_threads=0)
            except Exception as e:
                print(f"Error loading video {local_path}: {e}")
                return None
                
            num_frames = min(video_frames, int(mapped_duration * args.video_fps + 1))
            start_interval = max(0, begin_frame_id / video._fps)
            end_interval = start_interval + (num_frames - 1) / args.video_fps
            
            try:
                raw_video, _ = video.get_clip(start_interval, end_interval, num_frames)
            except Exception as e:
                print(f"Error extracting clip from {local_path}: {e}")
                return None
                
            _, h, w, c = raw_video.shape
        else:
            local_path = meta['image_path']
            raw_video = [cv2.imread(local_path)]
            h, w, c = raw_video[0].shape
            num_frames = 1
            ext = '.jpg'
            mapped_duration = 0
            
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h / w))]
        args.mapped_h_div_w_template = h_div_w_template_
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames - 1) // 4 + 1]
        
        return raw_video, scale_schedule, num_frames, mapped_duration, ext, local_path

    # For pure generation without ground truth
    h_div_w = meta.get('h_div_w', 1.0 if test_training_prompt == 'geneval' else 0.571)
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    args.mapped_h_div_w_template = h_div_w_template_
    mapped_duration = 5
    num_frames = min(video_frames, mapped_duration * args.video_fps + 1)
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames - 1) // 4 + 1]
    ext = '.jpg' if num_frames == 1 else '.mp4'
    
    return None, scale_schedule, num_frames, mapped_duration, ext, ""


def main():
    args = get_default_config()
    test_training_prompt = args.test_training_prompt
    print(f'[test_training_prompt] is {test_training_prompt}')
    
    video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)
    args.other_device = 'cuda'
    
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    model_test = load_transformer(vae, args)
    
    save_dir_root = get_save_dir_root(args, test_training_prompt)
    dataset = load_dataset(test_training_prompt, args, save_dir_root)
    
    save_file_meta = os.path.join(save_dir_root, 'meta.jsonl')
    os.makedirs(os.path.dirname(save_file_meta), exist_ok=True)
    save_file_obj = open(save_file_meta, 'w')
    
    print(args)
    
    for i, meta in enumerate(dataset):
        prompt = setup_prompt(meta, args, test_training_prompt)
        
        dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.train_h_div_w_list, args.video_frames)
        
        media_info = process_video_or_image(meta, args, test_training_prompt, h_div_w_templates, dynamic_resolution_h_w)
        if media_info is None and test_training_prompt in [1, 3, 4, 5, 6]:
            continue
            
        raw_video, scale_schedule, num_frames, mapped_duration, ext, local_path = media_info if media_info else (None, None, 1, 5, '.mp4', "")
        
        args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
        args.tower_split_index = args.first_full_spatial_size_scale_index + 1
        context_info = get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)
        
        noise_list = None
        gt_ls_Bl = None
        recons_video_compose = None
        recons_video_raw = None
        recons_video = None
        img_T3HW = None
        
        if test_training_prompt in [1, 3, 4, 5, 6] and raw_video is not None:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule] if args.apply_spatial_patchify else scale_schedule
            vae_stride = int(args.dynamic_scale_schedule.split('vae_stride')[-1]) if 'vae_stride' in args.dynamic_scale_schedule else 16
            tgt_h, tgt_w = scale_schedule[-1][1] * vae_stride, scale_schedule[-1][2] * vae_stride
            
            img_T3HW_list = [transform(Image.fromarray(frame[:,:,::-1]), tgt_h, tgt_w) for frame in raw_video]
            img_T3HW_t = torch.stack(img_T3HW_list, 0)
            img_bcthw = img_T3HW_t.permute(1,0,2,3).unsqueeze(0)
            
            img_T3HW = img_T3HW_t.permute(0,2,3,1)
            img_T3HW = (img_T3HW+1)/2
            img_T3HW = img_T3HW.mul_(255).to(torch.uint8).flip(dims=(3,)).cpu().numpy()
            
            noise_list, recons_video_raw, all_bit_indices, _, _, context_info = video_encode(
                vae, img_bcthw.to(args.other_device), vae_features=None, self_correction=None, 
                args=args, infer_mode=True, dynamic_resolution_h_w=dynamic_resolution_h_w
            )
            torch.cuda.empty_cache()
            
            recons_video = recons_video_raw
            gt_ls_Bl = all_bit_indices
            
        # Generation paths setup
        if test_training_prompt == 'vbench':
            save_dir = os.path.join(save_dir_root, 'vbench_videos')
            save_file_template = os.path.join(save_dir, meta['prompt'])
        elif test_training_prompt == 1:
            item_name = f'{i:03d}_t{mapped_duration:.1f}s_'+prompt.replace(' ', '_')[:40]
            save_dir = os.path.join(save_dir_root, 'test', item_name)
            save_dir_download = os.path.join(save_dir_root, 'test', 'videos')
            save_dir_download_withgt = os.path.join(save_dir_root, 'test', 'videos_with_gt')
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(save_dir_download, exist_ok=True)
            os.makedirs(save_dir_download_withgt, exist_ok=True)
            
        try:
            vae = vae.float().to('cuda')
        except:
            pass
            
        generated_image_list = []
        n_samples_list = list(range(meta.get('num_samples', args.n_sampes)))
        np.random.shuffle(n_samples_list)
        
        for index in n_samples_list:
            class_token_id = int(meta.get('digit', index))
            
            if test_training_prompt == 'vbench':
                save_file = save_file_template + f'_{meta["index"]}.mp4'
                if os.path.exists(save_file): continue
            elif test_training_prompt == 'geneval':
                save_file = os.path.join(meta['save_dir'], f"{index:05}.jpg")
                if os.path.exists(save_file): continue
                
            args.meta = json.dumps(meta)
            generated_image = gen_one_example(
                model_test, vae, text_tokenizer, text_encoder, [prompt],
                negative_prompt='', g_seed=None, gt_leak=args.gt_leak, gt_ls_Bl=gt_ls_Bl,
                cfg_list=args.cfg_val, tau_list=args.tau, scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer], vae_latent_dim=args.vae_latent_dim,
                sampling_per_bits=args.sampling_per_bits, enable_positive_prompt=0,
                args=args, get_visual_rope_embeds=get_visual_rope_embeds,
                context_info=context_info, noise_list=noise_list, class_token_id=class_token_id,
            )
            
            if len(generated_image.shape) == 3:
                generated_image = generated_image.unsqueeze(0)
                
            if test_training_prompt in ['vbench', 'geneval']:
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
                images2video(generated_image.cpu().numpy(), fps=args.fps, save_filepath=save_file)
                
            generated_image_list.append(generated_image)
            
        torch.cuda.empty_cache()
        if test_training_prompt in ['vbench', 'geneval']:
            continue
            
        if not generated_image_list:
            continue
            
        generated_image = torch.cat(generated_image_list, 2).cpu().numpy()
        images2video(generated_image, fps=args.fps, save_filepath=os.path.join(save_dir, f'./{item_name}_tmp_pred{ext}'))
        suffix = prompt.replace(' ', '_')[:40]
        images2video(generated_image, fps=args.fps, save_filepath=os.path.join(save_dir_download, f'{i:03d}_{index:03d}_{suffix}{ext}'))
        
        entrophy_statistic_file = os.path.join(save_dir, '../../entrophy_statistic.json')
        with open(entrophy_statistic_file, 'w') as f:
            json.dump(model_test.entrophy_statistics, f, indent=4)
            
        with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=4)
            
        if test_training_prompt in [0, 2]:
            continue
            
        if recons_video_raw is not None and recons_video is not None:
            recons_video_raw = imgs_tensor2uint8_imgs(recons_video_raw[0].cpu().data)
            recons_video = imgs_tensor2uint8_imgs(recons_video[0].cpu().data)
            
            if recons_video_compose is not None:
                images2video(imgs_tensor2uint8_imgs(recons_video_compose[0]), fps=args.fps, save_filepath=os.path.join(save_dir, f'./recons_video_compose{ext}'))
            
            _, h, w, c = generated_image.shape
            n_sampes_len = len(n_samples_list)
            recons_video = [cv2.resize(item, (w // n_sampes_len, h)) for item in recons_video]
            img_T3HW_resized = [cv2.resize(item, (w // n_sampes_len, h)) for item in img_T3HW]
            
            gt_recons_pred_video = np.concatenate([generated_image, recons_video, img_T3HW_resized], axis=2)
            images2video(gt_recons_pred_video, fps=args.fps, save_filepath=os.path.join(save_dir, f'./{item_name}_tmp_pred_recons_gt{ext}'))
            images2video(np.array(recons_video), fps=args.fps, save_filepath=os.path.join(save_dir, f'./{item_name}_tmp_recons{ext}'))
            images2video(np.array(img_T3HW_resized), fps=args.fps, save_filepath=os.path.join(save_dir, f'./{item_name}_tmp_gt{ext}'))
            images2video(gt_recons_pred_video, fps=args.fps, save_filepath=os.path.join(save_dir_download_withgt, f'{i:03d}_{index:03d}_{suffix}{ext}'))
            
            key = 'video_path' if ext == '.mp4' else 'image_path'
            save_file_obj.write(json.dumps({
                key: os.path.abspath(os.path.join(save_dir, f'./tmp_pred_recons_gt{ext}')),
                'local_path': os.path.abspath(local_path),
                'prompt': prompt,
            }) + '\n')
            
    save_file_obj.close()


if __name__ == '__main__':
    main()
