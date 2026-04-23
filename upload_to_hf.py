#!/usr/bin/env python3
"""
上传模型文件到 Hugging Face Hub

此脚本将三个大文件上传到 bytedance-research/grn 仓库
- model_path: /tmp/weights/9a8a674133266e996d8d56e784a10d67.pth
- vae_path: /tmp/weights/HBQ_tokenizer_64dim_M4.ckpt
- text_encoder_ckpt: /tmp/weights/umt5-xxl

使用方法:
1. 确保已安装 huggingface_hub: pip install huggingface_hub
2. 运行: python upload_to_hf.py
"""

import os
import time
from huggingface_hub import HfApi, HfFileSystem

def upload_file(api, repo_id, local_path, remote_path, chunk_size=50 * 1024 * 1024):
    """上传单个文件到 Hugging Face Hub"""
    print(f"开始上传: {local_path} -> {remote_path}")
    start_time = time.time()
    
    # 检查文件大小
    file_size = os.path.getsize(local_path)
    print(f"文件大小: {file_size / (1024*1024*1024):.2f} GB")
    
    try:
        # 上传文件
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model",
            chunk_size=chunk_size
        )
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"上传完成! 耗时: {duration:.2f} 秒")
        return True
    except Exception as e:
        print(f"上传失败: {e}")
        return False

def upload_directory(api, repo_id, local_dir, remote_dir):
    """上传目录到 Hugging Face Hub"""
    print(f"开始上传目录: {local_dir} -> {remote_dir}")
    
    fs = HfFileSystem()
    
    # 确保远程目录存在
    if not fs.exists(f"hf://datasets/{repo_id}/{remote_dir}"):
        fs.makedirs(f"hf://datasets/{repo_id}/{remote_dir}")
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            remote_path = os.path.join(remote_dir, relative_path).replace('\\', '/')
            
            print(f"上传文件: {local_path} -> {remote_path}")
            
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote_path,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"  ✅ 上传成功")
            except Exception as e:
                print(f"  ❌ 上传失败: {e}")
    
    print(f"目录上传完成: {local_dir}")

def main():
    # 配置
    repo_id = "bytedance-research/grn"
    files_to_upload = [
        {
            "local_path": "/tmp/weights/9a8a674133266e996d8d56e784a10d67.pth",
            "remote_path": "model.pth"
        },
        {
            "local_path": "/tmp/weights/HBQ_tokenizer_64dim_M4.ckpt",
            "remote_path": "hbq_tokenizer.ckpt"
        },
        {
            "local_path": "/tmp/weights/umt5-xxl",
            "remote_path": "umt5-xxl",
            "is_directory": True
        }
    ]
    
    # 初始化 API
    api = HfApi()
    
    # 检查仓库是否存在
    try:
        api.get_repo_info(repo_id=repo_id, repo_type="model")
        print(f"✅ 仓库 {repo_id} 存在")
    except Exception as e:
        print(f"⚠️  仓库 {repo_id} 可能不存在，尝试创建...")
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", private=False)
            print(f"✅ 仓库 {repo_id} 创建成功")
        except Exception as create_error:
            print(f"❌ 创建仓库失败: {create_error}")
            return
    
    # 上传文件
    for item in files_to_upload:
        if item.get("is_directory", False):
            upload_directory(api, repo_id, item["local_path"], item["remote_path"])
        else:
            upload_file(api, repo_id, item["local_path"], item["remote_path"])
        print("-" * 60)
    
    print("\n🎉 所有文件上传完成！")
    print(f"\n可以在以下地址查看:")
    print(f"https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    # 检查 huggingface_hub 是否安装
    try:
        import huggingface_hub
    except ImportError:
        print("请先安装 huggingface_hub:")
        print("pip install huggingface_hub")
        exit(1)
    
    # 检查是否已登录
    try:
        HfApi().whoami()
        print("✅ 已登录 Hugging Face")
    except Exception:
        print("请先登录 Hugging Face:")
        print("huggingface-cli login")
        exit(1)
    
    main()
