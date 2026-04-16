import torch
from grn_pipeline import GRNPipeline

# 加载 pipeline - 像 DiffusionPipeline 一样简单！
pipe = GRNPipeline.from_pretrained(
    model_path='./weights/model.pth',
    vae_path='./weights/hbq_tokenizer.ckpt',
    text_encoder_ckpt='./weights/umt5-xxl',
    torch_dtype=torch.bfloat16
)

# 移动到设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipe = pipe.to(device)

# 生成图像
result = pipe(
    prompt="A cute cat playing in the garden, high quality",
    negative_prompt="",
    guidance_scale=3.0,
    num_inference_steps=50,
    width=512,
    height=512,
    generator=None,
    content_type='image',
    seed=42
)

# 获取结果
image = result.images[0]
image.save('generated_image.jpg')
print("Image saved as generated_image.jpg")

# 生成视频
result = pipe(
    prompt="A dog chasing a butterfly in a meadow",
    guidance_scale=3.0,
    content_type='video',
    seed=123
)

# 获取结果
video_path = result.videos[0]
print(f"Video saved at: {video_path}")
