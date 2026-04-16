# Hugging Face Space 部署指南 (简化版)

## 快速开始

### 1. 创建 Hugging Face Space

1. 访问 [Hugging Face Spaces](https://huggingface.co/spaces)
2. 点击 "Create new Space"
3. 配置：
   - **Space name**: `GRN-T2IV` 
   - **SDK**: **Gradio**
   - **Hardware**: 推荐 `A10G` 或 `A100`
4. 点击 "Create space"

### 2. 上传文件到 Space

将以下文件推送到您的 Space：
```
your-space/
├── app.py                  # Gradio 应用
├── grn_pipeline.py         # 简化的 Pipeline 接口
├── requirements.txt        # 依赖
├── grn/                    # 模型代码
└── weights/                # 权重文件
    ├── model.pth
    ├── hbq_tokenizer.ckpt
    └── umt5-xxl/
```

### 3. 使用 Pipeline (像 DiffusionPipeline 一样简单)

```python
import torch
from grn_pipeline import GRNPipeline

# 加载
pipe = GRNPipeline.from_pretrained(
    model_path='./weights/model.pth',
    vae_path='./weights/hbq_tokenizer.ckpt',
    text_encoder_ckpt='./weights/umt5-xxl'
)

# 移动到设备
pipe = pipe.to('cuda')

# 生成图像
result = pipe(
    prompt="A cute cat playing in the garden",
    guidance_scale=3.0,
    seed=42,
    content_type='image'
)
image = result.images[0]

# 生成视频
result = pipe(
    prompt="A dog chasing a butterfly",
    content_type='video'
)
video = result.videos[0]
```

### 4. 详细部署步骤

```bash
# 克隆 Space
git clone https://huggingface.co/spaces/your-username/GRN-T2IV
cd GRN-T2IV

# 复制文件
cp -r /path/to/GRN/app.py .
cp -r /path/to/GRN/grn_pipeline.py .
cp -r /path/to/GRN/requirements.txt .
cp -r /path/to/GRN/grn .
mkdir -p weights
# 复制权重文件到 weights/

# 推送
git add .
git commit -m "Deploy GRN"
git push
```

## 配置说明

### 权重配置

在 `grn_pipeline.py` 的 `from_pretrained()` 方法或 `app.py` 的 `load_pipeline()` 中修改路径：

```python
pipe = GRNPipeline.from_pretrained(
    model_path='./weights/model.pth',          # 您的模型路径
    vae_path='./weights/hbq_tokenizer.ckpt',  # VAE 路径
    text_encoder_ckpt='./weights/umt5-xxl'    # 文本编码器路径
)
```

### Space 硬件建议

| 模型规模 | 推荐硬件 |
|---------|---------|
| 小模型   | T4      |
| 中等模型 | A10G    |
| 大模型   | A100    |

## 调试

1. **本地测试**：先在本地运行 `python app.py` 测试
2. **查看日志**：在 Space 的 Settings → Logs 中查看构建日志
3. **检查权重**：确保所有权重文件都正确上传

## 完成！

现在您的 GRN 模型已经部署完成，用户可以通过网页进行文本生图和文本生视频了！
