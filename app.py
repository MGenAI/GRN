import os
import sys
import torch
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grn_pipeline import GRNPipeline

# Global pipeline
pipe = None

def load_pipeline():
    global pipe
    print("Loading GRN pipeline...")
    # 从 Hugging Face Hub 下载权重
    pipe = GRNPipeline.from_pretrained(
        hf_repo_id="bytedance-research/grn",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("Pipeline loaded successfully!")
    return pipe

def generate(prompt, content_type="image", guidance_scale=3.0, seed=42, width=512, height=512):
    if pipe is None:
        return "Pipeline not loaded!"
    
    result = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        content_type=content_type,
        seed=seed,
        width=width,
        height=height
    )
    
    if content_type == "image" and hasattr(result, 'images'):
        return result.images[0]
    elif content_type == "video" and hasattr(result, 'videos'):
        return result.videos[0]
    return None

def create_demo():
    with gr.Blocks(title="GRN: Generative Refinement Networks", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# GRN: Generative Refinement Networks")
        gr.Markdown("Text-to-Image and Text-to-Video generation using GRN")
        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Enter your prompt here...",
                    value="A cute cat playing in the garden"
                )
                
                content_type = gr.Radio(
                    choices=["image", "video"],
                    value="image",
                    label="Content Type"
                )
                
                with gr.Accordion("Settings", open=False):
                    guidance_scale = gr.Slider(minimum=0, maximum=10, value=3.0, label="Guidance Scale")
                    seed = gr.Number(value=42, label="Seed", precision=0)
                    width = gr.Number(value=512, label="Width", precision=0)
                    height = gr.Number(value=512, label="Height", precision=0)
                
                generate_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                output = gr.Gallery(label="Output")
        
        def generate_and_display(prompt, content_type, guidance_scale, seed, width, height):
            result = generate(prompt, content_type, guidance_scale, seed, width, height)
            if result:
                return [result]
            return []
        
        generate_btn.click(
            fn=generate_and_display,
            inputs=[prompt_input, content_type, guidance_scale, seed, width, height],
            outputs=output
        )
        
        gr.Examples(
            examples=[
                ["A majestic lion standing on a cliff at sunset", "image", 3.0, 42, 512, 512],
                ["A dog chasing a butterfly in a meadow", "video", 3.0, 123, 512, 512],
            ],
            inputs=[prompt_input, content_type, guidance_scale, seed, width, height],
            cache_examples=False
        )
    
    return demo

if __name__ == "__main__":
    try:
        load_pipeline()
    except Exception as e:
        print(f"Error loading pipeline: {e}")
    
    demo = create_demo()
    demo.launch()
