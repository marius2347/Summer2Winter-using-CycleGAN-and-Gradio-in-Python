import gradio as gr
from PIL import Image
import numpy as np
from model import ImageTransformer, winter_filter, summer_filter

transformer = ImageTransformer()

def transform_image(image, direction, use_ai):
    if image is None:
        return None, "Please upload an image"
    
    try:
        if use_ai:
            result, status = transformer.transform(image, direction)
            if result:
                return result, status
        
        if direction == "summer_to_winter":
            result = winter_filter(image)
            return result, "Winter filter applied"
        else:
            result = summer_filter(image)
            return result, "Summer filter applied"
            
    except Exception as e:
        return None, f"Error: {str(e)}"

css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
}
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
}
.demo-box {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    margin: 1rem 0;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as app:
    
    gr.HTML("""
    <div class="main-header">
        <h1>Summer ↔ Winter Transformer</h1>
        <p>Transform images between seasons using AI</p>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="demo-box">')
            
            input_image = gr.Image(
                type="pil",
                label="Input Image",
                height=350
            )
            
            direction = gr.Radio(
                choices=[
                    ("Summer → Winter", "summer_to_winter"),
                    ("Winter → Summer", "winter_to_summer")
                ],
                value="summer_to_winter",
                label="Direction"
            )
            
            use_ai = gr.Checkbox(
                label="Use AI Model",
                value=True
            )
            
            transform_btn = gr.Button(
                "Transform Image",
                variant="primary"
            )
            
            gr.HTML('</div>')

        with gr.Column():
            gr.HTML('<div class="demo-box">')
            
            output_image = gr.Image(
                label="Transformed Image",
                height=350
            )
            
            status = gr.Textbox(
                label="Status",
                value="Ready to transform",
                lines=2
            )
            
            gr.HTML('</div>')

    transform_btn.click(
        fn=transform_image,
        inputs=[input_image, direction, use_ai],
        outputs=[output_image, status]
    )

    gr.HTML("""
    <div style="text-align: center; margin-top: 2rem; color: #666;">
        <p>Built with PyTorch CycleGAN and Gradio</p>
    </div>
    """)

if __name__ == "__main__":
    app.launch(share=True)