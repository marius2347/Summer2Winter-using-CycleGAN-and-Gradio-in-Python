# import necessary libraries for gradio app
import gradio as gr
from PIL import Image
from model import ImageTransformer, winter_filter, summer_filter

# function to handle image transformation
def transform_image(image, direction, use_ai):
    if image is None:
        return None, "Please upload an image"
    
    try:
        if use_ai:
            if direction == "summer_to_winter":
                transformer = ImageTransformer(
                    model_path="./checkpoints/summer2winter_yosemite_pretrained/latest_net_G.pth"
                )
            else:
                transformer = ImageTransformer(
                    model_path="./checkpoints/winter2summer_yosemite_pretrained/latest_net_G.pth"
                )
            transformed_img = transformer.transform_image(image)
            return transformed_img, "AI transformation applied"
        
        # fallback to preset filters
        if direction == "summer_to_winter":
            return winter_filter(image), "Winter filter applied"
        else:
            return summer_filter(image), "Summer filter applied"
            
    except Exception as e:
        return None, f"Error: {str(e)}"

# custom CSS for better styling
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

# building the gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=css) as app:

    # HEADER
    gr.HTML("""
    <div class="main-header">
        <h1>Summer ↔ Winter Transformer</h1>
        <p>Transform images between seasons using AI or preset filters</p>
    </div>
    """)

    # INPUT/OUTPUT SECTION
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

    # BUTTON ACTION
    transform_btn.click(
        fn=transform_image,
        inputs=[input_image, direction, use_ai],
        outputs=[output_image, status]
    )

    # FOOTER
    gr.HTML("""
    <div style="text-align: center; margin-top: 2rem; color: #666;">
        <p>Built with PyTorch CycleGAN and Gradio</p>
    </div>
    """)

# launch the app
if __name__ == "__main__":
    app.launch(share=True)
