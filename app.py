import gradio as gr
from ultralytics import YOLO
import cv2

# Ensure model is loaded
model = YOLO('/content/runs/detect/train5/weights/best.pt')

def detect_image(image):
    results = model(image)
    annotated = results[0].plot()
    return annotated

# Gradio interface setup with horizontal layout
with gr.Blocks() as demo:
    gr.Markdown("# **Weapon vs Umbrella Detection**")

    with gr.Tab("Image"):
        with gr.Row():
            img_input = gr.Image(type="numpy", label="Input Image")
            img_output = gr.Image(label="Detection Result")
        
        btn = gr.Button("Detect", variant="primary")
        btn.click(detect_image, inputs=img_input, outputs=img_output)

demo.launch(share=True)