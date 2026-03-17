Weapon vs Umbrella Detection using YOLOv8
🔍 Fine-tuned YOLOv8 model to distinguish between a person holding a weapon and a person holding an umbrella — a challenging visual similarity task.

📌 Overview
This project fine-tunes a pre-trained YOLOv8 model on a custom dataset to detect three classes:

person

weapon

umbrella

The main challenge is the visual similarity between rifles and closed umbrellas, which the model learns to differentiate through fine-tuning.

🧠 Model
Base model: yolov8l.pt

Trained for 25 epochs on a custom dataset (80/20 train/val split)

Final weights saved as best.pt

📊 Results
Class	mAP50
person	0.982
weapon	0.885
umbrella	0.917
Overall	0.928
🖼️ Demo
A simple Gradio interface is included to upload an image and get real-time detection results.

Run the demo:

bash
python app.py
Or open the provided Colab notebook and launch the Gradio share link.

📁 Files
Weapon_vs_Umbrella_Detection.ipynb – Full training and inference notebook

best.pt – Trained model weights

app.py – Gradio demo script

train.txt / val.txt – Image splits

data.yaml – Dataset configuration

🚀 How to Use
Clone the repo

Install dependencies: pip install ultralytics gradio

Run the demo: python app.py

Upload an image and see the results
