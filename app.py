### 1. Imports and class names setup ###
import gradio as gr
import os
import torch

from model import create_resnet50_model  # changed to your ResNet50 model creator
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
with open("class_names.txt", "r") as f:  # reading them in from class_names.txt
    class_names = [animal_name.strip() for animal_name in f.readlines()]

### 2. Model and transforms preparation ###

# Create model instance with number of classes matching your dataset (15 animals)
resnet50_animals, resnet50_transforms = create_resnet50_model(
    num_classes=15,  # update according to your animal classes
)

# Load saved weights for your Animal ResNet50 model
resnet50_animals.load_state_dict(
    torch.load(
        f="Animal_Resnet50.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 3. Predict function ###

def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken."""
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = resnet50_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    resnet50_animals.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into probabilities
        pred_probs = torch.softmax(resnet50_animals(img), dim=1)

    # Create prediction dictionary for Gradio Label output
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate prediction time
    pred_time = round(timer() - start_time, 5)

    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "Animal Classification ResNet50 🐾"
description = "A ResNet50 computer vision model trained to classify images into 15 different animal classes."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description
)

# Launch the app!
demo.launch()
