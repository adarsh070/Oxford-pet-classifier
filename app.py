# app.py

import gradio as gr
from fastai.vision.all import *
import skimage
import pickle


with open('vocab.pkl', 'rb') as f:
    raw_labels = pickle.load(f)

human_labels = {False: "Dog", True: "Cat"}
labels = [human_labels[l] for l in raw_labels]

dls = ImageDataLoaders.from_lists(
    path=".", 
    fnames=['cat_example.jpg'], 
    labels=['cat'],             
    valid_pct=0,
    bs=1
)

model_path = Path.cwd()/'model'
learn = vision_learner(dls, resnet18, metrics=error_rate, path=model_path)


learn.load('model_weights.pth', with_path=True)




def predict(img):
    """
    Gets predictions and returns a dictionary of human-readable labels and probabilities.
    """
    pred, pred_idx, probs = learn.predict(img)
    # The 'labels' variable is now ['Dog', 'Cat'], which matches the order of 'probs'
    return {label: float(prob) for label, prob in zip(labels, probs)}



title = "Pet Breed Classifier (Cat vs. Dog)"
description = "A pet classifier trained on the Oxford Pets dataset with fastai. Upload an image of a cat or a dog to see the model's prediction."
article = "<p style='text-align: center'><a href='https://www.robots.ox.ac.uk/~vgg/data/pets/' target='_blank'>Oxford-IIIT Pet Dataset</a> | <a href='https://github.com/fastai/fastbook' target='_blank'>Fast.ai course</a></p>"

gr.Interface(
    fn=predict, 
    inputs=gr.Image(height=224, width=224), 
    outputs=gr.Label(num_top_classes=2),
    title=title,
    description=description,
    article=article,
    examples=['cat_example.jpg'], 
    allow_flagging="never"
).launch()