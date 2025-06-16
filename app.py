# app.py

import gradio as gr
from fastai.vision.all import *
import skimage
import pickle

with open('vocab.pkl', 'rb') as f:
    labels = pickle.load(f)

dls = ImageDataLoaders.from_lists(
    path=".", 
    fnames=['cat_example.jpg'], 
    labels=['cat'],             
    valid_pct=0,
    bs=1
)

learn = vision_learner(dls, resnet34, metrics=error_rate)

learn.load('model_weights.pth', with_path=True)


def predict(img):
   
    pred, pred_idx, probs = learn.predict(img)
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