import gradio as gr
from geoguessr import DIR_MODELS_PROD , DIR_UPLOADS , DIR_DATA
from fastai.vision.all import load_learner
from geoguessr.config import prod_model_name
import pandas as pd


DIR_EXAMPLES = DIR_DATA.joinpath("examples")
model_path = DIR_MODELS_PROD.joinpath(prod_model_name)
learner = load_learner(model_path)
examples = ["us.jpg","italy.jpg"]
examples_path = [DIR_EXAMPLES.joinpath(i) for i in examples]

def model_prediction(img):
    country , index , preds = learner.predict(img)
    df = pd.DataFrame( {'country':learner.dls.vocab , 'probabilities' : preds.numpy()} )
    df.sort_values(by='probabilities', ascending=False)
    return df



def classify_image(img):
    
    img_path = DIR_UPLOADS.joinpath('test.jpg')
    img.save(img_path)
    
    df_preds = model_prediction(img_path)
    confidences = {df_preds['country'].loc[i]:float(df_preds['probabilities'].loc[i]) for i in range(len(df_preds)) }
    return confidences


gr.Interface(fn=classify_image, 
             inputs=gr.Image(type = 'pil'),
             outputs=gr.Label(num_top_classes=5),
             examples=examples_path).launch()

