import base64
import time
import pickle
import streamlit as st
import numpy as np
from PIL import Image 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

@st.cache(allow_output_mutation=True)
def load_model_and_class_names():
    """ Load model and class_names"""
    model = load_model("./models/xception_hypermodel_tuned.h5", compile=False)
    class_names = pickle.load(open("./models/class_names.save", "rb"))
    return model, class_names

def load_image(img):
    """ transform into array and preprocess image """
    img = img.resize((299,299))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = preprocess_input(img_tensor)
    return img_tensor

def get_prediction(model, img, class_names):
    """ Make prediction using model """
    preds = model.predict(img)
    pred_label = class_names[np.argmax(preds)]
    return pred_label

def main():
    model, class_names = load_model_and_class_names()
    st.title("Dog Breed Prediction from a Picture")
    file = st.file_uploader("Load a picture of your dog")
    img_placeholder = st.empty()
    success = st.empty()
    submit_placeholder = st.empty()
    submit=False

    if file is not None:
        with st.spinner("Loading..."):  
            model, class_names = load_model_and_class_names()
            img = Image.open(file)
            img_placeholder.image(img, width=299)
        submit = submit_placeholder.button("Launch prediction")

    if submit:
        with st.spinner('Waiting for result...'):    
            submit_placeholder.empty()
            img_tensor = load_image(img)
            res = get_prediction(model=model, img=img_tensor, class_names=class_names)
            success.success("{}".format(res))

if __name__ == "__main__":
    main()



