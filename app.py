# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 17:43:39 202"""
#importing the essential library
import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model

#load the model
model = load_model("plant_disease.h5")

#image label
all_labels = ["Corn-Common_rust","Potato-Early_blight","Tomato-Bacterial_spot"]

#creating the framework on the streamlit
st.title("Plant Disease Prediction")
st.markdown("Upload a image here to predict here")


#image upload using streamlit
plant_image = st.file_uploader("choose a image",type="jpg")
submit = st.button("Predict")

if submit:
    
    if plant_image is not None:
        file_bytes=np.asarray(bytearray(plant_image.read()),dtype=np.uint8)
        cv2_image = cv2.imdecode(file_bytes,1)
        
        
        st.image(cv2_image,channels="BGR")
        st.write(cv2_image.shape)
        #resize the shape of the image
        cv2_image=cv2.resize(cv2_image,(250,250))
        cv2_image.shape =(1,250,250,3)
        #predicting the image with the model that we made using cnn
        y_pred = model.predict(cv2_image)
        result = all_labels[np.argmax(y_pred)]
        st.title(str("this is "+result.split("-")[0]+" leaf with " +result.split("-")[1]))
        
        