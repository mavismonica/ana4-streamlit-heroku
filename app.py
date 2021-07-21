import streamlit as st
import tensorflow as tf
from keras.models import load_model
# st.set_option('deprecation.showfileUploaderEncoding',False)
# @st.cache(allow_output_mutation=True)
# def load_model():
#     #from keras.models import load_model
# #     model = tf.keras.model.load_model('C:/Users/mavis/Documents/ana4/trained_VGG_model.h5')
#     model = tf.keras.model.load_model('C:/Users/mavis/Documents/ana4/trained_VGG_model.h5')
#     return model

# model = load_model()
# model = load_model('C:/Users/mavis/Documents/ana4/trained_VGG_model.h5')
model = load_model('VGG_model.hdf5')

st.write("""
# Grocery item classification
""")
file = st.file_uploader("Please upload an grocery item image", type =["jpg","png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    size =(250,250)
    image = ImageOps.fit(image_data,size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    
    return prediction

if file is None:
    st.text("Please upload an image")
else:
    image = Image.open(file)
    st.image(image, use_column_width = True)
    predictions = import_and_predict(image,model)
    class_names = ['BEANS','CAKE','CANDY','CEREAL','CHIPS','CHOCOLATE','COFFEE','CORN','FISH','FLOUR','HONEY',
                    'JAM','JUICE','MILK','NUTS','OIL','PASTA','RICE','SODA','SPICES','SUGAR','TEA','TOMATO_SAUCE',
                    'VINEGAR','WATER']
    string = "This image most likely is "+class_names[np.argmax(predictions)]
    st.success(string)
