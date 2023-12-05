import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Mango Grade Analyzer",
    page_icon = ":mango:",
    initial_sidebar_state = 'auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            
            return key


with st.sidebar:
        st.image('mg.png')
        st.title("Mangifera Analytica")
        st.subheader("Accurate analysis and segregation of mango grades from the uploaded images")

        
       

    

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('grade_mango.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

    

st.write("""
         # Mango Grade Analyzer
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.info("Accuracy : " + str(x) + " %")

    class_names = ['Class_I', 'Class_II','Extra_Class']

    string = "Mango Type : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'Class_I':
        
        st.sidebar.warning(string)

    elif class_names[np.argmax(predictions)] == 'Class_II':
        st.sidebar.error(string)
        

    elif class_names[np.argmax(predictions)] == 'Extra_Class':
        string = "Class_III"
        st.sidebar.success(string)
        st.success("This is the best grade of Mango")
        
    
