import streamlit as st

import fasttext
import pandas as pd

from pycountry import languages

st.set_page_config(page_title = "Language Identification App")

PRETRAINED_MODEL_PATH = './lid.176.ftz'
model = fasttext.load_model(PRETRAINED_MODEL_PATH)

st.title('Language Detector App')

user_input = st.text_area("Enter your text here!")

if st.button("Identify Language"):
    prediction = model.predict(user_input)

    language_label = prediction[0][0][9:] #some error in this statement
    lang_name = languages.get(alpha_2 = language_label).name
    accuracy = round(prediction[1][0]*100)

    st.write(f"The identified language is **{lang_name}** with **{accuracy} %** accuracy.")