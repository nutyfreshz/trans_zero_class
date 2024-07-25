import subprocess
import streamlit as st
import pandas as pd
import numpy as np

st.sidebar.header("Part 1.0) Upload XLSX Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["xlsx"])

st.sidebar.header("Part 1.1) Enter Sheet name")
sheet_name_ex = st.sidebar.text_input("Enter Sheet name in Excel")

df = pd.read_excel(uploaded_file
                     , sheet_name = sheet_name_ex)

st.markdown("### Data Sample")
st.write(df.head())

st.sidebar.header("Part 1.2) Select Columns to Classify Topics")
cols_option = st.sidebar.selectbox("Select Columns", df.columns.tolist())

st.sidebar.header("Part 2) Enter Topics")
text_label = st.sidebar.text_input("Enter Topics split with comma e.g. positive,negative")
text_list = text_label.split(",")[:-1]

###########################################################

from langdetect import detect

def translate_text(text):
    # Detect the language of the text
    language = detect(text)
    
    # If the text is in English, return it as is
    if language == 'en':
        return text
    
    # If the text is in Thai, translate it
    if language == 'th':
        # Tokenize the text
        translated_tokens = model.generate(**tokenizer(text, return_tensors="pt"))
        # Decode the translated tokens
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    
    # If the text is in neither English nor Thai, handle accordingly
    return "Unsupported language"

df['cmnt_new'] = df[cols_option].apply(translate_text)

###########################################################

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-th-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-th-en")

###########################################################

def predict_sentiment(df, text_column, text_labels):
    result_list = []
    for index, row in df.iterrows():
        sequence_to_classify = row[text_column]
        result = classifier(sequence_to_classify, text_labels, multi_label = False)
        result['sentiment'] = result['labels'][0]
        result['score'] = result['scores'][0]
        result_list.append(result)
    result_df = pd.DataFrame(result_list)[['sequence','sentiment', 'score']]
    result_df = pd.merge(df, result_df, left_on = text_column, right_on="sequence", how = "left")
    return result_df

###########################################################

results_df = predict_sentiment(df = df
                               , text_column = 'cmnt_new'
                               , text_labels = text_list)

st.markdown("### Data Output")
st.write(results_df.head())




