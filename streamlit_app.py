import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore FutureWarnings for deprecated features

# Initialize tokenizer and model for translation
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-th-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-th-en")

# Initialize classifier
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

st.sidebar.header("Part 1.0) Upload CSV Data")
uploaded_file = st.sidebar.file_uploader("Upload an CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.markdown("### Data Sample")
    st.write(df.head())
    
    st.sidebar.header("Part 1.2) Select Columns to Classify Topics")
    cols_option = st.sidebar.selectbox("Select Columns", df.columns.tolist())
    
    st.sidebar.header("Part 2) Enter Topics")
    text_label = st.sidebar.text_input("Enter Topics split with comma e.g. positive,negative")
    text_list = [label.strip() for label in text_label.split(",") if label.strip()]  # Ensure labels are stripped and not empty
    
    def translate_text(text):
        if not text or pd.isna(text):
            return "Empty or NaN text"

        try:
            language = detect(text)
        except Exception as e:
            return f"Detection error: {e}"
        
        if language == 'en':
            return text
        
        if language == 'th':
            translated_tokens = model.generate(**tokenizer(text, return_tensors="pt"))
            translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            return translated_text
        
        return "Unsupported language"

    df['cmnt_new'] = df[cols_option].astype(str).apply(translate_text)  # Ensure column values are strings

    def predict_sentiment(df, text_column, text_labels):
        result_list = []
        for index, row in df.iterrows():
            sequence_to_classify = row[text_column]
            result = classifier(sequence_to_classify, text_labels, multi_label=False)
            result['sentiment'] = result['labels'][0]
            result['score'] = result['scores'][0]
            result_list.append(result)
        result_df = pd.DataFrame(result_list)[['sequence', 'sentiment', 'score']]
        result_df = pd.merge(df, result_df, left_on=text_column, right_on="sequence", how="left")
        return result_df

    results_df = predict_sentiment(df=df, text_column='cmnt_new', text_labels=text_list)
    
    st.markdown("### Data Output")
    st.write(results_df.head())
