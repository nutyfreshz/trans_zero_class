import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-th-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-th-en")

# Sidebar for uploading file and entering sheet name
st.sidebar.header("Part 1.0) Upload XLSX Data")
uploaded_file = st.sidebar.file_uploader("Upload an XLSX file", type=["xlsx"])

if uploaded_file is not None:
    st.sidebar.header("Part 1.1) Enter Sheet name")
    sheet_name_ex = st.sidebar.text_input("Enter Sheet name in Excel")
    
    if sheet_name_ex:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name_ex)
        st.markdown("### Data Sample")
        st.write(df.head())

        st.sidebar.header("Part 1.2) Select Columns to Classify Topics")
        cols_option = st.sidebar.selectbox("Select Columns", df.columns.tolist())

        st.sidebar.header("Part 2) Enter Topics")
        text_label = st.sidebar.text_input("Enter Topics split with comma e.g. positive,negative")
        text_list = text_label.split(",")

        if cols_option and text_list:
            # Function to translate text if needed
            def translate_text(text):
                language = detect(text)
                if language == 'en':
                    return text
                elif language == 'th':
                    translated_tokens = model.generate(**tokenizer(text, return_tensors="pt"))
                    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                    return translated_text
                return "Unsupported language"

            df['cmnt_new'] = df[cols_option].apply(translate_text)

            # Placeholder for classifier (assumed to be defined elsewhere)
            # from transformers import pipeline
            # classifier = pipeline("zero-shot-classification")

            # Function to predict sentiment (needs classifier to be defined)
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

            # Assuming classifier is defined, else this part needs the classifier definition
            # results_df = predict_sentiment(df=df, text_column='cmnt_new', text_labels=text_list)
            # st.markdown("### Data Output")
            # st.write(results_df.head())
        else:
            st.sidebar.warning("Please enter sheet name, select column, and enter topics.")

else:
    st.sidebar.warning("Please upload an Excel file.")
