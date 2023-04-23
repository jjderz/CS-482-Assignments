import streamlit as st
import pandas as pd
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    pipeline,
)
from huggingface_hub import HfApi

st.title("Toxic Comment Classifier")

default_text = "Enter your text here."
input_text = st.text_area("Input text", default_text, height=275)

hf_api = HfApi()
models = hf_api.list_models(author="jjderz")

model_names = []
for model in models:
    model_names.append(model.model_id)

model_choice = st.selectbox(
    "Select the model you want to use below.",
    model_names,
)

tokenizer = AutoTokenizer.from_pretrained(model_choice)
sequence_classifier = TFAutoModelForSequenceClassification.from_pretrained(model_choice)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=sequence_classifier,
    tokenizer=tokenizer,
    return_all_scores=True,
)

if st.button("Submit", type="primary"):
    sentiment_results = sentiment_pipeline(input_text)[0]

    # Find the highest scoring toxicity type
    highest_toxicity_label = ""
    highest_toxicity_score = 0
    for result in sentiment_results:
        if result["score"] > highest_toxicity_score:
            highest_toxicity_label = result["label"]
            highest_toxicity_score = result["score"]

    # Create a DataFrame to display the results
    data = {
        "Tweet": [input_text[:50] + "..." if len(input_text) > 50 else input_text],
        "Toxicity Class": [highest_toxicity_label],
        "Probability": [f"{highest_toxicity_score:.2f}"],
    }
    df = pd.DataFrame(data)
    st.write(df)
