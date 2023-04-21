import streamlit as st
import pandas as pd
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    pipeline,
)

st.title("Toxic Comment Classifier")

default_text = "Enter your text here."
input_text = st.text_area("Input text", default_text, height=275)

model_choice = st.selectbox(
    "Select the model you want to use below.",
    (
        "finetuned_toxic_model",
    ),
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
    toxic_score = sentiment_results["scores"][1]

    # Create a DataFrame to display the results
    data = {
        "Tweet": [input_text[:50] + "..." if len(input_text) > 50 else input_text],
        "Toxicity Class": ["Toxic" if toxic_score > 0.5 else "Non-toxic"],
        "Probability": [f"{toxic_score:.2f}"],
    }
    df = pd.DataFrame(data)
    st.write(df)
