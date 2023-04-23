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
   
)

model_choice = {
    "Fine-tuned Toxicity Model": "jjderz/toxic-classifier",
}

chosen_model = st.selectbox("Select Model", options=list(model_choice.keys()))

tokenizer = AutoTokenizer.from_pretrained(chosen_model)
sequence_classifier = TFAutoModelForSequenceClassification.from_pretrained(chosen_model)
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
