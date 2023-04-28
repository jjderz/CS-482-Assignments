import streamlit as st
import pandas as pd
from transformers import AutoTokenizer
from transformers import (
    TFAutoModelForSequenceClassification as SequenceClassificationModel,
)
from transformers import pipeline

st.title("Toxic Comment Classifier")

sample_text = "I hate you, you suck."

comment = ""
proceed = False
selected_model = ""

with st.container():
    selected_model = st.selectbox(
        "Choose the desired model from the dropdown menu.",
        ("jjderz/toxic-classifier",),
    )
    proceed = st.button("Submit", type="primary")

token_processor = AutoTokenizer.from_pretrained(selected_model)
model = SequenceClassificationModel.from_pretrained(selected_model)
classifier = pipeline(
    "sentiment-analysis", model=model, tokenizer=token_processor, return_all_scores=True
)

comment = st.text_area("Enter text", sample_text, height=275)

input_data = token_processor(comment, return_tensors="tf")

if proceed:
    result_data = dict(d.values() for d in classifier(comment)[0])
    categories = {k: result_data[k] for k in result_data.keys() if not k == "toxic"}

    top_category = max(categories, key=categories.get)

    # Create a DataFrame for the table
    table_data = {
        "Tweet": [comment[:50]],
        "Toxicity Category": [top_category],
        "Probability": [f"{categories[top_category]:.2f}%"],
    }
    results_df = pd.DataFrame(table_data)

    # Display the table
    st.write(results_df)

    expand_section = st.expander("Detailed output")
    expand_section.write(result_data)
