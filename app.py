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

token_processor = AutoTokenizer.from_pretrained(selected_model)
model = SequenceClassificationModel.from_pretrained(selected_model)
classifier = pipeline(
    "sentiment-analysis", model=model, tokenizer=token_processor, return_all_scores=True
)

comment = st.text_area("Enter text", sample_text, height=275)
proceed = st.button("Submit", type="primary")

input_data = token_processor(comment, return_tensors="tf")

# Initialize session state with an empty DataFrame
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=["Tweet", "Toxicity Category", "Probability"])

if proceed:
    result_data = dict(d.values() for d in classifier(comment)[0])
    categories = {k: result_data[k] for k in result_data.keys() if not k == "toxic"}

    if result_data["toxic"] < 0.5:
        top_category = "Not Toxic"
        probability = result_data["toxic"]
    else:
        top_category = max(categories, key=categories.get)
        probability = categories[top_category]

    # Append a new row to the DataFrame
    new_row = {
        "Tweet": comment[:50],
        "Toxicity Category": top_category,
        "Probability": f"{probability:.2f}%",
    }
    st.session_state.results_df = st.session_state.results_df.append(new_row, ignore_index=True)

    # Display the updated table
    st.write(st.session_state.results_df)

    expand_section = st.expander("Detailed output")
    expand_section.write(result_data)
