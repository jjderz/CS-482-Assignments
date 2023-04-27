import streamlit as st
from transformers import AutoTokenizer
from transformers import (
    TFAutoModelForSequenceClassification as SequenceClassificationModel,
)
from transformers import pipeline

st.title("Toxic Comment Classifier")

sample_text = "I hate you, you suck."

tweet_text = ""
proceed = False
selected_model = ""
col1, col2, col3 = st.columns([2,1,1])

with st.container():
    selected_model = st.selectbox(
        "Choose the desired model from the dropdown menu.",
        ("ogtega/tweet-toxicity-classifier",),
    )
    proceed = st.button("Submit", type="primary", use_container_width=True)

token_processor = AutoTokenizer.from_pretrained(selected_model)
model = SequenceClassificationModel.from_pretrained(selected_model)
classifier = pipeline(
    "sentiment-analysis", model=model, tokenizer=token_processor, return_all_scores=True
)

with col1:
    st.subheader("Comment")
    tweet_text = st.text_area("Enter text", sample_text, height=275)

with col2:
    st.subheader("Category")

with col3:
    st.subheader("Likelihood")


input_data = token_processor(tweet_text, return_tensors="tf")

if proceed:
    result_data = dict(d.values() for d in classifier(tweet_text)[0])
    categories = {k: result_data[k] for k in result_data.keys() if not k == "toxic"}

    top_category = max(categories, key=categories.get)

    with col2:
        st.write(f"#### {top_category}")

    with col3:
        st.write(f"#### **{categories[top_category]:.2f}%**")

    if result_data["toxic"] < 0.5:
        st.success("This tweet is likely not harmful!", icon=":white_check_mark:")
    else:
        st.warning("This tweet might be harmful.", icon=":warning:")
    
    expand_section = st.expander("Detailed output")
    expand_section.write(result_data)
