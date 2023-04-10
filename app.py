import streamlit as st
from transformers import AutoTokenizer
from transformers import (
    TFAutoModelForSequenceClassification as AutoModelForSequenceClassification,
)
from transformers import pipeline

st.title("Toxic Tweets")

demo = """Just took a stroll through the park and the sun is shining bright ☀️🌳 There's something so rejuvenating about being surrounded by nature! Hope you all have a wonderful day! 🌸😊 #grateful #happiness #naturelovers#flexinonthekids"""

text = st.text_area("Input text", demo, height=275)

model_name = st.selectbox(
    "Select the model you want to use below.",
    (
        "distilbert-base-uncased-finetuned-sst-2-english",
    ),
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
clf = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True
)

input = tokenizer(text, return_tensors="tf")

if st.button("Submit", type="primary"):
    results = clf(text)[0]
    # st.write(f"The sentiment is {results}.")
