import streamlit as s
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline

s.title("Toxic Tweets")

default_text = """Just took a stroll through the park and the sun is shining bright ☀️🌳 There's something so rejuvenating about being surrounded by nature! Hope you all have a wonderful day! 🌸😊 #grateful #happiness #naturelovers#flexinonthekids"""

input_text = s.text_area("Input text", default_text, height=275)

model_choice = s.selectbox(
    "Select the model you want to use below.",
    (
        "distilbert-base-uncased-finetuned-sst-2-english",
    ),
)

token = AutoTokenizer.from_pretrained(model_choice)
sequence_classifier = TFAutoModelForSequenceClassification.from_pretrained(model_choice)
sentiment_pipeline = pipeline(
    "sentiment-analysis", model=sequence_classifier, tokenizer=token, return_all_scores=True
)

tensor_input = token(input_text, return_tensors="tf")

if s.button("Submit", type="primary"):
    sentiment_results = sentiment_pipeline(input_text)[0]
    s.write(f"The sentiment is {sentiment_results}.")
