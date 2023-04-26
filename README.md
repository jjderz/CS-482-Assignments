---
title: Finetuning Language Models Tweets
emoji: 😻
colorFrom: indigo
colorTo: indigo
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
---
Google Sites Landing Page: https://sites.google.com/njit.edu/finetunedtoxiccommentmodel/home

Hugging Face: https://huggingface.co/spaces/jjderz/finetuning-language-models-tweets


### Toxic Comment Classifier

This project is designed to identify various forms of toxicity in comments using a deep learning model built on the BERT architecture. It employs a pre-trained BERT model that is fine-tuned on a dataset of labeled Wikipedia comments. The project consists of two main components: training the model in `toxic_trainer.ipynb` and deploying the model as a web application using Streamlit in `app.py`.

##### toxic_trainer.ipynb

This Jupyter notebook contains the code for training a BERT-based model to perform multi-label classification of toxic comments. The primary elements of this notebook are:

1. **Imports**: Incorporating necessary libraries and modules.

2. **Drive Mounting**: Connecting to Google Drive to store the model and checkpoints.

3. **Labels and dictionaries**: Specifying labels and generating dictionaries for label-to-index and index-to-label mapping.

4. **Dataset loading**: Retrieving the dataset from a CSV file and tokenizing it using a pre-trained BERT tokenizer.

5. **Preprocessing**: Developing a function to preprocess the data by tokenizing text and encoding labels.

6. **Model definition**: Establishing the BERT-based model for multi-label classification.

7. **Dataset encoding**: Encoding the dataset using the preprocessing function.

8. **Model training**: Assembling and training the model using the encoded dataset.

9. **Upload to Huggingface**: Using Huggingface cli to push the model and tokenizer to the hub.

##### app.py

This Python script is utilized to launch the trained model as a web application with the help of the Streamlit library. The primary components of this script are:

1. **Imports**: Incorporating necessary libraries and modules.

2. **Title**: Exhibiting the title of the web application.

3. **Input text area**: Generating a text area for users to enter text for toxicity classification.

4. **Model choice**: Offering a selection for users to pick their preferred model for classification.

5. **Tokenizer and model loading**: Retrieving the tokenizer and model based on user selection.

6. **Sentiment pipeline**: Developing a sentiment analysis pipeline using the loaded tokenizer and model.

7. **Submit button**: Introducing a button for submitting the input text for classification.

8. **Classification and result display**: When the submit button is activated, the input text is classified using the sentiment pipeline, and the results are presented in a table.

##### Dependencies

- Python 3.6+
- TensorFlow 2.x
- transformers
- datasets
- streamlit
- pandas

