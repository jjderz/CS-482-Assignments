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




Hugging Face: https://huggingface.co/spaces/jjderz/finetuning-language-models-tweets


### Toxic Comment Classifier

This project aims to detect various types of toxicity in comments using a deep learning model based on the BERT architecture. It uses a pre-trained BERT model and fine-tunes it on a dataset of labeled Wikipedia comments. The project is divided into two parts: training the model in `toxic_trainer.ipynb` and deploying the model as a web application using Streamlit in `app.py`.

##### toxic_trainer.ipynb

This Jupyter notebook contains the code to train a BERT-based model for multi-label classification of toxic comments. The main components of this notebook are:

1. **Imports**: Importing the necessary libraries and modules.

2. **Drive mounting**: Mounting the Google Drive to save the model and checkpoints.

3. **Labels and dictionaries**: Defining the labels and creating dictionaries for label-to-index and index-to-label conversions.

4. **Dataset loading**: Loading the dataset from a CSV file and tokenizing it using a pre-trained BERT tokenizer.

5. **Preprocessing**: Creating a function to preprocess the data by tokenizing text and encoding labels.

6. **Model definition**: Defining the BERT-based model for multi-label classification.

7. **Dataset encoding**: Encoding the dataset using the preprocessing function.

8. **Model training**: Compiling and training the model using the encoded dataset.

##### app.py

This Python script is used to deploy the trained model as a web application using the Streamlit library. The main components of this script are:

1. **Imports**: Importing the necessary libraries and modules.

2. **Title**: Displaying the title of the web application.

3. **Input text area**: Creating a text area for users to input their text for toxicity classification.

4. **Model choice**: Providing a choice for users to select the desired model for classification.

5. **Tokenizer and model loading**: Loading the tokenizer and model based on the user's choice.

6. **Sentiment pipeline**: Creating a sentiment analysis pipeline using the loaded tokenizer and model.

7. **Submit button**: Adding a button to submit the input text for classification.

8. **Classification and result display**: When the submit button is pressed, the input text is classified using the sentiment pipeline, and the results are displayed in a table.

##### Dependencies

- Python 3.6+
- TensorFlow 2.x
- transformers
- datasets
- streamlit
- pandas

