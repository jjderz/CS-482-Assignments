import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Load the dataset and preprocess
dataset = load_dataset("csv", data_files={"train": "train.csv"})["train"]
dataset = dataset.map(
    lambda e: {"labels": e["toxic"], "text": e["comment_text"]},
    remove_columns=["id", "comment_text", "toxic"],
)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize the dataset
dataset = dataset.map(
    lambda x: tokenizer(x["text"], padding="max_length", truncation=True),
    batched=True,
)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Split the dataset into training and validation sets
train_dataset, val_dataset = dataset.train_test_split(test_size=0.1).values()

# Define the model and training arguments
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="logs",
    logging_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
)

# Create a Trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()

# Save the fine-tuned model
model.save_pretrained("finetuned_toxic_model")
tokenizer.save_pretrained("finetuned_toxic_model")
