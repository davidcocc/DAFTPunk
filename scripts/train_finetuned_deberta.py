from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
import evaluate
import os

# 🎯 Sentiment labels
LABELS = ["joy", "sadness", "rage", "love", "nostalgia", "hope", "fear"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

# 🔧 Model config
MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = "./model/deberta_sentiment"

# 📂 Load dataset
df = pd.read_json("data/raw/lyrics_enriched.json")

# 🧹 Filter and encode
df = df[df["sentiment"].isin(LABELS)].reset_index(drop=True)
df["label"] = df["sentiment"].map(label2id)

# 📊 Create dataset
dataset = Dataset.from_pandas(df[["lyrics", "label"]])
dataset = dataset.train_test_split(test_size=0.2)

# 🧠 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["lyrics"], truncation=True, padding="max_length", max_length=256)

dataset = dataset.map(tokenize)

# 🧠 Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)

# 📈 Evaluation
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# ⚙️ Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

# 🏋️ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 🚀 Start training
trainer.train()
