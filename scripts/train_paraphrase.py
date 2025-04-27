from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk
import torch
import os

# Config
MODEL_NAME = "gsarti/it5-base"
DATASET_PATH = "data/processed/paraphrase_dataset"
OUTPUT_DIR = "model/it5_paraphrase"
BATCH_SIZE = 4
EPOCHS = 4

# Load dataset and tokenizer
dataset = load_from_disk(DATASET_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=20,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
    overwrite_output_dir=True
)

# Define trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Start training
trainer.train()

# Save final model and tokenizer
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("\n✅ Model and tokenizer saved to", OUTPUT_DIR)
