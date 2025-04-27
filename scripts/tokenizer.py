import json
from datasets import Dataset
from transformers import AutoTokenizer

DATA_PATH = "data/raw/lyrics_enriched.json"
MODEL_NAME = "gsarti/it5-base"
PREFIX = "parafrasa: "

with open(DATA_PATH, encoding="utf-8") as f:
    raw_data = json.load(f)

examples = []
for entry in raw_data:
    if "lyrics" in entry and "paraphrase" in entry:
        examples.append({
            "input": PREFIX + entry["lyrics"].strip(),
            "target": entry["paraphrase"].strip()
        })

hf_dataset = Dataset.from_list(examples)
print(f"✅ Loaded {len(hf_dataset)} examples")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    inputs = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=256)
    targets = tokenizer(batch["target"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = hf_dataset.map(tokenize, batched=True, remove_columns=["input", "target"])
tokenized_dataset.save_to_disk("data/processed/paraphrase_dataset")
print("✅ Dataset tokenizzato e salvato!")
