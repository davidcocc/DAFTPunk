from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
import evaluate
import os
import json
import logging
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"../results/training_log_{timestamp}.txt"
os.makedirs("../results", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

logger.info("Avvio training sentiment analysis")
logger.info("=" * 60)

LABELS = ["joy", "sadness", "rage", "love", "nostalgia", "hope", "fear"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = "../model/deberta_sentiment_v3"

logger.info(f"Modello base: {MODEL_NAME}")
logger.info(f"Salvataggio in: {OUTPUT_DIR}")
logger.info(f"Log file: {log_file}")

logger.info("Caricamento dataset...")
try:
    with open("data\\raw\\lyrics_enriched.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Entries totali nel dataset: {len(data)}")
except Exception as e:
    logger.error(f"Errore nel caricamento del dataset: {e}")
    raise

logger.info("Filtraggio entries con sentiment validi...")
valid_entries = []
invalid_sentiments = []
null_empty_sentiments = 0

for entry in data:
    sentiment = entry.get("sentiment", "")

    if not sentiment or not sentiment.strip():
        null_empty_sentiments += 1
        continue
        
    sentiment_clean = sentiment.strip().lower()
    
    if sentiment_clean in LABELS:
        valid_entries.append({
            "lyrics": entry["lyrics"],
            "sentiment": sentiment_clean,
            "artist": entry.get("artist", ""),
            "title": entry.get("title", "")
        })
    else:
        invalid_sentiments.append(sentiment_clean if sentiment_clean else "EMPTY")

logger.info(f"Entries valide: {len(valid_entries)}")
logger.info(f"Entries scartate (sentiment nullo/vuoto): {null_empty_sentiments}")
logger.info(f"Entries scartate (sentiment non valido): {len(invalid_sentiments)}")
logger.info(f"Totale entries scartate: {null_empty_sentiments + len(invalid_sentiments)}")

if len(valid_entries) == 0:
    logger.error("ERRORE: Nessuna entry valida trovata nel dataset!")
    raise ValueError("Dataset vuoto dopo il filtraggio")

sentiment_counts = Counter([entry["sentiment"] for entry in valid_entries])
logger.info("Distribuzione sentiment:")
for sentiment, count in sorted(sentiment_counts.items(), key=lambda x: x[1], reverse=True):
    pct = (count / len(valid_entries)) * 100
    logger.info(f"   {sentiment.ljust(10)}: {count:4} ({pct:5.1f}%)")

if invalid_sentiments:
    invalid_counts = Counter(invalid_sentiments)
    logger.info("Sentiment non validi trovati:")
    for sentiment, count in invalid_counts.most_common(5):
        logger.info(f"   '{sentiment}': {count}")

df = pd.DataFrame(valid_entries)
df["label"] = df["sentiment"].map(label2id)

logger.info(f"Dataset finale: {len(df)} entries per training")

if df["label"].isnull().any():
    logger.error("ERRORE: Alcuni sentiment non sono stati mappati correttamente!")
    unmapped = df[df["label"].isnull()]["sentiment"].unique()
    logger.error(f"Sentiment non mappati: {unmapped}")
    raise ValueError("Errore nel mapping dei sentiment")

dataset = Dataset.from_pandas(df[["lyrics", "label"]])
dataset = dataset.train_test_split(test_size=0.2, seed=42)

logger.info(f"Training set: {len(dataset['train'])} entries")
logger.info(f"Test set: {len(dataset['test'])} entries")

logger.info("Caricamento tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info("Tokenizer caricato con successo")
except Exception as e:
    logger.error(f"Errore nel caricamento del tokenizer: {e}")
    raise

def tokenize(batch):
    return tokenizer(batch["lyrics"], truncation=True, padding="max_length", max_length=256)

logger.info("Tokenizzazione dataset...")
dataset = dataset.map(tokenize, batched=True)
logger.info("Tokenizzazione completata")

logger.info("Caricamento modello...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )
    logger.info("Modello caricato con successo")
except Exception as e:
    logger.error(f"Errore nel caricamento del modello: {e}")
    raise

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

logger.info("Configurazione training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="f1", 
    report_to="none",
    save_total_limit=3,
    seed=42
)

logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
logger.info(f"Epoche: {training_args.num_train_epochs}")
logger.info(f"Learning rate: {training_args.learning_rate}")
logger.info(f"Metrica principale: F1 Score")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

logger.info("=" * 60)
logger.info("INIZIO TRAINING")
logger.info("=" * 60)

start_time = datetime.now()
try:
    training_result = trainer.train()
    logger.info("Training completato con successo!")
except Exception as e:
    logger.error(f"Errore durante il training: {e}")
    raise

end_time = datetime.now()
training_time = end_time - start_time

logger.info("=" * 60)
logger.info("TRAINING COMPLETATO!")
logger.info("=" * 60)

try:
    trainer.save_model()
    logger.info(f"Modello salvato in: {OUTPUT_DIR}")
except Exception as e:
    logger.error(f"Errore nel salvataggio del modello: {e}")

logger.info("Esecuzione valutazione finale...")
try:
    eval_results = trainer.evaluate()
    
    predictions = trainer.predict(dataset["test"])
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    class_report = classification_report(
        y_true, y_pred, 
        target_names=LABELS, 
        output_dict=True
    )
    
    logger.info("Generazione grafici...")
    
    log_history = trainer.state.log_history

    train_loss = []
    eval_loss = []
    eval_f1 = []
    eval_accuracy = []
    epochs = []
    
    for entry in log_history:
        if 'train_loss' in entry:
            train_loss.append(entry['train_loss'])
        if 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
            eval_f1.append(entry['eval_f1'])
            eval_accuracy.append(entry['eval_accuracy'])
            epochs.append(entry['epoch'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1.plot(train_loss, 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(epochs, eval_loss, 'r-', label='Validation Loss')
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()
    
    ax3.plot(epochs, eval_f1, 'g-', label='F1 Score')
    ax3.set_title('F1 Score')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1')
    ax3.grid(True)
    ax3.legend()
    
    ax4.plot(epochs, eval_accuracy, 'orange', label='Accuracy')
    ax4.set_title('Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    
    plot_file = f"../results/training_metrics_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Grafici salvati in: {plot_file}")
    plt.close()
    
    metrics_file = f"../results/metrics_{timestamp}.json"
    clean_metrics = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "training_time_minutes": round(training_time.total_seconds() / 60, 2),
        "dataset_size": len(valid_entries),
        "train_size": len(dataset['train']),
        "test_size": len(dataset['test']),
        "final_metrics": {
            "accuracy": round(eval_results['eval_accuracy'], 4),
            "precision": round(eval_results['eval_precision'], 4),
            "recall": round(eval_results['eval_recall'], 4),
            "f1": round(eval_results['eval_f1'], 4),
            "loss": round(eval_results['eval_loss'], 4)
        },
        "per_class_f1": {label: round(class_report[label]['f1-score'], 3) for label in LABELS if label in class_report},
        "training_config": {
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "weight_decay": training_args.weight_decay
        }
    }
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(clean_metrics, f, indent=2, ensure_ascii=False)
    
    logger.info("RISULTATI FINALI:")
    logger.info(f"   Tempo training: {training_time}")
    logger.info(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"   Precision: {eval_results['eval_precision']:.4f}")
    logger.info(f"   Recall: {eval_results['eval_recall']:.4f}")
    logger.info(f"   F1 Score: {eval_results['eval_f1']:.4f}")
    logger.info(f"   Loss: {eval_results['eval_loss']:.4f}")
    logger.info(f"Metriche salvate in: {metrics_file}")
    
except Exception as e:
    logger.error(f"Errore durante la valutazione: {e}")
    
logger.info("Processo completato con successo!")
