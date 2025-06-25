from transformers import DebertaV2TokenizerFast, DebertaV2ForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
import os
import json
import logging
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from datetime import datetime

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"results/training_log_italian_{timestamp}.txt"
    os.makedirs("results", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return timestamp

logger = logging.getLogger(__name__)

def main():
    timestamp = setup_logging()
    
    logger.info("Avvio training sentiment analysis")
    logger.info("=" * 60)

    LABELS = ["joy", "sadness", "rage", "love", "nostalgia", "hope", "fear"]
    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    MODEL_NAME = "osiria/deberta-base-italian"
    OUTPUT_DIR = "model/deberta_italian_v2"

    logger.info(f"Modello base: {MODEL_NAME}")
    logger.info(f"Salvataggio in: {OUTPUT_DIR}")

    logger.info("Caricamento dataset...")
    try:
        with open("data/raw/lyrics_enriched.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Entries totali nel dataset: {len(data)}")
    except Exception as e:
        logger.error(f"Errore nel caricamento del dataset: {e}")
        raise

    logger.info("Filtraggio entries con sentiment validi...")
    valid_entries = []
    for entry in data:
        sentiment = entry.get("sentiment", "")
        if sentiment and sentiment.strip().lower() in LABELS:
            valid_entries.append({
                "lyrics": entry["lyrics"],
                "sentiment": sentiment.strip().lower(),
                "artist": entry.get("artist", ""),
                "title": entry.get("title", "")
            })

    logger.info(f"Entries valide: {len(valid_entries)}")

    if len(valid_entries) == 0:
        logger.error("ERRORE: Nessuna entry valida trovata nel dataset!")
        raise ValueError("Dataset vuoto dopo il filtraggio")
    


    sentiment_counts = Counter([entry["sentiment"] for entry in valid_entries])
    logger.info("Distribuzione sentiment:")
    for sentiment, count in sorted(sentiment_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(valid_entries)) * 100
        logger.info(f"   {sentiment.ljust(10)}: {count:4} ({pct:5.1f}%)")

    df = pd.DataFrame(valid_entries)
    df["label"] = df["sentiment"].map(label2id)

    logger.info(f"Dataset finale: {len(df)} entries per training")

    dataset = Dataset.from_pandas(df[["lyrics", "label"]])
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    logger.info(f"Training set: {len(dataset['train'])} entries")
    logger.info(f"Test set: {len(dataset['test'])} entries")

    logger.info("Caricamento tokenizer...")
    tokenizer = DebertaV2TokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["lyrics"], truncation=True, padding="max_length", max_length=256)

    logger.info("Tokenizzazione dataset...")
    dataset = dataset.map(tokenize, batched=True)

    logger.info("Caricamento modello...")
    model = DebertaV2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )

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
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_steps=50,
        seed=71,
        warmup_steps=100,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    logger.info("INIZIO TRAINING")
    start_time = datetime.now()
    training_result = trainer.train()
    end_time = datetime.now()
    training_time = end_time - start_time

    logger.info("TRAINING COMPLETATO")

    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Modello salvato in: {OUTPUT_DIR}")

    logger.info("Esecuzione valutazione finale...")
    predictions = trainer.predict(dataset["test"])
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    eval_results = {
        'eval_accuracy': accuracy,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1,
        'eval_loss': predictions.metrics['test_loss']
    }
    
    class_report = classification_report(
        y_true, y_pred, 
        target_names=LABELS, 
        output_dict=True
    )
    
    # Generate training plots
    log_history = trainer.state.log_history
    
    train_loss = [entry['train_loss'] for entry in log_history if 'train_loss' in entry]
    eval_data = [entry for entry in log_history if 'eval_loss' in entry]
    
    if train_loss and eval_data:
        epochs = [entry['epoch'] for entry in eval_data]
        eval_loss = [entry['eval_loss'] for entry in eval_data]
        eval_accuracy = [entry['eval_accuracy'] for entry in eval_data]
        eval_f1 = [entry['eval_f1'] for entry in eval_data]
        eval_precision = [entry['eval_precision'] for entry in eval_data]
        eval_recall = [entry['eval_recall'] for entry in eval_data]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Training Loss
        ax1.plot(train_loss, 'b-', linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # Validation Loss  
        ax2.plot(epochs, eval_loss, 'r-', linewidth=2)
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        
        # Accuracy and F1
        ax3.plot(epochs, eval_accuracy, 'g-', linewidth=2, label='Accuracy')
        ax3.plot(epochs, eval_f1, 'orange', linewidth=2, label='F1 Score')
        ax3.set_title('Accuracy & F1 Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Precision and Recall
        ax4.plot(epochs, eval_precision, 'purple', linewidth=2, label='Precision')
        ax4.plot(epochs, eval_recall, 'brown', linewidth=2, label='Recall')
        ax4.set_title('Precision & Recall')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.suptitle('Training Metrics - DeBERTa Italiana')
        plt.tight_layout()
        
        plot_file = f"results/training_metrics_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Grafici salvati in: {plot_file}")
    
    # Save metrics
    metrics_file = f"results/metrics_italian_{timestamp}.json"
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
            "learning_rate": training_args.learning_rate
        }
    }
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(clean_metrics, f, indent=2, ensure_ascii=False)
    
    logger.info("RISULTATI FINALI:")
    logger.info(f"   Tempo training: {training_time}")
    logger.info(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"   F1 Score: {eval_results['eval_f1']:.4f}")
    logger.info(f"Metriche salvate in: {metrics_file}")
    
    logger.info("Performance per classe:")
    for label in LABELS:
        if label in class_report:
            metrics = class_report[label]
            logger.info(f"   {label}: F1={metrics['f1-score']:.3f}")
    
    logger.info("PROCESSO COMPLETATO")

if __name__ == '__main__':
    main() 