# DAFT.punk – DeBERTa for Analysis of Feelings in Texts 🎶

**Project for the Natural Language Processing (NLP) exam – University of Salerno**

## 📝 Project Overview

DAFT.punk is a sentiment analysis system for Italian song lyrics, developed as part of the NLP course at the University of Salerno. The project leverages a fine-tuned DeBERTa model to classify lyrics into seven emotional categories, providing both a web interface and scripts for data processing and model training.

## 🚀 How to Run the Application

```bash
python gradio_app.py
```

The web interface will be available at `http://localhost:7860`

## 📁 Project Structure

- `gradio_app.py` – Main web interface (Gradio)
- `scripts/`
  - `process_lyrics.py` – Script for labeling lyrics with emotion using GPT
  - `train_finetuned_deberta.py` – Script for training the DeBERTa model
  - `genius_fragments_scraper.py` – Scraper for collecting and fragmenting lyrics from Genius
- `model/deberta_sentiment_v2/` – Fine-tuned DeBERTa model
- `data/`
  - `raw/lyrics_enriched.json` – Main dataset (Italian lyrics with emotion labels)
  - `processed/` – Training reports and metric plots

## 🎯 Features

- Sentiment analysis on Italian song lyrics with 7 emotions: joy, sadness, rage, love, nostalgia, hope, fear
- Graphical visualization of emotion probabilities
- Scripts for data scraping, processing, training, and evaluation

## 📊 Model & Dataset

- **Model:** `osiria/deberta-base-italian`, fine-tuned on a custom dataset of Italian song lyrics
- **Dataset:** ~3MB of Italian lyrics, each labeled with one of 7 emotions (labels generated via GPT-4 and manual reviewed)

## 🏆 Model Performance (DeBERTa Sentiment v2)

Please keep in mind that learning has been heavily limited for economical and hardware reasons.

- **Accuracy:** 0.6767
- **Epochs:** 10
- **Batch size:** 16

## ⚙️ Installation & Requirements

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

**Main dependencies:**
- torch
- transformers
- datasets
- gradio
- pandas
- numpy
- openai
- beautifulsoup4

## 📚 Usage

- To launch the web app: `python gradio_app.py`
- To process and label lyrics: `python scripts/process_lyrics.py`
- To train the model: `python scripts/train_finetuned_deberta.py`
