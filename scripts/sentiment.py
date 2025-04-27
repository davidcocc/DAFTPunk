import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch
import random
import matplotlib.pyplot as plt
import io
from PIL import Image

# 📂 Load models and tokenizers
SENTIMENT_MODEL_DIR = "./model/deberta_sentiment/checkpoint-1072"
PARAPHRASE_MODEL_DIR = "./model/it5_paraphrase"

SENTIMENT_LABELS = ["joy", "sadness", "rage", "love", "nostalgia", "hope", "fear"]
EMOJIS = {
    "joy": "🌟",
    "sadness": "😢",
    "rage": "😡",
    "love": "❤️",
    "nostalgia": "🍂",
    "hope": "🌈",
    "fear": "😱"
}
COLORS = {
    "joy": "#FFD700",
    "sadness": "#1E90FF",
    "rage": "#FF4500",
    "love": "#FF69B4",
    "nostalgia": "#FFB6C1",
    "hope": "#32CD32",
    "fear": "#9370DB"
}

PLACEHOLDERS = [
    "La pioggia scende fredda su di me, Pesaro è una donna intelligente...",
    "La verità è che ti fa paura l'idea di scomparire...",
    "Non avrò bisogno delle medicine, degli psicofarmaci del Lexotan, dei rimedi in casa, della valeriana...",
    "Ho tirato pugni da ogni parte solo per uscire da un sacchetto di carta...",
    "Mister Tamburino, non ho voglia di scherzare, rimettiamoci la maglia i tempi stanno per cambiare...",
    "Andrai comunque, andrai come se n'è andato chiunque, andrai comunque, per come sei potresti andare ovunque...",
    "Fiorivi, sfiorivano le viole...",
    "Sapessi che felicità mi dà l'idea di non vederti più, di non pensarti più qualsiasi cosa mi dirai...",
    "Se c'è una cosa che odio di più è che non posso vederti quando ti spogli con una canzone dei Doors...",
    "E se io muoio da partigiano, o bella ciao, bella ciao, bella ciao ciao ciao, tu mi devi seppellir...",
    "Non è più inverno per noi ma dammi un'altra coperta...",
    "Curami, curami, curami! Prendimi in cura da te, che ti venga voglia di me...",
    "Da grande voglio fare il giovinastro, ma forse non farò bene neanche quello...",
    "Parco Sempione, verde e marrone, dentro la mia città..."
]

# Load tokenizers and models
tokenizer_sentiment = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_DIR)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_DIR)
model_sentiment.eval()

tokenizer_paraphrase = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL_DIR)
model_paraphrase = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL_DIR)
model_paraphrase.eval()

# Predict function
def generate_paraphrase(text):
    input_text = "parafrasa: " + text
    inputs = tokenizer_paraphrase(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        outputs = model_paraphrase.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    paraphrase = tokenizer_paraphrase.decode(outputs[0], skip_special_tokens=True)
    return paraphrase

def predict_sentiment(text):
    inputs = tokenizer_sentiment(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        outputs = model_sentiment(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].numpy()

    label_id = probs.argmax()
    sentiment = SENTIMENT_LABELS[label_id]
    emoji = EMOJIS[sentiment]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [COLORS[label] for label in SENTIMENT_LABELS]
    ax.bar(SENTIMENT_LABELS, probs, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title("Distribuzione delle emozioni", color="white", fontsize=14)
    ax.set_ylabel("Probabilità", color="white")
    ax.set_xticklabels(SENTIMENT_LABELS, rotation=45, color="white")
    ax.set_yticklabels([f"{x:.0%}" for x in ax.get_yticks()], color="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.tick_params(colors="white")
    fig.patch.set_facecolor('#0f0f1c')
    ax.set_facecolor('#0f0f1c')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    img = Image.open(buf)

    prediction_text = f"<div style='font-size: 24px; font-weight: bold; color: #00ffe5; text-align: center;'>{emoji} Predicted sentiment: <strong>{sentiment.upper()}</strong></div>"
    return prediction_text, img

def full_pipeline(text):
    paraphrase = generate_paraphrase(text)
    sentiment_text, sentiment_plot = predict_sentiment(paraphrase)
    return paraphrase, sentiment_text, sentiment_plot

# Custom CSS
custom_css = """
body {
    background-color: #0f0f1c;
    color: white;
    font-family: 'Orbitron', sans-serif;
}
.gradio-container {
    font-family: 'Orbitron', sans-serif;
}
textarea, .output-markdown {
    background-color: #1a1a2e !important;
    color: #00ffe5 !important;
    border: 1px solid #00ffe5;
}
button {
    background-color: #1f1f3b !important;
    border: 1px solid #00ffe5;
    color: #00ffe5;
}
img.logo {
    display: block;
    margin: auto;
    max-width: 300px;
    padding-bottom: 1rem;
}
.output-image, .output-markdown {
    margin-top: 20px;
    text-align: center;
}
"""

# Gradio UI with Blocks for vertical layout
def create_interface():
    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("""
        <img class='logo' src='https://upload.wikimedia.org/wikipedia/en/9/9c/Daft_Punk_logo.svg'>
        <h1 style='text-align: center;'>✨ DAFT.punk – Lyrics Analyzer</h1>
        <p style='text-align: center;'>Genera una parafrasi poetica di un verso musicale italiano e analizza il sentimento. Powered by DeBERTa & it5 🎧</p>
        """)
        input_box = gr.Textbox(
            label="🎶 Paste an Italian lyric excerpt",
            lines=4,
            placeholder=random.choice(PLACEHOLDERS)
        )
        submit_btn = gr.Button("🎧 Genera Parafrasi e Analizza")
        output_paraphrase = gr.Textbox(label="📝 Paraphrase Output")
        output_sentiment = gr.HTML()
        output_img = gr.Image(label="Emotion Probabilities")

        submit_btn.click(fn=full_pipeline, inputs=input_box, outputs=[output_paraphrase, output_sentiment, output_img])

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
