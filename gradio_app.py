import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import random
import matplotlib.pyplot as plt
import io
from PIL import Image

SENTIMENT_MODEL_DIR = "./model/deberta_italian_v2/"

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

print("🔄 Caricamento modello sentiment...")
tokenizer_sentiment = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_DIR)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_DIR)
model_sentiment.eval()
print("✅ Modello sentiment caricato con successo!")

def predict_sentiment(text, temperature=2.0):
    """
    Predict sentiment with temperature scaling to reduce overconfidence
    
    Args:
        text (str): Input text
        temperature (float): Temperature for softmax scaling (higher = less confident)
    """
    inputs = tokenizer_sentiment(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        outputs = model_sentiment(**inputs)
        logits_scaled = outputs.logits / temperature
        probs = torch.nn.functional.softmax(logits_scaled, dim=1)[0].numpy()

    label_id = probs.argmax()
    sentiment = SENTIMENT_LABELS[label_id]
    emoji = EMOJIS[sentiment]
    confidence = probs[label_id] * 100

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = [COLORS[label] for label in SENTIMENT_LABELS]
    wedges, texts, autotexts = ax.pie(
        probs,
        labels=SENTIMENT_LABELS,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 2 else '',
        colors=colors,
        startangle=140,
        textprops={'color': 'white', 'fontsize': 12, 'weight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    ax.set_title("Distribuzione delle emozioni", color="white", fontsize=16)
    fig.patch.set_facecolor('#0f0f1c')
    ax.set_facecolor('#0f0f1c')
    plt.setp(texts, color='white')
    plt.setp(autotexts, color='white', weight='bold')
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight', transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    prediction_text = f"<div style='font-size: 24px; font-weight: bold; color: #00ffe5; text-align: center;'>{emoji} Predicted sentiment: <strong>{sentiment.upper()}</strong> ({confidence:.1f}%)</div>"
    return prediction_text, img

def full_pipeline(text):
    if not text.strip():
        return "⚠️ Inserisci del testo per iniziare!", None, None
    
    try:
        sentiment_text, sentiment_plot = predict_sentiment(text)
        return sentiment_text, sentiment_plot
    except Exception as e:
        return f"❌ Errore durante l'elaborazione: {str(e)}", None

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto+Mono:wght@300;400;500&display=swap');

* {
    transition: all 0.3s ease;
}

body {
    background: linear-gradient(135deg, #0a0a15 0%, #1a1a2e 50%, #16213e 100%);
    color: #ffffff;
    font-family: 'Roboto Mono', monospace;
    overflow-x: hidden;
    margin: 0;
    padding: 0;
    width: 100vw;
    min-height: 100vh;
}

.gradio-container {
    background: rgba(10, 10, 21, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 0;
    box-shadow: none;
    border: none;
    max-width: 100vw;
    width: 100vw;
    min-height: 100vh;
    margin: 0;
    padding: 2rem;
}

.main-header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: linear-gradient(45deg, rgba(0, 255, 229, 0.1), rgba(255, 20, 147, 0.1));
    border-radius: 15px;
    border: 1px solid rgba(0, 255, 229, 0.3);
}

.main-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: 3rem;
    background: linear-gradient(45deg, #00ffe5, #ff1493, #ffd700);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient 3s ease infinite;
    text-shadow: 0 0 30px rgba(0, 255, 229, 0.5);
    margin: 0;
}

.subtitle {
    font-family: 'Roboto Mono', monospace;
    font-size: 1.2rem;
    color: #00ffe5;
    margin-top: 1rem;
    opacity: 0.9;
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.input-container {
    background: rgba(26, 26, 46, 0.6);
    border-radius: 15px;
    padding: 1.5rem;
    border: 1px solid rgba(0, 255, 229, 0.3);
    margin: 0 1rem 2rem 0;
    height: fit-content;
}

textarea {
    background: rgba(31, 31, 59, 0.8) !important;
    color: #00ffe5 !important;
    border: 2px solid rgba(0, 255, 229, 0.5) !important;
    border-radius: 10px !important;
    font-family: 'Roboto Mono', monospace !important;
    font-size: 1rem !important;
    padding: 1rem !important;
    transition: all 0.3s ease !important;
}

textarea:focus {
    border-color: #00ffe5 !important;
    box-shadow: 0 0 20px rgba(0, 255, 229, 0.3) !important;
    transform: translateY(-2px) !important;
}

.primary {
    background: linear-gradient(45deg, #00ffe5, #ff1493) !important;
    border: none !important;
    color: #0a0a15 !important;
    font-weight: bold !important;
    border-radius: 25px !important;
    padding: 0.8rem 2rem !important;
    font-family: 'Orbitron', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
}

.primary:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 25px rgba(0, 255, 229, 0.4) !important;
    filter: brightness(1.1) !important;
}

.output-container {
    background: rgba(26, 26, 46, 0.6);
    border-radius: 15px;
    padding: 1.5rem;
    border: 1px solid rgba(0, 255, 229, 0.3);
    margin: 0 0 2rem 1rem;
    height: fit-content;
}

.output-html {
    background: rgba(15, 15, 28, 0.8) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    border: 1px solid rgba(0, 255, 229, 0.3) !important;
}

.output-image {
    border-radius: 10px !important;
    border: 1px solid rgba(0, 255, 229, 0.3) !important;
    overflow: hidden !important;
}

label {
    color: #00ffe5 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin-bottom: 0.5rem !important;
}

.pulse {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(0, 255, 229, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(0, 255, 229, 0); }
    100% { box-shadow: 0 0 0 0 rgba(0, 255, 229, 0); }
}
"""

def create_interface():
    with gr.Blocks(css=custom_css, title="DAFT.punk - Sentiment Analyzer", theme=gr.themes.Base()) as demo:

        with gr.Column(elem_classes="main-header"):
            gr.HTML("""
            <div class="main-title">DAFT.punk</div>
            <div class="subtitle">🎵 DeBERTa for Analysis of Feeling in italian lyrical Texts 🤖</div>
            """)
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes="input-container"):
                gr.HTML("<h3 style='color: #00ffe5; text-align: center; margin-bottom: 1rem;'>📝 Input</h3>")
                
                input_box = gr.Textbox(
                    label="🎶 Inserisci un verso musicale italiano",
                    lines=8,
                    placeholder=random.choice(PLACEHOLDERS),
                    elem_classes="input-field"
                )
                
                submit_btn = gr.Button(
                    "🎧 Analizza", 
                    variant="primary",
                    elem_classes="pulse",
                    size="lg"
                )
            
            with gr.Column(scale=1, elem_classes="output-container"):
                gr.HTML("<h3 style='color: #00ffe5; text-align: center; margin-bottom: 1rem;'>📊 Results</h3>")
                
                output_sentiment = gr.HTML(elem_classes="output-html")
                
                output_img = gr.Image(
                    label="📈 Distribuzione delle Probabilità Emotive",
                    elem_classes="output-image",
                    height=500
                )

        submit_btn.click(
            fn=full_pipeline, 
            inputs=input_box, 
            outputs=[output_sentiment, output_img]
        )

    return demo

def main():
    print("Avvio...")
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,      
        share=False,            
        debug=False
    )

if __name__ == "__main__":
    main() 