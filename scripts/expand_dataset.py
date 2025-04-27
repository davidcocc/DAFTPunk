import os
import json
import time
import openai
from dotenv import load_dotenv

# ⚙️ Configurations
OUTPUT_PATH = "data/raw/lyrics_enriched.json"
BATCH_SIZE = 20
PAUSE_SEC = 1.0
MODEL_NAME = "gpt-4o"
TARGET_SENTIMENTS = ["joy", "sadness", "rage", "love", "nostalgia", "hope", "fear"]

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not found")

client = openai.OpenAI()

# Cost tracking variables
total_input_tokens = 0
total_output_tokens = 0

# 🔥 Prompt to GPT
def build_prompt():
    return [
        {"role": "system", "content": "You are an expert Italian poet and songwriter."},
        {"role": "user", "content": (
            "Inventa un breve estratto originale di una canzone italiana (4-6 righe).\n"
            "Poi parafrasalo poeticamente, mantenendo l'emozione e l'immaginario.\n"
            "Infine, assegna uno ed un solo sentimento predominante tra esclusivamente: joy, sadness, rage, love, nostalgia, hope, fear.\n"
            "Rispondi SOLO in JSON come questo formato:\n"
            '{ "original": "...", "paraphrase": "...", "sentiment": "joy" }\n'
            "Tutti i testi devono essere esclusivamente in lingua italiana."
        )}
    ]

# 🧠 Call GPT

def call_gpt(messages):
    global total_input_tokens, total_output_tokens
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.9,
        max_tokens=300
    )
    usage = response.usage
    total_input_tokens += usage.prompt_tokens
    total_output_tokens += usage.completion_tokens
    return response.choices[0].message.content.strip()

# 🗂️ Load existing dataset (if any)
def load_existing():
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []

# 💰 Calculate approximate cost
def estimate_cost():
    input_cost = (total_input_tokens / 1000) * 0.0005
    output_cost = (total_output_tokens / 1000) * 0.0015
    total = input_cost + output_cost
    return total

# 🚀 Main loop
def generate_batch(n_batches=5):
    dataset = load_existing()
    print(f"✅ Loaded {len(dataset)} existing entries.")

    for batch in range(n_batches):
        print(f"\n🔁 Batch {batch+1}/{n_batches}")
        for i in range(BATCH_SIZE):
            try:
                messages = build_prompt()
                reply = call_gpt(messages)
                item = json.loads(reply)

                if (item["sentiment"] not in TARGET_SENTIMENTS or
                    not item["original"].strip() or
                    not item["paraphrase"].strip()):
                    print(f"⚠️ Skipped invalid entry {i+1}")
                    continue

                dataset.append({
                    "lyrics": item["original"].strip(),
                    "paraphrase": item["paraphrase"].strip(),
                    "sentiment": item["sentiment"]
                })
                print(f"✅ Added entry {len(dataset)}: {item['sentiment']}")
            except Exception as e:
                print(f"⚠️ Error at entry {i+1}: {e}")
                continue
            time.sleep(PAUSE_SEC)

        # Save after each batch
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(dataset)} entries to {OUTPUT_PATH}")
        print(f"💰 Estimated cost so far: €{estimate_cost():.4f}")

if __name__ == "__main__":
    n_batches = int(input("How many batches to generate? (each batch = 20 examples): ").strip())
    generate_batch(n_batches=n_batches)
