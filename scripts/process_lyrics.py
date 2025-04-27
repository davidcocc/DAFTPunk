from dotenv import load_dotenv
import os
import json
import time
import openai

# 1. Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("❌ OPENAI_API_KEY not found in .env!")

# 2. Configuration
INPUT_PATH  = "data/raw/lyrics.json"
OUTPUT_PATH = "data/raw/lyrics_enriched.json"
MODEL_NAME  = "gpt-4o"  # GPT-4.1 mini
PAUSE_SEC   = 1.0
SAVE_EVERY  = 25        # salvataggio parziale ogni 25 entry

client = openai.OpenAI()  # Nuova interfaccia SDK >= 1.x

# 3. Prompt builder
def build_prompt(entry):
    lyrics = entry["lyrics"]
    return [
        {"role": "system", "content": "You are an expert Italian poet and lyricist."},
        {"role": "user", "content": (
            "Paraphrase the following Italian song excerpt, "
            "preserving emotional nuance, and imagery, in a couple of phrases at maximum. Avoid generic statements.\n\n"
            f"Excerpt: “{lyrics}”\n\n"
            "Then choose the most fitting label from: joy, sadness, rage, love, nostalgia, hope, fear.\n"
            "Format your reply as JSON:\n"
            '{ "paraphrase": "...", "sentiment": "joy" }'
        )}
    ]

# 4. Call GPT safely
def call_gpt(messages):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.8,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# 5. Main enrichment loop
def main():
    # Carica dati originali
    with open(INPUT_PATH, encoding="utf-8") as f:
        entries = json.load(f)

    enriched = []

    for i, entry in enumerate(entries, 1):
        try:
            messages = build_prompt(entry)
            reply = call_gpt(messages)

            # parsing sicuro
            import re

            # Estrai il blocco JSON "pulito" dal testo di risposta
            json_match = re.search(r'\{.*?\}', reply, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in GPT reply")

            parsed = json.loads(json_match.group())

            entry["paraphrase"] = parsed.get("paraphrase", "").strip()
            entry["sentiment"] = parsed.get("sentiment", "").strip().lower()

            enriched.append(entry)
            print(f"✅ [{i}/{len(entries)}] {entry['artist']} – {entry['title']} parafrasata!")

            # salvataggio parziale
            if i % SAVE_EVERY == 0:
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(enriched, f, ensure_ascii=False, indent=2)
                print("💾 Salvataggio parziale eseguito.")

        except json.JSONDecodeError:
            print(f"⚠️ GPT response not valid JSON at entry {i}. Skipping.")
        except Exception as e:
            print(f"❌ Error at entry {i}: {e}")
        time.sleep(PAUSE_SEC)

    # salvataggio finale
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Enrichment completo. Salvato tutto in '{OUTPUT_PATH}'")

if __name__ == "__main__":
    main()
