from dotenv import load_dotenv
import os
import json
import time
import openai
from datetime import datetime, timedelta

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env!")

INPUT_PATH  = "../data/raw/lyrics_enriched.json"  
OUTPUT_PATH = "../data/raw/lyrics_enriched.json"  
MODEL_NAME  = "gpt-4o"  
PAUSE_SEC   = 1.0
SAVE_EVERY  = 25        

client = openai.OpenAI()  

def build_prompt(entry):
    lyrics = entry["lyrics"]
    return [
        {"role": "system", "content": "You are an expert in Italian literature and emotion analysis."},
        {"role": "user", "content": (
            "Analyze the emotional sentiment of this Italian song excerpt.\n\n"
            f"Excerpt: \"{lyrics}\"\n\n"
            "Choose ONLY ONE label from: joy, sadness, rage, love, nostalgia, hope, fear.\n"
            "Consider the context, word choice, imagery, and emotional tone.\n"
            "Format your reply as JSON:\n"
            '{ "sentiment": "nostalgia" }'
        )}
    ]

def call_gpt(messages):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.8,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def main():
    start_time = datetime.now()
    print("AVVIO SENTIMENT ANALYSIS")
    print("=" * 60)
    
    print("Caricamento dataset...")
    with open(INPUT_PATH, encoding="utf-8") as f:
        entries = json.load(f)

    new_entries = [entry for entry in entries if not entry.get("sentiment", "").strip()]
    
    print(f"Entries totali nel dataset: {len(entries)}")
    print(f"Entries senza sentiment (da processare): {len(new_entries)}")
    print(f"Entries già processate: {len(entries) - len(new_entries)}")
    
    if not new_entries:
        print("Tutte le entries hanno già sentiment!")
        return

    processed_count = 0
    sentiment_stats = {"joy": 0, "sadness": 0, "rage": 0, "love": 0, "nostalgia": 0, "hope": 0, "fear": 0}
    errors = 0

    print(f"Inizio alle {start_time.strftime('%H:%M:%S')}")
    print("=" * 60)

    for i, entry in enumerate(new_entries, 1):
        try:
            progress_pct = (i / len(new_entries)) * 100
            elapsed = datetime.now() - start_time
            if i > 1:
                avg_time_per_entry = elapsed.total_seconds() / (i - 1)
                remaining_entries = len(new_entries) - i
                eta = datetime.now() + timedelta(seconds=avg_time_per_entry * remaining_entries)
                eta_str = eta.strftime('%H:%M:%S')
            else:
                eta_str = "calcolando..."

            print(f"\n[{i:4}/{len(new_entries)}] ({progress_pct:5.1f}%) - ETA: {eta_str}")
            print(f"   {entry['artist']} - {entry['title']}")
            print(f"   \"{entry['lyrics'][:80]}{'...' if len(entry['lyrics']) > 80 else ''}\"")
            
            messages = build_prompt(entry)
            reply = call_gpt(messages)

            import re
            json_match = re.search(r'\{.*?\}', reply, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in GPT reply")

            parsed = json.loads(json_match.group())
            sentiment = parsed.get("sentiment", "").strip().lower()
            
            # Valida che sia uno dei 7 sentiment permessi
            valid_sentiments = ["joy", "sadness", "rage", "love", "nostalgia", "hope", "fear"]
            if sentiment not in valid_sentiments:
                print(f"   Sentiment non valido '{sentiment}', uso 'nostalgia' come default")
                sentiment = "nostalgia"

            entry["sentiment"] = sentiment
            processed_count += 1
            sentiment_stats[sentiment] += 1

            print(f"   Sentiment: {sentiment.upper()}")

            if i % SAVE_EVERY == 0:
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(entries, f, ensure_ascii=False, indent=2)
                print(f"\nCHECKPOINT #{i//SAVE_EVERY}")
                print(f"   Processate: {processed_count} | Errori: {errors}")
                print(f"   Statistiche sentiment:")
                for sent, count in sentiment_stats.items():
                    if count > 0:
                        print(f"      {sent}: {count}")

        except json.JSONDecodeError as e:
            errors += 1
            print(f"   JSON non valido: {str(e)[:50]}...")
        except Exception as e:
            errors += 1
            print(f"   Errore: {str(e)[:50]}...")
        
        time.sleep(PAUSE_SEC)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    end_time = datetime.now()
    total_time = end_time - start_time

    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS COMPLETATA!")
    print("=" * 60)
    print(f"Entries processate: {processed_count}")
    print(f"Errori: {errors}")
    print(f"Dataset totale: {len(entries)} entries")
    print(f"Tempo totale: {total_time}")
    print(f"Media per entry: {total_time.total_seconds()/len(new_entries):.1f}s")
    print("\nDISTRIBUZIONE SENTIMENT:")
    for sentiment, count in sorted(sentiment_stats.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = (count / processed_count) * 100
            print(f"   {sentiment.ljust(10)}: {count:4} ({pct:5.1f}%)")

if __name__ == "__main__":
    main()
