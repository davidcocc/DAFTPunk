import csv
import os
import math

INPUT_CSV = "data/raw/lyrics.csv"
OUTPUT_DIR = "data/prompts"
BLOCK_SIZE = 5  # Numero di prompt per batch

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_lyrics(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def create_prompt(entry, index):
    text = entry["excerpt"].replace("\n", " ")
    return (
        f"[{index}]\n"
        f"🎵 Original text:\n{text}\n\n"
        f"🧠 Paraphrase this text keeping the poetic meaning.\n"
        f"❤️ Also label the main emotion (choose from: joy, sadness, rage, love, nostalgia).\n"
        f"✍️ Format the answer as:\n"
        f"Paraphrase: <your version>\nEmotion: <label>\n"
        f"{'-'*60}\n"
    )

def generate_blocks(data, block_size):
    return [data[i:i + block_size] for i in range(0, len(data), block_size)]

def main():
    lyrics_data = load_lyrics(INPUT_CSV)
    print(f"🔍 Loaded {len(lyrics_data)} lyrics excerpts.")

    prompt_blocks = generate_blocks(lyrics_data, BLOCK_SIZE)

    for i, block in enumerate(prompt_blocks, 1):
        prompt_text = ""
        for j, entry in enumerate(block):
            index = (i - 1) * BLOCK_SIZE + j + 1
            prompt_text += create_prompt(entry, index)

        output_path = os.path.join(OUTPUT_DIR, f"prompt_block_{i}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)

        print(f"✅ Saved prompt block {i} to '{output_path}'")

    print("\n🚀 Done! Open the files in 'data/prompts' and paste into ChatGPT.")

if __name__ == "__main__":
    main()
