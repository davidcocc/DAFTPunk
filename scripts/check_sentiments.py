import json

VALID_SENTIMENTS = {"love", "hope", "fear", "joy", "rage", "sadness", "nostalgia"}
INPUT_PATH = "data/raw/lyrics_enriched.json"

with open(INPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)

invalid_entries = [entry for entry in data if entry.get("sentiment") not in VALID_SENTIMENTS]

print(f"🚨 Found {len(invalid_entries)} entries with invalid sentiment labels:\n")
for entry in invalid_entries:
    print(f"- {entry['artist']} – {entry['title']} → {entry.get('sentiment')}")
