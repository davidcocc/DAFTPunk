import requests
from bs4 import BeautifulSoup
import json
import os
import time
import hashlib

BASE_URL    = "https://www.lyrics.com/"
SEARCH_URL  = BASE_URL + "serp.php?st={query}&qtype=2"
HEADERS     = {"User-Agent": "Mozilla/5.0"}
OUTPUT_PATH = "data/raw/lyrics.json"
MAX_SONGS   = 30
EXCERPT_LINES = 6

def get_song_links(artist_query, max_songs=MAX_SONGS):
    url = SEARCH_URL.format(query=artist_query.replace(" ", "+"))
    resp = requests.get(url, headers=HEADERS)
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, "html.parser")

    artist_anchor = soup.find("a", href=True, text=lambda t: t and artist_query.lower() in t.lower())
    if not artist_anchor:
        print(f"❌ Artist '{artist_query}' not found.")
        return []
    artist_url = BASE_URL + artist_anchor["href"]

    resp = requests.get(artist_url, headers=HEADERS)
    soup = BeautifulSoup(resp.content, "html.parser", from_encoding="ISO-8859-1")

    links = []
    for td in soup.find_all("td", class_="tal qx")[:max_songs]:
        a = td.find("a", href=True)
        if a:
            links.append(BASE_URL + a["href"])
    return links

def extract_first_excerpt(url):
    resp = requests.get(url, headers=HEADERS)
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    div = soup.find("pre", id="lyric-body-text")
    if not div:
        return title, None

    full = div.get_text(separator="\n").strip()
    stanzas = [s.strip() for s in full.split("\n\n") if s.strip()]
    if not stanzas:
        return title, None

    lines = [l.strip() for l in stanzas[0].splitlines() if l.strip()]
    excerpt = " ".join(lines[:EXCERPT_LINES])
    return title, excerpt

def hash_entry(e):
    return hashlib.sha256((e["artist"] + e["title"] + e["lyrics"]).encode("utf-8")).hexdigest()

def load_existing_hashes(path):
    seen = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for entry in json.load(f):
                seen.add(hash_entry(entry))
    return seen

def scrape_artist(artist, seen_hashes):
    links = get_song_links(artist, MAX_SONGS)
    new_entries = []

    for i, url in enumerate(links, 1):
        print(f"🎵 [{i}/{len(links)}] {artist} → {url}")
        title, excerpt = extract_first_excerpt(url)
        if excerpt:
            entry = {"artist": artist, "title": title, "url": url, "lyrics": excerpt}
            h = hash_entry(entry)
            if h not in seen_hashes:
                new_entries.append(entry)
                seen_hashes.add(h)
            else:
                print("⚠️ Duplicate; skipped.")
        time.sleep(1)

    return new_entries

def scrape_multiple_artists(artists, output_path=OUTPUT_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    seen_hashes = load_existing_hashes(output_path)
    all_new = []

    for artist in artists:
        print(f"\n🎤 Scraping songs for: {artist}")
        new_entries = scrape_artist(artist, seen_hashes)
        all_new.extend(new_entries)

    if not all_new:
        print("ℹ️ No new lyrics added.")
        return

    all_data = []
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            all_data = json.load(f)
    all_data.extend(all_new)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Appended {len(all_new)} entries to '{output_path}'")

if __name__ == "__main__":
    artist_list = [
        "Fulminacci",
        "Franco126",
        "Frah Quintale",
        "Mecna",
        "Offlaga Disco Pax",
        "Andrea Laszlo de Simone",
        "Postino"
    ]
    scrape_multiple_artists(artist_list)
    print("\n🎉 Scraping completo.")
