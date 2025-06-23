import json
import requests
from bs4 import BeautifulSoup
import time
import re
import random

class GeniusFragmentsScraper:
    def __init__(self):
        self.genius_token = "YOUR_GENIUS_TOKEN"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Authorization': f'Bearer {self.genius_token}'
        }
        
        self.existing_lyrics = set()
        self.processed_songs = set()

    def load_existing_dataset(self):
        try:
            with open('../data/raw/lyrics_enriched.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            songs = []
            seen_songs = set()
            
            for entry in data:
                lyrics_clean = re.sub(r'[^\w\s]', '', entry.get('lyrics', '').lower())
                self.existing_lyrics.add(lyrics_clean)
                
                if 'artist' in entry and 'title' in entry:
                    song_key = f"{entry['artist']}|{entry['title']}"
                    if song_key not in seen_songs:
                        songs.append({
                            'artist': entry['artist'],
                            'title': entry['title']
                        })
                        seen_songs.add(song_key)
            
            print(f"Dataset entries: {len(data)}")
            print(f"Canzoni uniche: {len(songs)}")
            print(f"Lyrics esistenti: {len(self.existing_lyrics)}")
            
            return data, songs
            
        except Exception as e:
            print(f"Errore caricamento dataset: {e}")
            return [], []

    def search_genius_api(self, artist, title):
        try:
            query = f"{artist} {title}"
            url = "https://api.genius.com/search"
            params = {'q': query}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                hits = data.get('response', {}).get('hits', [])
                
                for hit in hits:
                    result = hit.get('result', {})
                    song_title = result.get('title', '').lower()
                    artist_name = result.get('primary_artist', {}).get('name', '').lower()
                    
                    if (artist.lower() in artist_name or artist_name in artist.lower()):
                        if (title.lower() in song_title or song_title in title.lower()):
                            return result.get('url')
                
                for hit in hits:
                    result = hit.get('result', {})
                    artist_name = result.get('primary_artist', {}).get('name', '').lower()
                    if artist.lower() in artist_name:
                        return result.get('url')
            
            return None
            
        except Exception as e:
            print(f"   Errore API: {e}")
            return None

    def extract_genius_lyrics(self, genius_url):
        try:
            response = requests.get(genius_url, headers={'User-Agent': self.headers['User-Agent']}, timeout=15)
            
            if response.status_code != 200:
                return ""
            
            soup = BeautifulSoup(response.text, 'html.parser')

            selectors = [
                'div[data-lyrics-container="true"]',
                'div[class*="Lyrics__Container"]', 
                'div.lyrics',
                'div[class*="lyrics"]'
            ]
            
            full_text = ""
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        text = element.get_text(separator='\n', strip=True)
                        full_text += text + '\n'
                    break
            
            if len(full_text.strip()) > 50:
                cleaned_text = self.clean_genius_metadata(full_text)
                return cleaned_text
            
            return ""
            
        except Exception as e:
            print(f"   Errore estrazione: {e}")
            return ""

    def clean_genius_metadata(self, text):
        patterns_to_remove = [
            r'\d+\s*Contributors?',
            r'Lyrics?$',
            r'^\d+\s*',
            r'"[^"]*" è (il|la|uno|una).*?singolo.*?\.',
            r'Il brano è stato pubblicato.*?\.',
            r'La canzone è stata pubblicata.*?\.',
            r'Produced by.*',
            r'Written by.*',
            r'Mixed by.*',
            r'Mastered by.*',
            r'Album:.*',
            r'Released:.*',
            r'Genre:.*',
            r'View.*Lyrics',
            r'See.*Live',
            r'Get tickets.*',
            r'About.*',
            r'.*genius\.com.*',
            r'.*Embed.*',
            r'.*transcript.*',
            r'.*annotation.*'
        ]
        
        cleaned = text
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)

        lines = cleaned.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not self.is_metadata_line(line):
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)

    def is_metadata_line(self, line):
        metadata_keywords = [
            'contributors', 'lyrics', 'produced', 'written', 'mixed', 'mastered',
            'album', 'released', 'genre', 'embed', 'transcript', 'annotation',
            'singolo estratto', 'è stato pubblicato', 'brano è stato', 'vanta la collaborazione'
        ]
        
        line_lower = line.lower()
        return any(keyword in line_lower for keyword in metadata_keywords)

    def create_fragments_by_linebreaks(self, full_lyrics):
        text = re.sub(r'\[.*?\]', '', full_lyrics)  # Rimuove [Verse], [Chorus], [Strofa], [Strumentale], [Ritornello] etc.
        text = re.sub(r'\(.*?\)', '', text)         # Rimuove (note)
        text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)  # Rimuove numerazione
        
        text = re.sub(r'\[Strofa.*?\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[Strumentale.*?\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[Ritornello.*?\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[Pre-Ritornello.*?\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[Bridge.*?\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[Outro.*?\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[Intro.*?\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[Testo.*?\]', '', text, flags=re.IGNORECASE)

        lines = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 5]
        
        fragments = []
        fragment_size = 4  
        start_from = 6
        
        for i in range(start_from, len(lines), fragment_size):
            if len(fragments) >= 10:
                break

            fragment_lines = lines[i:i + fragment_size]
            
            if len(fragment_lines) >= 2:
                fragment = ' '.join(fragment_lines)
                
                if self.is_good_fragment(fragment):
                    fragments.append(fragment)

        for i in range(start_from, len(lines) - 2):

            if len(fragments) >= 10:
                break
                
            fragment_lines = lines[i:i + 3]
            fragment = ' '.join(fragment_lines)
            
            if self.is_good_fragment(fragment) and fragment not in fragments:
                fragments.append(fragment)
        
        return fragments

    def is_good_fragment(self, fragment, min_words=8, max_words=50):
        words = fragment.split()
        if len(words) < min_words or len(words) > max_words:
            return False
        
        fragment_clean = re.sub(r'[^\w\s]', '', fragment.lower())
        if fragment_clean in self.existing_lyrics:
            return False
        
        if len(set(words)) / len(words) < 0.6: 
            return False
        
        common_words = set(['la', 'il', 'di', 'che', 'e', 'a', 'un', 'per', 'con', 'da', 'su', 'del', 'al', 'le', 'in'])
        unique_words = set(word.lower() for word in words) - common_words
        if len(unique_words) < 3:
            return False
        
        special_chars = sum(1 for char in fragment if not char.isalnum() and char != ' ')
        if special_chars > len(fragment) * 0.2:
            return False
        
        return True

    def process_single_song(self, artist, title):
        song_key = f"{artist}|{title}"
        if song_key in self.processed_songs:
            return []
        
        print(f"   Processando: {artist} - {title}")
        
        try:
            genius_url = self.search_genius_api(artist, title)
            
            if not genius_url:
                print(f"      Non trovata su Genius")
                self.processed_songs.add(song_key)
                return []
            
            print(f"      Trovata: {genius_url}")
            
            full_lyrics = self.extract_genius_lyrics(genius_url)
            
            if not full_lyrics:
                print(f"      Nessun testo estratto")
                self.processed_songs.add(song_key)
                return []
            
            print(f"      Estratto testo di {len(full_lyrics)} caratteri")
            
            fragments = self.create_fragments_by_linebreaks(full_lyrics)
            
            if not fragments:
                print(f"      Nessun frammento valido")
                self.processed_songs.add(song_key)
                return []
            
            new_entries = []
            for fragment in fragments:
                new_entries.append({
                    'artist': artist,
                    'title': title,
                    'lyrics': fragment,
                    'sentiment': '' 
                })
            
            print(f"      Creati {len(new_entries)} frammenti")
            self.processed_songs.add(song_key)
            
            return new_entries
            
        except Exception as e:
            print(f"      Errore: {e}")
            self.processed_songs.add(song_key)
            return []

    def concatenate_to_dataset(self, new_entries):
        if not new_entries:
            return
        
        try:
            with open('../data/raw/lyrics_enriched.json', 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            combined_data = existing_data + new_entries
            
            with open('../data/raw/lyrics_enriched.json', 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, ensure_ascii=False, indent=2)
            
            print(f"   Dataset aggiornato: {len(existing_data)} -> {len(combined_data)} entries")
            
        except Exception as e:
            print(f"   Errore nel salvataggio: {e}")

    def run(self):
        print("Avvio Genius Fragments Scraper")
        print("=" * 50)
        
        existing_data, unique_songs = self.load_existing_dataset()
        
        if not unique_songs:
            print("Nessuna canzone trovata nel dataset!")
            return
        
        print(f"Canzoni da processare: {len(unique_songs)}")
        print("=" * 50)
        
        all_new_entries = []
        processed_count = 0
        errors = 0
        
        random.shuffle(unique_songs)
        
        for i, song in enumerate(unique_songs, 1):
            try:
                print(f"\n[{i}/{len(unique_songs)}] ({(i/len(unique_songs)*100):.1f}%)")
                
                new_entries = self.process_single_song(song['artist'], song['title'])
                
                if new_entries:
                    all_new_entries.extend(new_entries)
                    processed_count += 1
                
                if processed_count > 0 and processed_count % 10 == 0:
                    print(f"\n   Checkpoint: salvando {len(all_new_entries)} nuove entries...")
                    self.concatenate_to_dataset(all_new_entries)
                    all_new_entries = [] 

                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                errors += 1
                print(f"   Errore generale: {e}")
                continue
        
        if all_new_entries:
            print(f"\nSalvataggio finale di {len(all_new_entries)} entries...")
            self.concatenate_to_dataset(all_new_entries)
        
        print("\n" + "=" * 50)
        print("SCRAPING COMPLETATO")
        print("=" * 50)
        print(f"Canzoni processate con successo: {processed_count}")
        print(f"Errori: {errors}")
        print(f"Nuove entries create: {len(all_new_entries)}")

if __name__ == "__main__":
    scraper = GeniusFragmentsScraper()
    scraper.run() 