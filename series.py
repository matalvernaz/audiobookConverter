#!/usr/bin/env python3
import os
import sys
import json
import shutil
import subprocess
import urllib.request
import urllib.parse
try:
    import termios  # POSIX only — used to flush stdin before interactive prompts
except ImportError:
    termios = None
import re
import signal
import argparse
from pathlib import Path
from difflib import SequenceMatcher

def clear_input_buffer():
    sys.stdout.flush()
    if termios is None:
        return
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass

def load_memory(target_dir):
    mem_file = Path(target_dir) / ".series_memory.json"
    if mem_file.exists():
        try:
            with open(mem_file, 'r') as f:
                data = json.load(f)
                if data and isinstance(list(data.values())[0], list):
                    return {}
                return data
        except Exception:
            return {}
    return {}

def save_memory(memory, target_dir):
    mem_file = Path(target_dir) / ".series_memory.json"
    with open(mem_file, 'w') as f:
        json.dump(memory, f, indent=4)

def get_audio_metadata(filepath):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(filepath)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        data = json.loads(result.stdout)
        tags = data.get('format', {}).get('tags', {})
        tags_lower = {k.lower(): v for k, v in tags.items()}
        
        author = tags_lower.get('artist', tags_lower.get('album_artist', ''))
        album = tags_lower.get('album', '')
        title = tags_lower.get('title', '')
        return author.strip(), album.strip(), title.strip()
    except Exception:
        return "", "", ""

def parse_filename(filename):
    parts = filename.split(" - ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None, filename.strip()

def extract_series_number(text_list):
    """Pull a series position from title/tag strings.

    Only matches numbers that are explicitly anchored to a series keyword
    (Book/Volume/Vol/Part) or a '#N' marker. A bare trailing number is NOT a
    series number — "Apollo 13", "Catch 22" and "1984" would all be misread,
    producing authoritative-looking but wrong filenames.
    """
    patterns = [
        r'(?:Book|Volume|Vol|Part)\s*#?\s*0*(\d{1,3})\b',
        r'#\s*0*(\d{1,3})\b',
    ]
    for text in text_list:
        if not text:
            continue
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                return str(int(match.group(1))).zfill(2)
    return None

def clean_title(title):
    cleaned = re.sub(r'\s*[\[\(]?(disc|cd|part|volume)\s*\d+[\]\)]?', '', title, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*[\[\(]?(book)\s*\d+[\]\)]?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*[\[\(]?(unabridged|abridged|isis audio books|corgi audio|bbc radio|128br|160br|192br|vbr|mp3|m4b)[\]\)]?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*[\(\[]?\d{1,2}/\d{1,2}/\d{2,4}.*[\)\]]?', '', cleaned)
    cleaned = re.sub(r'^\d+(?:\.\d+)?\s*[\-\.]\s+', '', cleaned)
    cleaned = re.sub(r'\s+series$', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip(' -_,')

def fetch_series_roster(author, series):
    titles = set()
    author_query = author if author and author.lower() != "unknown" else ""
    
    try:
        query = urllib.parse.quote(f"{author_query} {series}".strip())
        url = f"https://itunes.apple.com/search?term={query}&media=audiobook&limit=25"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())
            for item in data.get('results', []):
                t = clean_title(item.get('collectionName', ''))
                if t: titles.add(t.lower())
    except Exception: pass

    try:
        q_str = f'inauthor:"{author_query}" "{series}"' if author_query else f'"{series}"'
        query = urllib.parse.quote(q_str)
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=25"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())
            for item in data.get('items', []):
                t = clean_title(item.get('volumeInfo', {}).get('title', ''))
                if t: titles.add(t.lower())
    except Exception: pass

    return list(titles)

# Patterns that extract a series name (group 1) from a title/collection string.
# Order matters — first match wins per string.
_series_patterns = [
    # "Series Name, Book 3: Title" (e.g. "The Bartimaeus Trilogy, Book Three: Ptolemy's Gate")
    re.compile(r'^(.*?),\s*(?:Book|Volume|Vol|Part)\s*(?:\d+|[A-Za-z]+)', re.IGNORECASE),
    # "Title: Series Name, Book 3"
    re.compile(r'[:\-]\s*(.*?),\s*(?:Book|Volume|Vol|Part)\s*(?:\d+|[A-Za-z]+)', re.IGNORECASE),
    # "(Series Name, Book 3)" or "(Series Name Book 3)"
    re.compile(r'\((.*?),?\s*(?:Book|Volume|Vol|Part)\s*(?:\d+|[A-Za-z]+)\)', re.IGNORECASE),
]

def hunt_for_series_clue(title, author):
    author_query = author if author and author.lower() != "unknown" else ""
    
    # 1. AUDNEXUS (Audible Bridge) - The Gold Standard
    try:
        t_query = urllib.parse.quote(title)
        a_query = urllib.parse.quote(author_query)
        url = f"https://api.audnex.us/books?title={t_query}&author={a_query}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=7) as response:
            data = json.loads(response.read())
            # Audnexus usually returns an array directly or inside 'data'/'results'
            results = data.get('data', data.get('results', [])) if isinstance(data, dict) else data
            for item in results:
                series_info = item.get('series', [])
                if series_info and isinstance(series_info, list):
                    return series_info[0].get('title', '').strip()
    except Exception: pass

    # 2. APPLE ITUNES FALLBACK
    try:
        query = urllib.parse.quote(f"{title} {author_query}".strip())
        url = f"https://itunes.apple.com/search?term={query}&media=audiobook&limit=3"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())
            for item in data.get('results', []):
                collection = item.get('collectionName', '')
                for pat in _series_patterns:
                    m = pat.search(collection)
                    if m: return m.group(1).strip()
    except Exception: pass

    # 3. GOOGLE BOOKS FALLBACK
    try:
        query = urllib.parse.quote(f"{title} {author_query}".strip())
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=3"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())
            for item in data.get('items', []):
                vol = item.get('volumeInfo', {})
                full_text = vol.get('title', '') + " " + vol.get('subtitle', '') + " " + vol.get('description', '')
                match = re.search(r'(?:Book|Volume|Part)\s*(?:\d+|[A-Za-z]+)\s*of\s*(?:the\s*)?([A-Z][a-zA-Z\s]+(?:Series|Chronicles|Cycle|Saga|Trilogy)?)', full_text, re.IGNORECASE)
                if match: return match.group(1).strip()
                match2 = re.search(r'\((.*?),\s*(?:Book|Volume|Part)\s*(?:\d+|[A-Za-z]+)\)', vol.get('title', ''), re.IGNORECASE)
                if match2: return match2.group(1).strip()
    except Exception: pass
    return None

def search_series(query):
    """Search APIs with a user query and return a list of unique series names found."""
    series_names = set()

    # iTunes: extract series from collection names
    try:
        q = urllib.parse.quote(query)
        url = f"https://itunes.apple.com/search?term={q}&media=audiobook&limit=25"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=7) as response:
            data = json.loads(response.read())
            for item in data.get('results', []):
                collection = item.get('collectionName', '')
                for pat in _series_patterns:
                    m = pat.search(collection)
                    if m:
                        series_names.add(m.group(1).strip())
    except Exception: pass

    # Google Books: extract series from title/subtitle/description
    try:
        q = urllib.parse.quote(query)
        url = f"https://www.googleapis.com/books/v1/volumes?q={q}&maxResults=20"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=7) as response:
            data = json.loads(response.read())
            for item in data.get('items', []):
                vol = item.get('volumeInfo', {})
                for text in [vol.get('title', ''), vol.get('subtitle', '')]:
                    for pat in _series_patterns:
                        m = pat.search(text)
                        if m:
                            series_names.add(m.group(1).strip())
                desc = vol.get('description', '')
                m = re.search(r'(?:Book|Volume|Part)\s*(?:\d+|[A-Za-z]+)\s*(?:of|in)\s*(?:the\s*)?([A-Z][a-zA-Z\s\']+?)(?:\s*series|\s*trilogy|\s*saga|\s*cycle|[,\.\!])', desc, re.IGNORECASE)
                if m:
                    series_names.add(m.group(1).strip())
    except Exception: pass

    # Audnexus
    try:
        q = urllib.parse.quote(query)
        url = f"https://api.audnex.us/books?title={q}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=7) as response:
            data = json.loads(response.read())
            results = data.get('data', data.get('results', [])) if isinstance(data, dict) else data
            for item in results:
                series_info = item.get('series', [])
                if series_info and isinstance(series_info, list):
                    name = series_info[0].get('title', '').strip()
                    if name:
                        series_names.add(name)
    except Exception: pass

    return sorted(series_names)

def interactive_series_router(filepath, target_dir, memory):
    file_author, file_title = parse_filename(filepath.stem)
    meta_author, meta_album, meta_title = get_audio_metadata(filepath)
    
    author = file_author if file_author else (meta_author if meta_author else "Unknown")
    
    best_title = meta_album if meta_album and not meta_album.lower().startswith("chapter") else file_title
    if best_title == file_title and meta_title: 
        best_title = meta_title
        
    candidates = {clean_title(t).lower() for t in [meta_album, meta_title, file_title] if t}
    
    series_num = extract_series_number([filepath.stem, meta_album, meta_title, file_title])
    clean_t = clean_title(best_title)
    
    print(f"\n{'='*50}")
    print(f"File:   {filepath.name}")
    print(f"Author: {author}")
    print(f"Title:  {clean_t}")
    if series_num:
        print(f"Detected Book #: {series_num}")
    print(f"{'='*50}")
    
    # Auto-route to a known series only on a HIGH-confidence, unambiguous match.
    # A loose 0.8 ratio on short titles silently misfiles books into the wrong
    # series folder, so require a strong score, a real title length, and a clear
    # margin over the runner-up before moving without asking.
    AUTO_MATCH_MIN_SCORE = 0.92
    AUTO_MATCH_MIN_MARGIN = 0.08
    AUTO_MATCH_MIN_LEN = 8
    if author in memory:
        scored = []  # (ratio, series_name)
        for known_series, known_titles in memory[author].items():
            for kt in known_titles:
                for c_t in candidates:
                    if c_t and len(c_t) >= AUTO_MATCH_MIN_LEN:
                        scored.append((SequenceMatcher(None, c_t, kt).ratio(), known_series))
        scored.sort(reverse=True)
        if scored and scored[0][0] >= AUTO_MATCH_MIN_SCORE:
            best_score, best_series = scored[0]
            # Unambiguous = no runner-up from a DIFFERENT series within the margin.
            rival = next((s for s, srs in scored[1:] if srs != best_series), 0.0)
            if best_score - rival >= AUTO_MATCH_MIN_MARGIN:
                print(f"    [+] Auto-matched to downloaded roster: {author} - {best_series}  (score {best_score:.2f})")
                return author, best_series, clean_t, series_num

    print("[*] Hunting for series clues online (Checking Audible...)")
    api_guess = hunt_for_series_clue(best_title, author)
    known_series_list = list(memory.get(author, {}).keys()) if author != "Unknown" else []
    
    clear_input_buffer()
    options = []
    
    if api_guess:
        options.append(("Accept API Detective Guess", api_guess))
        
    for s in known_series_list:
        if s != api_guess:
            options.append(("Use Known Series", s))
            
    search_idx = len(options) + 1
    manual_idx = search_idx + 1
    standalone_idx = manual_idx + 1
    skip_idx = standalone_idx + 1

    if options:
        print("\n--- Detected Options ---")
        for i, (desc, val) in enumerate(options, 1):
            print(f" {i}) {desc}: '{val}'")

    print("\n--- Manual Options ---")
    print(f" {search_idx}) Search - Look up a series name online")
    print(f" {manual_idx}) Manual - Type a completely new Series Name")
    print(f" {standalone_idx}) Standalone - Move to '{author}/Standalone/'")
    print(f" {skip_idx}) Skip - Leave file exactly where it is")

    while True:
        try:
            choice = input(f"\nSelect [1-{skip_idx}]: ").strip()
            if not choice: continue
            ival = int(choice)

            if 1 <= ival <= len(options):
                return author, options[ival-1][1], clean_t, series_num
            elif ival == search_idx:
                search_query = input("    [?] Search query (e.g. author + series or book title): ").strip()
                if not search_query:
                    continue
                print(f"    [*] Searching for '{search_query}'...")
                results = search_series(search_query)
                if not results:
                    print("    [!] No series found. Try a different query or use Manual.")
                    continue
                print("\n    --- Search Results ---")
                for j, name in enumerate(results, 1):
                    print(f"     {j}) {name}")
                print(f"     0) None of these — back to main menu")
                pick = input(f"\n    Pick [0-{len(results)}]: ").strip()
                try:
                    pick_val = int(pick)
                    if 1 <= pick_val <= len(results):
                        picked_series = results[pick_val - 1]
                        new_author = input(f"    [?] Author (press Enter to keep '{author}'): ").strip()
                        return new_author if new_author else author, picked_series, clean_t, series_num
                except ValueError: pass
                continue
            elif ival == manual_idx:
                new_series = input("    [?] Type the correct Series Name: ").strip()
                new_author = input(f"    [?] Type the correct Author (press Enter to keep '{author}'): ").strip()
                final_author = new_author if new_author else author
                return final_author, new_series, clean_t, series_num
            elif ival == standalone_idx:
                return author, "Standalone", clean_t, series_num
            elif ival == skip_idx:
                return None, None, None, None
            print("Invalid choice.")
        except ValueError: pass

def unique_path(path):
    """Return `path`, or `path` with a ' (N)' suffix if it already exists, so a
    name collision keeps both files instead of silently dropping one."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    i = 2
    while True:
        candidate = parent / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def cleanup_empty_dirs(root, dry_run=False):
    removed = 0
    for dirpath, dirnames, filenames in os.walk(str(root), topdown=False):
        d = Path(dirpath)
        if d == root:
            continue
        if not any(d.iterdir()):
            if dry_run:
                print(f"    [~] DRY RUN: Would remove empty folder -> {d.relative_to(root)}")
            else:
                d.rmdir()
                print(f"    [-] Removed empty folder: {d.relative_to(root)}")
            removed += 1
    return removed

def process_directory(target_dir, dry_run=False):
    root = Path(target_dir).resolve()
    memory = load_memory(target_dir)

    m4b_files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == '.m4b']

    if not m4b_files:
        print(f"No loose .m4b files found in {target_dir}.")
        print("\n[*] Checking for empty folders...")
        cleanup_empty_dirs(root, dry_run)
        return

    print(f"Found {len(m4b_files)} audiobook(s) to organize.")
    if dry_run:
        print("[!] RUNNING IN DRY-RUN MODE. NO FILES WILL BE MOVED.")

    # Don't save-and-exit from inside the signal handler — a Ctrl+C landing
    # mid-shutil.move can leave a half-copied file on a cross-filesystem move.
    # Instead flag the cancel and stop cleanly after the current file finishes.
    cancelled = {'flag': False}

    def request_cancel(signum, frame):
        cancelled['flag'] = True
        print("\n    [!] Cancel requested — finishing the current file, then stopping…")

    signal.signal(signal.SIGINT, request_cancel)

    def memorize(author, series, filepath):
        """Fetch + cache the series roster. In a real run this is called only
        after the file is actually placed, so memory never claims a book was
        organized when its move failed."""
        if series == "Standalone" or author == "Unknown":
            return
        memory.setdefault(author, {})
        if series in memory[author]:
            return
        print(f"    [*] Downloading book roster for '{series}'...")
        fetched_titles = fetch_series_roster(author, series)
        fetched_titles.append(clean_title(filepath.stem).lower())
        memory[author][series] = fetched_titles
        if not dry_run:
            save_memory(memory, target_dir)
        print(f"    [+] Memorized {len(fetched_titles)} titles for this series!")

    try:
        for filepath in sorted(m4b_files):
            if cancelled['flag']:
                break
            author, series, clean_t, series_num = interactive_series_router(filepath, root, memory)

            if not author or not series:
                print("    [~] Skipping file.")
                continue

            safe_author = re.sub(r'[\\/*?:"<>|]', "", author).strip()
            safe_series = re.sub(r'[\\/*?:"<>|]', "", series).strip()
            safe_title = re.sub(r'[\\/*?:"<>|]', "", clean_t).strip() or filepath.stem

            if series == "Standalone":
                new_folder = root / safe_author / "Standalone"
                new_filename = f"{safe_title}.m4b"
            else:
                new_folder = root / safe_author / safe_series
                if series_num:
                    new_filename = f"{series_num} - {safe_title}.m4b"
                else:
                    new_filename = f"{safe_title}.m4b"

            new_filepath = new_folder / new_filename

            if dry_run:
                print(f"    [~] DRY RUN: Would move to -> {safe_author}/{safe_series}/{new_filename}")
                memorize(author, series, filepath)  # in-memory only; primes dry-run preview
                continue

            new_folder.mkdir(parents=True, exist_ok=True)
            dest = unique_path(new_filepath)
            shutil.move(str(filepath), str(dest))
            print(f"    [+] Moved to: {safe_author}/{safe_series}/{dest.name}")
            if dest.name != new_filename:
                print(f"    [!] (a file named '{new_filename}' already existed — kept both)")
            memorize(author, series, dest)
    finally:
        # Safety net: flush any in-memory roster growth on normal exit or cancel.
        if memory and not dry_run:
            save_memory(memory, target_dir)

    if cancelled['flag']:
        print("\n    [!] Stopped early at your request.")

    print("\n[*] Checking for empty folders...")
    cleanup_empty_dirs(root, dry_run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audiobook Series Organizer")
    parser.add_argument("folder", help="The folder containing your loose .m4b files")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Simulate the process without moving any files.")
    args = parser.parse_args()
    
    if not Path(args.folder).is_dir():
        print(f"Error: {args.folder} is not a valid directory.")
        sys.exit(1)
        
    process_directory(args.folder, args.dry_run)
