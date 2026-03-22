#!/usr/bin/env python3
"""
audiobook_converter.py
Converts directories of audio files into a single chaptered .m4b audiobook file.
Fetches metadata and cover art from iTunes, Google Books, and Open Library.

Usage:
    python audiobook_converter.py <input_dir> [-o <output_dir>] [-b <bitrate>]
                                  [--auto-lookup] [--no-lookup] [--dry-run]
"""

import html
import os
import re
import sys
import json
import subprocess
import argparse
import tempfile
import urllib.request
import urllib.parse
import termios
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from html.parser import HTMLParser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIO_EXTS = {'.mp3', '.m4a', '.m4b', '.aac', '.ogg', '.opus', '.flac', '.wav', '.wma'}

_PLACEHOLDER_ARTISTS = frozenset({
    'artist', 'unknown', 'unknown author', 'unknown artist',
    'various', 'various artists', 'author', 'narrator', 'n/a', 'na',
})

STOPWORDS = {
    'the', 'a', 'an', 'of', 'and', 'in', 'to', 'is', 'it', 'at', 'on',
    'by', 'for', 'with', 'from', 'or', 'as', 'be', 'this', 'that',
}

STRUCTURAL_FOLDER_RE = re.compile(
    r'(?:^|\s)(cd|disc|disk|part|volume|vol)\s*\d+$'
    r'|^(unabridged|abridged|mp3|audiobooks?)$',
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


def title_words(s: str) -> set:
    words = set(re.sub(r'[^\w\s]', '', s.lower()).split())
    return words - STOPWORDS


def titles_match(a: str, b: str, word_threshold: float = 0.85, seq_threshold: float = 0.90) -> bool:
    words_a = title_words(a)
    words_b = title_words(b)
    if words_a and words_b:
        overlap = len(words_a & words_b)
        smaller = min(len(words_a), len(words_b))
        if (overlap / smaller) >= word_threshold:
            return True
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= seq_threshold


def strip_author_prefix(filename_stem: str) -> str:
    if ' - ' in filename_stem:
        return filename_stem.split(' - ', 1)[1].strip()
    return filename_stem.strip()


def strip_author_from_title(title: str, author: str) -> str:
    """Remove a leading author name from a title string if present."""
    if not author or author.lower() in ('unknown author', 'unknown'):
        return title
    if title.lower().startswith(author.lower()):
        stripped = title[len(author):].strip(' -_')
        if stripped:
            return stripped
    return title


def clean_title(title: str) -> str:
    title = re.sub(r'\s*[\[\(]?(disc|disk|cd|part|volume|vol)\s*\d+[\]\)]?', '', title, flags=re.IGNORECASE)
    title = re.sub(
        r'\s*[\[\(]?(unabridged|abridged|unb\b|isis audio ?books?|corgi audio|bbc radio|'
        r'full[ -]cast drama|full cast|\d{2,3}br|vbr|mp3|m4b)[\]\)]?',
        '', title, flags=re.IGNORECASE,
    )
    title = re.sub(r'\s*[\(\[]?\d{1,2}/\d{1,2}/\d{2,4}.*?[\)\]]?', '', title)
    title = re.sub(r'\(\s*[uU]\s*\d+\.\d+\s*\)', '', title)
    title = re.sub(r'\s*\{[^}]+\}', '', title)                                     # {106mb} file-size tags
    title = re.sub(r'\s+\d{2}[:.]\d{2}[:.]\d{2}', '', title)                      # HH:MM:SS or HH.MM.SS duration
    title = re.sub(r'\s+\d+k\b', '', title, flags=re.IGNORECASE)                   # bitrate e.g. 62k 128K
    title = re.sub(r'^\s*[A-Z]{1,5}-\d+\s*[-–]\s*', '', title)                    # series codes e.g. MR-02 -
    title = re.sub(r'([a-zA-Z])\d{1,2}(?=\s+-\s)', r'\1', title)                  # strip number embedded in series name: Flamel02 → Flamel
    title = re.sub(r'\s+[A-Z]\.\s*[A-Z][a-z]+\.?\s*$', '', title)                 # trailing narrator J.Johnson
    title = re.sub(r'^\d+[.\s]+(?=\d)', '', title)
    title = re.sub(r'^\d+\s+[AB]BY\s*[-–—]?\s*', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^\d+(?:\.\d+)?\s*[\-\.]?\s*', '', title)
    return title.strip(' -_.')


def _ffmeta_escape(value: str) -> str:
    """Escape a value for the ffmetadata format (backslash, equals, newline)."""
    return value.replace('\\', '\\\\').replace('=', '\\=').replace('\n', ' ')


def safe_filename(author: str, title: str) -> str:
    forbidden = r'[\\/*?:"<>|]'
    return f"{re.sub(forbidden, '', author)} - {re.sub(forbidden, '', title)}.m4b"


def truncate_author(author: str, max_len: int = 50) -> str:
    if len(author) <= max_len and ',' not in author:
        return author
    primary = author.split(',')[0].split('&')[0].split(' and ')[0].strip()
    return primary + ' and Others'


# ---------------------------------------------------------------------------
# Author name normalisation
# ---------------------------------------------------------------------------

def normalise_author(raw: str) -> str:
    """
    Convert tag-style author names to natural order and clean up separators.

    Examples:
        "Pratchett, Terry"           -> "Terry Pratchett"
        "Le Guin, Ursula K."         -> "Ursula K. Le Guin"
        "Adams, Douglas; Jones, Jim" -> "Douglas Adams"  (keeps primary only)
    """
    # Take only the first author if multiple are separated by ; or /
    primary = re.split(r'[;/]', raw)[0].strip()

    # Flip "Last, First" -> "First Last"
    if re.match(r'^[^,]+,\s+\S', primary):
        parts   = primary.split(',', 1)
        primary = f"{parts[1].strip()} {parts[0].strip()}"

    return primary.strip(' .,')


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts = []

    def handle_data(self, data):
        self._parts.append(data)

    def handle_entityref(self, name):
        self._parts.append(html.unescape(f"&{name};"))

    def handle_charref(self, name):
        self._parts.append(html.unescape(f"&#{name};"))


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities from a string."""
    if not text or '<' not in text:
        return text
    stripper = _HTMLStripper()
    try:
        stripper.feed(text)
    except Exception:
        return re.sub(r'<[^>]+>', ' ', text).strip()
    return ' '.join(''.join(stripper._parts).split())


# ---------------------------------------------------------------------------
# Metadata search — result scoring
# ---------------------------------------------------------------------------

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _score_result(result: dict, query_title: str, query_author: str) -> float:
    """
    Return a 0–1 score for how well a metadata result matches the query.
    Title similarity is weighted more heavily than author similarity.
    Cover art and a description give a small quality bonus.

    Title scoring uses the BEST match across combinations of:
      - query variants: full title, and the part after the first ' - '
        (the bare book title when a series prefix is present)
      - result variants: full title, and the part before the first ':'
        (the primary title when a subtitle/series qualifier follows)

    This lets "Series 01 - BookTitle" correctly match "BookTitle: Series Qualifier"
    even when the long forms score poorly against each other.

    When the author is unknown the score is based on title only, since adding
    "Unknown Author" to the author weight only penalises the score unfairly.
    """
    rt            = result.get('title', '')
    quality_bonus = (0.02 if result.get('cover_url') else 0) + (0.02 if result.get('desc') else 0)

    # Build title variants
    q_variants = [query_title]
    if ' - ' in query_title:
        short = query_title.split(' - ', 1)[1].strip()
        if short:
            q_variants.append(short)

    rt_variants = [rt]
    if ':' in rt:
        primary = rt.split(':', 1)[0].strip()
        if len(primary) > 5:
            rt_variants.append(primary)

    ts = max(_similarity(qv, rv) for qv in q_variants for rv in rt_variants)

    is_unknown = query_author.lower() in ('', 'unknown', 'unknown author')
    if is_unknown:
        return ts + quality_bonus

    author_score = _similarity(query_author, result.get('author', ''))
    return ts * 0.65 + author_score * 0.33 + quality_bonus


# ---------------------------------------------------------------------------
# Metadata sources
# ---------------------------------------------------------------------------

def _fetch_json(url: str, timeout: int = 6) -> dict:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _search_itunes(query: str) -> list:
    try:
        data = _fetch_json(
            f"https://itunes.apple.com/search?term={query}&media=audiobook&limit=5"
        )
        return [
            {
                'title':     clean_title(item.get('collectionName', 'Unknown')),
                'author':    item.get('artistName', 'Unknown'),
                'year':      item.get('releaseDate', '')[:4],
                'cover_url': item.get('artworkUrl100', '').replace('100x100', '600x600'),
                'desc':      strip_html(item.get('description', '')),
                'source':    'iTunes',
            }
            for item in data.get('results', [])
        ]
    except Exception:
        return []


def _search_google_books(query: str) -> list:
    try:
        data = _fetch_json(
            f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=5"
        )
        results = []
        for item in data.get('items', []):
            vol = item.get('volumeInfo', {})
            results.append({
                'title':     clean_title(vol.get('title', 'Unknown')),
                'author':    (vol.get('authors') or ['Unknown'])[0],
                'year':      vol.get('publishedDate', '')[:4],
                'cover_url': vol.get('imageLinks', {}).get('thumbnail', '').replace('zoom=1', 'zoom=3'),
                'desc':      strip_html(vol.get('description', '')),
                'source':    'Google Books',
            })
        return results
    except Exception:
        return []


def _search_open_library(query: str) -> list:
    """Search Open Library. Returns cover art where a cover_i ID is available."""
    try:
        data = _fetch_json(
            f"https://openlibrary.org/search.json"
            f"?q={query}&fields=title,author_name,first_publish_year,cover_i&limit=5"
        )
        results = []
        for doc in data.get('docs', []):
            cover_id  = doc.get('cover_i')
            cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg" if cover_id else ''
            results.append({
                'title':     clean_title(doc.get('title', 'Unknown')),
                'author':    (doc.get('author_name') or ['Unknown'])[0],
                'year':      str(doc.get('first_publish_year', '')),
                'cover_url': cover_url,
                'desc':      '',   # search endpoint doesn't return descriptions
                'source':    'Open Library',
            })
        return results
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Metadata cache + orchestration
# ---------------------------------------------------------------------------

_search_cache: dict = {}


def search_metadata(title: str, author: str) -> list:
    """
    Query iTunes, Google Books, and Open Library in parallel.
    De-duplicates results and ranks them by similarity to the query.
    Falls back to a title-only search if the combined query returns nothing.
    """
    cache_key = (title.strip().lower(), author.strip().lower())
    if cache_key in _search_cache:
        return _search_cache[cache_key]

    def _run(title_q: str, author_q: str) -> list:
        query = urllib.parse.quote(f"{title_q} {author_q}".strip())
        seen: set   = set()
        combined: list = []

        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = [
                ex.submit(_search_itunes,       query),
                ex.submit(_search_google_books, query),
                ex.submit(_search_open_library, query),
            ]
            for f in as_completed(futures):
                for item in f.result():
                    key = (item['title'].lower(), item['author'].lower())
                    if key not in seen:
                        seen.add(key)
                        combined.append(item)

        return combined

    results = _run(title, author)

    if not results:
        print("    [!] No results for title+author — retrying with title only …")
        results = _run(title, '')

    # For series-prefix titles like "Series N - BookTitle", ALWAYS also run a
    # short-title search and merge the results.  The full-query "Series - Book
    # Unknown Author" often returns irrelevant results, leaving the correct book
    # out of the candidate list entirely.
    # Also try the prefix segment (before ' - ') for "Franchise - Subtitle"
    # patterns (e.g. "Dungeon Crawler Carl - Audio Immersion Tunnel Season 1").
    if ' - ' in title:
        existing = {(r['title'].lower(), r['author'].lower()) for r in results}
        def _merge(extra_results: list) -> None:
            for item in extra_results:
                key = (item['title'].lower(), item['author'].lower())
                if key not in existing:
                    results.append(item)
                    existing.add(key)

        short_title = title.split(' - ', 1)[1].strip()
        if short_title and short_title != title:
            _merge(_run(short_title, ''))

        first_part = title.split(' - ', 1)[0].strip()
        if first_part and first_part != title and first_part != short_title and len(first_part) >= 8:
            _merge(_run(first_part, ''))
    # Rank best match first
    results.sort(key=lambda r: _score_result(r, title, author), reverse=True)

    _search_cache[cache_key] = results
    return results


# ---------------------------------------------------------------------------
# Interactive lookup
# ---------------------------------------------------------------------------

def _flush_stdin():
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


def interactive_lookup(
    title: str,
    author: str,
    auto_lookup: bool = False,
    no_lookup: bool = False,
) -> tuple:
    """
    Interactively (or automatically) select metadata for a book.
    Returns (title, author, cover_url, description, aborted).
    """
    if no_lookup:
        return title, author, None, '', False

    _flush_stdin()
    norm_author = normalise_author(author)
    print(f"\n[*] Searching online for: '{title}' by {norm_author} …")
    results = search_metadata(title, norm_author)

    if not results:
        print("    [!] No results found online.")
        if auto_lookup:
            print("    [~] Auto-lookup: skipping online metadata, using local info.")
            return title, norm_author, None, '', False
    else:
        if auto_lookup:
            res   = results[0]
            score = _score_result(res, title, norm_author)
            if score < 0.40:
                print(f"    [~] Auto-lookup: best match score {score:.2f} is too low — using local info.")
                return title, norm_author, None, '', False
            flags = ('[Cover]' if res['cover_url'] else '') + (' [Summary]' if res['desc'] else '')
            print(f"    [+] Auto-selected: {res['title']}  score={score:.2f}  {flags}")
            return res['title'], res['author'], res['cover_url'], res['desc'], False

        if (
            len(results) == 1
            and SequenceMatcher(None, title.lower(), results[0]['title'].lower()).ratio() > 0.85
        ):
            res   = results[0]
            flags = ('[Cover]' if res['cover_url'] else '') + (' [Summary]' if res['desc'] else '')
            print(f"    [+] Auto-selecting match: {res['title']}  {flags}")
            return res['title'], res['author'], res['cover_url'], res['desc'], False

        print("\n" + "=" * 60 + "\n ONLINE RESULTS (best match first)\n" + "=" * 60)
        for i, res in enumerate(results, 1):
            flags = ('[Cover]' if res['cover_url'] else '') + (' [Summary]' if res['desc'] else '')
            score = _score_result(res, title, norm_author)
            print(
                f"  {i}) [{res['source']:12s}] {res['title']} ({res['year']}) "
                f"— {res['author']}  {flags}  score={score:.2f}"
            )

    n = len(results)
    skip_opt, manual_opt, abort_opt = n + 1, n + 2, n + 3
    print(f"  {skip_opt}) Skip – use local info")
    print(f"  {manual_opt}) Manual search")
    print(f"  {abort_opt}) Abort – skip this folder")

    while True:
        try:
            raw = input(f"\nSelect [1–{abort_opt}]: ").strip()
            if not raw:
                continue
            choice = int(raw)
            if 1 <= choice <= n:
                s = results[choice - 1]
                return s['title'], s['author'], s['cover_url'], s['desc'], False
            elif choice == skip_opt:
                return title, norm_author, None, '', False
            elif choice == manual_opt:
                new_title  = input("    New title: ").strip() or title
                new_author = input("    New author (blank = keep): ").strip() or norm_author
                return interactive_lookup(new_title, new_author, auto_lookup=False, no_lookup=False)
            elif choice == abort_opt:
                return None, None, None, None, True
        except (ValueError, EOFError):
            pass


# ---------------------------------------------------------------------------
# Audiobook discovery
# ---------------------------------------------------------------------------

def find_audiobooks(input_dir: Path) -> dict:
    root       = input_dir.resolve()
    all_files  = [p for p in root.rglob('*') if p.suffix.lower() in AUDIO_EXTS]
    books: dict = {}
    loose_count = 0

    for file in all_files:
        parent   = file.parent
        book_dir = parent.parent if STRUCTURAL_FOLDER_RE.search(parent.name) else parent
        if book_dir == root:
            loose_count += 1
            continue
        books.setdefault(book_dir, []).append(file)

    if loose_count:
        print(f"[!] Ignored {loose_count} loose audio file(s) sitting directly in the root folder.")

    return books


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def probe_file(filepath: Path):
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_format', '-show_streams',
        str(filepath),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data   = json.loads(result.stdout)
        fmt    = data.get('format', {})
        tags   = {k.lower(): v for k, v in fmt.get('tags', {}).items()}
        stream = next(
            (s for s in data.get('streams', []) if s.get('codec_type') == 'audio'),
            data.get('streams', [{}])[0],
        )
        return {
            'path':        filepath,
            'duration':    float(fmt.get('duration', 0)),
            'codec':       stream.get('codec_name', ''),
            'sample_rate': stream.get('sample_rate', ''),
            'channels':    stream.get('channels', 2),
            'title':       tags.get('title', filepath.stem),
            'artist':      (lambda a: 'Unknown Author' if a.lower() in _PLACEHOLDER_ARTISTS else a)(
                               tags.get('artist') or tags.get('album_artist') or 'Unknown Author'
                           ),
            'album':       tags.get('album', 'Unknown Audiobook'),
        }
    except Exception:
        return None


def transcode_worker(input_path: Path, output_path: Path, bitrate: str) -> Path:
    cmd = [
        'ffmpeg', '-y', '-nostdin', '-loglevel', 'quiet',
        '-i', str(input_path),
        '-c:a', 'aac', '-b:a', bitrate,
        '-vn',
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg transcode failed for {input_path.name}: "
            + result.stderr.decode(errors='replace').strip()
        )
    return output_path


def run_ffmpeg_with_progress(cmd: list, total_duration_sec: float, task_name: str = 'Assembling'):
    clean_cmd = [c for c in cmd if c not in ('-loglevel', 'error', 'quiet')]
    clean_cmd = clean_cmd[:1] + ['-loglevel', 'error', '-stats'] + clean_cmd[1:]

    time_re = re.compile(r'time=(\d+):(\d+):(\d+(?:\.\d+)?)')
    process = subprocess.Popen(clean_cmd, stderr=subprocess.PIPE, universal_newlines=True)

    for line in process.stderr:
        m = time_re.search(line)
        if m:
            h, mn, s = m.groups()
            current  = int(h) * 3600 + int(mn) * 60 + float(s)
            pct      = min(100.0, current / total_duration_sec * 100.0) if total_duration_sec > 0 else 0
            filled   = int(40 * pct / 100)
            bar      = '=' * filled + '-' * (40 - filled)
            print(f"\r    [~] {task_name}: [{bar}] {pct:.1f}%", end='', flush=True)

    process.wait()
    print(f"\r    [~] {task_name}: [{'=' * 40}] 100.0%", flush=True)

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg assembly step failed (exit {process.returncode})")


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def already_exists(check_title: str, existing_stems: list) -> bool:
    if not check_title or len(check_title) <= 3:
        return False
    return any(titles_match(check_title, stem) for stem in existing_stems)


def build_existing_stems(output_dir: Path) -> list:
    return [strip_author_prefix(f.stem.lower()) for f in output_dir.glob('*.m4b')]


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_book(
    book_dir: Path,
    files: list,
    output_dir: Path,
    bitrate: str = '192k',
    dry_run: bool = False,
    auto_lookup: bool = False,
    no_lookup: bool = False,
    existing_stems: list | None = None,
):
    print(f"\n{'=' * 60}")
    print(f"  Audiobook: {book_dir.name}")
    print(f"{'=' * 60}")

    files = sorted(files, key=lambda f: natural_sort_key(f))

    initial    = probe_file(files[0])
    raw_album  = initial['album']  if initial else book_dir.name
    raw_artist = initial['artist'] if initial else 'Unknown Author'

    use_folder   = 'unknown' in raw_album.lower() or len(book_dir.name) > len(raw_album) + 8
    early_title  = clean_title(book_dir.name if use_folder else raw_album)
    early_author = normalise_author(raw_artist)
    early_title  = strip_author_from_title(early_title, early_author)

    check_title = strip_author_prefix(early_title.lower()) if ' - ' in early_title else early_title.lower()

    if existing_stems is not None:
        print(f"[?] Checking for existing file matching: '{check_title}' …")
        if already_exists(check_title, existing_stems):
            print("[!] Match found in output folder — skipping.")
            return
        print("[*] No match found. Proceeding …")

    print(f"[*] Probing {len(files)} file(s) …")
    track_data: list = []
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as ex:
        futures = {ex.submit(probe_file, f): f for f in files}
        done = 0
        for future in as_completed(futures):
            done += 1
            res = future.result()
            if res:
                track_data.append(res)
            print(f"\r    [~] Probing: {int(done / len(files) * 100)}%", end='', flush=True)
    print()

    track_data.sort(key=lambda t: natural_sort_key(t['path']))

    if not track_data:
        print("[!] No valid audio files found. Skipping.")
        return

    book_title, book_author, cover_url, book_desc, abort = interactive_lookup(
        early_title, early_author, auto_lookup, no_lookup,
    )
    if abort:
        print("[!] Aborted by user.")
        return

    safe_author     = truncate_author(book_author)
    output_filename = safe_filename(safe_author, book_title)
    output_file     = output_dir / output_filename

    print(f"[*] Output: {output_file}")

    if dry_run:
        print("[~] Dry run — nothing written.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp         = Path(tmpdir)
        concat_list = tmp / 'concat.txt'
        meta_file   = tmp / 'metadata.txt'
        cover_file  = None

        if cover_url:
            try:
                cover_file = tmp / 'cover.jpg'
                req = urllib.request.Request(cover_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as r, open(cover_file, 'wb') as out:
                    out.write(r.read())
                print("    [+] Cover art downloaded.")
            except Exception as e:
                print(f"    [!] Cover art download failed: {e}")
                cover_file = None

        codecs       = {t['codec'] for t in track_data}
        sample_rates = {t['sample_rate'] for t in track_data}
        can_copy     = (codecs == {'aac'} and len(sample_rates) == 1)

        if not can_copy:
            print(f"    [~] Transcoding {len(track_data)} chapter(s) to AAC {bitrate} …")
            transcode_errors = []
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as xc:
                future_map = {}
                for i, t in enumerate(track_data):
                    out = tmp / f"{i:04d}.m4a"
                    t['target'] = out
                    future_map[xc.submit(transcode_worker, t['path'], out, bitrate)] = i

                done = 0
                for future in as_completed(future_map):
                    done += 1
                    try:
                        future.result()
                    except RuntimeError as e:
                        transcode_errors.append(str(e))
                    print(f"\r    [~] Transcoding: {int(done / len(track_data) * 100)}%", end='', flush=True)
            print()

            if transcode_errors:
                print("[!] Some files failed to transcode:")
                for err in transcode_errors:
                    print(f"    {err}")
        else:
            print("    [+] Source is already AAC — stream copying (no re-encode).")
            for t in track_data:
                t['target'] = t['path']

        total_sec = sum(t['duration'] for t in track_data)
        curr_ms   = 0

        with open(concat_list, 'w', encoding='utf-8') as fc, \
             open(meta_file,   'w', encoding='utf-8') as fm:

            fm.write(';FFMETADATA1\n')
            fm.write(f"title={_ffmeta_escape(book_title)}\n")
            fm.write(f"artist={_ffmeta_escape(book_author)}\n")
            fm.write(f"album={_ffmeta_escape(book_title)}\n")
            fm.write("genre=Audiobook\n")
            if book_desc:
                desc_escaped = _ffmeta_escape(book_desc)
                fm.write(f"comment={desc_escaped}\n")
                fm.write(f"description={desc_escaped}\n")
            fm.write('\n')

            for i, t in enumerate(track_data):
                safe_path     = str(t['target']).replace('\\', '\\\\').replace("'", "\\'")
                fc.write(f"file '{safe_path}'\n")
                dur_ms        = int(t['duration'] * 1000)
                chapter_title = t['title'] if t['title'] != t['path'].stem else f"Chapter {i + 1}"
                fm.write(f"[CHAPTER]\nTIMEBASE=1/1000\n")
                fm.write(f"START={curr_ms}\nEND={curr_ms + dur_ms}\n")
                fm.write(f"title={_ffmeta_escape(chapter_title)}\n\n")
                curr_ms += dur_ms

        cmd = [
            'ffmpeg', '-y', '-nostdin',
            '-f', 'concat', '-safe', '0', '-i', str(concat_list),
            '-i', str(meta_file),
        ]
        if cover_file and cover_file.exists():
            cmd += [
                '-i', str(cover_file),
                '-map', '0:a',
                '-map', '2:0',
                '-c:v', 'mjpeg',
                '-disposition:v:0', 'attached_pic',
            ]
        else:
            cmd += ['-map', '0:a']

        cmd += [
            '-map_metadata', '1',
            '-map_chapters', '1',
            '-c:a', 'copy',
            '-movflags', '+faststart',
            str(output_file),
        ]

        try:
            run_ffmpeg_with_progress(cmd, total_sec, task_name='Assembling')
        except RuntimeError as e:
            print(f"\n[!] Assembly failed: {e}")
            return

    print(f"[+] Created: {output_file.name}")

    if existing_stems is not None:
        existing_stems.append(strip_author_prefix(output_file.stem.lower()))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Convert directories of audio files into chaptered .m4b audiobooks.'
    )
    parser.add_argument('input',           help='Input directory containing audiobook folders')
    parser.add_argument('-o', '--output',  default='.', help='Output directory (default: current dir)')
    parser.add_argument('-b', '--bitrate', default='192k', help='AAC bitrate for transcoding (default: 192k)')
    parser.add_argument('-n', '--dry-run', action='store_true', help='Scan and report without writing files')
    parser.add_argument('--auto-lookup',   action='store_true', help='Auto-select the top metadata result')
    parser.add_argument('--no-lookup',     action='store_true', help='Skip all online metadata lookups')
    args = parser.parse_args()

    in_p  = Path(args.input).resolve()
    out_p = Path(args.output).resolve()

    if not in_p.is_dir():
        print(f"[!] Input path does not exist or is not a directory: {in_p}")
        sys.exit(1)

    if not args.dry_run:
        out_p.mkdir(parents=True, exist_ok=True)

    books = find_audiobooks(in_p)

    if not books:
        print("[!] No audiobook folders found.")
        sys.exit(0)

    print(f"[*] Found {len(books)} audiobook folder(s).")

    existing_stems = build_existing_stems(out_p) if not args.dry_run else []

    for book_dir, book_files in sorted(books.items(), key=lambda kv: natural_sort_key(kv[0])):
        process_book(
            book_dir, book_files, out_p,
            bitrate=args.bitrate,
            dry_run=args.dry_run,
            auto_lookup=args.auto_lookup,
            no_lookup=args.no_lookup,
            existing_stems=existing_stems,
        )

    print("\n[+] All done.")


if __name__ == '__main__':
    main()
