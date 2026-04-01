#!/usr/bin/env python3
"""
audiobook_converter.py
Converts directories of audio files into a single chaptered .m4b audiobook file.
Fetches metadata and cover art from iTunes, Google Books, Open Library, and Audnexus.
Embeds series, narrator, cover art, and description into the output .m4b.
Optionally detects chapter boundaries via speech recognition (--chapterize, requires vosk).

Usage:
    python audiobook_converter.py <input_dir> [-o <output_dir>] [-b <bitrate>]
                                  [--auto-lookup] [--no-lookup] [--dry-run]
                                  [--chapterize]
"""

import html
import logging
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
import threading
from datetime import datetime
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
    # Franchise/series names sometimes embedded in artist tags by rippers
    'star wars', 'marvel', 'dc comics', 'bbc', 'audible',
})

STOPWORDS = {
    'the', 'a', 'an', 'of', 'and', 'in', 'to', 'is', 'it', 'at', 'on',
    'by', 'for', 'with', 'from', 'or', 'as', 'be', 'this', 'that',
}

STRUCTURAL_FOLDER_RE = re.compile(
    r'(?:^|\s)(cd|disc|disk|part|volume|vol)\s*\d+$'
    r'|\[(cd|disc|disk|part|volume|vol)\s*\d+\]$'
    r'|^(unabridged|abridged|mp3|audiobooks?)$',
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger('ab')


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        '%(asctime)s  %(levelname)-7s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.setLevel(logging.DEBUG)
    log.info(f"Log file: {log_path}")


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
        r'podium|audible studios|listening library|macmillan audio|tantor|brilliance audio|'
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
    title = re.sub(r'^Year\s+\d+(?:\s*[-–]\s*\d+)?\s*[-–—]\s*', '', title, flags=re.IGNORECASE)  # Year 36 -, Year 12-13 -
    title = re.sub(r'^Book\s+[Tt]he\s+\d+(?:st|nd|rd|th)\s*[-–—]?\s*', '', title, flags=re.IGNORECASE)  # Book The 1st-, Book the 2nd -
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
    if ' - ' in rt:
        last_seg = rt.rsplit(' - ', 1)[1].strip()
        if len(last_seg) > 5 and last_seg not in rt_variants:
            rt_variants.append(last_seg)

    ts = max(_similarity(qv, rv) for qv in q_variants for rv in rt_variants)
    # Small tiebreaker: reward results whose full title matches the full query well.
    # This helps when many books share the same short title (e.g. "Invincible") —
    # the one whose full title also contains the series context scores slightly higher.
    full_bonus = _similarity(query_title, rt) * 0.01

    is_unknown = query_author.lower() in ('', 'unknown', 'unknown author')
    if is_unknown:
        return ts + full_bonus + quality_bonus

    author_score = _similarity(query_author, result.get('author', ''))
    return ts * 0.65 + author_score * 0.33 + full_bonus + quality_bonus


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
                'series':    '',
                'narrator':  '',
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
                'series':    '',
                'narrator':  '',
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
                'series':    '',
                'narrator':  '',
            })
        return results
    except Exception:
        return []


def _search_audnexus(title: str, author: str) -> list:
    """Search Audnexus (Audible data bridge) — audiobook-specific, returns series/position info."""
    try:
        params = f"title={urllib.parse.quote(title.strip())}"
        if author.strip():
            params += f"&author={urllib.parse.quote(author.strip())}"
        data  = _fetch_json(f"https://api.audnex.us/books?{params}", timeout=7)
        items = data if isinstance(data, list) else data.get('data', data.get('results', []))
        out   = []
        for item in items:
            authors    = item.get('authors', [])
            author_str = authors[0].get('name', 'Unknown') if authors else 'Unknown'
            series_parts = item.get('series', [])
            series_str   = ''
            if series_parts and isinstance(series_parts, list):
                s          = series_parts[0]
                s_title    = s.get('title', '')
                s_pos      = s.get('position', '')
                series_str = f"{s_title} #{s_pos}" if s_pos else s_title
            narrators    = item.get('narrators', [])
            narrator_str = narrators[0].get('name', '') if narrators else ''
            out.append({
                'title':     clean_title(item.get('title', 'Unknown')),
                'author':    author_str,
                'year':      (item.get('releaseDate', '') or '')[:4],
                'cover_url': item.get('image', ''),
                'desc':      strip_html(item.get('summary', '')),
                'source':    'Audnexus',
                'series':    series_str,
                'narrator':  narrator_str,
            })
        return out
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

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [
                ex.submit(_search_itunes,       query),
                ex.submit(_search_google_books, query),
                ex.submit(_search_open_library, query),
                ex.submit(_search_audnexus,     title_q, author_q),
            ]
            for f in as_completed(futures):
                for item in f.result():
                    key = (item['title'].lower(), item['author'].lower())
                    if key not in seen:
                        seen.add(key)
                        combined.append(item)

        return combined

    # Don't include placeholder authors (e.g. "Unknown Author") in the API
    # query — they pollute search results and prevent finding the correct book.
    search_author = '' if author.lower() in _PLACEHOLDER_ARTISTS else author
    results = _run(title, search_author)

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
    Returns (title, author, cover_url, description, series, narrator, aborted).
    """
    if no_lookup:
        return title, author, None, '', '', '', False

    _flush_stdin()
    norm_author = normalise_author(author)
    print(f"\n[*] Searching online for: '{title}' by {norm_author} …")
    results = search_metadata(title, norm_author)

    if not results:
        print("    [!] No results found online.")
        if auto_lookup:
            print("    [~] Auto-lookup: skipping online metadata, using local info.")
            return title, norm_author, None, '', '', '', False
    else:
        if auto_lookup:
            res   = results[0]
            score = _score_result(res, title, norm_author)
            if score < 0.40:
                print(f"    [~] Auto-lookup: best match score {score:.2f} is too low — using local info.")
                return title, norm_author, None, '', '', '', False
            flags      = ('[Cover]' if res['cover_url'] else '') + (' [Summary]' if res['desc'] else '')
            series_tag = f"  [{res['series']}]" if res.get('series') else ''
            narrator_tag = f"  (narrated by {res['narrator']})" if res.get('narrator') else ''
            print(f"    [+] Auto-selected: {res['title']}{series_tag}{narrator_tag}  score={score:.2f}  {flags}")
            log.info(f"Metadata [{res['source']}]: \"{res['title']}\" by {res['author']}{series_tag}{narrator_tag}  score={score:.2f}{flags}")
            return res['title'], res['author'], res['cover_url'], res['desc'], res.get('series', ''), res.get('narrator', ''), False

        if (
            len(results) == 1
            and SequenceMatcher(None, title.lower(), results[0]['title'].lower()).ratio() > 0.85
        ):
            res   = results[0]
            flags      = ('[Cover]' if res['cover_url'] else '') + (' [Summary]' if res['desc'] else '')
            series_tag = f"  [{res['series']}]" if res.get('series') else ''
            narrator_tag = f"  (narrated by {res['narrator']})" if res.get('narrator') else ''
            print(f"    [+] Auto-selecting match: {res['title']}{series_tag}{narrator_tag}  {flags}")
            log.info(f"Metadata [{res['source']}]: \"{res['title']}\" by {res['author']}{series_tag}{narrator_tag}{flags}")
            return res['title'], res['author'], res['cover_url'], res['desc'], res.get('series', ''), res.get('narrator', ''), False

        print("\n" + "=" * 60 + "\n ONLINE RESULTS (best match first)\n" + "=" * 60)
        for i, res in enumerate(results, 1):
            flags        = ('[Cover]' if res['cover_url'] else '') + (' [Summary]' if res['desc'] else '')
            score        = _score_result(res, title, norm_author)
            series_tag   = f"  [{res['series']}]" if res.get('series') else ''
            narrator_tag = f"  (narrated by {res['narrator']})" if res.get('narrator') else ''
            print(
                f"  {i}) [{res['source']:12s}] {res['title']}{series_tag} ({res['year']}) "
                f"— {res['author']}{narrator_tag}  {flags}  score={score:.2f}"
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
                s            = results[choice - 1]
                series_tag   = f"  [{s['series']}]" if s.get('series') else ''
                narrator_tag = f"  (narrated by {s['narrator']})" if s.get('narrator') else ''
                score        = _score_result(s, title, norm_author)
                log.info(f"Metadata [{s['source']}]: \"{s['title']}\" by {s['author']}{series_tag}{narrator_tag}  score={score:.2f}")
                return s['title'], s['author'], s['cover_url'], s['desc'], s.get('series', ''), s.get('narrator', ''), False
            elif choice == skip_opt:
                log.info(f"Metadata [local]: \"{title}\" by {norm_author}")
                return title, norm_author, None, '', '', '', False
            elif choice == manual_opt:
                new_title  = input("    New title: ").strip() or title
                new_author = input("    New author (blank = keep): ").strip() or norm_author
                return interactive_lookup(new_title, new_author, auto_lookup=False, no_lookup=False)
            elif choice == abort_opt:
                log.warning(f"Aborted by user")
                return None, None, None, None, None, None, True
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
# Speech-based chapter detection
# ---------------------------------------------------------------------------

_ORDINALS = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
    'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
    'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5',
    'sixth': '6', 'seventh': '7', 'eighth': '8', 'ninth': '9', 'tenth': '10',
}

_STANDALONE_MARKERS = frozenset({
    'prologue', 'epilogue', 'interlude', 'afterword', 'foreword',
    'introduction', 'preface', 'postscript',
})

_CHAPTER_WORDS = frozenset({'chapter', 'part', 'book'}) | _STANDALONE_MARKERS


def _find_silence_ends(audio_file: Path, noise_db: int = -40, min_dur: float = 1.0) -> list[float]:
    """Return timestamps (seconds) where silence ends — potential chapter start points."""
    cmd = [
        'ffmpeg', '-nostdin', '-loglevel', 'quiet',
        '-i', str(audio_file),
        '-af', f'silencedetect=noise={noise_db}dB:d={min_dur}',
        '-f', 'null', '-',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return [float(m.group(1)) for m in re.finditer(r'silence_end: (\d+\.?\d*)', result.stderr)]


def detect_chapters_speech(audio_file: Path, tmpdir: Path) -> list[tuple[float, str]]:
    """
    Scan an audio file for spoken chapter markers using vosk speech recognition.
    Only analyses short clips after silence gaps, so it's fast even for long books.

    Candidates are filtered to a minimum spacing to skip intra-paragraph silences,
    then processed in parallel (Vosk releases the GIL so threads run truly concurrently).

    Returns a sorted list of (start_seconds, chapter_title) tuples.
    Requires: pip install vosk  +  a vosk model at ~/.vosk/model or $VOSK_MODEL.
    """
    try:
        import vosk
    except ImportError:
        print("    [!] vosk not installed — run: pip install vosk")
        return []

    model_path = os.environ.get('VOSK_MODEL') or os.path.expanduser('~/.vosk/model')
    if not os.path.isdir(model_path):
        print(f"    [!] Vosk model not found at: {model_path}")
        print(f"        Download a model from https://alphacephei.com/vosk/models")
        print(f"        and place (or symlink) it at ~/.vosk/model  (or set $VOSK_MODEL)")
        return []

    CLIP_SEC    = 20   # seconds of audio to examine after each silence
    MIN_SPACING = 120  # ignore silence gaps within 2 min of the previous kept candidate

    log.debug(f"Chapterize: source={audio_file.name}  clip_sec={CLIP_SEC}  min_spacing={MIN_SPACING}s")

    print("    [~] Finding silence gaps …")
    silence_ends = _find_silence_ends(audio_file)
    log.debug(f"Chapterize: {len(silence_ends)} raw silence gap(s) found")

    # Always check t=0; then keep only candidates spaced MIN_SPACING apart to
    # skip intra-paragraph silences that can't be chapter breaks.
    raw_candidates = sorted({0.0} | set(silence_ends))
    candidates: list[float] = [raw_candidates[0]]
    for ts in raw_candidates[1:]:
        if ts - candidates[-1] >= MIN_SPACING:
            candidates.append(ts)

    skipped = len(raw_candidates) - len(candidates)
    log.debug(f"Chapterize: {len(candidates)} candidate(s) after spacing filter  ({skipped} dropped)")
    print(f"    [~] {len(candidates)} candidate(s) to scan "
          f"({skipped} skipped — closer than {MIN_SPACING}s to previous) …")

    model = vosk.Model(model_path)

    # Use half the CPU count to leave headroom; ffmpeg + Vosk together are heavy
    max_workers = max(1, (os.cpu_count() or 2) // 2)
    log.debug(f"Chapterize: {max_workers} parallel worker(s)")

    # --- worker: extract one clip and run Vosk ----------------------------
    def _scan_clip(ts: float) -> tuple[float, list]:
        wav_path = tmpdir / f'clip_{int(ts * 1000):012d}.wav'
        cmd = [
            'ffmpeg', '-y', '-nostdin', '-loglevel', 'quiet',
            '-ss', str(ts), '-t', str(CLIP_SEC),
            '-i', str(audio_file),
            '-ar', '16000', '-ac', '1', '-f', 'wav', str(wav_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            err = result.stderr.decode(errors='replace').strip()
            log.debug(f"Chapterize: clip extraction failed at {ts:.1f}s — {err or 'no stderr'}")
            return ts, []

        rec = vosk.KaldiRecognizer(model, 16000)
        rec.SetWords(True)
        words = []
        with open(wav_path, 'rb') as wf:
            wf.read(44)  # skip WAV header
            while True:
                data = wf.read(8000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    words.extend(json.loads(rec.Result()).get('result', []))
        words.extend(json.loads(rec.FinalResult()).get('result', []))
        wav_path.unlink(missing_ok=True)

        if words:
            transcript = ' '.join(w.get('word', '') for w in words[:12])
            log.debug(f"Chapterize: {ts:.1f}s → \"{transcript}{'…' if len(words) > 12 else ''}\"")

        return ts, words

    # --- run workers in parallel, updating a shared progress counter ------
    n_total   = len(candidates)
    n_done    = 0
    done_lock = threading.Lock()
    raw_results: list[tuple[float, list]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(_scan_clip, ts): ts for ts in candidates}
        for future in as_completed(future_map):
            ts, words = future.result()
            raw_results.append((ts, words))
            with done_lock:
                n_done += 1
                d = n_done
            m_ts, s_ts = divmod(int(ts), 60)
            h_ts, m_ts = divmod(m_ts, 60)
            print(f"\r    [~] Scanned {d}/{n_total}  (last: {h_ts:02d}:{m_ts:02d}:{s_ts:02d})", end='', flush=True)

    print()  # newline after progress line
    log.debug(f"Chapterize: all {n_total} clip(s) scanned")

    # --- parse results in timestamp order; deduplicate chapter numbers -----
    raw_results.sort(key=lambda x: x[0])
    chapters: list[tuple[float, str]] = []
    seen_nums: set = set()

    for ts, words in raw_results:
        for i, w in enumerate(words):
            word = w.get('word', '').lower()
            if word not in _CHAPTER_WORDS:
                continue
            word_ts = ts + w.get('start', 0)

            if word in _STANDALONE_MARKERS:
                chapters.append((word_ts, word.capitalize()))
                log.debug(f"Chapterize: standalone marker '{word}' at {word_ts:.1f}s")
                break

            next_words = [words[j].get('word', '').lower() for j in range(i + 1, min(i + 4, len(words)))]
            num = None
            for nw in next_words:
                if nw.isdigit():
                    num = nw
                    break
                if nw in _ORDINALS:
                    num = _ORDINALS[nw]
                    break
            if num and num not in seen_nums:
                seen_nums.add(num)
                title = f"{word.capitalize()} {num}"
                chapters.append((word_ts, title))
                log.debug(f"Chapterize: '{title}' at {word_ts:.1f}s")
                break

    log.debug(f"Chapterize: {len(chapters)} chapter(s) found after deduplication")
    chapters.sort(key=lambda c: c[0])
    return chapters


def retag_m4b(
    source_file: Path,
    book_title: str,
    book_author: str,
    cover_url: str | None,
    book_desc: str,
    book_series: str = '',
    book_narrator: str = '',
) -> bool:
    """Overwrite metadata tags on an existing .m4b in-place (stream-copy, no re-encode)."""
    import shutil
    tmp_out = source_file.with_suffix('.retag.m4b')

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp       = Path(tmpdir)
        meta_file = tmp / 'metadata.txt'
        cover_file = None

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

        with open(meta_file, 'w', encoding='utf-8') as fm:
            fm.write(';FFMETADATA1\n')
            fm.write(f"title={_ffmeta_escape(book_title)}\n")
            fm.write(f"artist={_ffmeta_escape(book_author)}\n")
            fm.write(f"album={_ffmeta_escape(book_title)}\n")
            fm.write("genre=Audiobook\n")
            if book_series:
                fm.write(f"grouping={_ffmeta_escape(book_series)}\n")
            if book_narrator:
                fm.write(f"composer={_ffmeta_escape(book_narrator)}\n")
            if book_desc:
                desc_escaped = _ffmeta_escape(book_desc)
                fm.write(f"comment={desc_escaped}\n")
                fm.write(f"description={desc_escaped}\n")

        cmd = ['ffmpeg', '-y', '-nostdin', '-loglevel', 'quiet',
               '-i', str(source_file), '-i', str(meta_file)]
        if cover_file and cover_file.exists():
            cmd += ['-i', str(cover_file),
                    '-map', '0:a', '-map', '2:0',
                    '-c:v', 'mjpeg', '-disposition:v:0', 'attached_pic']
        else:
            cmd += ['-map', '0:a']
        cmd += ['-map_metadata', '1', '-map_chapters', '0',
                '-c:a', 'copy', '-movflags', '+faststart', str(tmp_out)]

        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"[!] Retag failed: {result.stderr.decode(errors='replace').strip()}")
            if tmp_out.exists():
                tmp_out.unlink()
            return False

    shutil.move(str(tmp_out), str(source_file))
    print(f"[+] Retagged in place: {source_file.name}")
    log.info(f"Retagged: {source_file.name}  →  {book_title} by {book_author}")
    return True


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
    chapterize: bool = False,
):
    print(f"\n{'=' * 60}")
    print(f"  Audiobook: {book_dir.name}")
    print(f"{'=' * 60}")
    log.info(f"--- {book_dir.name}  ({len(files)} file(s))")

    files = sorted(files, key=lambda f: natural_sort_key(f))

    initial    = probe_file(files[0])
    raw_album  = initial['album']  if initial else book_dir.name
    raw_artist = initial['artist'] if initial else 'Unknown Author'

    # Only prefer the folder name when the album tag is absent, a placeholder,
    # or very short (< 15 chars) — avoids using noisy folder names over clean
    # album tags that happen to be shorter (e.g. "Album" vs "Album - NoisySuffix").
    use_folder   = 'unknown' in raw_album.lower() or (len(raw_album) < 15 and len(book_dir.name) > len(raw_album) + 8)
    early_title  = clean_title(book_dir.name if use_folder else raw_album)
    early_author = normalise_author(raw_artist)
    early_title  = strip_author_from_title(early_title, early_author)

    check_title = strip_author_prefix(early_title.lower()) if ' - ' in early_title else early_title.lower()

    if existing_stems is not None:
        print(f"[?] Checking for existing file matching: '{check_title}' …")
        if already_exists(check_title, existing_stems):
            print("[!] Match found in output folder — skipping.")
            log.info(f"Skipped (duplicate): {book_dir.name}")
            return
        print("[*] No match found. Proceeding …")

    # --- Single .m4b: already converted -----------------------------------
    if len(files) == 1 and files[0].suffix.lower() == '.m4b':
        import shutil
        safe_author     = truncate_author(early_author)
        output_filename = safe_filename(safe_author, early_title)
        output_file     = output_dir / output_filename

        if output_file.exists():
            print("[!] Output file already exists — skipping.")
            log.info(f"Skipped (exists): {output_filename}")
            return

        print(f"[*] Single .m4b: {files[0].name}")
        print(f"    Title:  {early_title}")
        print(f"    Author: {early_author}")

        if not auto_lookup:
            _flush_stdin()
            raw = input("    Copy to output? [Y/n]: ").strip().lower()
            if raw not in ('', 'y', 'yes'):
                print("[~] Skipping.")
                log.info(f"Skipped (user declined): {book_dir.name}")
                return

        if dry_run:
            print(f"[~] Dry run — would copy to: {output_file}")
            log.info(f"Dry run: would copy {files[0].name} → {output_filename}")
            return

        shutil.copy2(str(files[0]), str(output_file))
        print(f"[+] Copied: {output_file.name}")
        log.info(f"Copied: {files[0].name}  →  {output_filename}")
        if existing_stems is not None:
            existing_stems.append(strip_author_prefix(output_file.stem.lower()))
        return
    # ----------------------------------------------------------------------

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
        log.warning(f"No valid audio files: {book_dir.name}")
        return

    book_title, book_author, cover_url, book_desc, book_series, book_narrator, abort = interactive_lookup(
        early_title, early_author, auto_lookup, no_lookup,
    )
    if abort:
        print("[!] Aborted by user.")
        return

    safe_author     = truncate_author(book_author)
    output_filename = safe_filename(safe_author, book_title)
    output_file     = output_dir / output_filename

    print(f"[*] Output: {output_file}")

    if output_file.exists():
        print("[!] Output file already exists — skipping.")
        log.info(f"Skipped (exists): {output_filename}")
        if existing_stems is not None:
            existing_stems.append(strip_author_prefix(output_file.stem.lower()))
        return

    if dry_run:
        print("[~] Dry run — nothing written.")
        log.info(f"Dry run: would create {output_filename}")
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
                    log.error(f"Transcode: {err}")
        else:
            print("    [+] Source is already AAC — stream copying (no re-encode).")
            for t in track_data:
                t['target'] = t['path']

        total_sec = sum(t['duration'] for t in track_data)
        curr_ms   = 0

        # Speech-based chapter detection for single-file audiobooks
        speech_chapters: list[tuple[float, str]] = []
        if chapterize and len(track_data) == 1 and track_data[0]['path'].suffix.lower() != '.m4b':
            print("    [~] Running speech chapter detection …")
            log.info("Chapterize: starting speech detection")
            detected = detect_chapters_speech(track_data[0]['path'], tmp)
            if detected:
                print(f"    [+] Detected {len(detected)} chapter marker(s):")
                log.info(f"Chapterize: detected {len(detected)} chapter marker(s)")
                for ch_ts, ch_title in detected:
                    m, s = divmod(int(ch_ts), 60)
                    h, m = divmod(m, 60)
                    print(f"        {h:02d}:{m:02d}:{s:02d}  {ch_title}")
                    log.info(f"Chapterize:   {h:02d}:{m:02d}:{s:02d}  {ch_title}")
                _flush_stdin()
                raw = input("    Use these chapters? [Y/n]: ").strip().lower()
                if raw in ('', 'y', 'yes'):
                    speech_chapters = detected
                    log.info("Chapterize: chapters accepted by user")
                else:
                    log.info("Chapterize: chapters rejected by user")
            else:
                print("    [!] No chapter markers detected via speech recognition.")
                log.info("Chapterize: no chapter markers detected")

        with open(concat_list, 'w', encoding='utf-8') as fc, \
             open(meta_file,   'w', encoding='utf-8') as fm:

            fm.write(';FFMETADATA1\n')
            fm.write(f"title={_ffmeta_escape(book_title)}\n")
            fm.write(f"artist={_ffmeta_escape(book_author)}\n")
            fm.write(f"album={_ffmeta_escape(book_title)}\n")
            fm.write("genre=Audiobook\n")
            if book_series:
                fm.write(f"grouping={_ffmeta_escape(book_series)}\n")
            if book_narrator:
                fm.write(f"composer={_ffmeta_escape(book_narrator)}\n")
                print(f"    [+] Narrator: {book_narrator}")
            if book_desc:
                desc_escaped = _ffmeta_escape(book_desc)
                fm.write(f"comment={desc_escaped}\n")
                fm.write(f"description={desc_escaped}\n")
            fm.write('\n')

            if speech_chapters:
                # Single file — write chapters from speech detection timestamps
                safe_path = str(track_data[0]['target']).replace('\\', '\\\\').replace("'", "\\'")
                fc.write(f"file '{safe_path}'\n")
                total_ms = int(total_sec * 1000)
                for idx, (ch_ts, ch_title) in enumerate(speech_chapters):
                    start_ms = int(ch_ts * 1000)
                    if idx + 1 < len(speech_chapters):
                        end_ms = int(speech_chapters[idx + 1][0] * 1000)
                    else:
                        end_ms = total_ms
                    fm.write(f"[CHAPTER]\nTIMEBASE=1/1000\n")
                    fm.write(f"START={start_ms}\nEND={end_ms}\n")
                    fm.write(f"title={_ffmeta_escape(ch_title)}\n\n")
            else:
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
            log.error(f"Assembly failed: {e}")
            return

    print(f"[+] Created: {output_file.name}")
    log.info(f"Created: {output_file.name}  ({len(track_data)} track(s), {'stream-copy' if can_copy else bitrate})")

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
    parser.add_argument('--chapterize',    action='store_true', help='Detect chapters via speech recognition for single-file audiobooks (requires vosk + model)')
    parser.add_argument('--log', metavar='FILE', help='Log file path (default: ab_TIMESTAMP.log in output dir)')
    args = parser.parse_args()

    in_p  = Path(args.input).resolve()
    out_p = Path(args.output).resolve()

    if not in_p.is_dir():
        print(f"[!] Input path does not exist or is not a directory: {in_p}")
        sys.exit(1)

    out_p.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log) if args.log else out_p / f"ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_path)
    log.info(f"Input:  {in_p}")
    log.info(f"Output: {out_p}")
    log.info(f"Flags:  bitrate={args.bitrate}  dry_run={args.dry_run}  auto_lookup={args.auto_lookup}  no_lookup={args.no_lookup}  chapterize={args.chapterize}")
    print(f"[*] Logging to: {log_path}")

    books = find_audiobooks(in_p)

    if not books:
        print("[!] No audiobook folders found.")
        log.info("No audiobook folders found.")
        sys.exit(0)

    print(f"[*] Found {len(books)} audiobook folder(s).")
    log.info(f"Found {len(books)} audiobook folder(s)")

    existing_stems = build_existing_stems(out_p) if not args.dry_run else []

    for book_dir, book_files in sorted(books.items(), key=lambda kv: natural_sort_key(kv[0])):
        process_book(
            book_dir, book_files, out_p,
            bitrate=args.bitrate,
            dry_run=args.dry_run,
            auto_lookup=args.auto_lookup,
            no_lookup=args.no_lookup,
            existing_stems=existing_stems,
            chapterize=args.chapterize,
        )

    log.info("Done")
    print("\n[+] All done.")


if __name__ == '__main__':
    main()
