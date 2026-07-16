#!/usr/bin/env python3
"""
audiobook_converter.py
Converts directories of audio files into a single chaptered .m4b audiobook file.
Fetches metadata and cover art from iTunes, Google Books, Open Library, and Audnexus.
Embeds series, narrator, cover art, and description into the output .m4b.
Optionally detects chapter boundaries via speech recognition (--chapterize, requires faster-whisper).

Usage:
    python audiobook_converter.py <input_dir> [-o <output_dir>] [-b <bitrate>]
                                  [--auto-lookup] [--no-lookup] [--dry-run]
                                  [--chapterize]
"""

import html
import logging
import os
import re
import shutil
import sys
import json
import subprocess
import argparse
import tempfile
import urllib.request
import urllib.parse
import threading
try:
    import termios  # POSIX only — used to flush stdin before interactive prompts
except ImportError:
    termios = None
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import concurrent.futures
from difflib import SequenceMatcher
from html.parser import HTMLParser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIO_EXTS = {'.mp3', '.m4a', '.m4b', '.aac', '.ogg', '.opus', '.flac', '.wav', '.wma'}

DEFAULT_BITRATE = '192k'
DEFAULT_WHISPER_MODEL = 'medium'
API_TIMEOUT = 6
API_SEARCH_LIMIT = 5
METADATA_SEARCH_TIMEOUT = 15          # max seconds to wait for all metadata APIs

# Audnexus has no title-search route (a bare /books?title= returns 404); it only
# serves /books/{asin}. So we resolve ASINs through Audible's own unauthenticated
# catalog search first, then fetch the normalized record per ASIN from Audnexus.
AUDIBLE_CATALOG_URL = 'https://api.audible.com/1.0/catalog/products'
AUDNEXUS_BOOK_URL   = 'https://api.audnex.us/books'
AUDNEXUS_REGION     = 'us'            # Audible marketplace for ASIN lookups
AUDNEXUS_MAX_DETAIL_WORKERS = 4       # parallel /books/{asin} detail fetches

# Scoring weights and thresholds
SCORE_TITLE_WEIGHT = 0.65             # how much title match contributes to score
SCORE_AUTHOR_WEIGHT = 0.33            # how much author match contributes to score
SCORE_AUTHOR_MISMATCH_WEIGHT = 0.35   # title weight when author doesn't match at all
SCORE_LAST_NAME_FALLBACK = 0.8        # multiplier for last-name-only author match
SCORE_QUALITY_COVER_BONUS = 0.02      # bonus for having cover art
SCORE_QUALITY_DESC_BONUS = 0.02       # bonus for having a description
SCORE_QUALITY_SERIES_BONUS = 0.03     # bonus for series info (audiobook-specific, breaks ties toward richer sources)
SCORE_QUALITY_NARRATOR_BONUS = 0.02   # bonus for a narrator (audiobook-specific)
SCORE_AUTO_SELECT_THRESHOLD = 0.55    # minimum score to auto-select a result
SCORE_SINGLE_RESULT_THRESHOLD = 0.55  # minimum score to auto-select when only one result
SCORE_KEYWORD_ANCHOR_CAP = 0.30       # cap when no query words appear in result
SHORT_TITLE_WORD_COUNT = 3            # titles with fewer words blend Jaccard + char similarity
TITLE_VARIANT_MIN_LEN = 5             # min chars for colon-split primary variant

# Duplicate detection thresholds
DUPE_WORD_THRESHOLD = 0.85
DUPE_SEQ_THRESHOLD = 0.90

# Output verification tolerances.
# Source-vs-output must be TIGHT — it's the same audio. A small ABSOLUTE slack
# absorbs AAC encoder priming/padding accumulated across concatenated files
# (tens of ms each); a percentage here would be a trap — 1% of a 40h book is 24
# minutes, enough to hide a whole dropped chapter behind an "OK".
VERIFY_SOURCE_TOLERANCE_SEC = 15
# AAC encoder priming/padding is tens of ms per concatenated file and
# accumulates across the book, so the source-vs-output tolerance grows with the
# track count on top of the absolute floor — otherwise a 300-track CD rip
# false-WARNs on ~15s of legitimate, unavoidable delta.
VERIFY_SOURCE_SLACK_PER_TRACK_SEC = 0.06
# Online editions genuinely differ (intro/outro, narrator pacing, rounding), so
# allow a percentage — but cap it, or 5% of a 25h omnibus (75 min) would hide a
# missing disc.
VERIFY_EDITION_TOLERANCE_PCT     = 0.05
VERIFY_EDITION_TOLERANCE_CAP_SEC = 900   # 15 min hard cap on edition slack
VERIFY_REPORT_SUFFIX = '.ab_report.txt'

# Chapterize settings
CHAPTERIZE_CLIP_SEC = 20              # seconds of audio to examine after each silence
CHAPTERIZE_MIN_SPACING = 120          # ignore silence gaps within 2 min of previous candidate
SILENCE_NOISE_DB = -30                # noise floor for silence detection
SILENCE_MIN_DURATION = 0.5            # minimum silence duration in seconds

# Folder name heuristic: prefer folder name over album tag when album is short
FOLDER_NAME_MIN_ADVANTAGE = 8        # folder must be this many chars longer than album

MAX_AUTHOR_LEN = 50                   # truncation limit for author in filenames
MAX_TITLE_LEN = 120                   # truncation limit for title in filenames (keeps paths under Windows MAX_PATH)
DECISION_CACHE_FILE = '.ab_decisions.json'  # persists interactive choices across restarts
MAX_TRANSCODE_WORKERS = None          # None = use all CPU cores for parallel transcoding
TRANSCODE_TIMEOUT = 3600              # per-file transcode timeout in seconds (60 min)
SILENCE_DETECT_TIMEOUT = 1800         # per-file silence-detection timeout (30 min)
CHAPTERIZE_CLIP_TIMEOUT = 60          # per-clip extraction timeout for chapterize
RETAG_TIMEOUT = 1800                  # m4b retag (stream-copy remux) timeout (30 min)
ASSEMBLY_TIMEOUT_FLOOR = 1800         # minimum assembly timeout; actual = max(floor, 2 * total_sec)
MERGE_CACHE_PREFIX = 'merge:'                     # decision cache key prefix for folder-merge prompts

# Loudness-normalization targets (EBU R128 / ffmpeg loudnorm).
# I = -18 LUFS matches Audible's audiobook mastering convention; TP/LRA leave
# enough headroom that linear-mode normalization works for typical speech.
LOUDNORM_I   = -18.0
LOUDNORM_TP  = -1.5
LOUDNORM_LRA = 11.0

# iTunes returns an ISO-3166 country (e.g. 'us'), not a language. The m4b
# `language` atom expects ISO-639-2 (e.g. 'eng'), which ABS/Plex/Apple use for
# their language filters. Map the common audiobook storefronts; unknown
# countries leave language blank rather than emit a bogus code.
ITUNES_COUNTRY_TO_LANG = {
    'us': 'eng', 'gb': 'eng', 'ca': 'eng', 'au': 'eng', 'ie': 'eng', 'nz': 'eng',
    'de': 'deu', 'at': 'deu', 'fr': 'fra', 'es': 'spa', 'mx': 'spa', 'it': 'ita',
    'br': 'por', 'pt': 'por', 'nl': 'nld', 'se': 'swe', 'no': 'nor', 'dk': 'dan',
    'fi': 'fin', 'jp': 'jpn', 'ru': 'rus', 'pl': 'pol',
}

# Language normalization → ISO-639-2/B (three-letter), the form the m4b
# `language` atom and ABS/Plex/Apple language filters expect. Providers hand us
# three different shapes: Audnexus full English names ('english'), Google Books
# two-letter ISO-639-1 ('en'), and Open Library / our iTunes country map
# (already three-letter). This maps every shape we see to one canonical code.
_LANGUAGE_TO_ISO639_2 = {
    'eng': 'eng', 'en': 'eng', 'english': 'eng',
    'deu': 'deu', 'ger': 'deu', 'de': 'deu', 'german': 'deu',
    'fra': 'fra', 'fre': 'fra', 'fr': 'fra', 'french': 'fra',
    'spa': 'spa', 'es': 'spa', 'spanish': 'spa',
    'ita': 'ita', 'it': 'ita', 'italian': 'ita',
    'por': 'por', 'pt': 'por', 'portuguese': 'por',
    'nld': 'nld', 'dut': 'nld', 'nl': 'nld', 'dutch': 'nld',
    'swe': 'swe', 'sv': 'swe', 'swedish': 'swe',
    'nor': 'nor', 'no': 'nor', 'norwegian': 'nor',
    'dan': 'dan', 'da': 'dan', 'danish': 'dan',
    'fin': 'fin', 'fi': 'fin', 'finnish': 'fin',
    'jpn': 'jpn', 'ja': 'jpn', 'japanese': 'jpn',
    'rus': 'rus', 'ru': 'rus', 'russian': 'rus',
    'pol': 'pol', 'pl': 'pol', 'polish': 'pol',
}

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

def _ensure_bundled_tools_on_path() -> None:
    """When packaged with PyInstaller, ffmpeg.exe/ffprobe.exe ship next to the
    exe. Prepend that dir to PATH so subprocess('ffmpeg', ...) finds them."""
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        os.environ['PATH'] = exe_dir + os.pathsep + os.environ.get('PATH', '')


_ensure_bundled_tools_on_path()


log = logging.getLogger('ab')

# Module-level: when True, any code path that would prompt the user instead
# logs a warning and aborts the affected book. Set by main() from
# --non-interactive. Used by the merge prompt, interactive metadata picker,
# and the chapterize confirm.
NON_INTERACTIVE = False


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


def _log_tool_versions() -> None:
    """Resolve and log the ffmpeg/ffprobe binaries actually in use.

    Catches misconfiguration loudly: a stale PyInstaller bundle path, an
    architecture-mismatched binary on PATH, an outdated ffmpeg without atoms
    we depend on (e.g. modern `show`/`episode_id`), etc.
    """
    for tool in ('ffmpeg', 'ffprobe'):
        resolved = shutil.which(tool) or '<not found on PATH>'
        version = '?'
        if resolved != '<not found on PATH>':
            try:
                out = subprocess.run([tool, '-version'], capture_output=True,
                                     text=True, timeout=10).stdout
                version = (out.splitlines() or ['?'])[0].strip()
            except Exception as e:
                version = f'<failed to query: {e}>'
        log.info(f"{tool}: {resolved}  ({version})")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


def normalize_language(raw: str) -> str:
    """Map a provider language value to an ISO-639-2/B three-letter code.

    Returns '' for anything we can't confidently map so callers emit no
    language rather than a bogus code. A three-letter input we don't recognise
    is trusted as-is (it's already in the target format)."""
    if not raw:
        return ''
    key = str(raw).strip().lower()
    mapped = _LANGUAGE_TO_ISO639_2.get(key)
    if mapped:
        return mapped
    if len(key) == 3 and key.isalpha():
        return key
    return ''


def title_words(s: str) -> set:
    words = set(re.sub(r'[^\w\s]', '', s.lower()).split())
    return words - STOPWORDS


def titles_match(a: str, b: str, word_threshold: float = DUPE_WORD_THRESHOLD, seq_threshold: float = DUPE_SEQ_THRESHOLD) -> bool:
    """True when two titles are close enough to be the same book.

    Word score is Jaccard (intersection / union), NOT intersection / smaller —
    the latter scores any subset title 1.0, so "Dune" would falsely match
    "Dune Messiah" and get skipped as a duplicate. Jaccard gives 1/2 = 0.5
    there, while genuine dupes (near-identical word sets) still clear the
    threshold. The SequenceMatcher fallback catches near-identical strings that
    the word set misses (minor punctuation / spelling differences).
    """
    words_a = title_words(a)
    words_b = title_words(b)
    if words_a and words_b:
        overlap = len(words_a & words_b)
        union   = len(words_a | words_b)
        if union and (overlap / union) >= word_threshold:
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
        rest = title[len(author):]
        # Only strip if author is followed by a separator, not mid-word/possessive
        if rest and rest[0] not in (' ', '-', '_', '\t'):
            return title
        stripped = rest.strip(' -_')
        if stripped:
            return stripped
    return title


def clean_title(title: str) -> str:
    title = re.sub(r'\s*[\[\(]?(disc|disk|cd|part|volume|vol)\s*\d+[\]\)]?', '', title, flags=re.IGNORECASE)
    title = re.sub(
        r'\s*[\[\(]?(?:unabridged|abridged|unb\b|isis audio ?books?|corgi audio|bbc radio|'
        r'podium|audible studios|listening library|macmillan audio|tantor|brilliance audio|'
        r'full[ -]cast drama|full cast|\d{2,3}br|vbr|mp3|m4b)(?:[- ]\d+)?[\]\)]?',
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
    title = re.sub(r'^\d+\.\s+(?=\d)', '', title)             # "04. 3954 BBY" but not "12.5 Side Jobs"
    title = re.sub(r'^\d+\s+[AB]BY\s*[-–—]?\s*', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^Year\s+\d+(?:\s*[-–]\s*\d+)?\s*[-–—]\s*', '', title, flags=re.IGNORECASE)  # Year 36 -, Year 12-13 -
    title = re.sub(r'^Book\s+[Tt]he\s+\d+(?:st|nd|rd|th)\s*[-–—]?\s*', '', title, flags=re.IGNORECASE)  # Book The 1st-, Book the 2nd -
    title = re.sub(r'^\d+(?:\.\d+)?\s*[\-\.]\s+', '', title)  # "01 - Title" / "12.5 - Title" but not bare "1984"
    title = re.sub(r'^\d+(?:\.\d+)\s+', '', title)          # "12.5 Side Jobs" (decimal prefix with space)
    title = re.sub(r'^0*\d{1,2}\s+(?=[A-Z])', '', title)    # "01 Monsters" but not "1984"
    title = re.sub(r'^\s*[A-Z]{1,5}-\d+\s*[-–]\s*', '', title)  # series codes exposed after number strip
    title = re.sub(r'\s*\[[A-Z]{1,4}\]', '', title)            # short bracket tags: [L], [FQ], [MP3]
    return title.strip(' -_.')


# ---------------------------------------------------------------------------
# Book metadata model
# ---------------------------------------------------------------------------

@dataclass
class BookMetadata:
    """Everything we know about a book, flowing from provider → cache → tags.

    All fields are strings (empty when unknown) so JSON serialization in the
    decision cache is symmetric. cover_path is local-only and never cached.
    """
    title: str = ''
    author: str = ''
    narrator: str = ''
    series: str = ''
    series_part: str = ''     # numeric position within series (may be decimal)
    description: str = ''
    publisher: str = ''
    date: str = ''            # year or ISO date string
    language: str = ''
    genre: str = 'Audiobook'
    asin: str = ''
    isbn: str = ''
    copyright_: str = ''      # avoid shadowing builtin
    cover_url: str = ''
    cover_path: Path | None = None
    runtime_min: str = ''     # online edition runtime in minutes (Audnexus); verify-only, never tagged

    @classmethod
    def from_decision(cls, d: dict | None) -> 'BookMetadata':
        """Construct from a decision-cache entry, tolerating the older schema
        (title/author/cover_url/desc/series/narrator only)."""
        if not d:
            return cls()
        return cls(
            title       = d.get('title', '')       or '',
            author      = d.get('author', '')      or '',
            narrator    = d.get('narrator', '')    or '',
            series      = d.get('series', '')      or '',
            series_part = d.get('series_part', '') or '',
            description = d.get('desc', '')        or d.get('description', '') or '',
            publisher   = d.get('publisher', '')   or '',
            date        = d.get('date', '')        or d.get('year', '') or '',
            language    = d.get('language', '')    or '',
            genre       = d.get('genre', '')       or 'Audiobook',
            asin        = d.get('asin', '')        or '',
            isbn        = d.get('isbn', '')        or '',
            copyright_  = d.get('copyright', '')   or '',
            cover_url   = d.get('cover_url')       or '',
            runtime_min = str(d.get('runtime_min', '') or ''),
        )

    def to_decision(self) -> dict:
        """Serialize to a decision-cache entry (JSON-safe)."""
        return {
            'title':       self.title,
            'author':      self.author,
            'narrator':    self.narrator,
            'series':      self.series,
            'series_part': self.series_part,
            'desc':        self.description,
            'publisher':   self.publisher,
            'date':        self.date,
            'language':    self.language,
            'genre':       self.genre,
            'asin':        self.asin,
            'isbn':        self.isbn,
            'copyright':   self.copyright_,
            'cover_url':   self.cover_url or None,
            'runtime_min': self.runtime_min or '',
        }


_SORT_LEADING_ARTICLES = ('the ', 'a ', 'an ')


def _sort_form(s: str) -> str:
    """Rotate leading articles for sort-tag values.

    'The Hobbit' -> 'Hobbit, The';  'A Game of Thrones' -> 'Game of Thrones, A'.
    Apple Books and iTunes use these to alphabetise; ABS/BookPlayer ignore but
    it's free to write.
    """
    if not s:
        return s
    low = s.lower()
    for art in _SORT_LEADING_ARTICLES:
        if low.startswith(art):
            return f"{s[len(art):]}, {s[:len(art)-1]}"
    return s


def render_book_ffmetadata(m: BookMetadata,
                           chapter_specs: list[tuple[int, int, str]] | None = None) -> str:
    """Render the full body of an FFMETADATA1 file for a given book.

    Emits every field that any of our target media servers / players reads:

      title, artist, album (= title), album_artist, composer (= narrator),
      genre, date, publisher, language, copyright, grouping/show/episode_id
      (series), comment/description/synopsis, sort_*, isbn, asin,
      media_type (stik=2), gapless_playback (pgap=1), encoder,
      plus [CHAPTER] blocks.
    """
    lines: list[str] = [';FFMETADATA1']

    def w(key: str, val) -> None:
        if val is None:
            return
        s = str(val).strip()
        if s:
            lines.append(f'{key}={_ffmeta_escape(s)}')

    # --- Core identification -----------------------------------------------
    w('title',        m.title)
    w('artist',       m.author)
    w('album',        m.title)              # ABS prefers album for title
    w('album_artist', m.author)             # Apple/ABS fallback/Plex match
    w('composer',     m.narrator)           # ABS reads composer as narrator
    w('genre',        m.genre or 'Audiobook')
    w('date',         m.date)
    w('publisher',    m.publisher)
    w('language',     m.language)
    w('copyright',    m.copyright_)

    # --- Series — write THREE forms so every consumer finds something:
    #     grouping    : ABS legacy fallback, Smart AudioBook Player
    #     show        : modern tvsh atom (ABS modern, generic mp4 readers)
    #     episode_id  : tves, numeric position
    if m.series:
        grouping_value = f"{m.series} #{m.series_part}" if m.series_part else m.series
        w('grouping',   grouping_value)
        w('show',       m.series)
        if m.series_part:
            w('episode_id', m.series_part)

    # --- Description: write comment + description + synopsis to satisfy
    #     ABS (description), pre-2024 ABS (comment), Apple ldes (synopsis).
    if m.description:
        w('comment',     m.description)
        w('description', m.description)
        w('synopsis',    m.description)

    # --- Sort tags (Apple Books library ordering) --------------------------
    if m.title:
        w('sort_name',  _sort_form(m.title))
        w('sort_album', _sort_form(m.title))
    if m.author:
        w('sort_artist',       _sort_form(m.author))
        w('sort_album_artist', _sort_form(m.author))

    # --- Identifiers (used by ABS rescans and Plex/Audnexus matching) ------
    w('isbn', m.isbn)
    w('asin', m.asin)

    # --- Apple audiobook flags ---------------------------------------------
    # Without media_type=2 (stik=Audiobook), Apple Books and AVFoundation
    # treat the file as a music album — wrong icon, wrong library section,
    # no chapter UI. gapless_playback=1 prevents micro-stutters between
    # back-to-back chapter playback on iOS.
    w('media_type',       '2')
    w('gapless_playback', '1')
    w('encoder',          'audiobookConverter')

    lines.append('')

    # --- Chapters ----------------------------------------------------------
    if chapter_specs:
        for start_ms, end_ms, title in chapter_specs:
            lines.append('[CHAPTER]')
            lines.append('TIMEBASE=1/1000')
            lines.append(f'START={start_ms}')
            lines.append(f'END={end_ms}')
            lines.append(f'title={_ffmeta_escape(title)}')
            lines.append('')

    return '\n'.join(lines)


# Cover normalization: scale longest side ≤2000, force JPEG, sRGB.
# Apple Books / AVFoundation choke on PNG or oversized covers; this filter
# fixes both in the same pass via ffmpeg (no Pillow dep needed).
COVER_NORMALIZE_FILTER = (
    "scale='if(gt(iw,ih),min(2000,iw),-2)':'if(gt(iw,ih),-2,min(2000,ih))'"
)


def cover_input_args(cover_file: Path, audio_input_index: int = 0,
                     cover_input_index: int = 2) -> list[str]:
    """ffmpeg args for attaching `cover_file` as a normalized JPEG cover.

    Assumes the caller has already added the audio input (typically the
    concat list at index 0) and the metadata file at index 1, so the cover
    is input #2 by default.
    """
    return [
        '-i', str(cover_file),
        '-map', f'{audio_input_index}:a',
        '-map', f'{cover_input_index}:0',
        '-c:v', 'mjpeg',
        '-pix_fmt', 'yuvj420p',
        '-vf', COVER_NORMALIZE_FILTER,
        '-q:v', '4',
        '-disposition:v:0', 'attached_pic',
    ]


def _ffmeta_escape(value: str) -> str:
    """Escape a value for the ffmetadata format.

    Per FFMETADATA1: backslash escapes special chars (`\\`, `=`, `;`, `#`,
    newline). Newlines stay in the output but with a backslash so the parser
    treats them as a continued line, preserving paragraph formatting in
    descriptions. `;` and `#` need escaping because at the start of a line
    they're parsed as comments.
    """
    return (
        value.replace('\\', '\\\\')
             .replace('=',  '\\=')
             .replace(';',  '\\;')
             .replace('#',  '\\#')
             .replace('\n', '\\\n')
    )


def _language_stream_args(meta: 'BookMetadata') -> list:
    """ffmpeg args that tag the audio stream's language.

    The global `language=` key in the FFMETADATA file is silently dropped by
    ffmpeg's mov/mp4 muxer, so the audio stream itself is the only place a
    language actually lands in an .m4b. Returns [] when we have no confident
    ISO-639-2 code, so a bogus code is never written."""
    lang = normalize_language(meta.language)
    return ['-metadata:s:a:0', f'language={lang}'] if lang else []


_WIN_RESERVED_BASENAMES = (
    {'CON', 'PRN', 'AUX', 'NUL'}
    | {f'COM{i}' for i in range(1, 10)}
    | {f'LPT{i}' for i in range(1, 10)}
)
_FILENAME_FORBIDDEN_RE = re.compile(r'[\\/*?:"<>|\x00-\x1f]')


def _sanitize_filename_part(s: str) -> str:
    """Make `s` safe to use as part of a cross-platform filename.

    Strips characters Windows rejects (the explicit set plus ASCII controls),
    trims trailing dots and whitespace (Windows silently drops them, which
    causes "Author - Vol. " → "Author - Vol" collisions), and prefixes
    Windows reserved basenames (CON, PRN, COM1…) with an underscore — those
    names are rejected on Windows even when given a suffix.
    """
    cleaned = _FILENAME_FORBIDDEN_RE.sub('', s).rstrip(' .')
    if cleaned.upper() in _WIN_RESERVED_BASENAMES:
        cleaned = '_' + cleaned
    return cleaned


def _truncate_title_part(title: str, max_len: int = MAX_TITLE_LEN) -> str:
    """Bound a title for use in a filename. Trims on a word boundary when one
    is reasonably close to the limit so we don't cut mid-word."""
    if len(title) <= max_len:
        return title
    clipped = title[:max_len]
    cut = clipped.rfind(' ')
    if cut >= max_len - 20:
        clipped = clipped[:cut]
    return clipped.rstrip(' -_.')


def safe_filename(author: str, title: str) -> str:
    safe_title = _truncate_title_part(_sanitize_filename_part(title))
    return f"{_sanitize_filename_part(author)} - {safe_title}.m4b"


def truncate_author(author: str, max_len: int = MAX_AUTHOR_LEN) -> str:
    if len(author) <= max_len and ',' not in author:
        return author
    primary = author.split(',')[0].split('&')[0].split(' and ')[0].strip()
    return primary + ' and Others'


# ---------------------------------------------------------------------------
# Decision cache — persist interactive choices across restarts
# ---------------------------------------------------------------------------

def _load_decision_cache(cache_path: Path) -> dict:
    """Load the decision cache from disk. Returns empty dict on any error.

    On JSON corruption we move the bad file aside as `<name>.corrupt` before
    returning empty — otherwise the next _save_decision call would atomically
    replace the corrupt cache with a fresh empty one, taking every cached
    decision down with it.
    """
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        backup = cache_path.with_suffix(cache_path.suffix + '.corrupt')
        try:
            cache_path.replace(backup)
            log.warning(f"Decision cache was corrupt; moved to {backup.name} ({e})")
        except OSError as move_err:
            log.warning(f"Decision cache corrupt and couldn't be backed up: {e} (move: {move_err})")
        return {}
    except OSError as e:
        log.warning(f"Decision cache unreadable: {e}")
        return {}


def _save_decision(cache_path: Path, cache: dict, key: str, decision: dict) -> None:
    """Write a single decision to the cache and flush to disk immediately."""
    cache[key] = decision
    try:
        tmp = cache_path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        tmp.replace(cache_path)
    except OSError as e:
        log.warning(f"Failed to save decision cache: {e}")


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

_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'but',
    'with', 'by', 'from', 'is', 'it', 'its', 'as', 'be', 'was', 'are', 'were',
    'that', 'this', 'not', 'no', 'so', 'up', 'do', 'if',
})


def _content_words(text: str) -> set:
    """Lowercase words minus stop words and non-alpha tokens."""
    return {w for w in re.sub(r"[^a-z0-9 ]", '', text.lower()).split()
            if w and w not in _STOP_WORDS}


def _jaccard(a_words: set, b_words: set) -> float:
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def _similarity(a: str, b: str) -> float:
    """Character-level similarity (kept for the single-result auto-select check)."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _title_variants(title: str) -> list[str]:
    variants = [title]
    if ' - ' in title:
        after = title.split(' - ', 1)[1].strip()
        if after:
            variants.append(after)
        before = title.split(' - ', 1)[0].strip()
        if before and before not in variants:
            variants.append(before)
    if ':' in title:
        primary = title.split(':', 1)[0].strip()
        if len(primary) > TITLE_VARIANT_MIN_LEN and primary not in variants:
            variants.append(primary)
    return variants


def _author_last_name(name: str) -> str:
    name = name.strip()
    if ',' in name:
        return name.split(',', 1)[0].strip().lower()
    parts = name.split()
    return parts[-1].lower() if parts else ''


def _score_result(result: dict, query_title: str, query_author: str) -> float:
    """
    Return a 0–1 score for how well a metadata result matches the query.

    Title score: best Jaccard word-overlap across query/result title variants.
    For short titles (<3 content words) blend 50/50 with SequenceMatcher so that
    single-word titles like "Abhorsen" still match well.

    Keyword anchor: if the query title has content words, at least one of the
    most-distinctive query words must appear somewhere in the result metadata
    (title + series + description). Results that share no content words at all
    with the query are capped at 0.30.

    Author: full-name Jaccard, with a last-name-only fallback when the full
    match is weak.

    Quality bonus: small reward for cover art, description, series, and
    narrator — enough to break near-ties toward richer, audiobook-specific
    results (e.g. an Audnexus hit over a bare Open Library edition).
    """
    rt            = result.get('title', '') or ''
    quality_bonus = (
        (SCORE_QUALITY_COVER_BONUS    if result.get('cover_url')   else 0)
        + (SCORE_QUALITY_DESC_BONUS     if result.get('desc')        else 0)
        + (SCORE_QUALITY_SERIES_BONUS   if result.get('series')      else 0)
        + (SCORE_QUALITY_NARRATOR_BONUS if result.get('narrator')    else 0)
    )

    q_variants = _title_variants(query_title)
    rt_variants = _title_variants(rt)

    # Word-overlap (Jaccard) across all variant pairs
    best_jaccard = max(
        _jaccard(_content_words(qv), _content_words(rv))
        for qv in q_variants for rv in rt_variants
    )

    # For very short titles blend with character-level similarity
    q_words = _content_words(query_title)
    if len(q_words) < SHORT_TITLE_WORD_COUNT:
        best_char = max(
            _similarity(qv, rv) for qv in q_variants for rv in rt_variants
        )
        ts = 0.5 * best_jaccard + 0.5 * best_char
    else:
        ts = best_jaccard

    # Keyword anchor: cap weak matches that share no content words
    result_blob = ' '.join(filter(None, [rt, result.get('series', ''), result.get('desc', '')]))
    result_words = _content_words(result_blob)
    if q_words and not (q_words & result_words):
        ts = min(ts, SCORE_KEYWORD_ANCHOR_CAP)

    # Author scoring
    is_unknown = query_author.lower() in ('', 'unknown', 'unknown author')
    if is_unknown:
        return ts + quality_bonus

    ra = result.get('author', '') or ''
    author_full  = _jaccard(_content_words(query_author), _content_words(ra))
    # Last-name fallback: if full-name match is weak, try last-name only
    q_last = _author_last_name(query_author)
    r_last = _author_last_name(ra)
    author_last  = 1.0 if (q_last and r_last and q_last == r_last) else 0.0
    author_score = max(author_full, author_last * SCORE_LAST_NAME_FALLBACK)

    # When author is known but matches zero words, penalise heavily — a perfect
    # title with the wrong author is likely a different book entirely.
    if ra and author_score == 0:
        return ts * SCORE_AUTHOR_MISMATCH_WEIGHT + quality_bonus

    return ts * SCORE_TITLE_WEIGHT + author_score * SCORE_AUTHOR_WEIGHT + quality_bonus


# ---------------------------------------------------------------------------
# Metadata sources
# ---------------------------------------------------------------------------

def _fetch_json(url: str, timeout: int = API_TIMEOUT) -> dict:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _search_itunes(query: str) -> list:
    try:
        data = _fetch_json(
            f"https://itunes.apple.com/search?term={query}&media=audiobook&limit={API_SEARCH_LIMIT}"
        )
        return [
            {
                'title':     clean_title(item.get('collectionName', 'Unknown')),
                'author':    item.get('artistName', 'Unknown'),
                'year':      item.get('releaseDate', '')[:4],
                'date':      item.get('releaseDate', '')[:10],
                'cover_url': item.get('artworkUrl100', '').replace('100x100', '1400x1400'),
                'desc':      strip_html(item.get('description', '')),
                'source':    'iTunes',
                'series':    '',
                'series_part': '',
                'narrator':  '',
                'publisher': '',
                'language':  ITUNES_COUNTRY_TO_LANG.get(item.get('country', '').lower(), ''),
                'genre':     item.get('primaryGenreName', '') or 'Audiobook',
                'asin':      '',
                'isbn':      '',
                'copyright': '',
            }
            for item in data.get('results', [])
        ]
    except Exception:
        return []


def _search_google_books(query: str) -> list:
    try:
        data = _fetch_json(
            f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={API_SEARCH_LIMIT}"
        )
        results = []
        for item in data.get('items', []):
            vol = item.get('volumeInfo', {})
            # Pull ISBN (prefer ISBN_13, fall back to ISBN_10) from
            # industryIdentifiers if present.
            isbn = ''
            for ident in vol.get('industryIdentifiers', []):
                if ident.get('type') == 'ISBN_13':
                    isbn = ident.get('identifier', '')
                    break
            if not isbn:
                for ident in vol.get('industryIdentifiers', []):
                    if ident.get('type') == 'ISBN_10':
                        isbn = ident.get('identifier', '')
                        break
            categories = vol.get('categories') or []
            results.append({
                'title':     clean_title(vol.get('title', 'Unknown')),
                'author':    (vol.get('authors') or ['Unknown'])[0],
                'year':      vol.get('publishedDate', '')[:4],
                'date':      vol.get('publishedDate', '')[:10],
                'cover_url': vol.get('imageLinks', {}).get('thumbnail', '').replace('zoom=1', 'zoom=3'),
                'desc':      strip_html(vol.get('description', '')),
                'source':    'Google Books',
                'series':    '',
                'series_part': '',
                'narrator':  '',
                'publisher': vol.get('publisher', '') or '',
                'language':  normalize_language(vol.get('language', '')),
                'genre':     categories[0] if categories else 'Audiobook',
                'asin':      '',
                'isbn':      isbn,
                'copyright': '',
            })
        return results
    except Exception:
        return []


def _search_open_library(query: str) -> list:
    """Search Open Library. Returns cover art where a cover_i ID is available."""
    try:
        data = _fetch_json(
            f"https://openlibrary.org/search.json"
            f"?q={query}&fields=title,author_name,first_publish_year,cover_i,publisher,isbn,language"
            f"&limit={API_SEARCH_LIMIT}"
        )
        results = []
        for doc in data.get('docs', []):
            cover_id  = doc.get('cover_i')
            cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg" if cover_id else ''
            publishers = doc.get('publisher') or []
            isbns      = doc.get('isbn') or []
            results.append({
                'title':       clean_title(doc.get('title', 'Unknown')),
                'author':      (doc.get('author_name') or ['Unknown'])[0],
                'year':        str(doc.get('first_publish_year', '')),
                'date':        str(doc.get('first_publish_year', '')),
                'cover_url':   cover_url,
                'desc':        '',   # search endpoint doesn't return descriptions
                'source':      'Open Library',
                'series':      '',
                'series_part': '',
                'narrator':    '',
                'publisher':   publishers[0] if publishers else '',
                # OL's search endpoint returns languages across ALL editions,
                # not this one, so it's unreliable per-edition — leave language
                # to iTunes/Audnexus, which are edition-specific.
                'language':    '',
                'genre':       'Audiobook',
                'asin':        '',
                # OL often returns 13-digit first when available
                'isbn':        next((i for i in isbns if len(i) == 13), isbns[0] if isbns else ''),
                'copyright':   '',
            })
        return results
    except Exception:
        return []


def _search_audible_asins(title: str, author: str) -> list:
    """Resolve candidate ASINs via Audible's public catalog search.

    Audnexus has no title search, so this is the entry point that turns a
    title/author into ASINs we can then look up in Audnexus.
    """
    params = {'num_results': API_SEARCH_LIMIT, 'products_sort_by': 'Relevance'}
    if title.strip():
        params['title'] = title.strip()
    if author.strip():
        params['author'] = author.strip()
    data = _fetch_json(f"{AUDIBLE_CATALOG_URL}?{urllib.parse.urlencode(params)}")
    return [p['asin'] for p in (data.get('products') or []) if p.get('asin')]


def _fetch_audnexus_book(asin: str) -> dict | None:
    """Fetch and normalise one Audnexus book record by ASIN.

    The book endpoint's schema differs from the old (dead) search endpoint:
    series lives in `seriesPrimary` ({name, position}), language is a full
    English word ('english'), copyright is an integer year, and `summary` is
    HTML while `description` is a clean one-liner.
    """
    data = _fetch_json(f"{AUDNEXUS_BOOK_URL}/{urllib.parse.quote(asin)}?region={AUDNEXUS_REGION}")
    if not isinstance(data, dict) or not data.get('title'):
        return None

    authors      = data.get('authors') or []
    author_str   = authors[0].get('name', 'Unknown') if authors else 'Unknown'
    narrators    = data.get('narrators') or []
    narrator_str = narrators[0].get('name', '') if narrators else ''

    series_str = series_pos = ''
    sp = data.get('seriesPrimary')
    if isinstance(sp, dict):
        series_str = sp.get('name', '') or ''
        series_pos = str(sp.get('position', '') or '')

    genres_list = data.get('genres') or []
    primary_genre = ''
    for g in genres_list:
        if g.get('type', '').lower() == 'genre' and g.get('name'):
            primary_genre = g['name']
            break
    if not primary_genre and genres_list:
        primary_genre = genres_list[0].get('name', '') or ''

    # Prefer the fuller HTML summary (stripped) over the short plain description.
    desc = strip_html(data.get('summary', '')) or (data.get('description', '') or '')
    copyright_val = data.get('copyright', '')

    return {
        'title':       clean_title(data.get('title', 'Unknown')),
        'author':      author_str,
        'year':        (data.get('releaseDate', '') or '')[:4],
        'date':        (data.get('releaseDate', '') or '')[:10],
        'cover_url':   data.get('image', '') or '',
        'desc':        desc,
        'source':      'Audnexus',
        'series':      series_str,
        'series_part': series_pos,
        'narrator':    narrator_str,
        'publisher':   data.get('publisherName', '') or '',
        'language':    normalize_language(data.get('language', '')),
        'genre':       primary_genre or 'Audiobook',
        'asin':        data.get('asin', '') or asin,
        'isbn':        data.get('isbn', '') or '',
        'copyright':   str(copyright_val) if copyright_val else '',
        'runtime_min': str(data.get('runtimeLengthMin', '') or ''),
    }


def _search_audnexus(title: str, author: str) -> list:
    """Search Audnexus (Audible data bridge) — audiobook-specific, returns
    series/position, narrator, ASIN, publisher, language, genres, runtime.

    Two-step: Audible catalog search resolves ASINs, then Audnexus returns a
    normalized record per ASIN. Detail fetches run in parallel, bounded to the
    catalog result count.
    """
    try:
        asins = _search_audible_asins(title, author)
    except Exception as e:
        log.debug(f"Audible catalog search failed for '{title}' / '{author}': {e}")
        return []
    if not asins:
        return []

    out: list = []
    with ThreadPoolExecutor(max_workers=min(AUDNEXUS_MAX_DETAIL_WORKERS, len(asins))) as ex:
        futures = {ex.submit(_fetch_audnexus_book, a): a for a in asins}
        for f in as_completed(futures):
            try:
                rec = f.result()
            except Exception as e:
                log.debug(f"Audnexus book fetch failed for {futures[f]}: {e}")
                continue
            if rec:
                out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Metadata cache + orchestration
# ---------------------------------------------------------------------------

_search_cache: dict = {}

# Fields worth backfilling when the same book comes back from more than one
# provider, so the record we keep is the union of everything we learned rather
# than whichever provider happened to answer first.
_MERGEABLE_RESULT_FIELDS = (
    'cover_url', 'desc', 'series', 'series_part', 'narrator', 'publisher',
    'language', 'genre', 'asin', 'isbn', 'copyright', 'runtime_min', 'year', 'date',
)


def _merge_result(kept: dict, other: dict) -> None:
    """Backfill empty fields on `kept` from a duplicate `other` (same title +
    author from a different provider). Only fills gaps — never overwrites a
    value kept already has, except 'genre' when kept's is the generic default.

    When the incoming record contributes series or narrator that kept lacked,
    the source label is promoted so the picker shows where the richer data came
    from (e.g. 'iTunes+Audnexus')."""
    gained_audiobook_data = False
    for field in _MERGEABLE_RESULT_FIELDS:
        new = (other.get(field) or '')
        if isinstance(new, str):
            new = new.strip()
        if not new:
            continue
        cur = (kept.get(field) or '')
        if isinstance(cur, str):
            cur = cur.strip()
        if field == 'genre':
            if cur in ('', 'Audiobook'):
                kept[field] = other[field]
            continue
        if not cur:
            kept[field] = other[field]
            if field in ('series', 'narrator'):
                gained_audiobook_data = True
    if gained_audiobook_data and other.get('source'):
        kept_src = kept.get('source', '') or ''
        if other['source'] not in kept_src:
            kept['source'] = f"{kept_src}+{other['source']}" if kept_src else other['source']


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
        by_key: dict   = {}
        combined: list = []

        ex = ThreadPoolExecutor(max_workers=4)
        try:
            # Submission order is also merge priority: the first provider to
            # return a given (title, author) is the base record, later providers
            # backfill its gaps. iTunes first (edition-specific language), then
            # Google, Open Library, Audnexus (series/narrator/runtime).
            futures = [
                ex.submit(_search_itunes,       query),
                ex.submit(_search_google_books, query),
                ex.submit(_search_open_library, query),
                ex.submit(_search_audnexus,     title_q, author_q),
            ]
            done, not_done = wait(futures, timeout=METADATA_SEARCH_TIMEOUT)
            for f in not_done:
                f.cancel()
            if not_done:
                log.warning(f"Metadata search: {len(not_done)} provider(s) timed out after {METADATA_SEARCH_TIMEOUT}s")
            # Iterate in submission order (not done-set order) for deterministic
            # merge results regardless of which provider finished first.
            for f in futures:
                if f not in done:
                    continue
                try:
                    for item in f.result():
                        key = (item['title'].lower(), item['author'].lower())
                        if key not in by_key:
                            by_key[key] = item
                            combined.append(item)
                        else:
                            _merge_result(by_key[key], item)
                except Exception as e:
                    log.debug(f"Metadata provider failed: {e}")
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

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
    if termios is None:
        return
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


def _result_to_meta(r: dict, fallback_title: str = '', fallback_author: str = '') -> BookMetadata:
    """Convert a provider search result dict into a BookMetadata."""
    return BookMetadata(
        title       = r.get('title', '') or fallback_title,
        author      = r.get('author', '') or fallback_author,
        narrator    = r.get('narrator', '') or '',
        series      = r.get('series', '') or '',
        series_part = r.get('series_part', '') or '',
        description = r.get('desc', '') or '',
        publisher   = r.get('publisher', '') or '',
        date        = r.get('date', '') or r.get('year', '') or '',
        language    = r.get('language', '') or '',
        genre       = r.get('genre', '') or 'Audiobook',
        asin        = r.get('asin', '') or '',
        isbn        = r.get('isbn', '') or '',
        copyright_  = r.get('copyright', '') or '',
        cover_url   = r.get('cover_url', '') or '',
        runtime_min = str(r.get('runtime_min', '') or ''),
    )


def _local_meta(title: str, author: str) -> BookMetadata:
    """Build a 'skip' / no-lookup BookMetadata from local info only."""
    return BookMetadata(title=title, author=author)


def interactive_lookup(
    title: str,
    author: str,
    auto_lookup: bool = False,
    no_lookup: bool = False,
) -> tuple[BookMetadata | None, bool]:
    """Interactively (or automatically) select metadata for a book.

    Returns (BookMetadata, aborted). When aborted=True, the metadata is None.
    """
    if no_lookup:
        return _local_meta(title, author), False

    if NON_INTERACTIVE and not auto_lookup:
        log.warning(f"Non-interactive mode and no cached decision for '{title}' — aborting this book")
        print(f"[!] No cached decision for '{title}' and non-interactive mode set — skipping.")
        return None, True

    _flush_stdin()
    norm_author = normalise_author(author)
    print(f"\n[*] Searching online for: '{title}' by {norm_author} …")
    results = search_metadata(title, norm_author)

    if not results:
        print("    [!] No results found online.")
        if auto_lookup:
            print("    [~] Auto-lookup: skipping online metadata, using local info.")
            return _local_meta(title, norm_author), False
    else:
        if auto_lookup:
            res   = results[0]
            score = _score_result(res, title, norm_author)
            if score < SCORE_AUTO_SELECT_THRESHOLD:
                print(f"    [~] Auto-lookup: best match score {score:.2f} is too low — using local info.")
                return _local_meta(title, norm_author), False
            flags        = ('[Cover]' if res['cover_url'] else '') + (' [Summary]' if res['desc'] else '')
            series_tag   = f"  [{res['series']}]" if res.get('series') else ''
            narrator_tag = f"  (narrated by {res['narrator']})" if res.get('narrator') else ''
            print(f"    [+] Auto-selected: {res['title']}{series_tag}{narrator_tag}  score={score:.2f}  {flags}")
            log.info(f"Metadata [{res['source']}]: \"{res['title']}\" by {res['author']}{series_tag}{narrator_tag}  score={score:.2f}{flags}")
            return _result_to_meta(res, title, norm_author), False

        if (
            len(results) == 1
            and _score_result(results[0], title, norm_author) >= SCORE_SINGLE_RESULT_THRESHOLD
        ):
            res          = results[0]
            flags        = ('[Cover]' if res['cover_url'] else '') + (' [Summary]' if res['desc'] else '')
            series_tag   = f"  [{res['series']}]" if res.get('series') else ''
            narrator_tag = f"  (narrated by {res['narrator']})" if res.get('narrator') else ''
            print(f"    [+] Auto-selecting match: {res['title']}{series_tag}{narrator_tag}  {flags}")
            log.info(f"Metadata [{res['source']}]: \"{res['title']}\" by {res['author']}{series_tag}{narrator_tag}{flags}")
            return _result_to_meta(res, title, norm_author), False

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
                return _result_to_meta(s, title, norm_author), False
            elif choice == skip_opt:
                log.info(f"Metadata [local]: \"{title}\" by {norm_author}")
                return _local_meta(title, norm_author), False
            elif choice == manual_opt:
                new_title  = input("    New title: ").strip() or title
                new_author = input("    New author (blank = keep): ").strip() or norm_author
                return interactive_lookup(new_title, new_author, auto_lookup=False, no_lookup=False)
            elif choice == abort_opt:
                log.warning(f"Aborted by user")
                return None, True
        except ValueError:
            pass
        except EOFError:
            log.error("Stdin closed (EOF). Cannot prompt for metadata. Exiting.")
            print("\n[!] Stdin closed — exiting (cannot prompt interactively).")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Audiobook discovery
# ---------------------------------------------------------------------------

def find_audiobooks(
    input_dir: Path,
    decision_cache: dict | None = None,
    cache_path: Path | None = None,
    prompt_merge=None,
) -> dict:
    """Discover audiobook folders under input_dir.

    prompt_merge: optional callable (parent_name: str, children: list[str]) -> bool.
    Used to intercept the "merge subfolders?" prompt when running from the GUI;
    if None, falls back to a stdin y/N prompt. Cached decisions are honoured
    in both modes.
    """
    root       = input_dir.resolve()
    all_files  = [p for p in root.rglob('*') if p.suffix.lower() in AUDIO_EXTS]
    books: dict = {}
    loose_files: list[Path] = []

    for file in all_files:
        parent   = file.parent
        book_dir = parent.parent if STRUCTURAL_FOLDER_RE.search(parent.name) else parent
        if book_dir == root:
            loose_files.append(file)
            continue
        books.setdefault(book_dir, []).append(file)

    # Files sitting directly in the root are normally stray junk in a library
    # tree and get ignored. But if the root contains ONLY loose files (no
    # per-book subfolders), the user pointed us at a single book's own folder —
    # treat the root itself as that book rather than finding nothing.
    if loose_files:
        if not books:
            books[root] = loose_files
            print(f"[*] Treating input folder as a single book ({len(loose_files)} file(s)).")
        else:
            print(f"[!] Ignored {len(loose_files)} loose audio file(s) sitting directly in the root folder.")

    # Consolidation: when a non-root folder has multiple child folders that
    # each contain a single audio file, ask the user whether to merge them
    # into one book.  Handles short-story collections split into per-story
    # subfolders (e.g. "Brief Cases/Story 1/story.mp3").
    parent_groups: dict[Path, list[Path]] = {}
    for book_dir in list(books):
        parent = book_dir.parent
        if parent != root:
            parent_groups.setdefault(parent, []).append(book_dir)

    cache = decision_cache or {}
    for parent, children in sorted(parent_groups.items(), key=lambda kv: natural_sort_key(kv[0])):
        if len(children) < 2 or not all(len(books[c]) == 1 for c in children):
            continue

        cache_key = MERGE_CACHE_PREFIX + str(parent)
        cached = cache.get(cache_key)

        if cached is not None:
            merge = cached.get('merge', False)
            if merge:
                print(f"[*] Merging {len(children)} subfolders under: {parent.name}  (cached)")
            else:
                print(f"[*] Keeping {len(children)} subfolders separate under: {parent.name}  (cached)")
        else:
            subfolder_names = sorted([c.name for c in children], key=natural_sort_key)
            print(f"\n{'=' * 60}")
            print(f"  {parent.name}")
            print(f"  {len(children)} subfolders, each with 1 audio file:")
            for name in subfolder_names[:6]:
                print(f"    • {name}")
            if len(subfolder_names) > 6:
                print(f"    … and {len(subfolder_names) - 6} more")
            print(f"{'=' * 60}")
            if prompt_merge is not None:
                merge = bool(prompt_merge(parent.name, subfolder_names))
            elif NON_INTERACTIVE:
                merge = False
                log.info(f"Non-interactive: defaulting merge=No for {parent.name}")
                print(f"[*] Non-interactive: keeping {len(children)} subfolder(s) separate.")
            else:
                _flush_stdin()
                try:
                    raw = input("  Merge into one book? [y/N]: ").strip().lower()
                except EOFError:
                    log.error("Stdin closed (EOF) during merge prompt. Exiting.")
                    print("\n[!] Stdin closed — exiting (cannot prompt interactively).")
                    sys.exit(1)
                merge = raw in ('y', 'yes')
            if decision_cache is not None and cache_path is not None:
                _save_decision(cache_path, decision_cache, cache_key, {
                    'merge': merge, 'timestamp': datetime.now().isoformat(),
                })

        if merge:
            merged_files: list = []
            for child in children:
                merged_files.extend(books.pop(child))
            if parent in books:
                books[parent].extend(merged_files)
            else:
                books[parent] = merged_files
            log.info(f"Merged {len(children)} single-file subfolder(s) under: {parent.name}")

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
    except Exception as e:
        log.warning(f"probe_file failed for {filepath.name}: {e}")
        return None


def _measure_loudness(input_path: Path) -> dict | None:
    """Pass-1 loudnorm scan. Returns the measurement dict for use as
    measured_* inputs in pass 2, or None if the scan failed."""
    cmd = [
        'ffmpeg', '-y', '-nostdin', '-hide_banner', '-loglevel', 'info',
        '-i', str(input_path),
        '-af',
        f'loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}:print_format=json',
        '-f', 'null', '-',
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE,
                                timeout=TRANSCODE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    # loudnorm prints the JSON at the very end of stderr, after the parsed_*
    # diagnostics. Find the last '{' through matching '}'.
    text  = result.stderr.decode(errors='replace')
    start = text.rfind('{')
    end   = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def transcode_worker(input_path: Path, output_path: Path, bitrate: str,
                     normalize: bool = False) -> Path:
    """Transcode to a .partial file and atomically rename on success.

    Why: a timed-out or failed ffmpeg can leave a truncated output, which would
    otherwise be picked up by an `exists()` check downstream and concatenated
    into a corrupt audiobook.

    When normalize=True a pass-1 loudnorm scan runs first; pass 2 applies a
    linear-mode loudnorm filter using those measurements so the per-file gain
    is constant (dynamics preserved). If the scan fails we skip normalization
    rather than fall through to dynamic mode.
    """
    partial = output_path.with_suffix(output_path.suffix + '.partial')
    partial.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)

    filter_args: list[str] = []
    if normalize:
        m = _measure_loudness(input_path)
        if m is not None:
            filter_args = [
                '-af',
                (
                    f'loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}'
                    f":measured_I={m.get('input_i')}"
                    f":measured_TP={m.get('input_tp')}"
                    f":measured_LRA={m.get('input_lra')}"
                    f":measured_thresh={m.get('input_thresh')}"
                    f":offset={m.get('target_offset')}"
                    ':linear=true:print_format=summary'
                ),
            ]

    # -f ipod forces the MP4 muxer ffmpeg uses for .m4a/.m4b output. Without
    # this, the `.partial` extension on the temp file confuses muxer detection
    # and ffmpeg refuses to open the output.
    cmd = [
        'ffmpeg', '-y', '-nostdin', '-loglevel', 'error',
        '-threads', '0',
        '-i', str(input_path),
        *filter_args,
        '-c:a', 'aac', '-b:a', bitrate,
        '-vn',
        '-f', 'ipod',
        str(partial),
    ]
    # Pass 1 doubles wall time; give pass 2 the same allowance either way.
    timeout = TRANSCODE_TIMEOUT
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                timeout=timeout)
    except subprocess.TimeoutExpired:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg transcode timed out after {timeout}s for {input_path.name}"
        )
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg transcode failed for {input_path.name}: "
            + result.stderr.decode(errors='replace').strip()
        )
    partial.replace(output_path)
    return output_path


def _strip_loglevel(args: list) -> list:
    """Remove `-loglevel <value>` pairs from an ffmpeg arg list.

    The previous implementation filtered every token equal to "error" or
    "quiet", which corrupted commands when a path/title happened to match.
    """
    out = []
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg == '-loglevel':
            skip = True
            continue
        out.append(arg)
    return out


# True only for an interactive terminal. When stdout is a pipe (the GUI runs
# ab.py as a subprocess) or a redirected log, carriage-return progress bars turn
# into thousands of appended lines a screen reader has to wade through, so we
# switch to occasional milestone lines instead.
_STDOUT_IS_TTY = bool(getattr(sys.stdout, 'isatty', lambda: False)())
PROGRESS_MILESTONE_STEP = 10          # percent between milestone lines when piped


def _print_progress(label: str, done: int, total: int, state: dict) -> None:
    """Emit loop progress: an in-place percentage on a tty, or milestone lines
    every PROGRESS_MILESTONE_STEP percent when piped. `state` is a per-loop dict
    that carries the last milestone across calls."""
    if total <= 0:
        return
    pct = int(done / total * 100)
    if _STDOUT_IS_TTY:
        print(f"\r    [~] {label}: {pct}%", end='', flush=True)
        if done >= total:
            print()
        return
    milestone = pct - (pct % PROGRESS_MILESTONE_STEP)
    if milestone > state.get('last', -1) and milestone > 0:
        state['last'] = milestone
        print(f"    [~] {label}: {milestone}%", flush=True)
    if done >= total and state.get('last', 0) < 100:
        state['last'] = 100
        print(f"    [~] {label}: done.", flush=True)


def run_ffmpeg_with_progress(cmd: list, total_duration_sec: float, task_name: str = 'Assembling'):
    clean_cmd = _strip_loglevel(cmd)
    clean_cmd = clean_cmd[:1] + ['-loglevel', 'error', '-stats'] + clean_cmd[1:]

    timeout_sec = max(ASSEMBLY_TIMEOUT_FLOOR, int(total_duration_sec * 2))
    time_re = re.compile(r'time=(\d+):(\d+):(\d+(?:\.\d+)?)')
    process = subprocess.Popen(clean_cmd, stderr=subprocess.PIPE, universal_newlines=True)

    # Watchdog: kills process if it exceeds timeout_sec wall-clock seconds.
    timed_out = threading.Event()
    def _watchdog():
        if not done_event.wait(timeout_sec):
            timed_out.set()
            try:
                process.kill()
            except Exception:
                pass
    done_event = threading.Event()
    wd = threading.Thread(target=_watchdog, daemon=True)
    wd.start()

    last_milestone = -1
    try:
        for line in process.stderr:
            m = time_re.search(line)
            if m:
                h, mn, s = m.groups()
                current  = int(h) * 3600 + int(mn) * 60 + float(s)
                pct      = min(100.0, current / total_duration_sec * 100.0) if total_duration_sec > 0 else 0
                if _STDOUT_IS_TTY:
                    filled = int(40 * pct / 100)
                    bar    = '=' * filled + '-' * (40 - filled)
                    print(f"\r    [~] {task_name}: [{bar}] {pct:.1f}%", end='', flush=True)
                else:
                    milestone = int(pct) - (int(pct) % PROGRESS_MILESTONE_STEP)
                    if milestone > last_milestone and milestone > 0:
                        last_milestone = milestone
                        print(f"    [~] {task_name}: {milestone}%", flush=True)

        process.wait()
    finally:
        done_event.set()
        wd.join(timeout=1)
        if process.stderr:
            process.stderr.close()

    if timed_out.is_set():
        raise RuntimeError(f"ffmpeg {task_name.lower()} timed out after {timeout_sec}s")

    if _STDOUT_IS_TTY:
        print(f"\r    [~] {task_name}: [{'=' * 40}] 100.0%", flush=True)
    else:
        print(f"    [~] {task_name}: done.", flush=True)

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
    # rglob, not glob: once a companion organiser (series.py) has sorted output
    # into Author/Series/ subfolders, a top-level-only scan would go blind and
    # re-convert everything on the next run.
    return [strip_author_prefix(f.stem.lower()) for f in output_dir.rglob('*.m4b')]


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


def _find_silence_ends(audio_file: Path, noise_db: int = SILENCE_NOISE_DB, min_dur: float = SILENCE_MIN_DURATION) -> list[float]:
    """Return timestamps (seconds) where silence ends — potential chapter start points."""
    cmd = [
        'ffmpeg', '-nostdin', '-loglevel', 'info',
        '-i', str(audio_file),
        '-af', f'silencedetect=noise={noise_db}dB:d={min_dur}',
        '-f', 'null', '-',
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=SILENCE_DETECT_TIMEOUT)
    except subprocess.TimeoutExpired:
        log.warning(f"silencedetect timed out after {SILENCE_DETECT_TIMEOUT}s for {audio_file.name}")
        return []
    if result.returncode != 0:
        log.warning(f"silencedetect failed for {audio_file.name}: {result.stderr[-2000:].strip()}")
        return []
    return [float(m.group(1)) for m in re.finditer(r'silence_end: (\d+\.?\d*)', result.stderr)]


def detect_chapters_speech(audio_file: Path, tmpdir: Path) -> list[tuple[float, str]]:
    """
    Scan an audio file for spoken chapter markers using faster-whisper.
    Only analyses short clips after silence gaps, so it's fast even for long books.

    Candidates are filtered to a minimum spacing to skip intra-paragraph silences,
    then processed sequentially through the Whisper model.

    Returns a sorted list of (start_seconds, chapter_title) tuples.
    Requires: pip install faster-whisper  (model auto-downloads on first use).
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("    [!] faster-whisper not installed — run: pip install faster-whisper")
        return []

    CLIP_SEC    = CHAPTERIZE_CLIP_SEC
    MIN_SPACING = CHAPTERIZE_MIN_SPACING
    MODEL_SIZE  = os.environ.get('WHISPER_MODEL', DEFAULT_WHISPER_MODEL)

    log.debug(f"Chapterize: source={audio_file.name}  clip_sec={CLIP_SEC}  min_spacing={MIN_SPACING}s  model={MODEL_SIZE}")

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

    print(f"    [~] Loading Whisper model '{MODEL_SIZE}' (downloads on first use) …")
    model = WhisperModel(MODEL_SIZE, device='cpu', compute_type='int8')

    # --- extract clips and transcribe sequentially (model is not thread-safe) ---
    n_total = len(candidates)
    raw_results: list[tuple[float, list]] = []

    for idx, ts in enumerate(candidates):
        wav_path = tmpdir / f'clip_{int(ts * 1000):012d}.wav'
        cmd = [
            'ffmpeg', '-y', '-nostdin', '-loglevel', 'quiet',
            '-ss', str(ts), '-t', str(CLIP_SEC),
            '-i', str(audio_file),
            '-ar', '16000', '-ac', '1', '-f', 'wav', str(wav_path),
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    timeout=CHAPTERIZE_CLIP_TIMEOUT)
        except subprocess.TimeoutExpired:
            log.debug(f"Chapterize: clip extraction timed out at {ts:.1f}s")
            raw_results.append((ts, []))
            continue
        if result.returncode != 0:
            err = result.stderr.decode(errors='replace').strip()
            log.debug(f"Chapterize: clip extraction failed at {ts:.1f}s — {err or 'no stderr'}")
            raw_results.append((ts, []))
            continue

        words = []
        try:
            segments, _ = model.transcribe(str(wav_path), word_timestamps=True)
            for segment in segments:
                for w in (segment.words or []):
                    words.append({'word': w.word.strip(), 'start': w.start})
        except Exception as e:
            log.debug(f"Chapterize: transcription failed at {ts:.1f}s — {e}")

        wav_path.unlink(missing_ok=True)

        if words:
            transcript = ' '.join(w['word'] for w in words[:12])
            log.debug(f"Chapterize: {ts:.1f}s → \"{transcript}{'…' if len(words) > 12 else ''}\"")

        raw_results.append((ts, words))

        m_ts, s_ts = divmod(int(ts), 60)
        h_ts, m_ts = divmod(m_ts, 60)
        print(f"\r    [~] Scanned {idx + 1}/{n_total}  (last: {h_ts:02d}:{m_ts:02d}:{s_ts:02d})", end='', flush=True)

    print()  # newline after progress line
    log.debug(f"Chapterize: all {n_total} clip(s) scanned")

    # --- parse results in timestamp order; deduplicate chapter numbers -----
    raw_results.sort(key=lambda x: x[0])
    chapters: list[tuple[float, str]] = []

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
            # Don't dedup by number: candidates are already ≥MIN_SPACING apart so
            # the same announcement can't appear twice, and a book that resets
            # numbering per part ("Part 2, Chapter 1") legitimately repeats a
            # number — dropping repeats would merge those chapters.
            if num:
                title = f"{word.capitalize()} {num}"
                chapters.append((word_ts, title))
                log.debug(f"Chapterize: '{title}' at {word_ts:.1f}s")
                break

    log.debug(f"Chapterize: {len(chapters)} chapter(s) found after deduplication")
    chapters.sort(key=lambda c: c[0])
    return chapters


# Magic-byte signatures for the image formats a cover download might return.
# ffmpeg's cover attach only accepts real images; a CDN error page served with
# HTTP 200 would otherwise be written as "cover.jpg" and fail the whole build.
_IMAGE_MAGIC = (
    b'\xff\xd8\xff',            # JPEG
    b'\x89PNG\r\n\x1a\n',       # PNG
    b'GIF87a', b'GIF89a',       # GIF
    b'RIFF',                    # WEBP (RIFF....WEBP)
    b'BM',                      # BMP
)


def _looks_like_image(data: bytes) -> bool:
    return any(data.startswith(sig) for sig in _IMAGE_MAGIC)


def _download_cover(cover_url: str, dest: Path) -> Path | None:
    """Download cover art to dest. Returns the path on success, None on failure.

    Validates the payload is actually an image (by magic bytes) before writing,
    so a CDN error page or HTML redirect served with a 200 status doesn't get
    saved as a bogus cover that later fails the ffmpeg cover-attach step."""
    try:
        req = urllib.request.Request(cover_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = r.read()
    except Exception as e:
        print(f"    [!] Cover art download failed: {e}")
        return None
    if not data or not _looks_like_image(data):
        print("    [!] Cover art download did not return a valid image — skipping cover.")
        log.warning(f"Cover art from {cover_url[:120]} was not a recognised image ({len(data)} bytes)")
        return None
    try:
        with open(dest, 'wb') as out:
            out.write(data)
    except OSError as e:
        print(f"    [!] Could not save cover art: {e}")
        return None
    print("    [+] Cover art downloaded.")
    return dest


def _read_all_tags(path: Path) -> dict:
    """Return all format-level tags from a media file, lowercased keys (or {})."""
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data = json.loads(result.stdout)
        return {k.lower(): v for k, v in (data.get('format', {}).get('tags') or {}).items()}
    except Exception:
        return {}


def _backfill_meta_from_source(meta: BookMetadata, source: Path) -> BookMetadata:
    """Return a copy of `meta` with empty fields filled from `source`'s existing
    tags. Retagging a book we found no (or only a sparse local) match for must
    not discard richer metadata the file already carried — narrator, series,
    description, publisher, etc. Never overwrites a value `meta` already has."""
    tags = _read_all_tags(source)
    if not tags:
        return meta
    m = replace(meta)

    def fill(attr: str, *keys: str) -> None:
        if getattr(m, attr, ''):
            return
        for k in keys:
            v = (tags.get(k) or '').strip()
            if v:
                setattr(m, attr, v)
                return

    fill('title',       'title', 'album')
    fill('author',      'artist', 'album_artist')
    fill('narrator',    'composer')
    fill('series',      'show')
    fill('series_part', 'episode_id')
    fill('description', 'description', 'comment', 'synopsis')
    fill('publisher',   'publisher')
    fill('date',        'date')
    fill('isbn',        'isbn')
    fill('asin',        'asin')
    fill('copyright_',  'copyright')

    # 'genre' has a non-empty default, so fill() would never replace it.
    if m.genre in ('', 'Audiobook'):
        g = (tags.get('genre') or '').strip()
        if g:
            m.genre = g

    # Legacy 'grouping' ("Series #N") as a series fallback when there's no
    # dedicated show/episode_id atom.
    if not m.series:
        grouping = (tags.get('grouping') or '').strip()
        if grouping:
            mo = re.match(r'^(.*?)\s*#\s*([\d.]+)\s*$', grouping)
            if mo:
                m.series = mo.group(1).strip()
                if not m.series_part:
                    m.series_part = mo.group(2)
            else:
                m.series = grouping
    return m


def retag_m4b(source_file: Path, meta: BookMetadata) -> bool:
    """Overwrite metadata tags on an existing .m4b in-place (stream-copy).

    Because `-map_metadata 1` replaces the file's global metadata wholesale,
    any tag not present in `meta` would be lost. So we first backfill `meta`'s
    empty fields from the file's existing tags — a book with a good narrator /
    description / series that got only a sparse (or skipped) match must not come
    out worse-tagged than it went in."""
    meta = _backfill_meta_from_source(meta, source_file)
    tmp_out = source_file.with_suffix('.retag.m4b')

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp       = Path(tmpdir)
        meta_file = tmp / 'metadata.txt'
        cover_file: Path | None = None

        if meta.cover_url:
            cover_file = _download_cover(meta.cover_url, tmp / 'cover.jpg')

        with open(meta_file, 'w', encoding='utf-8') as fm:
            fm.write(render_book_ffmetadata(meta))

        cmd = ['ffmpeg', '-y', '-nostdin', '-loglevel', 'error',
               '-i', str(source_file), '-i', str(meta_file)]
        if cover_file and cover_file.exists():
            # New cover replaces whatever was embedded. Normalize to JPEG
            # ≤2000px so Apple Books / BookPlayer don't choke on PNGs or
            # oversized images.
            cmd += cover_input_args(cover_file, audio_input_index=0, cover_input_index=2)
        else:
            # No new cover supplied — preserve everything the source has
            # (audio + the existing attached_pic + any extra streams).
            cmd += ['-map', '0', '-c:v', 'copy']
        cmd += ['-map_metadata', '1', '-map_chapters', '0',
                '-c:a', 'copy', *_language_stream_args(meta),
                '-movflags', '+faststart', str(tmp_out)]

        try:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    timeout=RETAG_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"[!] Retag timed out after {RETAG_TIMEOUT}s")
            log.warning(f"Retag timed out for {source_file.name}")
            if tmp_out.exists():
                tmp_out.unlink()
            return False
        if result.returncode != 0:
            print(f"[!] Retag failed: {result.stderr.decode(errors='replace').strip()}")
            if tmp_out.exists():
                tmp_out.unlink()
            return False

    shutil.move(str(tmp_out), str(source_file))
    print(f"[+] Retagged in place: {source_file.name}")
    log.info(f"Retagged: {source_file.name}  →  {meta.title} by {meta.author}")
    return True


# ---------------------------------------------------------------------------
# Output verification — a screen-reader-friendly "trust but verify" report.
# Nothing here mutates the audiobook; it only probes the finished file and
# compares it to what we expected to produce.
# ---------------------------------------------------------------------------

def _fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _within_tolerance(actual: float, expected: float, abs_tol: float, pct_tol: float) -> bool:
    """True when `actual` is within max(abs_tol, expected*pct_tol) of `expected`."""
    tol = max(abs_tol, expected * pct_tol)
    return abs(actual - expected) <= tol


def _safe_float(value, default: float = 0.0) -> float:
    """Parse a float from ffprobe output, tolerating None / 'N/A' / junk."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _probe_for_verify(path: Path) -> dict | None:
    """ffprobe a finished file for verification (format, streams, chapters,
    tags). Returns None if the file can't be opened at all."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', '-show_chapters', str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except Exception:
        return None


def _probe_source_chapters(path: Path) -> list[tuple[int, int, str]]:
    """Return a file's embedded chapters as (start_ms, end_ms, title) tuples.

    Empty if the file has none. Lets a single-file source keep real chapter
    markers through the re-mux instead of collapsing to one whole-book chapter."""
    data = _probe_for_verify(path)
    if not data:
        return []
    specs: list[tuple[int, int, str]] = []
    for c in data.get('chapters', []):
        start_ms = int(_safe_float(c.get('start_time')) * 1000)
        end_ms   = int(_safe_float(c.get('end_time')) * 1000)
        if end_ms <= start_ms:
            continue
        title = (c.get('tags') or {}).get('title', '') or f"Chapter {len(specs) + 1}"
        specs.append((start_ms, end_ms, title))
    return specs


def _emit_verify_report(output_file: Path, header: str,
                        checks: list[tuple[str, str]], write_report: bool) -> dict:
    """Print, log, and (optionally) write a sidecar for a list of
    (level, message) checks. level is one of OK / WARN / FAIL."""
    n_ok   = sum(1 for lvl, _ in checks if lvl == 'OK')
    n_warn = sum(1 for lvl, _ in checks if lvl == 'WARN')
    n_fail = sum(1 for lvl, _ in checks if lvl == 'FAIL')
    summary = f"{n_ok} OK" + (f", {n_warn} warning(s)" if n_warn else '') + (f", {n_fail} failure(s)" if n_fail else '')

    console_marks = {'OK': '[+]', 'WARN': '[!]', 'FAIL': '[X]'}
    print(f"\n[verify] {header}")
    for lvl, msg in checks:
        print(f"    {console_marks.get(lvl, '[?]')} {msg}")
    print(f"    => {summary}")

    log.info(f"Verify: {header} — {summary}")
    for lvl, msg in checks:
        (log.warning if lvl in ('WARN', 'FAIL') else log.info)(f"Verify:   {lvl} {msg}")

    report_text = '\n'.join([f"Verification: {header}"]
                            + [f"    {lvl:4s} {msg}" for lvl, msg in checks]
                            + [f"    -> {summary}"])
    if write_report:
        try:
            report_path = output_file.parent / (output_file.stem + VERIFY_REPORT_SUFFIX)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text + '\n')
        except OSError as e:
            log.warning(f"Could not write verify report: {e}")

    return {'ok': n_ok, 'warn': n_warn, 'fail': n_fail, 'checks': checks, 'text': report_text}


def verify_output(output_file: Path, meta: BookMetadata,
                  expected_duration_sec: float, expected_chapters: int | None,
                  write_report: bool = True, source_tracks: int = 1) -> dict:
    """Check a finished .m4b against expectations and emit a linear report.

    Checks file validity, source-vs-output duration, online-edition runtime
    (when Audnexus gave us one), chapter count + ordering, cover art, codec and
    the core tags. The duration-delta check is the load-bearing one: it catches
    ffmpeg truncations and dropped source files that are otherwise invisible
    until you hit a gap mid-listen.
    """
    header = (meta.title or output_file.stem) + (f" by {meta.author}" if meta.author else '')
    data = _probe_for_verify(output_file)
    if data is None:
        return _emit_verify_report(
            output_file, header,
            [('FAIL', 'Output could not be opened by ffprobe — it may be corrupt.')],
            write_report,
        )

    fmt      = data.get('format', {})
    streams  = data.get('streams', [])
    chapters = data.get('chapters', [])
    audio    = [s for s in streams if s.get('codec_type') == 'audio']
    video    = [s for s in streams if s.get('codec_type') == 'video']
    tags     = {k.lower(): v for k, v in (fmt.get('tags') or {}).items()}
    checks: list[tuple[str, str]] = []

    # 1. Validity + codec + stream count
    if not audio:
        checks.append(('FAIL', 'No audio stream in output.'))
    elif len(audio) != 1:
        checks.append(('WARN', f'{len(audio)} audio streams found; expected exactly 1.'))
    else:
        codec = audio[0].get('codec_name', '?')
        checks.append(('OK', 'Valid file, 1 audio stream (AAC).') if codec == 'aac'
                       else ('WARN', f'Audio codec is {codec}, expected AAC.'))

    out_dur = _safe_float(fmt.get('duration'))

    # Source tolerance grows with track count: AAC priming accumulates a little
    # per concatenated file, so a many-track rip legitimately drifts more.
    src_tol = max(VERIFY_SOURCE_TOLERANCE_SEC,
                  source_tracks * VERIFY_SOURCE_SLACK_PER_TRACK_SEC)

    # 2. Duration vs source audio — tight absolute tolerance, no percentage.
    if out_dur <= 0:
        checks.append(('WARN', 'Output duration could not be read.'))
    elif expected_duration_sec > 0:
        delta = abs(out_dur - expected_duration_sec)
        if _within_tolerance(out_dur, expected_duration_sec, src_tol, 0.0):
            checks.append(('OK', f'Duration {_fmt_hms(out_dur)} matches source (delta {int(delta)}s).'))
        else:
            direction = 'shorter' if out_dur < expected_duration_sec else 'longer'
            checks.append(('WARN', f'Output is {_fmt_hms(delta)} {direction} than the source '
                                   f'({_fmt_hms(out_dur)} vs {_fmt_hms(expected_duration_sec)}) — possible truncation or dropped file.'))

    # 3. Duration vs online edition runtime (Audnexus, when known)
    try:
        runtime_min = int(float(meta.runtime_min)) if meta.runtime_min else 0
    except (TypeError, ValueError):
        runtime_min = 0
    if runtime_min > 0 and out_dur > 0:
        edition_sec = runtime_min * 60
        delta = abs(out_dur - edition_sec)
        edition_tol = min(edition_sec * VERIFY_EDITION_TOLERANCE_PCT, VERIFY_EDITION_TOLERANCE_CAP_SEC)
        if delta <= edition_tol:
            checks.append(('OK', f'Matches the online edition runtime (~{_fmt_hms(edition_sec)}).'))
        else:
            direction = 'shorter' if out_dur < edition_sec else 'longer'
            checks.append(('WARN', f'Output is {_fmt_hms(delta)} {direction} than the online edition '
                                   f'({_fmt_hms(out_dur)} vs ~{_fmt_hms(edition_sec)}) — could be a missing file/disc, or just a different edition.'))

    # 4. Chapters
    if chapters:
        starts = [_safe_float(c.get('start_time')) for c in chapters]
        ends   = [_safe_float(c.get('end_time'))   for c in chapters]
        # Non-decreasing starts; a same-start pair is a zero-length chapter,
        # which the `positive` check reports with a clearer message.
        ordered  = all(starts[i] <= starts[i + 1] for i in range(len(starts) - 1))
        positive = all(ends[i] > starts[i] for i in range(len(chapters)))
        # Coverage: chapters should span the file — first near t=0, last ending
        # near the audio end. A gap means part of the book has no chapter, which
        # breaks nonvisual navigation.
        covers_start = starts[0] <= src_tol
        covers_end   = (out_dur <= 0) or (ends[-1] >= out_dur - src_tol)
        if expected_chapters is not None and len(chapters) != expected_chapters:
            checks.append(('WARN', f'{len(chapters)} chapters in output, expected {expected_chapters}.'))
        elif not ordered:
            checks.append(('WARN', f'{len(chapters)} chapters present but not in time order.'))
        elif not positive:
            checks.append(('WARN', f'{len(chapters)} chapters present but at least one has zero length.'))
        elif not (covers_start and covers_end):
            checks.append(('WARN', f'{len(chapters)} chapters present but they do not span the whole file '
                                   f'(first starts at {_fmt_hms(starts[0])}, last ends at {_fmt_hms(ends[-1])} of {_fmt_hms(out_dur)}).'))
        else:
            checks.append(('OK', f'{len(chapters)} chapters, in order and spanning the file.'))
    elif expected_chapters:
        checks.append(('WARN', f'No chapters in output, expected {expected_chapters}.'))
    else:
        checks.append(('WARN', 'No chapters in output.'))

    # 5. Cover art. Only a real attached_pic counts; a non-cover video stream
    # must NOT pass as a cover. And "no cover" is only a problem if we actually
    # had one to embed — otherwise it's expected, not a warning.
    def _is_attached_pic(v) -> bool:
        val = (v.get('disposition') or {}).get('attached_pic')
        return val == 1 or val is True or str(val) == '1'

    if any(_is_attached_pic(v) for v in video):
        checks.append(('OK', 'Cover art embedded.'))
    elif video:
        checks.append(('WARN', 'A video stream is present but not marked as cover art.'))
    elif meta.cover_url:
        checks.append(('WARN', 'Cover art was available but is not embedded in the output.'))
    else:
        checks.append(('OK', 'No cover art (none was available).'))

    # 6. Core tags. Author is written to both artist and album_artist, and some
    # muxers/players expose only one — accept any author-ish key so a healthy
    # file doesn't false-WARN.
    title_present  = bool(tags.get('title') or tags.get('album'))
    author_present = bool(tags.get('artist') or tags.get('album_artist') or tags.get('author'))
    present = [name for name, key in (('title', 'title'), ('author', 'artist'),
                                      ('album', 'album'), ('narrator', 'composer'),
                                      ('series', 'show')) if tags.get(key)]
    missing_core = ([] if title_present else ['title']) + ([] if author_present else ['author'])
    if missing_core:
        checks.append(('WARN', f"Missing core tag(s): {', '.join(missing_core)}."))
    else:
        checks.append(('OK', f"Tags present: {', '.join(present) or 'title, author'}."))

    return _emit_verify_report(output_file, header, checks, write_report)


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
    accept_chapters: bool = False,
    skip_transcode_errors: bool = False,
    normalize: bool = False,
    decision_cache: dict | None = None,
    cache_path: Path | None = None,
    verify: bool = True,
):
    print(f"\n{'=' * 60}")
    print(f"  Audiobook: {book_dir.name}")
    print(f"{'=' * 60}")
    log.info(f"--- {book_dir.name}  ({len(files)} file(s))")

    files = sorted(files, key=lambda f: natural_sort_key(f))

    initial    = probe_file(files[0])
    raw_album  = initial['album']  if initial else book_dir.name
    raw_artist = initial['artist'] if initial else 'Unknown Author'

    # Files from multiple subdirectories means a merged collection — the album
    # tag of any one file is a chapter/story name, not the collection title.
    is_collection = len({f.parent for f in files}) > 1

    # Only prefer the folder name when the album tag is absent, a placeholder,
    # very short (< 15 chars), or the book is a merged collection.
    use_folder   = is_collection or 'unknown' in raw_album.lower() or (len(raw_album) < 15 and len(book_dir.name) > len(raw_album) + FOLDER_NAME_MIN_ADVANTAGE)
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

    # --- Decision cache: check for a previous abort/choice ----------------
    cache_key = str(book_dir)
    cached_decision = (decision_cache or {}).get(cache_key)
    if cached_decision and cached_decision.get('aborted'):
        print("    [*] Previously aborted — skipping. (Use --clear-cache to reset)")
        log.info(f"Skipped (cached abort): {book_dir.name}")
        return

    # --- Single .m4b: already converted -----------------------------------
    # When --chapterize is requested we skip this fast copy path and fall
    # through to the general path, which can run speech detection and re-mux
    # the chapters in (still a stream-copy for an AAC .m4b, so no re-encode).
    if len(files) == 1 and files[0].suffix.lower() == '.m4b' and not chapterize:
        if cached_decision and not cached_decision.get('aborted'):
            meta = BookMetadata.from_decision(cached_decision)
            print(f"    [*] Using cached decision: \"{meta.title}\" by {meta.author}")
            abort = False
        else:
            meta, abort = interactive_lookup(early_title, early_author, auto_lookup, no_lookup)
            # Don't cache aborts: otherwise the user can't retry a single
            # aborted folder without --clear-cache.
            if not abort and meta is not None and decision_cache is not None and cache_path is not None:
                entry = meta.to_decision()
                entry['timestamp'] = datetime.now().isoformat()
                _save_decision(cache_path, decision_cache, cache_key, entry)
        if abort or meta is None:
            print("[!] Aborted.")
            return

        safe_author     = truncate_author(meta.author)
        output_filename = safe_filename(safe_author, meta.title)
        output_file     = output_dir / output_filename

        if output_file.exists():
            print("[!] Output file already exists — skipping.")
            log.info(f"Skipped (exists): {output_filename}")
            return

        print(f"[*] Single .m4b: {files[0].name}")
        print(f"    Title:  {meta.title}")
        print(f"    Author: {meta.author}")

        if dry_run:
            print(f"[~] Dry run — would copy to: {output_file}")
            log.info(f"Dry run: would copy {files[0].name} → {output_filename}")
            return

        shutil.copy2(str(files[0]), str(output_file))
        print(f"[+] Copied: {output_file.name}")
        log.info(f"Copied: {files[0].name}  →  {output_filename}")

        # Always retag the copy — even when the user 'skipped' lookup, we want
        # to embed our standard atom set (stik=Audiobook, pgap, sort tags,
        # genre=Audiobook). retag is a stream-copy remux, so it's cheap.
        retag_m4b(output_file, meta)

        if verify:
            src_dur = float(initial.get('duration', 0) or 0) if initial else 0.0
            # Source chapters are preserved as-is on a stream-copy retag, so we
            # don't assert a specific count for the single-file path.
            verify_output(output_file, meta, src_dur, expected_chapters=None)

        if existing_stems is not None:
            existing_stems.append(strip_author_prefix(output_file.stem.lower()))
        return
    # ----------------------------------------------------------------------

    print(f"[*] Probing {len(files)} file(s) …")
    track_data: list = []
    probe_progress: dict = {}
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as ex:
        futures = {ex.submit(probe_file, f): f for f in files}
        done = 0
        for future in as_completed(futures):
            done += 1
            res = future.result()
            if res:
                track_data.append(res)
            _print_progress('Probing', done, len(files), probe_progress)

    track_data.sort(key=lambda t: natural_sort_key(t['path']))

    if not track_data:
        print("[!] No valid audio files found. Skipping.")
        log.warning(f"No valid audio files: {book_dir.name}")
        return

    if cached_decision and not cached_decision.get('aborted'):
        meta = BookMetadata.from_decision(cached_decision)
        print(f"    [*] Using cached decision: \"{meta.title}\" by {meta.author}")
        abort = False
    else:
        meta, abort = interactive_lookup(early_title, early_author, auto_lookup, no_lookup)
        if not abort and meta is not None and decision_cache is not None and cache_path is not None:
            entry = meta.to_decision()
            entry['timestamp'] = datetime.now().isoformat()
            _save_decision(cache_path, decision_cache, cache_key, entry)
    if abort or meta is None:
        print("[!] Aborted.")
        return

    safe_author     = truncate_author(meta.author)
    output_filename = safe_filename(safe_author, meta.title)
    output_file     = output_dir / output_filename

    # Re-check duplicates with the final (post-lookup) title
    if existing_stems is not None:
        final_check = strip_author_prefix(meta.title.lower()) if ' - ' in meta.title else meta.title.lower()
        if final_check != check_title and already_exists(final_check, existing_stems):
            print(f"[!] Post-lookup title '{meta.title}' matches an existing file — skipping.")
            log.info(f"Skipped (duplicate after lookup): {book_dir.name}")
            return

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
        cover_file: Path | None = None

        if meta.cover_url:
            cover_file = _download_cover(meta.cover_url, tmp / 'cover.jpg')

        codecs       = {t['codec'] for t in track_data}
        sample_rates = {t['sample_rate'] for t in track_data}
        channels     = {t['channels'] for t in track_data}
        # Stream-copy fast path is unavailable when normalizing — loudnorm
        # has to run during a re-encode.
        can_copy     = (
            not normalize
            and codecs == {'aac'}
            and len(sample_rates) == 1
            and len(channels) == 1
        )

        if not can_copy:
            label = f"AAC {bitrate}" + (' + loudnorm' if normalize else '')
            print(f"    [~] Transcoding {len(track_data)} chapter(s) to {label} …")
            transcode_errors = []
            successful_indexes: set[int] = set()
            with ThreadPoolExecutor(max_workers=MAX_TRANSCODE_WORKERS or (os.cpu_count() or 1)) as xc:
                future_map = {}
                for i, t in enumerate(track_data):
                    out = tmp / f"{i:04d}.m4a"
                    t['target'] = out
                    future_map[xc.submit(transcode_worker, t['path'], out, bitrate, normalize)] = i

                done = 0
                transcode_progress: dict = {}
                for future in as_completed(future_map):
                    done += 1
                    idx = future_map[future]
                    try:
                        future.result()
                        successful_indexes.add(idx)
                    except RuntimeError as e:
                        transcode_errors.append(str(e))
                    _print_progress('Transcoding', done, len(track_data), transcode_progress)

            if transcode_errors:
                print(f"[!] {len(transcode_errors)}/{len(track_data)} file(s) failed to transcode:")
                for err in transcode_errors:
                    print(f"    {err}")
                    log.error(f"Transcode: {err}")
                if not skip_transcode_errors:
                    log.warning(f"Aborting (transcode errors): {book_dir.name}")
                    return
                missing = len(track_data) - len(successful_indexes)
                track_data = [t for i, t in enumerate(track_data) if i in successful_indexes]
                if not track_data:
                    log.error(f"All files failed to transcode: {book_dir.name}")
                    return
                print(f"\n[!!!] WARNING: Assembling with {missing} MISSING track(s) — audiobook will be incomplete.")
                log.warning(f"Assembled with {missing} missing track(s): {book_dir.name}")
        else:
            print("    [+] Source is already AAC — stream copying (no re-encode).")
            for t in track_data:
                t['target'] = t['path']

        # Stage every input under a numbered name inside tmpdir so the concat
        # list never contains apostrophes / spaces / special characters from
        # the source filenames. Try cheapest first: symlink → hardlink → copy.
        # Windows non-admin accounts can't create symlinks without Developer
        # Mode; hardlinks fail across volumes; copy always works.
        for i, t in enumerate(track_data):
            link_name = f"input_{i:04d}{t['target'].suffix or '.m4a'}"
            link_path = tmp / link_name
            if link_path == t['target']:
                t['concat_target'] = t['target']
                continue
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            try:
                link_path.symlink_to(t['target'])
            except (OSError, NotImplementedError):
                try:
                    os.link(t['target'], link_path)
                except (OSError, NotImplementedError):
                    shutil.copy2(t['target'], link_path)
                    log.debug(f"Concat staging copied (no symlink/hardlink): {t['target'].name}")
            t['concat_target'] = link_path

        total_sec = sum(t['duration'] for t in track_data)
        curr_ms   = 0

        # Speech-based chapter detection for single-file audiobooks
        speech_chapters: list[tuple[float, str]] = []
        if chapterize and len(track_data) == 1:
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
                if accept_chapters or NON_INTERACTIVE:
                    speech_chapters = detected
                    print("    [+] Auto-accepting detected chapters.")
                    log.info("Chapterize: chapters auto-accepted")
                else:
                    _flush_stdin()
                    try:
                        raw = input("    Use these chapters? [Y/n]: ").strip().lower()
                    except EOFError:
                        log.error("Stdin closed (EOF) during chapter prompt. Exiting.")
                        print("\n[!] Stdin closed — exiting (cannot prompt interactively).")
                        sys.exit(1)
                    if raw in ('', 'y', 'yes'):
                        speech_chapters = detected
                        log.info("Chapterize: chapters accepted by user")
                    else:
                        log.info("Chapterize: chapters rejected by user")
            else:
                print("    [!] No chapter markers detected via speech recognition.")
                log.info("Chapterize: no chapter markers detected")

        # --- Build the chapter list (speech-detected, embedded, or per-track) --
        chapter_specs: list[tuple[int, int, str]] = []
        if speech_chapters:
            total_ms = int(total_sec * 1000)
            for idx, (ch_ts, ch_title) in enumerate(speech_chapters):
                # Force chapter 1 to t=0 so there's no unchaptered region at
                # the start of the file (Apple Books and some Android players
                # glitch otherwise).
                start_ms = 0 if idx == 0 else int(ch_ts * 1000)
                end_ms = (int(speech_chapters[idx + 1][0] * 1000)
                          if idx + 1 < len(speech_chapters) else total_ms)
                chapter_specs.append((start_ms, end_ms, ch_title))
        else:
            # A single source file (e.g. an .m4a or chapterless-looking .m4b
            # taking the concat path) may already carry real chapter markers.
            # The per-track fallback would flatten those into one whole-book
            # chapter, so preserve the embedded ones when there's real structure.
            embedded = _probe_source_chapters(track_data[0]['path']) if len(track_data) == 1 else []
            if len(embedded) > 1:
                print(f"    [+] Preserving {len(embedded)} embedded chapter(s) from source.")
                log.info(f"Preserving {len(embedded)} embedded chapter(s) from source file")
                chapter_specs = embedded
            else:
                for i, t in enumerate(track_data):
                    dur_ms = int(t['duration'] * 1000)
                    chapter_title = t['title'] if t['title'] != t['path'].stem else f"Chapter {i + 1}"
                    chapter_specs.append((curr_ms, curr_ms + dur_ms, chapter_title))
                    curr_ms += dur_ms

        # --- Write concat list and FFMETADATA file -------------------------
        # Paths inside the staging tmpdir are numbered (input_NNNN.ext) so
        # they never contain spaces or special chars themselves, but the
        # tmpdir prefix can — e.g. a Windows username containing an
        # apostrophe makes %TEMP% include one too. ffmpeg's concat demuxer
        # tokenizer treats a bare `'` as end-of-quoted-string, so escape
        # the only character that can break parsing.
        def _concat_quote(p: Path) -> str:
            return str(p).replace("'", "'\\''")

        with open(concat_list, 'w', encoding='utf-8') as fc:
            if speech_chapters:
                fc.write(f"file '{_concat_quote(track_data[0]['concat_target'])}'\n")
            else:
                for t in track_data:
                    fc.write(f"file '{_concat_quote(t['concat_target'])}'\n")

        with open(meta_file, 'w', encoding='utf-8') as fm:
            fm.write(render_book_ffmetadata(meta, chapter_specs))

        if meta.narrator:
            print(f"    [+] Narrator: {meta.narrator}")
        if meta.series:
            series_disp = f"{meta.series} #{meta.series_part}" if meta.series_part else meta.series
            print(f"    [+] Series:   {series_disp}")
        if meta.asin:
            print(f"    [+] ASIN:     {meta.asin}")

        # --- Build the ffmpeg assembly command -----------------------------
        # Assemble to a .partial first, then atomically rename on success. An
        # interrupted or killed ffmpeg (Ctrl-C, power loss, GUI cancel) must
        # never leave a truncated .m4b at the real path — a later run would see
        # it via exists() and skip the book as "already done", hiding a
        # half-built audiobook in the library permanently. `-f ipod` is required
        # because the muxer can't infer MP4 from the .partial extension (same
        # reason transcode_worker needs it).
        assembly_partial = output_file.with_suffix(output_file.suffix + '.partial')
        assembly_partial.unlink(missing_ok=True)

        def _build_assembly_cmd(with_cover: bool) -> list:
            c = [
                'ffmpeg', '-y', '-nostdin',
                '-f', 'concat', '-safe', '0', '-i', str(concat_list),
                '-i', str(meta_file),
            ]
            if with_cover and cover_file and cover_file.exists():
                c += cover_input_args(cover_file, audio_input_index=0, cover_input_index=2)
            else:
                c += ['-map', '0:a']
            c += [
                '-map_metadata', '1',
                '-map_chapters', '1',
                '-c:a', 'copy',
                *_language_stream_args(meta),
                '-movflags', '+faststart',
                '-f', 'ipod',
                str(assembly_partial),
            ]
            return c

        have_cover = bool(cover_file and cover_file.exists())
        try:
            run_ffmpeg_with_progress(_build_assembly_cmd(have_cover), total_sec, task_name='Assembling')
        except RuntimeError as e:
            assembly_partial.unlink(missing_ok=True)
            # A bad cover image shouldn't cost the whole (possibly hours-long)
            # assembly — retry once without it before giving up.
            if have_cover:
                print(f"\n[!] Assembly failed with cover art ({e}); retrying without cover …")
                log.warning(f"Assembly failed with cover; retrying coverless: {e}")
                try:
                    run_ffmpeg_with_progress(_build_assembly_cmd(False), total_sec, task_name='Assembling')
                except RuntimeError as e2:
                    print(f"\n[!] Assembly failed: {e2}")
                    log.error(f"Assembly failed (coverless retry): {e2}")
                    assembly_partial.unlink(missing_ok=True)
                    return
            else:
                print(f"\n[!] Assembly failed: {e}")
                log.error(f"Assembly failed: {e}")
                return

        assembly_partial.replace(output_file)

    print(f"[+] Created: {output_file.name}")
    log.info(f"Created: {output_file.name}  ({len(track_data)} track(s), {'stream-copy' if can_copy else bitrate})")

    if verify:
        verify_output(output_file, meta, total_sec, expected_chapters=len(chapter_specs),
                      source_tracks=len(track_data))

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
    parser.add_argument('-b', '--bitrate', default=DEFAULT_BITRATE, help=f'AAC bitrate for transcoding (default: {DEFAULT_BITRATE})')
    parser.add_argument('-n', '--dry-run', action='store_true', help='Scan and report without writing files')
    parser.add_argument('--auto-lookup',   action='store_true', help='Auto-select the top metadata result')
    parser.add_argument('--no-lookup',     action='store_true', help='Skip all online metadata lookups')
    parser.add_argument('--chapterize',    action='store_true', help='Detect chapters via speech recognition for single-file audiobooks (requires: pip install faster-whisper)')
    parser.add_argument('--accept-chapters', action='store_true', help='When --chapterize is used, accept the detected chapters without prompting (for non-interactive / GUI use)')
    parser.add_argument('--non-interactive', action='store_true', help='Fail fast instead of prompting. Use when launching from a GUI / cron / pipe — any cache-miss that would otherwise need a prompt aborts the book cleanly.')
    parser.add_argument('--skip-transcode-errors', action='store_true', help='Continue assembly even when some files fail to transcode')
    parser.add_argument('--normalize', action='store_true', help=f'Loudness-normalize each file to {LOUDNORM_I:g} LUFS (EBU R128, two-pass, linear). Skip for full-cast productions where dynamic range is intentional.')
    parser.add_argument('--no-verify', action='store_true', help='Skip the post-build verification report (duration/chapter/tag/cover checks + .ab_report.txt sidecar).')
    parser.add_argument('--clear-cache',  action='store_true', help='Clear cached interactive decisions and re-prompt for everything')
    parser.add_argument('--re-prompt', metavar='PATH', nargs='+', default=[],
                        help='Drop cached decisions for these specific folder paths so the script will re-prompt for them; leaves the rest of the cache intact. May be passed multiple paths.')
    parser.add_argument('--log', metavar='FILE', help='Log file path (default: ab_TIMESTAMP.log in output dir)')
    args = parser.parse_args()

    global NON_INTERACTIVE
    NON_INTERACTIVE = bool(args.non_interactive)

    # Validate bitrate up front — a typo like '192kk' or '19x' otherwise fails
    # every transcode with a confusing per-file ffmpeg error deep into a run.
    if not re.fullmatch(r'\d+k?', args.bitrate.strip(), flags=re.IGNORECASE):
        print(f"[!] Invalid bitrate '{args.bitrate}'. Use a form like 192k, 128k, or a plain bps value.")
        sys.exit(1)

    in_p  = Path(args.input).resolve()
    out_p = Path(args.output).resolve()

    if not in_p.is_dir():
        print(f"[!] Input path does not exist or is not a directory: {in_p}")
        sys.exit(1)

    out_p.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log) if args.log else out_p / f"ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_path)
    _log_tool_versions()
    log.info(f"Input:  {in_p}")
    log.info(f"Output: {out_p}")
    log.info(f"Flags:  bitrate={args.bitrate}  dry_run={args.dry_run}  auto_lookup={args.auto_lookup}  no_lookup={args.no_lookup}  chapterize={args.chapterize}  accept_chapters={args.accept_chapters}  non_interactive={args.non_interactive}  skip_transcode_errors={args.skip_transcode_errors}  normalize={args.normalize}")
    print(f"[*] Logging to: {log_path}")

    # Decision cache — remembers interactive choices across restarts
    cache_path = out_p / DECISION_CACHE_FILE
    if args.clear_cache and cache_path.exists():
        cache_path.unlink()
        print("[*] Decision cache cleared.")
        log.info("Decision cache cleared by --clear-cache")
    decision_cache = _load_decision_cache(cache_path)
    if decision_cache:
        print(f"[*] Loaded {len(decision_cache)} cached decision(s) from previous run.")
        log.info(f"Loaded {len(decision_cache)} cached decision(s)")

    if args.re_prompt:
        removed_total = 0
        for raw_path in args.re_prompt:
            resolved = str(Path(raw_path).resolve())
            keys = [resolved, MERGE_CACHE_PREFIX + resolved]
            removed = [k for k in keys if decision_cache.pop(k, None) is not None]
            if removed:
                removed_total += len(removed)
                print(f"[*] Re-prompt: cleared {len(removed)} cached decision(s) for: {resolved}")
                log.info(f"Re-prompt cleared cache key(s) for: {resolved}")
            else:
                print(f"[!] Re-prompt: no cached decisions found for: {resolved}")
        if removed_total:
            try:
                tmp = cache_path.with_suffix('.tmp')
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump(decision_cache, f, indent=2, ensure_ascii=False)
                tmp.replace(cache_path)
            except OSError as e:
                log.warning(f"Failed to save decision cache after --re-prompt: {e}")

    books = find_audiobooks(in_p, decision_cache=decision_cache, cache_path=cache_path)

    if not books:
        print("[!] No audiobook folders found.")
        log.info("No audiobook folders found.")
        sys.exit(0)

    print(f"[*] Found {len(books)} audiobook folder(s).")
    log.info(f"Found {len(books)} audiobook folder(s)")

    existing_stems = build_existing_stems(out_p)

    failures = 0
    for book_dir, book_files in sorted(books.items(), key=lambda kv: natural_sort_key(kv[0])):
        try:
            process_book(
                book_dir, book_files, out_p,
                bitrate=args.bitrate,
                dry_run=args.dry_run,
                auto_lookup=args.auto_lookup,
                no_lookup=args.no_lookup,
                existing_stems=existing_stems,
                chapterize=args.chapterize,
                accept_chapters=args.accept_chapters,
                skip_transcode_errors=args.skip_transcode_errors,
                normalize=args.normalize,
                decision_cache=decision_cache,
                cache_path=cache_path,
                verify=not args.no_verify,
            )
        except Exception as e:
            # One book's unexpected error (disk full mid-copy, an NFS blip, a
            # corrupt source) must not sink the rest of an overnight batch.
            # SystemExit / KeyboardInterrupt derive from BaseException and are
            # deliberately NOT caught here, so a closed stdin or Ctrl-C still
            # aborts the whole run.
            failures += 1
            log.exception(f"Unhandled error processing {book_dir.name}: {e}")
            print(f"[!] Error processing '{book_dir.name}': {e} — skipping to next book.")

    log.info(f"Done ({failures} book(s) failed)" if failures else "Done")
    if failures:
        print(f"\n[+] All done — {failures} book(s) failed with errors (see the log).")
    else:
        print("\n[+] All done.")


if __name__ == '__main__':
    main()
