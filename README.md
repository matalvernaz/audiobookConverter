# audiobookConverter (ab.py)

Converts directories of audio files into single, chaptered `.m4b` audiobooks. Pulls cover art, descriptions, series, and narrator from iTunes / Google Books / Open Library / Audnexus. Interactive by default — you review and pick the metadata match for each book.

## Highlights

- **Inputs**: any folder tree containing MP3, M4A, M4B, AAC, OGG, OPUS, FLAC, WAV, or WMA
- **Outputs**: one `.m4b` per book, with embedded chapters, cover art, description, series, and narrator
- **Stream-copy fast path** when the source is already AAC at a uniform sample rate / channel count
- **Speech-based chapter detection** (`--chapterize`) for single-file books, via `faster-whisper`
- **Multi-disc / multi-part folders** auto-collapse into one book
- **Short-story collections** split across subfolders can be merged on prompt
- **Decision cache** remembers your interactive choices across runs; per-folder invalidation via `--re-prompt`
- **Duplicate detection** skips books already present in the output folder

## Requirements

- Python 3.10+ (uses `list[...]` / `dict[...]` / `X | Y` generics)
- `ffmpeg` and `ffprobe` on `PATH`
- *Optional* `faster-whisper` if you want `--chapterize`:
  ```
  pip install faster-whisper
  ```

Install ffmpeg on Debian/Ubuntu:

```
sudo apt install ffmpeg
```

## Usage

```
python3 ab.py <input_dir> [options]
```

The script walks `<input_dir>` recursively, groups files into books (one folder = one book, with multi-disc/multi-part folders merged), and for each book opens an interactive picker so you can confirm the metadata.

### Common invocations

Interactive (default):

```
python3 ab.py /path/to/audiobooks -o /path/to/output
```

Auto-select the top result for every book (still skips when confidence is too low):

```
python3 ab.py /path/to/audiobooks -o /path/to/output --auto-lookup
```

Skip metadata lookups entirely, use folder/tag info as-is:

```
python3 ab.py /path/to/audiobooks -o /path/to/output --no-lookup
```

Preview what would happen, write nothing:

```
python3 ab.py /path/to/audiobooks -o /path/to/output --dry-run
```

Speech-based chapter detection for a single-file audiobook:

```
python3 ab.py /path/to/audiobooks -o /path/to/output --chapterize
```

Re-prompt for one specific folder while keeping every other cached decision:

```
python3 ab.py /path/to/audiobooks -o /path/to/output --re-prompt "/path/to/audiobooks/Book Title"
```

### All flags

| Flag | What it does |
|------|--------------|
| `input` | Input directory containing audiobook folders. |
| `-o`, `--output PATH` | Output directory (default: current dir). Created if missing. |
| `-b`, `--bitrate RATE` | AAC bitrate for transcoded books (default: `192k`). Ignored on stream-copy fast path. |
| `-n`, `--dry-run` | Scan and report what would happen; write nothing. |
| `--auto-lookup` | Auto-pick the top metadata result if it scores above the confidence threshold. |
| `--no-lookup` | Skip all online metadata lookups; tag from folder name and existing file tags. |
| `--chapterize` | For single-file audiobooks, run `faster-whisper` over silence-gap candidates to detect spoken chapter markers ("Chapter Three", "Prologue", etc.). |
| `--skip-transcode-errors` | If some source files fail to transcode, assemble the audiobook anyway (loudly warns; the output will be missing those tracks). |
| `--clear-cache` | Delete the entire decision cache and re-prompt for every book. |
| `--re-prompt PATH [PATH …]` | Drop cached decisions for the listed folder paths only; leaves the rest of the cache intact. Use this when one book got the wrong match. |
| `--log FILE` | Override the log file path (default: `ab_<TIMESTAMP>.log` in the output dir). |

### Environment

- `WHISPER_MODEL` — model size for `--chapterize` (default `medium`; `tiny` / `base` / `small` are much faster on slow CPUs).

## How it works

1. **Discover** — recursively scan the input folder, group files into book directories. Subdirectories named `Disc 1` / `CD 02` / `Part 3` etc. collapse into their parent.
2. **Probe** — `ffprobe` every file for codec, sample rate, channels, duration, and existing tags.
3. **Identify** — for each book, query iTunes, Google Books, Open Library, and Audnexus in parallel, score the results by title and author similarity, and show you a ranked picker.
4. **Decide** — you pick a result, skip lookup, search manually, or abort the folder. Your choice is cached so a re-run doesn't re-prompt.
5. **Assemble** — stream-copy AAC sources, or transcode non-AAC sources to AAC in parallel using all available cores. Each transcode writes to a `.partial` file and is atomically renamed on success.
6. **Tag** — write an FFMETADATA file with title / author / album / series / narrator / description / cover art, plus per-chapter timestamps, and run a final `ffmpeg` pass to produce the `.m4b`.

## Decision cache

Interactive choices are saved to `<output_dir>/.ab_decisions.json`. Re-running `ab.py` against the same input folder will reuse those choices and skip the prompt. To override:

- `--re-prompt /path/to/Book` — re-ask for one folder, keep the rest cached.
- `--clear-cache` — wipe the entire cache.

Aborted books are *not* cached, so just running the script again is enough to retry an aborted folder.

## Project layout

```
README.md      — this file
ab.py          — the converter (single file, no package)
```

## License

Personal use. No license declared — open an issue if you want to use this.
