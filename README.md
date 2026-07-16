# audiobookConverter

Converts directories of audio files into single, chaptered `.m4b` audiobooks. Pulls cover art, descriptions, series, and narrator from iTunes / Google Books / Open Library / Audnexus.

Ships as two front-ends sharing one engine:

- **`ab.py`** — the CLI. Interactive numbered picker per book; the original tool.
- **`Audiobook Converter`** — wxPython desktop GUI for Windows users who'd rather click than type. Wraps the CLI; same metadata sources, same output.

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
| `--accept-chapters` | With `--chapterize`, accept the detected chapters without prompting (for non-interactive / GUI use). |
| `--normalize` | Loudness-normalize each file to ≈-18 LUFS (EBU R128, two-pass, linear). Skip for full-cast productions where dynamic range is intentional. |
| `--skip-transcode-errors` | If some source files fail to transcode, assemble the audiobook anyway (loudly warns; the output will be missing those tracks). |
| `--no-verify` | Skip the post-build verification report (duration/chapter/tag/cover checks + `.ab_report.txt` sidecar). |
| `--non-interactive` | Never prompt — any book that would need a prompt (uncached, not auto-lookup) is skipped cleanly. Use when launching from a GUI, cron, or a pipe. |
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

## GUI (Windows)

`Audiobook Converter` is a desktop front-end for non-CLI users. It wraps `ab.py` — same conversion logic — but presents the metadata picker, merge prompt, and progress as accessible wxPython dialogs (NVDA-friendly).

**Workflow:** pick an input and output folder, click **Scan**, look up metadata for each detected book in a dialog (or click **Look up all pending books** to step through them one after another), then **Convert all** runs `ab.py` in the background and streams its log into a window.

**Install (end-user, Windows):** download `AudiobookConverter-<version>-Windows.zip` from the [Releases page](https://github.com/matalvernaz/audiobookConverter/releases), unzip anywhere, run `Audiobook Converter.exe`. The release bundle includes ffmpeg/ffprobe — no separate install needed.

**Run from source (development):**

```
pip install wxPython
python gui.py
```

**Build the Windows release locally (rare):**

```
# On Windows, with ffmpeg.exe + ffprobe.exe in vendor/ffmpeg/
pip install wxPython pyinstaller
pyinstaller audiobookConverter.spec
# Output in dist/AudiobookConverter/
```

Releases are built automatically by `.github/workflows/release-windows.yml` when a `v*` tag is pushed.

### GUI feature parity

| CLI flag | GUI |
|----------|-----|
| `--auto-lookup` | "Auto-pick top match for unset books" checkbox |
| `--no-lookup` | Implicit when every book is skipped via "Skip" |
| `--dry-run` | Not exposed (run a small test folder instead) |
| `--chapterize` | **Not in V1** — coming as a runtime-install button |
| `--skip-transcode-errors` | "Continue past transcode errors" checkbox |
| `--clear-cache` | "Re-prompt selected" per book; full clear via deleting `.ab_decisions.json` |
| `--re-prompt` | "Re-prompt selected" button |

## Project layout

```
README.md                        — this file
ab.py                            — the CLI / conversion engine
gui.py                           — the wxPython desktop GUI
series.py                        — companion organiser: sorts loose .m4b files into Author/Series/Title.m4b
test_ab.py                       — unit + ffmpeg end-to-end tests (python3 test_ab.py)
audiobookConverter.spec          — PyInstaller spec (builds ab.exe + Audiobook Converter.exe)
.github/workflows/release-windows.yml — CI: build & publish Windows release on v* tag
.github/workflows/test.yml       — CI: run the test suite on every push
```

## License

Personal use. No license declared — open an issue if you want to use this.
