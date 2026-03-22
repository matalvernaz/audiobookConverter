# audiobookConverter

A Python script that converts directories of audio files into single chaptered `.m4b` audiobook files, with automatic metadata and cover art fetching.

## Features

- Converts MP3, M4A, M4B, AAC, OGG, OPUS, FLAC, WAV, and WMA files
- Merges multi-disc/multi-part audiobooks into a single file
- Fetches metadata (title, author, description, cover art) from iTunes, Google Books, and Open Library
- Embeds chapter markers from individual tracks
- Skips already-converted books via fuzzy duplicate detection
- Stream-copies AAC files (no re-encode) when possible
- Progress bars for transcoding and assembly

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe`

Install ffmpeg on Debian/Ubuntu:
```
sudo apt install ffmpeg
```

## Usage

```
python3 ab.py <input_dir> [-o <output_dir>] [-b <bitrate>] [--auto-lookup] [--no-lookup] [--dry-run]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `input` | Directory containing audiobook folders |
| `-o`, `--output` | Output directory (default: current directory) |
| `-b`, `--bitrate` | AAC bitrate for transcoding (default: `192k`) |
| `-n`, `--dry-run` | Scan and report without writing any files |
| `--auto-lookup` | Automatically select the top metadata result |
| `--no-lookup` | Skip all online metadata lookups |

### Examples

Preview what would be converted without writing anything:
```
python3 ab.py /path/to/audiobooks -o /path/to/output --dry-run --no-lookup
```

Convert everything, auto-selecting metadata:
```
python3 ab.py /path/to/audiobooks -o /path/to/output --auto-lookup
```

Convert with interactive metadata selection:
```
python3 ab.py /path/to/audiobooks -o /path/to/output
```

## How It Works

1. Scans the input directory recursively for audiobook folders
2. Groups multi-disc folders (e.g. `Book Title Disc 1`, `Book Title Disc 2`) into a single book
3. Probes audio files with `ffprobe` to read existing metadata
4. Searches iTunes, Google Books, and Open Library for the best metadata match
5. Transcodes non-AAC files to AAC (or stream-copies if already AAC)
6. Assembles everything into a single `.m4b` with embedded chapters, cover art, and metadata
