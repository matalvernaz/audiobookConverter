# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — builds both the CLI (ab.exe) and the GUI
(Audiobook Converter.exe) into a single dist folder.

Run:  pyinstaller audiobookConverter.spec

Expects ffmpeg.exe and ffprobe.exe to be sitting at vendor/ffmpeg/ffmpeg.exe
and vendor/ffmpeg/ffprobe.exe at build time. The GitHub Actions workflow
downloads them automatically. For local builds on Windows, drop the two
exes into vendor/ffmpeg/ yourself first.
"""

import os
from pathlib import Path

SPEC_DIR = Path(SPECPATH)

# --- ffmpeg bundling -------------------------------------------------------

vendor_ffmpeg = SPEC_DIR / 'vendor' / 'ffmpeg'
ffmpeg_binaries = []
for name in ('ffmpeg.exe', 'ffprobe.exe'):
    p = vendor_ffmpeg / name
    if p.exists():
        ffmpeg_binaries.append((str(p), '.'))
if not ffmpeg_binaries:
    print('[spec] WARNING: vendor/ffmpeg/ffmpeg.exe and ffprobe.exe not found; '
          'the build will not include them.')


# --- CLI (ab.exe) ----------------------------------------------------------

a_cli = Analysis(
    ['ab.py'],
    pathex=[str(SPEC_DIR)],
    binaries=ffmpeg_binaries,
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=['wx', 'wx.lib', 'numpy', 'faster_whisper', 'ctranslate2'],
    noarchive=False,
)
pyz_cli = PYZ(a_cli.pure, a_cli.zipped_data)
exe_cli = EXE(
    pyz_cli,
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name='ab',
    console=True,
    icon=None,
)


# --- GUI (Audiobook Converter.exe) -----------------------------------------

a_gui = Analysis(
    ['gui.py'],
    pathex=[str(SPEC_DIR)],
    binaries=[],
    datas=[],
    hiddenimports=['wx._xml', 'wx._html'],
    hookspath=[],
    runtime_hooks=[],
    excludes=['numpy', 'faster_whisper', 'ctranslate2'],
    noarchive=False,
)
pyz_gui = PYZ(a_gui.pure, a_gui.zipped_data)
exe_gui = EXE(
    pyz_gui,
    a_gui.scripts,
    [],
    exclude_binaries=True,
    name='Audiobook Converter',
    console=False,
    icon=None,
)


# --- Collect both into one dist folder -------------------------------------

coll = COLLECT(
    exe_cli,
    a_cli.binaries,
    a_cli.zipfiles,
    a_cli.datas,
    exe_gui,
    a_gui.binaries,
    a_gui.zipfiles,
    a_gui.datas,
    strip=False,
    upx=False,
    name='AudiobookConverter',
)
