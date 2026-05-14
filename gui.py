#!/usr/bin/env python3
"""
gui.py — Windows GUI for the audiobookConverter (ab.py).

Built for users who'd rather click than type. The CLI (`ab.py`) is the source
of truth for the actual conversion work; this front-end:

  1. Walks the input folder using ab.py's own discovery logic
  2. Asks the user about merges and metadata via accessible wx dialogs
  3. Writes those choices into ab.py's decision cache (.ab_decisions.json)
  4. Launches ab.py as a subprocess to do the conversion — by then the cache
     is fully populated so ab.py runs without ever prompting

That gives a polished GUI without forking the conversion logic.

Accessibility notes: all controls have labels, the book list uses wx.ListCtrl
(reads cleanly under NVDA), every dialog is keyboard-navigable, and the
conversion log is a wx.TextCtrl that screen readers can browse.
"""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import wx

import ab  # CLI module — we reuse its discovery, scoring, and metadata search


APP_NAME = 'Audiobook Converter'
BITRATE_PRESETS = ['64k', '96k', '128k', '192k', '256k', '320k']
DEFAULT_BITRATE = '192k'

# Decision-cache filename (matches ab.py)
DECISION_CACHE_FILE = '.ab_decisions.json'
MERGE_CACHE_PREFIX = 'merge:'


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def _get_ab_command() -> list[str]:
    """Return the command-line prefix used to invoke ab.py as a subprocess.

    In a PyInstaller frozen build we expect ab.exe sitting next to gui.exe.
    In development we just invoke the script with the running Python.
    """
    if getattr(sys, 'frozen', False):
        ab_exe = Path(sys.executable).parent / 'ab.exe'
        if ab_exe.exists():
            return [str(ab_exe)]
    # Fallback / dev mode
    return [sys.executable, str(Path(__file__).resolve().parent / 'ab.py')]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BookEntry:
    """A detected book + its conversion decision."""
    book_dir: Path
    files: list = field(default_factory=list)
    detected_title: str = ''
    detected_author: str = ''
    track_count: int = 0
    total_duration_sec: float = 0.0
    decision: dict | None = None
    status: str = 'pending'  # 'pending' | 'cached' | 'picked' | 'skipped'

    @property
    def display_title(self) -> str:
        if self.decision and self.decision.get('title'):
            return self.decision['title']
        return self.detected_title or self.book_dir.name

    @property
    def display_author(self) -> str:
        if self.decision and self.decision.get('author'):
            return self.decision['author']
        return self.detected_author or 'Unknown'

    @property
    def status_label(self) -> str:
        return {
            'pending': 'Needs metadata',
            'cached':  'Cached',
            'picked':  'Picked',
            'skipped': 'Skip (use local info)',
        }.get(self.status, self.status)

    @property
    def duration_label(self) -> str:
        secs = int(self.total_duration_sec)
        h, m = divmod(secs // 60, 60)
        return f"{h:d}:{m:02d}"


# ---------------------------------------------------------------------------
# Merge dialog
# ---------------------------------------------------------------------------

class MergeDialog(wx.Dialog):
    """Asked during scan when we find N single-file subfolders under one parent.

    Mirrors the CLI's "Merge into one book? [y/N]" prompt.
    """

    def __init__(self, parent: wx.Window, parent_name: str, children: list[str]):
        super().__init__(
            parent, title='Merge subfolders into one book?',
            size=(560, 420),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )

        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        intro = wx.StaticText(
            panel,
            label=(
                f'The folder "{parent_name}" contains {len(children)} '
                f'subfolders, each with a single audio file. This is often '
                f'used for short-story collections.\n\n'
                f'Should they be merged into one audiobook?'
            ),
        )
        intro.Wrap(520)
        sizer.Add(intro, 0, wx.ALL | wx.EXPAND, 12)

        list_label = wx.StaticText(panel, label='Subfolders:')
        sizer.Add(list_label, 0, wx.LEFT | wx.RIGHT, 12)

        list_box = wx.ListBox(panel, choices=children)
        list_box.SetMinSize((-1, 180))
        sizer.Add(list_box, 1, wx.ALL | wx.EXPAND, 12)

        # Buttons
        btn_sizer = wx.StdDialogButtonSizer()
        self.btn_yes = wx.Button(panel, wx.ID_YES, label='&Merge')
        self.btn_no  = wx.Button(panel, wx.ID_NO,  label='&Keep separate')
        btn_sizer.AddButton(self.btn_yes)
        btn_sizer.AddButton(self.btn_no)
        btn_sizer.Realize()
        sizer.Add(btn_sizer, 0, wx.ALL | wx.ALIGN_RIGHT, 12)

        panel.SetSizer(sizer)
        self.SetEscapeId(wx.ID_NO)
        self.btn_no.SetDefault()

        self.btn_yes.Bind(wx.EVT_BUTTON, lambda e: self.EndModal(wx.ID_YES))
        self.btn_no.Bind(wx.EVT_BUTTON,  lambda e: self.EndModal(wx.ID_NO))


# ---------------------------------------------------------------------------
# Metadata picker dialog
# ---------------------------------------------------------------------------

class MetadataDialog(wx.Dialog):
    """Pick the metadata match for one book.

    Shows a list of ranked results from iTunes / Google Books / Open Library /
    Audnexus, with the description and other fields previewed for whichever
    row is focused. Mirrors the CLI's numbered picker.
    """

    def __init__(self, parent: wx.Window, book: BookEntry):
        super().__init__(
            parent,
            title=f'Pick metadata: {book.book_dir.name}',
            size=(900, 640),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )
        self.book = book
        self.results: list[dict] = []
        self.choice: dict | None = None  # output

        self._query_title  = book.detected_title or book.book_dir.name
        self._query_author = book.detected_author

        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        # --- Query row -----------------------------------------------------
        query_box = wx.StaticBoxSizer(wx.VERTICAL, panel, label='Search query')
        grid = wx.FlexGridSizer(rows=2, cols=2, vgap=6, hgap=6)
        grid.AddGrowableCol(1, 1)

        grid.Add(wx.StaticText(panel, label='Title:'),  0, wx.ALIGN_CENTER_VERTICAL)
        self.title_ctrl = wx.TextCtrl(panel, value=self._query_title)
        grid.Add(self.title_ctrl, 1, wx.EXPAND)

        grid.Add(wx.StaticText(panel, label='Author:'), 0, wx.ALIGN_CENTER_VERTICAL)
        self.author_ctrl = wx.TextCtrl(panel, value=self._query_author)
        grid.Add(self.author_ctrl, 1, wx.EXPAND)

        query_box.Add(grid, 0, wx.ALL | wx.EXPAND, 6)

        self.search_btn = wx.Button(panel, label='&Search')
        self.search_btn.Bind(wx.EVT_BUTTON, self.on_search)
        query_box.Add(self.search_btn, 0, wx.ALL, 6)

        sizer.Add(query_box, 0, wx.ALL | wx.EXPAND, 10)

        # --- Results list --------------------------------------------------
        results_label = wx.StaticText(panel, label='Results (best match first):')
        sizer.Add(results_label, 0, wx.LEFT | wx.RIGHT, 10)

        self.results_list = wx.ListCtrl(
            panel,
            style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.LC_HRULES,
        )
        self.results_list.InsertColumn(0, 'Score', width=60)
        self.results_list.InsertColumn(1, 'Source', width=110)
        self.results_list.InsertColumn(2, 'Title', width=280)
        self.results_list.InsertColumn(3, 'Author', width=180)
        self.results_list.InsertColumn(4, 'Year', width=60)
        self.results_list.InsertColumn(5, 'Series', width=140)
        sizer.Add(self.results_list, 2, wx.ALL | wx.EXPAND, 10)

        self.results_list.Bind(wx.EVT_LIST_ITEM_SELECTED,   self.on_result_focus)
        self.results_list.Bind(wx.EVT_LIST_ITEM_ACTIVATED,  self.on_pick)

        # --- Detail preview ------------------------------------------------
        preview_box = wx.StaticBoxSizer(wx.VERTICAL, panel, label='Selected result')
        self.detail_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_BESTWRAP,
        )
        self.detail_ctrl.SetMinSize((-1, 120))
        preview_box.Add(self.detail_ctrl, 1, wx.ALL | wx.EXPAND, 6)
        sizer.Add(preview_box, 1, wx.ALL | wx.EXPAND, 10)

        # --- Action buttons ------------------------------------------------
        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.pick_btn   = wx.Button(panel, label='&Pick selected')
        self.skip_btn   = wx.Button(panel, label='S&kip (use local info)')
        self.cancel_btn = wx.Button(panel, wx.ID_CANCEL, label='&Cancel')
        self.pick_btn.Bind(wx.EVT_BUTTON,   self.on_pick)
        self.skip_btn.Bind(wx.EVT_BUTTON,   self.on_skip)
        self.cancel_btn.Bind(wx.EVT_BUTTON, lambda e: self.EndModal(wx.ID_CANCEL))
        btn_row.Add(self.pick_btn,   0, wx.RIGHT, 8)
        btn_row.Add(self.skip_btn,   0, wx.RIGHT, 8)
        btn_row.AddStretchSpacer(1)
        btn_row.Add(self.cancel_btn, 0)
        sizer.Add(btn_row, 0, wx.ALL | wx.EXPAND, 10)

        panel.SetSizer(sizer)

        # Auto-search on open so users see results immediately.
        wx.CallAfter(self.on_search, None)

    # ----------------------------------------------------------------------

    def on_search(self, _evt):
        title  = self.title_ctrl.GetValue().strip()
        author = self.author_ctrl.GetValue().strip()
        if not title:
            wx.MessageBox('Please enter a title to search for.', 'Missing title',
                          wx.OK | wx.ICON_INFORMATION, self)
            return

        self.search_btn.Disable()
        self.search_btn.SetLabel('Searching…')
        self.results_list.DeleteAllItems()
        self.detail_ctrl.SetValue('')

        def worker():
            try:
                norm = ab.normalise_author(author)
                results = ab.search_metadata(title, norm)
            except Exception as e:
                results = []
                wx.CallAfter(wx.MessageBox,
                             f'Search failed: {e}', 'Error',
                             wx.OK | wx.ICON_ERROR, self)
            wx.CallAfter(self._populate, results, title, author)

        threading.Thread(target=worker, daemon=True).start()

    def _populate(self, results: list[dict], query_title: str, query_author: str):
        self.results = results
        norm_author = ab.normalise_author(query_author)
        for r in results:
            score = ab._score_result(r, query_title, norm_author)
            idx = self.results_list.InsertItem(self.results_list.GetItemCount(), f'{score:.2f}')
            self.results_list.SetItem(idx, 1, r.get('source', '') or '')
            self.results_list.SetItem(idx, 2, r.get('title', '')  or '')
            self.results_list.SetItem(idx, 3, r.get('author', '') or '')
            self.results_list.SetItem(idx, 4, str(r.get('year', '') or ''))
            self.results_list.SetItem(idx, 5, r.get('series', '') or '')

        self.search_btn.Enable()
        self.search_btn.SetLabel('&Search')

        if results:
            self.results_list.Select(0)
            self.results_list.Focus(0)
            self.results_list.SetFocus()
        else:
            self.detail_ctrl.SetValue('No results found. You can change the title/author and search again, or Skip.')

    def on_result_focus(self, evt):
        idx = evt.GetIndex()
        if 0 <= idx < len(self.results):
            r = self.results[idx]
            lines = [
                f"Title:    {r.get('title', '')}",
                f"Author:   {r.get('author', '')}",
                f"Year:     {r.get('year', '')}",
                f"Source:   {r.get('source', '')}",
                f"Series:   {r.get('series', '') or '—'}",
                f"Narrator: {r.get('narrator', '') or '—'}",
                f"Cover:    {'yes' if r.get('cover_url') else 'no'}",
                '',
                'Description:',
                r.get('desc', '') or '(no description)',
            ]
            self.detail_ctrl.SetValue('\n'.join(lines))

    def on_pick(self, _evt):
        idx = self.results_list.GetFirstSelected()
        if idx < 0 or idx >= len(self.results):
            wx.MessageBox('Pick a result first, or click Skip.', 'No selection',
                          wx.OK | wx.ICON_INFORMATION, self)
            return
        r = self.results[idx]
        norm_author = ab.normalise_author(self.author_ctrl.GetValue().strip())
        self.choice = {
            'title':     r.get('title', '')  or self.title_ctrl.GetValue().strip(),
            'author':    r.get('author', '') or norm_author,
            'cover_url': r.get('cover_url') or None,
            'desc':      r.get('desc', '')   or '',
            'series':    r.get('series', '') or '',
            'narrator':  r.get('narrator','') or '',
            'source':    r.get('source', ''),
        }
        self.EndModal(wx.ID_OK)

    def on_skip(self, _evt):
        # Skip = use local info. Stored as a decision with the early title/author
        # and no online enrichment, so ab.py won't re-prompt.
        norm_author = ab.normalise_author(self.author_ctrl.GetValue().strip())
        self.choice = {
            'title':     self.title_ctrl.GetValue().strip() or self._query_title,
            'author':    norm_author,
            'cover_url': None,
            'desc':      '',
            'series':    '',
            'narrator':  '',
            'source':    'local',
        }
        self.EndModal(wx.ID_OK)


# ---------------------------------------------------------------------------
# Conversion dialog
# ---------------------------------------------------------------------------

class ConversionDialog(wx.Dialog):
    """Live progress + log while ab.py runs the conversion."""

    def __init__(self, parent: wx.Window, command: list[str], cwd: str | None = None):
        super().__init__(
            parent, title='Converting…',
            size=(820, 520),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )
        self.command = command
        self.cwd = cwd
        self.process: subprocess.Popen | None = None
        self._cancelled = False
        self._reader_thread: threading.Thread | None = None

        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.status = wx.StaticText(panel, label='Starting…')
        sizer.Add(self.status, 0, wx.ALL, 10)

        log_label = wx.StaticText(panel, label='Conversion log:')
        sizer.Add(log_label, 0, wx.LEFT | wx.RIGHT, 10)

        self.log = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL,
        )
        self.log.SetMinSize((-1, 320))
        sizer.Add(self.log, 1, wx.ALL | wx.EXPAND, 10)

        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.cancel_btn = wx.Button(panel, label='&Cancel')
        self.close_btn  = wx.Button(panel, wx.ID_OK, label='&Close')
        self.close_btn.Disable()
        self.cancel_btn.Bind(wx.EVT_BUTTON, self.on_cancel)
        self.close_btn.Bind(wx.EVT_BUTTON,  lambda e: self.EndModal(wx.ID_OK))
        btn_row.AddStretchSpacer(1)
        btn_row.Add(self.cancel_btn, 0, wx.RIGHT, 8)
        btn_row.Add(self.close_btn,  0)
        sizer.Add(btn_row, 0, wx.ALL | wx.EXPAND, 10)

        panel.SetSizer(sizer)

        self.Bind(wx.EVT_CLOSE, self.on_close)

    def start(self):
        # Put the child in its own process group/session so we can kill its
        # whole tree on cancel — otherwise on Windows the ffmpeg subprocess
        # survives the Python wrapper and keeps running.
        popen_kwargs = {}
        if platform.system() == 'Windows':
            popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs['start_new_session'] = True

        # Force the child to flush stdout per line and emit UTF-8 — without
        # these env vars Windows can batch output for seconds at a time and
        # mojibake any non-ASCII title/author in the log pane.
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'

        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                bufsize=1,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env,
                **popen_kwargs,
            )
        except Exception as e:
            self.log.AppendText(f'Failed to start ab.py: {e}\n')
            self.status.SetLabel('Failed to start.')
            self.cancel_btn.Disable()
            self.close_btn.Enable()
            return

        self.status.SetLabel('Running…')
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()

    def _kill_process_tree(self):
        """Kill the conversion subprocess and any ffmpeg children.

        terminate() on the wrapper alone leaves ffmpeg running on Windows;
        we have to tear the whole process group/job down.
        """
        if not self.process or self.process.poll() is not None:
            return
        try:
            if platform.system() == 'Windows':
                subprocess.run(
                    ['taskkill', '/T', '/F', '/PID', str(self.process.pid)],
                    capture_output=True,
                    timeout=10,
                )
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        except Exception:
            # Last-ditch: kill the wrapper at least.
            try:
                self.process.kill()
            except Exception:
                pass

    def _reader(self):
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            wx.CallAfter(self.log.AppendText, line)
        self.process.wait()
        wx.CallAfter(self._on_finished)

    def _on_finished(self):
        rc = self.process.returncode if self.process else -1
        if self._cancelled:
            self.status.SetLabel('Cancelled.')
        elif rc == 0:
            self.status.SetLabel('Done.')
            self.log.AppendText('\n[+] Conversion finished successfully.\n')
            wx.Bell()  # NVDA-friendly: audible "I need you" / "I'm done" cue
        else:
            self.status.SetLabel(f'Failed (exit code {rc}).')
            self.log.AppendText(f'\n[!] ab.py exited with code {rc}.\n')
            wx.Bell()
        self.cancel_btn.Disable()
        self.close_btn.Enable()
        self.close_btn.SetFocus()

    def on_cancel(self, _evt):
        if self.process and self.process.poll() is None:
            if wx.MessageBox(
                'Cancel the in-progress conversion? Any partly-built audiobook '
                'will be left in the output folder and may be incomplete.',
                'Confirm cancel',
                wx.YES_NO | wx.ICON_QUESTION,
                self,
            ) == wx.YES:
                self._cancelled = True
                self._kill_process_tree()

    def on_close(self, evt):
        if self.process and self.process.poll() is None:
            evt.Veto()
            self.on_cancel(None)
            return
        evt.Skip()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title=APP_NAME, size=(1100, 720))

        self.books: list[BookEntry] = []
        self.decision_cache: dict = {}
        self.cache_path: Path | None = None

        self._build_ui()
        self._restore_prefs()
        self.Bind(wx.EVT_CLOSE, self.on_close)

    # ----------------------------------------------------------------------
    # UI construction
    # ----------------------------------------------------------------------

    def _build_ui(self):
        panel = wx.Panel(self)
        outer = wx.BoxSizer(wx.VERTICAL)

        # --- Folder pickers -----------------------------------------------
        folders = wx.StaticBoxSizer(wx.VERTICAL, panel, label='Folders')

        in_row = wx.BoxSizer(wx.HORIZONTAL)
        in_row.Add(wx.StaticText(panel, label='Input folder:',  size=(120, -1)),
                   0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.input_ctrl = wx.TextCtrl(panel)
        in_row.Add(self.input_ctrl, 1, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 6)
        in_btn = wx.Button(panel, label='&Browse…')
        in_btn.Bind(wx.EVT_BUTTON, self.on_pick_input)
        in_row.Add(in_btn, 0, wx.ALIGN_CENTER_VERTICAL)
        folders.Add(in_row, 0, wx.ALL | wx.EXPAND, 6)

        out_row = wx.BoxSizer(wx.HORIZONTAL)
        out_row.Add(wx.StaticText(panel, label='Output folder:', size=(120, -1)),
                    0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.output_ctrl = wx.TextCtrl(panel)
        out_row.Add(self.output_ctrl, 1, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 6)
        out_btn = wx.Button(panel, label='B&rowse…')
        out_btn.Bind(wx.EVT_BUTTON, self.on_pick_output)
        out_row.Add(out_btn, 0, wx.ALIGN_CENTER_VERTICAL)
        folders.Add(out_row, 0, wx.ALL | wx.EXPAND, 6)

        outer.Add(folders, 0, wx.ALL | wx.EXPAND, 10)

        # --- Options ------------------------------------------------------
        options = wx.StaticBoxSizer(wx.HORIZONTAL, panel, label='Options')

        options.Add(wx.StaticText(panel, label='Bitrate:'),
                    0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.bitrate_ctrl = wx.ComboBox(
            panel, value=DEFAULT_BITRATE, choices=BITRATE_PRESETS,
            style=wx.CB_DROPDOWN,
        )
        options.Add(self.bitrate_ctrl, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 16)

        self.auto_lookup_ctrl = wx.CheckBox(panel, label='Auto-pick top match for unset books')
        options.Add(self.auto_lookup_ctrl, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 16)

        self.skip_transcode_ctrl = wx.CheckBox(panel, label='Continue past transcode errors')
        options.Add(self.skip_transcode_ctrl, 0, wx.ALIGN_CENTER_VERTICAL)

        outer.Add(options, 0, wx.ALL | wx.EXPAND, 10)

        # --- Books list ---------------------------------------------------
        books_box = wx.StaticBoxSizer(wx.VERTICAL, panel, label='Detected books')

        action_row = wx.BoxSizer(wx.HORIZONTAL)
        self.scan_btn = wx.Button(panel, label='&Scan input folder')
        self.scan_btn.Bind(wx.EVT_BUTTON, self.on_scan)
        action_row.Add(self.scan_btn, 0, wx.RIGHT, 6)

        self.lookup_btn = wx.Button(panel, label='&Look up metadata for selected')
        self.lookup_btn.Bind(wx.EVT_BUTTON, self.on_lookup_selected)
        self.lookup_btn.Disable()
        action_row.Add(self.lookup_btn, 0, wx.RIGHT, 6)

        self.lookup_all_btn = wx.Button(panel, label='Look up &all pending books')
        self.lookup_all_btn.Bind(wx.EVT_BUTTON, self.on_lookup_all)
        self.lookup_all_btn.Disable()
        action_row.Add(self.lookup_all_btn, 0, wx.RIGHT, 6)

        self.skip_btn = wx.Button(panel, label='S&kip selected (use local info)')
        self.skip_btn.Bind(wx.EVT_BUTTON, self.on_skip_selected)
        self.skip_btn.Disable()
        action_row.Add(self.skip_btn, 0, wx.RIGHT, 6)

        self.clear_btn = wx.Button(panel, label='&Re-prompt selected')
        self.clear_btn.Bind(wx.EVT_BUTTON, self.on_reprompt_selected)
        self.clear_btn.Disable()
        action_row.Add(self.clear_btn, 0)

        books_box.Add(action_row, 0, wx.ALL, 6)

        self.books_list = wx.ListCtrl(
            panel,
            style=wx.LC_REPORT | wx.LC_HRULES | wx.LC_SINGLE_SEL,
        )
        self.books_list.InsertColumn(0, 'Status', width=140)
        self.books_list.InsertColumn(1, 'Title',  width=320)
        self.books_list.InsertColumn(2, 'Author', width=220)
        self.books_list.InsertColumn(3, 'Tracks', width=60)
        self.books_list.InsertColumn(4, 'Length', width=80)
        self.books_list.InsertColumn(5, 'Folder', width=240)
        self.books_list.Bind(wx.EVT_LIST_ITEM_SELECTED,   self.on_book_select)
        self.books_list.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.on_book_select)
        self.books_list.Bind(wx.EVT_LIST_ITEM_ACTIVATED,  self.on_lookup_selected)

        books_box.Add(self.books_list, 1, wx.ALL | wx.EXPAND, 6)

        outer.Add(books_box, 1, wx.ALL | wx.EXPAND, 10)

        # --- Convert row --------------------------------------------------
        convert_row = wx.BoxSizer(wx.HORIZONTAL)
        self.summary = wx.StaticText(panel, label='No books scanned yet.')
        convert_row.Add(self.summary, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 12)
        self.convert_btn = wx.Button(panel, label='&Convert all')
        self.convert_btn.Bind(wx.EVT_BUTTON, self.on_convert)
        self.convert_btn.Disable()
        convert_row.Add(self.convert_btn, 0, wx.ALIGN_CENTER_VERTICAL)
        outer.Add(convert_row, 0, wx.ALL | wx.EXPAND, 10)

        panel.SetSizer(outer)

        # Status bar (NVDA reads these announcements naturally)
        self.CreateStatusBar(1)
        self.SetStatusText('Pick an input folder and click Scan.')

    # ----------------------------------------------------------------------
    # Preferences
    # ----------------------------------------------------------------------

    def _prefs_path(self) -> Path:
        return Path(wx.StandardPaths.Get().GetUserConfigDir()) / 'audiobookConverter' / 'gui_prefs.json'

    def _restore_prefs(self):
        try:
            with open(self._prefs_path(), encoding='utf-8') as f:
                prefs = json.load(f)
            self.input_ctrl.SetValue(prefs.get('input', ''))
            self.output_ctrl.SetValue(prefs.get('output', ''))
            self.bitrate_ctrl.SetValue(prefs.get('bitrate', DEFAULT_BITRATE))
            self.auto_lookup_ctrl.SetValue(prefs.get('auto_lookup', False))
            self.skip_transcode_ctrl.SetValue(prefs.get('skip_transcode_errors', False))
        except Exception:
            pass

    def _save_prefs(self):
        try:
            path = self._prefs_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({
                    'input':                 self.input_ctrl.GetValue(),
                    'output':                self.output_ctrl.GetValue(),
                    'bitrate':               self.bitrate_ctrl.GetValue(),
                    'auto_lookup':           self.auto_lookup_ctrl.GetValue(),
                    'skip_transcode_errors': self.skip_transcode_ctrl.GetValue(),
                }, f, indent=2)
        except Exception:
            pass  # prefs are not load-bearing

    # ----------------------------------------------------------------------
    # Decision-cache helpers (shared with ab.py)
    # ----------------------------------------------------------------------

    def _load_cache(self):
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, encoding='utf-8') as f:
                self.decision_cache = json.load(f)
        except (json.JSONDecodeError, OSError, FileNotFoundError):
            self.decision_cache = {}

    def _save_cache(self):
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self.decision_cache, f, indent=2, ensure_ascii=False)
            tmp.replace(self.cache_path)
        except OSError as e:
            wx.MessageBox(f'Could not write decision cache: {e}',
                          'Cache write failed', wx.OK | wx.ICON_ERROR, self)

    # ----------------------------------------------------------------------
    # Folder picker handlers
    # ----------------------------------------------------------------------

    def on_pick_input(self, _evt):
        dlg = wx.DirDialog(self, 'Choose the folder containing audiobook folders',
                           defaultPath=self.input_ctrl.GetValue() or '')
        if dlg.ShowModal() == wx.ID_OK:
            self.input_ctrl.SetValue(dlg.GetPath())
        dlg.Destroy()

    def on_pick_output(self, _evt):
        dlg = wx.DirDialog(self, 'Choose where converted .m4b files go',
                           defaultPath=self.output_ctrl.GetValue() or '')
        if dlg.ShowModal() == wx.ID_OK:
            self.output_ctrl.SetValue(dlg.GetPath())
        dlg.Destroy()

    # ----------------------------------------------------------------------
    # Scan
    # ----------------------------------------------------------------------

    def on_scan(self, _evt):
        in_p  = self.input_ctrl.GetValue().strip()
        out_p = self.output_ctrl.GetValue().strip()
        if not in_p or not Path(in_p).is_dir():
            wx.MessageBox('Pick an input folder first.', 'Missing input',
                          wx.OK | wx.ICON_INFORMATION, self)
            return
        if not out_p:
            wx.MessageBox('Pick an output folder first.', 'Missing output',
                          wx.OK | wx.ICON_INFORMATION, self)
            return

        in_path  = Path(in_p).resolve()
        out_path = Path(out_p).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        self.cache_path = out_path / DECISION_CACHE_FILE
        self._load_cache()

        # Disable UI during scan
        self.scan_btn.Disable()
        self.convert_btn.Disable()
        self.books_list.DeleteAllItems()
        self.books.clear()
        self.SetStatusText('Scanning…')

        def merge_callback(parent_name: str, children: list[str]) -> bool:
            """Called from the scan thread; needs to show a wx dialog on the
            main thread and wait for the answer."""
            done = threading.Event()
            result = {'ans': False}

            def show():
                dlg = MergeDialog(self, parent_name, children)
                result['ans'] = (dlg.ShowModal() == wx.ID_YES)
                dlg.Destroy()
                done.set()

            wx.CallAfter(show)
            done.wait()
            return result['ans']

        def worker():
            try:
                books = ab.find_audiobooks(
                    in_path,
                    decision_cache=self.decision_cache,
                    cache_path=self.cache_path,
                    prompt_merge=merge_callback,
                )
            except Exception as e:
                wx.CallAfter(self._scan_failed, str(e))
                return

            # Probe each book for track count + duration
            entries: list[BookEntry] = []
            for book_dir, files in books.items():
                files = sorted(files, key=lambda f: ab.natural_sort_key(f))
                entries.append(BookEntry(book_dir=book_dir, files=files, track_count=len(files)))

            # Probe in parallel — cheap I/O
            with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as ex:
                fut_map = {ex.submit(ab.probe_file, e.files[0]): e for e in entries}
                for fut in as_completed(fut_map):
                    e = fut_map[fut]
                    initial = fut.result()
                    if initial:
                        raw_album  = initial['album']  if initial else e.book_dir.name
                        raw_artist = initial['artist'] if initial else 'Unknown Author'
                        is_collection = len({f.parent for f in e.files}) > 1
                        use_folder = (
                            is_collection
                            or 'unknown' in raw_album.lower()
                            or (len(raw_album) < 15
                                and len(e.book_dir.name) > len(raw_album) + ab.FOLDER_NAME_MIN_ADVANTAGE)
                        )
                        title  = ab.clean_title(e.book_dir.name if use_folder else raw_album)
                        author = ab.normalise_author(raw_artist)
                        title  = ab.strip_author_from_title(title, author)
                        e.detected_title  = title
                        e.detected_author = author
                        e.total_duration_sec = float(initial.get('duration', 0) or 0) * e.track_count

                # Apply cached decisions
                for e in entries:
                    key = str(e.book_dir)
                    cached = self.decision_cache.get(key)
                    if cached and not cached.get('aborted'):
                        e.decision = cached
                        e.status = 'cached'

            wx.CallAfter(self._scan_done, entries)

        threading.Thread(target=worker, daemon=True).start()

    def _scan_failed(self, msg: str):
        self.scan_btn.Enable()
        wx.MessageBox(f'Scan failed: {msg}', 'Error', wx.OK | wx.ICON_ERROR, self)
        self.SetStatusText('Scan failed.')

    def _scan_done(self, entries: list[BookEntry]):
        self.books = sorted(entries, key=lambda e: ab.natural_sort_key(e.book_dir))
        self._rebuild_list()
        self.scan_btn.Enable()
        self.convert_btn.Enable(bool(self.books))
        self.lookup_all_btn.Enable(any(e.status == 'pending' for e in self.books))
        if self.books:
            self.SetStatusText(f'Found {len(self.books)} book(s). Look up metadata, then click Convert all.')
        else:
            self.SetStatusText('No audiobook folders found.')

    def _rebuild_list(self):
        self.books_list.DeleteAllItems()
        for e in self.books:
            idx = self.books_list.InsertItem(self.books_list.GetItemCount(), e.status_label)
            self.books_list.SetItem(idx, 1, e.display_title)
            self.books_list.SetItem(idx, 2, e.display_author)
            self.books_list.SetItem(idx, 3, str(e.track_count))
            self.books_list.SetItem(idx, 4, e.duration_label)
            self.books_list.SetItem(idx, 5, e.book_dir.name)
        self._update_summary()

    def _update_summary(self):
        if not self.books:
            self.summary.SetLabel('No books scanned yet.')
            return
        pending = sum(1 for e in self.books if e.status == 'pending')
        cached  = sum(1 for e in self.books if e.status == 'cached')
        picked  = sum(1 for e in self.books if e.status == 'picked')
        skipped = sum(1 for e in self.books if e.status == 'skipped')
        self.summary.SetLabel(
            f'{len(self.books)} book(s): {pending} pending, {cached} cached, '
            f'{picked} picked, {skipped} skip.'
        )

    # ----------------------------------------------------------------------
    # Per-book actions
    # ----------------------------------------------------------------------

    def _selected_book(self) -> BookEntry | None:
        idx = self.books_list.GetFirstSelected()
        if idx < 0 or idx >= len(self.books):
            return None
        return self.books[idx]

    def on_book_select(self, _evt):
        has_sel = self.books_list.GetFirstSelected() >= 0
        self.lookup_btn.Enable(has_sel)
        self.skip_btn.Enable(has_sel)
        self.clear_btn.Enable(has_sel)

    def on_lookup_selected(self, _evt):
        e = self._selected_book()
        if not e:
            return
        self._lookup_one(e)

    def _lookup_one(self, e: BookEntry):
        dlg = MetadataDialog(self, e)
        try:
            if dlg.ShowModal() == wx.ID_OK and dlg.choice:
                e.decision = dlg.choice
                e.status = 'skipped' if dlg.choice.get('source') == 'local' else 'picked'
                self._persist_decision(e)
                self._rebuild_list()
        finally:
            dlg.Destroy()

    def on_lookup_all(self, _evt):
        if not any(e.status == 'pending' for e in self.books):
            return
        for e in list(self.books):
            if e.status != 'pending':
                continue
            self.books_list.Select(self.books.index(e), 1)
            self.books_list.EnsureVisible(self.books.index(e))
            self._lookup_one(e)

    def on_skip_selected(self, _evt):
        e = self._selected_book()
        if not e:
            return
        e.decision = {
            'title':     e.detected_title or e.book_dir.name,
            'author':    e.detected_author or 'Unknown Author',
            'cover_url': None,
            'desc':      '',
            'series':    '',
            'narrator':  '',
            'source':    'local',
        }
        e.status = 'skipped'
        self._persist_decision(e)
        self._rebuild_list()

    def on_reprompt_selected(self, _evt):
        e = self._selected_book()
        if not e:
            return
        key = str(e.book_dir)
        self.decision_cache.pop(key, None)
        self._save_cache()
        e.decision = None
        e.status = 'pending'
        self._rebuild_list()

    def _persist_decision(self, e: BookEntry):
        key = str(e.book_dir)
        d = e.decision or {}
        self.decision_cache[key] = {
            'title':     d.get('title', '')     or '',
            'author':    d.get('author', '')    or '',
            'cover_url': d.get('cover_url')     or None,
            'desc':      d.get('desc', '')      or '',
            'series':    d.get('series', '')    or '',
            'narrator':  d.get('narrator', '')  or '',
            'timestamp': datetime.now().isoformat(),
        }
        self._save_cache()

    # ----------------------------------------------------------------------
    # Convert
    # ----------------------------------------------------------------------

    def on_convert(self, _evt):
        if not self.books:
            return
        in_p  = self.input_ctrl.GetValue().strip()
        out_p = self.output_ctrl.GetValue().strip()
        if not in_p or not out_p:
            return

        pending = [e for e in self.books if e.status == 'pending']
        if pending and not self.auto_lookup_ctrl.GetValue():
            if wx.MessageBox(
                f'{len(pending)} book(s) still need metadata. They will be '
                f'skipped (no online lookup, no embed). Continue?',
                'Books still pending',
                wx.YES_NO | wx.ICON_QUESTION, self,
            ) != wx.YES:
                return

        cmd = _get_ab_command() + [
            in_p,
            '-o', out_p,
            '-b', self.bitrate_ctrl.GetValue() or DEFAULT_BITRATE,
            # Belt-and-braces: the decision cache is fully populated before we
            # launch, so ab.py shouldn't need to prompt. But if something
            # slipped through, --non-interactive makes the CLI fail fast on
            # that book instead of hanging on input() waiting for a tty.
            '--non-interactive',
            '--accept-chapters',
        ]
        if self.auto_lookup_ctrl.GetValue():
            cmd.append('--auto-lookup')
        if self.skip_transcode_ctrl.GetValue():
            cmd.append('--skip-transcode-errors')

        self._save_prefs()

        dlg = ConversionDialog(self, cmd)
        dlg.start()
        dlg.ShowModal()
        dlg.Destroy()

        # Re-scan to refresh statuses after conversion
        self.SetStatusText('Conversion finished. Re-scan to see updated state.')

    # ----------------------------------------------------------------------

    def on_close(self, evt):
        self._save_prefs()
        evt.Skip()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = wx.App(False)
    app.SetAppName(APP_NAME)
    frame = MainFrame()
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
