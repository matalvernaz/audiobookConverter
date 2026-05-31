#!/usr/bin/env python3
"""Unit tests for the pure (non-network, non-ffmpeg) logic in ab.py and series.py.

These cover the heuristic functions where the real bugs live — title cleaning,
duplicate detection, author normalisation, filename safety, metadata scoring and
series-number parsing. Run with either:

    python3 test_ab.py
    pytest test_ab.py
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import ab

_HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

# series.py is a local-only utility (gitignored, not shipped with the converter),
# so its tests only run where the file is present.
try:
    import series
    HAVE_SERIES = True
except ImportError:
    HAVE_SERIES = False


class TitlesMatchTests(unittest.TestCase):
    """titles_match underpins duplicate detection. The key regression: a short
    title must NOT be treated as a duplicate of a longer title that contains it."""

    def test_identical_titles_match(self):
        self.assertTrue(ab.titles_match("The Hobbit", "The Hobbit"))

    def test_article_difference_still_matches(self):
        # 'the' is a stopword, so word sets are identical.
        self.assertTrue(ab.titles_match("The Hobbit", "Hobbit"))

    def test_subset_title_is_not_a_duplicate(self):
        # Jaccard 1/2 = 0.5 < 0.85 — "Dune" is a different book from "Dune Messiah".
        self.assertFalse(ab.titles_match("Dune", "Dune Messiah"))

    def test_unrelated_titles_do_not_match(self):
        self.assertFalse(ab.titles_match("Dune", "Neuromancer"))

    def test_near_identical_string_matches_via_sequencematcher(self):
        self.assertTrue(ab.titles_match("Mistborn The Final Empire",
                                        "Mistborn: The Final Empire"))


class AlreadyExistsTests(unittest.TestCase):
    def test_short_title_against_superset_not_skipped(self):
        # Regression: converting "Dune" must not be skipped because a
        # "dune messiah" stem already exists in the output folder.
        self.assertFalse(ab.already_exists("dune", ["dune messiah", "neuromancer"]))

    def test_real_duplicate_is_detected(self):
        self.assertTrue(ab.already_exists("the final empire", ["the final empire"]))

    def test_too_short_title_is_never_a_dupe(self):
        self.assertFalse(ab.already_exists("it", ["it", "the stand"]))


class NormaliseAuthorTests(unittest.TestCase):
    def test_last_first_flip(self):
        self.assertEqual(ab.normalise_author("Pratchett, Terry"), "Terry Pratchett")

    def test_middle_initial_preserved(self):
        self.assertEqual(ab.normalise_author("Le Guin, Ursula K."), "Ursula K. Le Guin")

    def test_multiple_authors_keeps_primary(self):
        self.assertEqual(ab.normalise_author("Adams, Douglas; Jones, Jim"), "Douglas Adams")

    def test_natural_order_untouched(self):
        self.assertEqual(ab.normalise_author("Neil Gaiman"), "Neil Gaiman")


class CleanTitleTests(unittest.TestCase):
    def test_strips_disc_marker(self):
        self.assertEqual(ab.clean_title("The Hobbit Disc 1"), "The Hobbit")

    def test_strips_unabridged(self):
        self.assertEqual(ab.clean_title("The Hobbit (Unabridged)"), "The Hobbit")

    def test_strips_track_number_prefix(self):
        self.assertEqual(ab.clean_title("01 - The Fellowship"), "The Fellowship")

    def test_keeps_year_only_title(self):
        # "1984" must survive — it's the title, not a track number.
        self.assertEqual(ab.clean_title("1984"), "1984")


class StripAuthorFromTitleTests(unittest.TestCase):
    def test_strips_leading_author(self):
        self.assertEqual(ab.strip_author_from_title("Brandon Sanderson - Mistborn",
                                                    "Brandon Sanderson"), "Mistborn")

    def test_does_not_strip_midword(self):
        # Author name as a prefix of a longer word must not be chopped.
        self.assertEqual(ab.strip_author_from_title("Kingkiller Chronicle", "King"),
                         "Kingkiller Chronicle")


class FilenameSafetyTests(unittest.TestCase):
    def test_forbidden_chars_removed(self):
        name = ab.safe_filename("AC/DC", 'Back: In "Black"?')
        for ch in '\\/:*?"<>|':
            self.assertNotIn(ch, name)
        self.assertTrue(name.endswith(".m4b"))

    def test_reserved_basename_prefixed(self):
        self.assertEqual(ab._sanitize_filename_part("CON"), "_CON")

    def test_long_title_truncated(self):
        long_title = "Word " * 80  # 400 chars
        out = ab.safe_filename("Author", long_title)
        # author + " - " + title + ".m4b"; title portion must be bounded.
        title_part = out[len("Author - "):-len(".m4b")]
        self.assertLessEqual(len(title_part), ab.MAX_TITLE_LEN)

    def test_short_title_not_truncated(self):
        self.assertEqual(ab.safe_filename("Author", "Short"), "Author - Short.m4b")

    def test_trailing_dot_space_trimmed(self):
        # Windows silently drops trailing dots/spaces — sanitizer must too.
        self.assertEqual(ab._sanitize_filename_part("Vol. "), "Vol")


class TruncateAuthorTests(unittest.TestCase):
    def test_single_author_untouched(self):
        self.assertEqual(ab.truncate_author("Neil Gaiman"), "Neil Gaiman")

    def test_multiple_authors_collapsed(self):
        self.assertEqual(ab.truncate_author("Neil Gaiman, Terry Pratchett"),
                         "Neil Gaiman and Others")


class FfmetaEscapeTests(unittest.TestCase):
    def test_escapes_special_chars(self):
        self.assertEqual(ab._ffmeta_escape("a=b;c#d"), "a\\=b\\;c\\#d")

    def test_escapes_backslash(self):
        self.assertEqual(ab._ffmeta_escape("a\\b"), "a\\\\b")


class CountryLanguageTests(unittest.TestCase):
    def test_known_country_maps_to_iso639(self):
        self.assertEqual(ab.ITUNES_COUNTRY_TO_LANG.get("us"), "eng")
        self.assertEqual(ab.ITUNES_COUNTRY_TO_LANG.get("de"), "deu")

    def test_unknown_country_absent(self):
        # Unknown -> not present, so the provider emits an empty language
        # rather than a bogus country code in the m4b language tag.
        self.assertNotIn("zz", ab.ITUNES_COUNTRY_TO_LANG)


class ScoreResultTests(unittest.TestCase):
    def _r(self, title, author="", **kw):
        base = {"title": title, "author": author, "series": "", "desc": ""}
        base.update(kw)
        return base

    def test_exact_match_scores_high(self):
        score = ab._score_result(self._r("Mistborn", "Brandon Sanderson"),
                                 "Mistborn", "Brandon Sanderson")
        self.assertGreaterEqual(score, ab.SCORE_AUTO_SELECT_THRESHOLD)

    def test_wrong_author_penalised(self):
        right = ab._score_result(self._r("Mistborn", "Brandon Sanderson"),
                                 "Mistborn", "Brandon Sanderson")
        wrong = ab._score_result(self._r("Mistborn", "Stephen King"),
                                 "Mistborn", "Brandon Sanderson")
        self.assertGreater(right, wrong)

    def test_keyword_anchor_caps_unrelated_result(self):
        score = ab._score_result(self._r("Completely Different Book"),
                                 "Mistborn", "")
        self.assertLessEqual(score, ab.SCORE_KEYWORD_ANCHOR_CAP + 1e-9)


class VerifyHelperTests(unittest.TestCase):
    def test_fmt_hms(self):
        self.assertEqual(ab._fmt_hms(0), "0:00:00")
        self.assertEqual(ab._fmt_hms(3723), "1:02:03")
        self.assertEqual(ab._fmt_hms(59.6), "0:01:00")  # rounds

    def test_within_tolerance_absolute_floor(self):
        # 3s delta on a tiny expected value passes via the 5s absolute floor.
        self.assertTrue(ab._within_tolerance(13, 10, 5, 0.01))

    def test_within_tolerance_percentage(self):
        # 1% of 40000s = 400s; a 300s delta is within tolerance.
        self.assertTrue(ab._within_tolerance(40300, 40000, 5, 0.01))

    def test_outside_tolerance(self):
        self.assertFalse(ab._within_tolerance(41000, 40000, 5, 0.01))


@unittest.skipUnless(_HAVE_FFMPEG, "ffmpeg/ffprobe not on PATH")
class VerifyOutputE2ETests(unittest.TestCase):
    """Build a real chaptered .m4b and run the verifier against it."""

    def _build_m4b(self, tmp: Path, seconds: int = 4) -> ab.BookMetadata:
        src = tmp / "src.m4a"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", f"sine=frequency=440:duration={seconds}",
             "-c:a", "aac", "-b:a", "64k", str(src)],
            check=True, capture_output=True, timeout=60,
        )
        meta = ab.BookMetadata(title="Test Book", author="Test Author")
        half = (seconds * 1000) // 2
        specs = [(0, half, "Chapter 1"), (half, seconds * 1000, "Chapter 2")]
        meta_file = tmp / "meta.txt"
        meta_file.write_text(ab.render_book_ffmetadata(meta, specs), encoding="utf-8")
        self.out = tmp / "Test Author - Test Book.m4b"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-i", str(meta_file),
             "-map", "0:a", "-map_metadata", "1", "-map_chapters", "1",
             "-c:a", "copy", "-f", "ipod", str(self.out)],
            check=True, capture_output=True, timeout=60,
        )
        return meta

    def test_clean_build_passes(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            meta = self._build_m4b(tmp, seconds=4)
            result = ab.verify_output(self.out, meta, expected_duration_sec=4.0,
                                      expected_chapters=2, write_report=False)
            self.assertEqual(result["fail"], 0)
            messages = " ".join(m for _, m in result["checks"])
            self.assertIn("2 chapters, in order", messages)
            self.assertIn("Tags present", messages)

    def test_short_output_flags_duration(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            meta = self._build_m4b(tmp, seconds=4)
            # Claim the source was 10 minutes — output is far shorter, must warn.
            result = ab.verify_output(self.out, meta, expected_duration_sec=600.0,
                                      expected_chapters=2, write_report=False)
            self.assertGreaterEqual(result["warn"], 1)
            self.assertTrue(any(lvl == "WARN" and "shorter" in msg
                                for lvl, msg in result["checks"]))


@unittest.skipUnless(HAVE_SERIES, "series.py not present (local-only utility)")
class ExtractSeriesNumberTests(unittest.TestCase):
    def test_book_keyword(self):
        self.assertEqual(series.extract_series_number(["Mistborn Book 2"]), "02")

    def test_hash_marker(self):
        self.assertEqual(series.extract_series_number(["Stormlight #3"]), "03")

    def test_bare_trailing_number_not_a_series_number(self):
        # Regression: these must NOT be parsed as series positions.
        self.assertIsNone(series.extract_series_number(["Apollo 13"]))
        self.assertIsNone(series.extract_series_number(["Catch 22"]))
        self.assertIsNone(series.extract_series_number(["1984"]))

    def test_first_anchored_match_wins(self):
        self.assertEqual(series.extract_series_number(["", "The Saga, Volume 4"]), "04")


@unittest.skipUnless(HAVE_SERIES, "series.py not present (local-only utility)")
class UniquePathTests(unittest.TestCase):
    def test_nonexistent_returned_as_is(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.m4b"
            self.assertEqual(series.unique_path(p), p)

    def test_collision_gets_suffix(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.m4b"
            p.write_bytes(b"x")
            out = series.unique_path(p)
            self.assertEqual(out.name, "a (2).m4b")


if __name__ == "__main__":
    unittest.main(verbosity=2)
