from __future__ import annotations

import unittest

from services.api.app.local_fast_path import normalize
from services.api.app.persona import SIMPLE_GREETING_PATTERN


class SimpleGreetingDeterminismTests(unittest.TestCase):
    def test_reviewed_greetings_still_match(self) -> None:
        valid = (
            "hi",
            " hello ",
            "HEY!!!",
            "hey ao",
            "hey    bunnelby?!",
            "hello\tao",
            "hi bunnelby",
            "good morning",
            "good   afternoon",
            "good\tevening",
            "good night",
            "hi   .?!",
        )
        for value in valid:
            with self.subTest(value=value):
                self.assertIsNotNone(SIMPLE_GREETING_PATTERN.match(value))

    def test_non_greetings_still_miss(self) -> None:
        invalid = (
            "",
            "hi there",
            "goodnight",
            "hey ao there",
            "hello bunnelby please",
            "hi ! ?",
            "hi . !",
        )
        for value in invalid:
            with self.subTest(value=value):
                self.assertIsNone(SIMPLE_GREETING_PATTERN.match(value))

    def test_large_whitespace_is_handled_deterministically(self) -> None:
        value = "hi" + (" " * 100_000)
        self.assertIsNotNone(SIMPLE_GREETING_PATTERN.match(value))


class LocalFastPathWrapperNormalizationTests(unittest.TestCase):
    def test_reviewed_wrappers_preserve_existing_normalization(self) -> None:
        cases = {
            "please open notepad": "open notepad",
            "open notepad please": "open notepad",
            "bunnelby open notepad": "open notepad",
            "hey bunnelby open notepad": "open notepad",
            "ok bunnelby open notepad": "open notepad",
            "bunnelby, open notepad": "open notepad",
            "bunnelby : open notepad": "open notepad",
            "hey bunnelby , open notepad": "open notepad",
            "bunnelby please open notepad": "open notepad",
            # Removal is intentionally one-pass and order-sensitive.
            "please bunnelby open notepad": "bunnelby open notepad",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize(raw), expected)

    def test_malformed_wrappers_are_not_broadened(self) -> None:
        cases = {
            "bunnelbyopen notepad": "bunnelbyopen notepad",
            "hey open notepad": "hey open notepad",
            "ok open notepad": "ok open notepad",
            "bunnelby,open notepad": "bunnelby,open notepad",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize(raw), expected)

    def test_large_whitespace_collapses_before_wrapper_processing(self) -> None:
        raw = "hey" + (" " * 100_000) + "bunnelby" + (" " * 100_000) + "open notepad please"
        self.assertEqual(normalize(raw), "open notepad")


if __name__ == "__main__":
    unittest.main()
