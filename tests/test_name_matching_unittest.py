# tests/test_name_matching_unittest.py
import unittest
from pathlib import Path

from src.normalisation import normalize_text
from src.name_matching import (
    load_names_csv,
    build_automaton,
    extract_names_from_query_norm,
)

class TestNameMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names_path = Path(__file__).parent / "sample_names.csv"
        names = load_names_csv(names_path)
        cls.automaton = build_automaton(names)

    def test_word_boundary(self):
        # "mia" must not match inside "miami"
        q = normalize_text("miami beach")
        self.assertEqual(extract_names_from_query_norm(q, self.automaton), "")

    def test_full_name_preferred(self):
        q = normalize_text("i love kiara knight videos")
        self.assertEqual(extract_names_from_query_norm(q, self.automaton), "kiara knight")

    def test_multiple_names(self):
        q = normalize_text("mia malkova alexis texas")
        self.assertEqual(extract_names_from_query_norm(q, self.automaton), "mia malkova,alexis texas")

if __name__ == "__main__":
    unittest.main()
