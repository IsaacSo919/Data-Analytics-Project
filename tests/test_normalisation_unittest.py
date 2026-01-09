import unittest
from pathlib import Path
from src.normalisation import normalize_text

class TestNormalization(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(normalize_text("  Mia   Malkova  "), "mia malkova")
        self.assertEqual(normalize_text("Kiara-Knight"), "kiara knight")
        self.assertEqual(normalize_text("miami!!!"), "miami")
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")

if __name__ == "__main__":
    unittest.main()
