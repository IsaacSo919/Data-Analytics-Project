import unittest

from src.name_matching import build_name_index, verify_pornstar


class TestPornstarVerification(unittest.TestCase):
    def setUp(self):
        names_norm = ["mia khalifa", "alexis texas", "mia malkova"]
        self.names_set, self.canon_map = build_name_index(names_norm)

    def test_verify_matches_after_normalisation(self):
        # extra spaces + weird case should still match via normalize_text inside verify_pornstar
        out = verify_pornstar("  Mia   Khalifa  ", self.names_set, self.canon_map)
        self.assertEqual(out, "mia khalifa")

    def test_verify_returns_none_if_not_in_list(self):
        out = verify_pornstar("Taylor Swift", self.names_set, self.canon_map)
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
