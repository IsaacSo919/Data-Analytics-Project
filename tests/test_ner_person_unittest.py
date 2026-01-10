import unittest
from unittest.mock import patch

from src.ner_person import PersonNER


class TestPersonNER(unittest.TestCase):
    def test_returns_empty_when_pipeline_unavailable(self):
        ner = PersonNER()

        # Force _lazy_load to set _pipe to False (simulate transformers/torch missing)
        with patch.object(PersonNER, "_lazy_load", lambda self: setattr(self, "_pipe", False)):
            out = ner.extract_person_candidates("mia khalifa")
            self.assertEqual(out, [])

    def test_extracts_only_person_and_dedupes(self):
        ner = PersonNER()

        fake_spans = [
            {"entity_group": "PER", "word": "Mia Khalifa"},
            {"entity_group": "ORG", "word": "Nike"},
            {"entity": "PER", "word": "Mia Khalifa"},  # duplicate
            {"entity_group": "PER", "word": "Alexis Texas"},
        ]

        def fake_pipe(_text):
            return fake_spans

        # Make _lazy_load set a fake callable pipeline
        with patch.object(PersonNER, "_lazy_load", lambda self: setattr(self, "_pipe", fake_pipe)):
            out = ner.extract_person_candidates("anything")
            self.assertEqual(set(out), {"Mia Khalifa", "Alexis Texas"})


if __name__ == "__main__":
    unittest.main()
