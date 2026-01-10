import unittest

from run_ner_only import extract_persons


class TestRunNerOnly(unittest.TestCase):
    def test_extract_persons_dedupes_and_orders(self):
        def fake_ner_pipe(_text):
            return [
                {"entity_group": "PER", "word": "Mia Khalifa"},
                {"entity_group": "ORG", "word": "Amazon"},
                {"entity_group": "PER", "word": "Mia Khalifa"},  # duplicate
                {"entity": "PER", "word": "Alexis Texas"},       # alternate key
            ]

        out = extract_persons("whatever", fake_ner_pipe, titlecase=False)
        self.assertEqual(out, "Mia Khalifa,Alexis Texas")

    def test_titlecase_passed_into_pipeline(self):
        seen = {"text": None}

        def fake_ner_pipe(text):
            seen["text"] = text
            return []

        _ = extract_persons("mia khalifa videos", fake_ner_pipe, titlecase=True)
        self.assertEqual(seen["text"], "Mia Khalifa Videos")


if __name__ == "__main__":
    unittest.main()
