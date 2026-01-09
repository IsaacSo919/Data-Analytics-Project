import unittest
from pathlib import Path
from src.data_ingestion import ingest_queries

class TestIngestion(unittest.TestCase):
    def test_ingestion_basic(self):
        csv_path = Path(__file__).parent / "sample_queries_raw.csv"  # ✅ 永遠指到 tests/ 底下
        df = ingest_queries(csv_path)

        self.assertIn("query", df.columns)
        self.assertIn("count", df.columns)
        #new checks: the normalisation test
        self.assertIn("query_norm", df.columns)
        self.assertEqual(df["query_norm"].isna().sum(), 0)
        self.assertEqual((df["query_norm"].str.strip() == "").sum(), 0)

        # 可選：抽幾筆確認 normalization 有在發生（至少應該全小寫）
        # draw a few samples to confirm normalization is happening (at least should be all lowercase)
        self.assertTrue((df["query_norm"] == df["query_norm"].str.lower()).all())
        '''query and count column checks'''
        self.assertEqual(df["query"].isna().sum(), 0)
        self.assertEqual((df["query"].str.strip() == "").sum(), 0)
        self.assertTrue(str(df["count"].dtype) in ("int64", "int32"))
        self.assertEqual((df["count"] < 0).sum(), 0)
        self.assertGreater(len(df), 0)

if __name__ == "__main__":
    unittest.main()
