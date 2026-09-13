import tempfile
import unittest
from pathlib import Path

from harness.memory import MemoryStore, chunk_text, hash_embed


class TestHashEmbed(unittest.TestCase):
    def test_deterministic_across_calls(self):
        self.assertEqual(hash_embed("hello world"), hash_embed("hello world"))

    def test_unit_norm(self):
        import math
        vec = hash_embed("some reasonably long piece of text to embed", dim=64)
        norm = math.sqrt(sum(v * v for v in vec))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_empty_text_is_zero_vector(self):
        vec = hash_embed("", dim=32)
        self.assertEqual(vec, [0.0] * 32)


class TestChunkText(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("short", chunk_chars=100), ["short"])

    def test_empty_text_has_no_chunks(self):
        self.assertEqual(chunk_text("   "), [])

    def test_long_text_is_split_with_overlap(self):
        text = "x" * 1000
        chunks = chunk_text(text, chunk_chars=300, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        # reconstructing without overlap should still cover the whole text
        self.assertGreaterEqual(sum(len(c) for c in chunks), len(text))


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "mem.sqlite3"
        self.store = MemoryStore(self.db_path, dim=64)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_add_and_count(self):
        self.store.add("The build script lives at scripts/build.py", source="note")
        self.assertEqual(self.store.count(), 1)

    def test_rejects_empty_content(self):
        with self.assertRaises(ValueError):
            self.store.add("   ")

    def test_search_finds_relevant_entry_over_irrelevant_one(self):
        self.store.add("The database migration tool is Alembic, configured in alembic.ini.", source="note")
        self.store.add("Bananas are a good source of potassium.", source="note")
        results = self.store.search("which tool handles database migrations?", top_k=5)
        self.assertTrue(results)
        self.assertIn("Alembic", results[0]["content"])

    def test_search_on_empty_store_returns_empty(self):
        self.assertEqual(self.store.search("anything"), [])

    def test_add_chunks_indexes_multiple_rows_for_long_text(self):
        text = "paragraph. " * 500
        n = self.store.add_chunks(text, source="docs/big.md", chunk_chars=200)
        self.assertGreater(n, 1)
        self.assertEqual(self.store.count(), n)

    def test_persists_across_reopen(self):
        self.store.add("persisted fact", source="note")
        self.store.close()
        reopened = MemoryStore(self.db_path, dim=64)
        try:
            self.assertEqual(reopened.count(), 1)
            results = reopened.search("persisted fact")
            self.assertTrue(results)
        finally:
            reopened.close()


if __name__ == "__main__":
    unittest.main()
