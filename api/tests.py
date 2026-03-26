from django.test import TestCase
from .views import chunk_text  

class ChunkTextTests(TestCase):
    
    def test_empty_string_returns_empty_list(self):
        """If we pass nothing, we should get nothing back."""
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n  "), [])
        self.assertEqual(chunk_text(None), [])

    def test_basic_sentence_splitting(self):
        """It should pack sentences into chunks up to max_chars."""
        text = "Sentence one. Sentence two. Sentence three."
        
        # Setting max_chars artificially low (20) to force it to split the sentences up
        # Setting overlap to 0 to keep the math simple for this test
        chunks = chunk_text(text, max_chars=20, overlap=0)
        
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "Sentence one.")
        self.assertEqual(chunks[1], "Sentence two.")
        self.assertEqual(chunks[2], "Sentence three.")

    def test_overlap_logic(self):
        """It should carry over the specified overlap characters to the next chunk."""
        text = "This is the first part. And this is the second part."
        
        chunks = chunk_text(text, max_chars=30, overlap=5)
        
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "This is the first part.")
        self.assertEqual(chunks[1], "part. And this is the second part.")