import unittest
from trigram_model import TrigramModel, get_ngrams

class TestTrigramModel(unittest.TestCase):

    def setUp(self):
        # Initializing a sample corpus for tests
        self.model = TrigramModel('/Users/yehudadicker/Downloads/NLP/GitHub/nlp/HW1/test_corpus.txt')
        print(self.model.unigramcounts)
        print(self.model.bigramcounts)

    # Test for get_ngrams
    def test_part_1(self):
        self.assertEqual(
            get_ngrams(["a", "b"], 2),
            [('START', 'a'), ('a', 'b'), ('b', 'END')]
        )

    # Test for raw_unigram_probability
    def test_part_2(self):
        self.assertAlmostEqual(
            self.model.raw_unigram_probability(("a",)),
            0.25  # There are 5 tokens: START, a, b, b, END. 'a' appears 2 times.
        )

    # Test for raw_bigram_probability
    def test_part_3(self):
        self.assertAlmostEqual(
            self.model.raw_bigram_probability(("a", "b")),
            0.5  # 'a b' appears once and 'a' appears twice.
        )

    # Test for raw_trigram_probability
    def test_part_4(self):
        self.assertAlmostEqual(
            self.model.raw_trigram_probability(("START", "a", "b")),
            1.0  # 'START a b' appears once and 'START a' appears once.
        )
''' 
    # For the smoothed_trigram_probability and sentence_logprob, the expected values are not straightforward
    # to determine without a sample corpus and would vary depending on lambda values.
    # However, I'm giving an example structure below:

    # Test for smoothed_trigram_probability
    def test_part_5(self):
        self.assertAlmostEqual(
            self.model.smoothed_trigram_probability(("START", "a", "b")),
            YOUR_EXPECTED_VALUE_HERE  # You need to compute and replace this
        )

    # Test for sentence_logprob
    def test_part_6(self):
        self.assertAlmostEqual(
            self.model.sentence_logprob(["a", "b"]),
            YOUR_EXPECTED_VALUE_HERE  # You need to compute and replace this
        )
'''

if __name__ == '__main__':
    unittest.main()
