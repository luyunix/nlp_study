import math
import unittest

from text_preprocessing_from_scratch.core import (
    build_vocabulary,
    cbow_examples,
    cosine_similarity,
    label_distribution,
    ngrams,
    normalize_length,
    one_hot,
    one_hot_corpus,
    skipgram_examples,
    word_frequencies,
)


class TextPreprocessingTests(unittest.TestCase):
    def test_vocabulary_preserves_first_occurrence(self):
        self.assertEqual(build_vocabulary(["今", "天", "今"]), {"今": 0, "天": 1})

    def test_one_hot(self):
        vocabulary = {"今天": 0, "天气": 1, "很好": 2}
        self.assertEqual(one_hot("天气", vocabulary), [0, 1, 0])

    def test_one_hot_corpus_shape(self):
        vocabulary, matrix = one_hot_corpus(["a", "b", "a"])
        self.assertEqual(vocabulary, {"a": 0, "b": 1})
        self.assertEqual(matrix, [[1, 0], [0, 1], [1, 0]])

    def test_cbow_context_predicts_center(self):
        examples = cbow_examples(["我", "爱", "自然", "语言", "处理"], window_size=1)
        self.assertEqual(examples[1], (("爱", "语言"), "自然"))

    def test_skipgram_center_predicts_context(self):
        pairs = skipgram_examples(["我", "爱", "NLP"], window_size=1)
        self.assertIn(("爱", "我"), pairs)
        self.assertIn(("爱", "NLP"), pairs)

    def test_ngrams_can_include_lower_orders(self):
        result = ngrams(["a", "b", "c"], 2, include_lower_orders=True)
        self.assertEqual(
            result,
            [("a",), ("b",), ("c",), ("a", "b"), ("b", "c")],
        )

    def test_cosine_similarity(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 1]), 1 / math.sqrt(2))
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0)

    def test_normalize_length(self):
        self.assertEqual(normalize_length([1, 2], 4, 0), [1, 2, 0, 0])
        self.assertEqual(
            normalize_length([1, 2, 3, 4], 2, 0, truncation="left"),
            [3, 4],
        )

    def test_distribution_and_frequencies(self):
        self.assertEqual(label_distribution(["正", "负", "正"]), {"正": 2, "负": 1})
        self.assertEqual(word_frequencies(["好", "天气", "好"]), [("好", 2), ("天气", 1)])
        self.assertEqual(
            word_frequencies(iter(["好", "天气", "好"])),
            [("好", 2), ("天气", 1)],
        )

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ngrams(["a"], 0)
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0], [1, 0])
        with self.assertRaises(ValueError):
            normalize_length([1], 2, 0, padding="middle")


if __name__ == "__main__":
    unittest.main()
