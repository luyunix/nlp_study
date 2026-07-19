import unittest

from fasttext_from_scratch.data import (
    format_labeled_line,
    normalize_text,
    parse_labeled_line,
)
from fasttext_from_scratch.huffman import huffman_codes, weighted_path_length
from fasttext_from_scratch.sampling import sample_negatives


class FastTextBasicsTest(unittest.TestCase):
    def test_normalize_and_format(self):
        self.assertEqual(normalize_text("Apple,  APPLE?"), "apple , apple ?")
        line = format_labeled_line(["fruit", "food"], "Apple!")
        self.assertEqual(line, "__label__fruit __label__food apple !")

    def test_parse_multilabel(self):
        labels, text = parse_labeled_line(
            "__label__sport __label__local Wuhan wins"
        )
        self.assertEqual(labels, ["sport", "local"])
        self.assertEqual(text, "Wuhan wins")

    def test_huffman_prefix_free_and_weighted_length(self):
        weights = {"A": 5, "B": 9, "C": 7, "D": 3}
        codes = huffman_codes(weights)
        for label, code in codes.items():
            for other, other_code in codes.items():
                if label != other:
                    self.assertFalse(other_code.startswith(code))
        self.assertEqual(weighted_path_length(weights, codes), 47)
        self.assertLessEqual(len(codes["B"]), len(codes["D"]))

    def test_negative_sampling_excludes_positive(self):
        negatives = sample_negatives(
            {"world": 100, "book": 60, "apple": 30, "river": 10},
            positive="world",
            k=2,
            seed=7,
        )
        self.assertEqual(len(negatives), 2)
        self.assertNotIn("world", negatives)
        self.assertEqual(len(set(negatives)), 2)


if __name__ == "__main__":
    unittest.main()
