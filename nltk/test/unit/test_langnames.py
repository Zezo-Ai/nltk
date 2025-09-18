import unittest

from nltk.langnames import q2tag, tag2q


class TestLangNames(unittest.TestCase):
    def test_tag2q_known(self):
        # Example from docs
        self.assertEqual(tag2q("nds-u-sd-demv"), "Q4289225")

    def test_tag2q_unknown(self):
        # Should return None for unknown tag
        self.assertIsNone(tag2q("zzzz-unknown-tag"))

    def test_q2tag_known(self):
        # Example from docs
        self.assertEqual(q2tag("Q4289225"), "nds-u-sd-demv")

    def test_q2tag_unknown(self):
        # Should return None for unknown Q-code
        self.assertIsNone(q2tag("Q0000000"))


if __name__ == "__main__":
    unittest.main()
