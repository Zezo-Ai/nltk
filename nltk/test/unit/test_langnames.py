import unittest

from nltk.langnames import lang2q, q2name, q2tag, tag2q


class TestLangNames(unittest.TestCase):
    def test_tag2q_known(self):
        self.assertEqual(tag2q("nds-u-sd-demv"), "Q4289225")

    def test_tag2q_unknown(self):
        self.assertIsNone(tag2q("zzzz-unknown-tag"))

    def test_q2tag_known(self):
        self.assertEqual(q2tag("Q4289225"), "nds-u-sd-demv")

    def test_q2tag_unknown(self):
        self.assertIsNone(q2tag("Q0000000"))

    def test_q2name_known(self):
        self.assertEqual(q2name("Q4289225"), "Low German: Mecklenburg-Vorpommern")
        self.assertEqual(q2name("Q4289225", "short"), "Low German")

    def test_q2name_unknown(self):
        self.assertIsNone(q2name("Q0000000"))

    def test_lang2q_known(self):
        self.assertEqual(lang2q("Low German"), "Q25433")

    def test_lang2q_unknown(self):
        self.assertIsNone(lang2q("NonexistentLanguage"))


if __name__ == "__main__":
    unittest.main()
