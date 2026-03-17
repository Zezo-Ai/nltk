import unittest
import warnings

from nltk.langnames import lang2q, langcode, langname, q2name, q2tag, tag2q


class TestTag2Q(unittest.TestCase):
    def test_known_tag(self):
        self.assertEqual(tag2q("nds-u-sd-demv"), "Q4289225")

    def test_unknown_tag(self):
        self.assertIsNone(tag2q("zzzz-unknown-tag"))

    def test_empty_string(self):
        self.assertIsNone(tag2q(""))


class TestQ2Tag(unittest.TestCase):
    def test_known_qcode(self):
        self.assertEqual(q2tag("Q4289225"), "nds-u-sd-demv")

    def test_unknown_qcode(self):
        self.assertIsNone(q2tag("Q0000000"))

    def test_empty_string(self):
        self.assertIsNone(q2tag(""))


class TestQ2Name(unittest.TestCase):
    def test_known_full(self):
        self.assertEqual(q2name("Q4289225"), "Low German: Mecklenburg-Vorpommern")

    def test_known_short(self):
        self.assertEqual(q2name("Q4289225", "short"), "Low German")

    def test_unknown_qcode(self):
        self.assertIsNone(q2name("Q0000000"))

    def test_empty_string(self):
        self.assertIsNone(q2name(""))


class TestLang2Q(unittest.TestCase):
    def test_known_language(self):
        self.assertEqual(lang2q("Low German"), "Q25433")

    def test_unknown_language(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIsNone(lang2q("NonexistentLanguage"))

    def test_empty_string(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIsNone(lang2q(""))


class TestLangName(unittest.TestCase):
    def test_known_tag(self):
        self.assertEqual(langname("ca-Latn-ES-valencia"), "Catalan: Latin: Spain: Valencian")

    def test_known_tag_short(self):
        self.assertEqual(langname("ca-Latn-ES-valencia", typ="short"), "Catalan")

    def test_retired_code(self):
        self.assertEqual(langname("fri"), "Western Frisian")

    def test_unknown_code(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = langname("zzz")
            self.assertIsNone(result)


class TestLangCode(unittest.TestCase):
    def test_known_name(self):
        self.assertEqual(langcode("Modern Greek (1453-)"), "el")

    def test_known_name_3letter(self):
        self.assertEqual(langcode("Modern Greek (1453-)", typ=3), "ell")

    def test_retired_name(self):
        self.assertEqual(langcode("Western Frisian"), "fy")

    def test_unknown_name(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIsNone(langcode("NotARealLanguage"))


if __name__ == "__main__":
    unittest.main()
