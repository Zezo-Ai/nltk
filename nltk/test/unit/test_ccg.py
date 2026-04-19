"""
Tests for the CCG (Combinatory Categorial Grammar) module.
"""

import unittest

from nltk.ccg import lexicon
from nltk.ccg.api import FunctionalCategory


class TestLexiconVariableDirection(unittest.TestCase):
    """Test that `lexicon.fromstring` accepts variable direction markers (`\\_`)
    in slash modifiers, as used in polymorphic categories like `(S\\_NP)/(S\\_NP)`.
    """

    def test_fromstring_parses_variable_direction(self):
        """Regression test: previously, APP_RE rejected `_` as a modifier,
        causing the lexer to raise AttributeError on any variable direction.
        """
        lex = lexicon.fromstring(
            r"""
            :- S, NP
            quickly => (S\_NP)/(S\_NP)
            """
        )

        categories = lex.categories("quickly")
        self.assertEqual(len(categories), 1)

        cat = categories[0].categ()
        self.assertIsInstance(cat, FunctionalCategory)
        # Outer structure: (S\_NP) / (S\_NP)
        self.assertTrue(cat.dir().is_forward())
        self.assertIsInstance(cat.res(), FunctionalCategory)
        self.assertIsInstance(cat.arg(), FunctionalCategory)
        # Inner direction carries the variable marker
        self.assertIn("_", str(cat.res().dir()))
        self.assertTrue(cat.res().dir().is_backward())
