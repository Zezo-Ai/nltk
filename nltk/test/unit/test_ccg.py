"""
Tests for the CCG (Combinatory Categorial Grammar) module.
"""

import unittest

from nltk.ccg.api import Direction, FunctionalCategory, PrimitiveCategory


class TestFunctionalCategorySubstitute(unittest.TestCase):
    """Test that FunctionalCategory.substitute() applies substitutions
    to all three components: result, argument, and direction."""

    def test_substitute_applies_to_direction(self):
        """Direction substitutions must propagate through substitute().

        Regression test: previously, substitute() computed the substituted
        direction but returned the original direction instead.
        """
        res = PrimitiveCategory("S")
        arg = PrimitiveCategory("NP")
        # A variable direction (restrictions="_") that should be substituted
        variable_dir = Direction("/", "_")

        category = FunctionalCategory(res, arg, variable_dir)

        # Substitute the variable direction with concrete restrictions
        new_restrictions = ["."]
        subs = [("_", new_restrictions)]
        result = category.substitute(subs)

        # The direction's restrictions should now be ["."], not "_"
        self.assertFalse(result.dir().is_variable())
        self.assertEqual(result.dir().restrs(), new_restrictions)

    def test_substitute_no_op_on_concrete_direction(self):
        """Substitution on a non-variable direction should be a no-op."""
        res = PrimitiveCategory("S")
        arg = PrimitiveCategory("NP")
        concrete_dir = Direction("\\", ["."])

        category = FunctionalCategory(res, arg, concrete_dir)

        result = category.substitute([("_", ",")])

        self.assertEqual(result.dir().restrs(), ["."])
