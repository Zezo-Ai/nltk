# Natural Language Toolkit: Combinatory Categorial Grammar
#
# Copyright (C) 2001-2026 NLTK Project
# Author: Tanin Na Nakorn (@tanin)
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""
Helper functions for CCG semantics computation
"""

import copy

from nltk.sem.logic import *


def barendregt_normalize(expr, counter=None, prioritized_vars=None):
    """
    Canonicalizes variables while preserving pedagogical names (F, P, x, y, z).
    Ensures alpha-equivalent formulas produce identical strings.
    """
    if expr is None:
        return None

    # Initialization for the root call
    if counter is None:
        expr = expr.simplify()
        counter = [0]
        # Default pedagogical sequence if no specific hint is provided
        if prioritized_vars is None:
            prioritized_vars = ["x", "y", "z", "P", "Q"]

    if isinstance(expr, VariableBinderExpression):
        # Pick name from priority list, else fallback to numbered e
        if counter[0] < len(prioritized_vars):
            name = prioritized_vars[counter[0]]
        else:
            name = f"e{counter[0] - len(prioritized_vars) + 1}"

        new_var = Variable(name)
        counter[0] += 1

        # Execute alpha-conversion to prevent capture, then recurse
        safe_expr = expr.alpha_convert(new_var)
        return safe_expr.__class__(
            safe_expr.variable,
            barendregt_normalize(safe_expr.term, counter, prioritized_vars),
        )

    elif isinstance(expr, ApplicationExpression):
        return ApplicationExpression(
            barendregt_normalize(expr.function, counter, prioritized_vars),
            barendregt_normalize(expr.argument, counter, prioritized_vars),
        )

    elif isinstance(expr, BooleanExpression):
        return expr.__class__(
            barendregt_normalize(expr.first, counter, prioritized_vars),
            barendregt_normalize(expr.second, counter, prioritized_vars),
        )

    elif isinstance(expr, NegatedExpression):
        return NegatedExpression(
            barendregt_normalize(expr.term, counter, prioritized_vars)
        )

    elif isinstance(expr, EqualityExpression):
        return expr.__class__(
            barendregt_normalize(expr.first, counter, prioritized_vars),
            barendregt_normalize(expr.second, counter, prioritized_vars),
        )

    return expr


from nltk.sem.logic import (
    ApplicationExpression,
    LambdaExpression,
    Variable,
    VariableExpression,
    unique_variable,
)


def compute_function_semantics(function, argument):
    if function is None or argument is None:
        return None
    return barendregt_normalize(
        ApplicationExpression(function, argument), prioritized_vars=["x", "y", "z"]
    )


def compute_type_raised_semantics(semantics):
    if semantics is None:
        return None
    core = unique_variable(pattern=Variable("F"))
    # Strictly pure type-raising: \F.F(semantics)
    return barendregt_normalize(
        LambdaExpression(
            core,
            ApplicationExpression(VariableExpression(core), copy.deepcopy(semantics)),
        ),
        prioritized_vars=["F", "x", "y", "z", "P"],
    )


def compute_composition_semantics(function, argument):
    if function is None or argument is None:
        return None

    # Required for NLTK's structural doctests
    assert isinstance(
        argument, LambdaExpression
    ), f"`{argument}` must be a lambda expression"

    v = unique_variable(pattern=Variable("z"))
    return barendregt_normalize(
        LambdaExpression(
            v,
            ApplicationExpression(
                function, ApplicationExpression(argument, VariableExpression(v))
            ),
        ),
        prioritized_vars=["x", "y", "z"],
    )


def compute_substitution_semantics(function, argument):
    if function is None or argument is None:
        return None

    # Required for NLTK's structural doctests
    assert isinstance(function, LambdaExpression) and isinstance(
        function.term, LambdaExpression
    ), f"`{function}` must be a lambda expression with 2 arguments"

    x_var = unique_variable(pattern=Variable("x"))
    return barendregt_normalize(
        LambdaExpression(
            x_var,
            ApplicationExpression(
                ApplicationExpression(function, VariableExpression(x_var)),
                ApplicationExpression(argument, VariableExpression(x_var)),
            ),
        ),
        prioritized_vars=["x", "y", "z"],
    )
