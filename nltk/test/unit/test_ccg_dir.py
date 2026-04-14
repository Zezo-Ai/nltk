def test_parse_variable_direction():
    """
    Ensures the string parser can successfully instantiate a Direction object
    that correctly identifies as a variable. This verifies that the application
    regex allows underscores and that the parsed string is properly evaluated.
    """
    from nltk.ccg import lexicon

    lex_str = r"""
    :- S, NP
    quickly => (S\_NP)/(S\_NP)
    """
    lex = lexicon.fromstring(lex_str)

    # Extract the Token, then the syntactic category
    quickly_token = lex.categories("quickly")[0]
    quickly_cat = quickly_token.categ()

    # Extract the result category (S\_NP) and check its direction (\_)
    var_direction = quickly_cat.res().dir()

    assert (
        var_direction.is_variable()
    ), "Lexer failed to properly parse the direction as a variable."
    assert (
        var_direction.restrs() == "_"
    ), f"Expected restriction '_', got {var_direction.restrs()!r}"


def test_variable_direction_can_unify():
    """
    Ensures that when a variable direction unifies with a concrete direction,
    the substitution mapping is correctly extracted and returned to the caller.
    """
    from nltk.ccg import lexicon

    lex_str = r"""
    :- S, NP
    walked => S\NP
    quickly => (S\_NP)/(S\_NP)
    """
    lex = lexicon.fromstring(lex_str)

    # Extract the syntactic categories from the Tokens
    walked_cat = lex.categories("walked")[0].categ()
    quickly_cat = lex.categories("quickly")[0].categ()

    # Attempt to unify (S\_NP) with S\NP
    subs = quickly_cat.res().can_unify(walked_cat)

    assert subs is not None, "Unification failed entirely."

    # Extract just the variables from the substitution list (the left side of the tuples)
    substituted_vars = [var for var, val in subs]

    assert "_" in substituted_vars, "can_unify dropped the direction variable mapping!"
