"""
Regression tests for ``nltk.tag.tnt.TnT``.
"""

import math

import pytest

from nltk.tag.tnt import (
    _BOS,
    _EOS,
    _LOG_FLOOR_2,
    TnT,
    _safe_inverse,
    _safe_log2,
    basic_sent_chop,
)

# Synthetic corpus covering the tag classes used below.
_TRAIN = [
    [
        ("the", "DT"),
        ("cat", "NN"),
        ("ran", "VBD"),
        (".", "."),
    ],
    [
        ("the", "DT"),
        ("dog", "NN"),
        ("jumped", "VBD"),
        (".", "."),
    ],
    [
        ("the", "DT"),
        ("happy", "JJ"),
        ("cat", "NN"),
        ("slept", "VBD"),
        (".", "."),
    ],
    [
        ("the", "DT"),
        ("running", "VBG"),
        ("dog", "NN"),
        ("barked", "VBD"),
        (".", "."),
    ],
    [
        ("Dianne", "NNP"),
        ("loves", "VBZ"),
        ("to", "TO"),
        ("hug", "VB"),
        (".", "."),
    ],
    [
        ("Pappy", "NNP"),
        ("is", "VBZ"),
        ("very", "RB"),
        ("loyal", "JJ"),
        (".", "."),
    ],
    [
        ("beagles", "NNS"),
        ("are", "VBP"),
        ("happy", "JJ"),
        ("to", "TO"),
        ("rest", "VB"),
        (".", "."),
    ],
]


class _CountingUnk:
    """Counts how many times ``train()`` is called."""

    def __init__(self):
        self.train_calls = 0

    def train(self, _data):
        self.train_calls += 1

    def tag(self, toks):
        return [(w, "NN") for w in toks]


@pytest.fixture
def tagger():
    """A ``TnT`` instance trained on the shared synthetic corpus."""
    t = TnT()
    t.train(_TRAIN)
    return t


def _assert_tag_output(words, out):
    """Check basic decode shape invariants."""
    assert len(out) == len(words)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in out)
    assert [w for w, _ in out] == words
    assert all(isinstance(tag, str) for _, tag in out)


# ---------------------------------------------------------------------
# Decode shape and beam pruning
# ---------------------------------------------------------------------


@pytest.mark.parametrize("sent", _TRAIN, ids=lambda s: " ".join(w for w, _ in s))
def test_known_words_decode_to_their_only_seen_tag(tagger, sent):
    """Sanity check on the unambiguous decode path. Each word in
    ``_TRAIN`` has a single observed tag, so any deviation points at a
    broken Viterbi pass rather than a modeling choice."""
    words = [w for w, _ in sent]
    assert tagger.tag(words) == sent


def test_threshold_pruning_does_not_empty_beam_under_ambiguity():
    """With local tag ambiguity and a tight beam, decode still returns
    a full tag sequence."""
    train = [
        [("the", "DT"), ("dogs", "NNS"), (".", ".")],
        [("the", "DT"), ("dogs", "VBZ"), (".", ".")],
        [("a", "DT"), ("cat", "NN"), (".", ".")],
    ]
    t = TnT(N=2)
    t.train(train)
    words = ["the", "dogs", "."]
    out = t.tag(words)
    _assert_tag_output(words, out)


# ---------------------------------------------------------------------
# EOS handling
# ---------------------------------------------------------------------


def test_eos_recorded_in_all_three_ngrams(tagger):
    """EOS is folded into the same n-gram model as every other tag, so
    it appears in unigram, bigram, and trigram counts. The unigram
    count must equal the number of non-empty training sentences."""
    expected_unigram = sum(1 for sent in _TRAIN if sent)
    assert tagger._tag_unigrams[_EOS] == expected_unigram
    assert any(_EOS in dist for dist in tagger._tag_bigrams.values())
    assert any(_EOS in dist for dist in tagger._tag_trigrams.values())


def _state(tagger, word, tag):
    return (
        tag,
        tagger._use_capitalization and bool(word) and word[0].isupper(),
    )


def test_eos_follows_sentence_final_punctuation(tagger):
    """EOS is attributed to the actual final tag history from the
    training data, not to a hardcoded predecessor context."""
    dot_state = _state(tagger, ".", ".")

    expected_bigram_count = 0
    expected_trigram_counts = {}

    for sent in _TRAIN:
        if not sent or sent[-1][0] != ".":
            continue

        expected_bigram_count += 1

        prev_state = _state(tagger, sent[-2][0], sent[-2][1])
        history = (prev_state, dot_state)
        expected_trigram_counts[history] = expected_trigram_counts.get(history, 0) + 1

    assert tagger._tag_bigrams[dot_state][_EOS] == expected_bigram_count

    for history, count in expected_trigram_counts.items():
        assert tagger._tag_trigrams[history][_EOS] == count


def test_empty_sentences_do_not_record_eos_at_bos():
    """EOS is not recorded with BOS as its bigram or trigram history,
    which would correspond to an empty sentence."""
    t = TnT()
    t.train([[], [("the", "DT"), ("cat", "NN"), (".", ".")], []])
    assert _EOS not in t._tag_bigrams[_BOS]
    assert _EOS not in t._tag_trigrams[(_BOS, _BOS)]


# ---------------------------------------------------------------------
# Training semantics
# ---------------------------------------------------------------------


def test_repeated_train_rebuilds_state():
    """``train()`` rebuilds internal state on each call rather than
    accumulating, so retraining on the same data is idempotent."""
    t = TnT()
    t.train(_TRAIN)
    snapshot = (
        t._tag_unigrams.N(),
        sum(d.N() for d in t._tag_bigrams.values()),
        sum(d.N() for d in t._tag_trigrams.values()),
        len(t._word_tag_freqs.conditions()),
        t._num_tag_tokens,
    )
    t.train(_TRAIN)
    after = (
        t._tag_unigrams.N(),
        sum(d.N() for d in t._tag_bigrams.values()),
        sum(d.N() for d in t._tag_trigrams.values()),
        len(t._word_tag_freqs.conditions()),
        t._num_tag_tokens,
    )
    assert snapshot == after


def test_external_unk_tagger_trains_once():
    """The optional ``unk`` tagger is trained only on the first
    ``train()`` call. Subsequent calls leave it alone so caller-managed
    state is preserved across passes."""
    cu = _CountingUnk()
    t = TnT(unk=cu)
    t.train(_TRAIN)
    assert cu.train_calls == 1
    t.train(_TRAIN)
    assert cu.train_calls == 1


def test_trained_constructor_flag_skips_external_unk_training():
    """``Trained=True`` tells the constructor that the optional ``unk``
    tagger is already trained, so subsequent ``train()`` calls do not
    retrain it."""
    cu = _CountingUnk()
    t = TnT(unk=cu, Trained=True)
    t.train(_TRAIN)
    assert cu.train_calls == 0


def test_external_unk_bypasses_decode_cache():
    """Stateful ``unk`` taggers are invoked on every call rather than
    memoized, otherwise caching collapses their varying output into a
    single result."""

    class AlternatingUnk:
        def __init__(self):
            self.flip = False

        def train(self, _data):
            pass

        def tag(self, toks):
            self.flip = not self.flip
            return [(w, "NN" if self.flip else "JJ") for w in toks]

    t = TnT(unk=AlternatingUnk())
    t.train(_TRAIN)
    seen = [t.tag(["xyzzy"])[0][1] for _ in range(3)]
    assert seen == ["NN", "JJ", "NN"]


# ---------------------------------------------------------------------
# Unknown-word handling
# ---------------------------------------------------------------------


def test_unknown_word_uses_suffix_model_not_literal_unk(tagger):
    """The suffix model is the actual unknown-word path. The literal
    ``"Unk"`` string is only a last-resort fallback for taggers with
    no priors, so a trained tagger should never reach it."""
    [(word, tag)] = tagger.tag(["xyzzy"])
    assert word == "xyzzy"
    assert tag in tagger._tag_prior_probs
    assert tag != "Unk"


def test_unknown_word_falls_back_to_unk_when_priors_empty():
    """Without training, the suffix model has no priors and tagging
    falls back to the literal ``"Unk"`` string. This keeps the API
    safe on unconfigured taggers."""
    t = TnT()
    assert t.tag(["xyzzy"]) == [("xyzzy", "Unk")]


def test_external_unk_returning_unseen_tag_survives_decode():
    """An external ``unk`` tagger may return a tag that was never seen
    during training. The candidate state has no unigram entry, so the
    transition cache falls through to the model floor; the path stays
    alive and the tag appears in the decoder's output."""

    class ConstantUnk:
        def train(self, _data):
            pass

        def tag(self, toks):
            return [(w, "NONESUCH") for w in toks]

    t = TnT(unk=ConstantUnk())
    t.train(_TRAIN)
    [(word, tag)] = t.tag(["xyzzy"])
    assert word == "xyzzy"
    assert tag == "NONESUCH"


# ---------------------------------------------------------------------
# Lambdas
# ---------------------------------------------------------------------


def test_empty_training_zeroes_lambdas():
    """``_compute_lambda()`` zeroes the weights on empty input rather
    than dividing by zero."""
    t = TnT()
    t.train([])
    assert (t._lambda1, t._lambda2, t._lambda3) == (0.0, 0.0, 0.0)


def test_lambdas_are_non_negative_and_sum_to_one(tagger):
    """On normal input, the deleted-interpolation weights form a valid
    probability distribution."""
    assert tagger._lambda1 >= 0
    assert tagger._lambda2 >= 0
    assert tagger._lambda3 >= 0
    assert math.isclose(
        tagger._lambda1 + tagger._lambda2 + tagger._lambda3,
        1.0,
        abs_tol=1e-12,
    )


# ---------------------------------------------------------------------
# N validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [0, -1, 0.5, math.nan, math.inf, -math.inf, True, False, "1000", None, [1000]],
)
def test_invalid_n_raises_value_error(bad):
    """Invalid ``N`` values raise ``ValueError`` at construction time."""
    with pytest.raises(ValueError):
        TnT(N=bad)


@pytest.mark.parametrize("ok", [1, 2, 1000, 1_000_000])
def test_valid_n_is_accepted(ok):
    """Integer ``N >= 1`` is accepted."""
    TnT(N=ok)


# ---------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("words", "segment", "expected_len"),
    [
        ([], False, 0),
        ([], True, 0),
        (["the", "cat", ".", "the", "dog", "."], True, 6),
        (["the", "cat", ".", "the", "dog"], True, 5),
    ],
)
def test_segmentation_shapes(tagger, words, segment, expected_len):
    """Segmented tagging preserves output shape across empty, split,
    and trailing-fragment inputs."""
    out = tagger.tag(words, segment=segment)
    assert len(out) == expected_len
    if expected_len > 0:
        _assert_tag_output(words, out)


def test_segment_true_matches_segment_false_for_single_segment(tagger):
    """Without mid-sequence punctuation, the segmented and unsegmented
    decode paths produce the same tag sequence."""
    words = [w for w, _ in _TRAIN[0]]
    assert tagger.tag(words) == tagger.tag(words, segment=True)


def test_basic_sent_chop_emits_trailing_fragment():
    """Without this, sentences that lack a final marker would be
    silently dropped during segmentation, costing real evaluation
    data."""
    assert basic_sent_chop(["a", "b", ".", "c", "d"]) == [["a", "b", "."], ["c", "d"]]


def test_basic_sent_chop_with_tagged_input():
    """``basic_sent_chop(raw=False)`` segments a list of ``(word, tag)`` tuples."""
    tagged = [("a", "DT"), ("b", "NN"), (".", "."), ("c", "DT"), ("d", "NN")]
    chopped = basic_sent_chop(tagged, raw=False)
    assert chopped == [
        [("a", "DT"), ("b", "NN"), (".", ".")],
        [("c", "DT"), ("d", "NN")],
    ]


def test_tagdata_segment_kwarg_forwards_to_tag(tagger):
    """Without forwarding, ``tagdata`` would silently ignore
    ``segment`` even though it accepts the kwarg."""
    inputs = [
        ["the", "cat", ".", "the", "dog", "."],
        ["the", "cat", "ran", "."],
    ]
    assert tagger.tagdata(inputs, segment=True) == [
        tagger.tag(s, segment=True) for s in inputs
    ]


# ---------------------------------------------------------------------
# Long-sentence and suffix-trie regressions
# ---------------------------------------------------------------------


def test_long_sentence_does_not_recurse(tagger):
    """Decode is iterative; a recursive implementation would raise
    ``RecursionError`` on long real-world inputs once the sentence
    exceeds Python's default recursion limit."""
    words = ["the", "cat"] * 1000
    out = tagger.tag(words)
    _assert_tag_output(words, out)


def test_unknown_word_decode_does_not_grow_suffix_trie(tagger):
    """Defaultdict-style access during suffix scoring could otherwise
    grow the trie unboundedly across calls, leaking memory and
    perturbing decode behavior on later inputs."""
    trie = tagger._suffix_trie_by_cap[False]
    before = set(trie.conditions())
    for word in ["xyzzy", "friendo", "doodad", "phlogiston"]:
        tagger.tag([word])
    after = set(trie.conditions())
    assert after == before


def test_decode_does_not_grow_ngram_conditions(tagger):
    """Tagging must not add conditions to ``_tag_bigrams`` or
    ``_tag_trigrams``. ``ConditionalFreqDist`` subscript access creates
    empty entries for unseen keys, which would silently mutate the
    trained model and break the read-only-decode contract."""
    before_bigrams = set(tagger._tag_bigrams.conditions())
    before_trigrams = set(tagger._tag_trigrams.conditions())
    tagger.tag(["xyzzy", "phlogiston", "doodad"])
    assert set(tagger._tag_bigrams.conditions()) == before_bigrams
    assert set(tagger._tag_trigrams.conditions()) == before_trigrams


def test_transition_logp_cache_matches_direct_formula(tagger):
    """The cached transition log-probabilities equal the direct
    interpolated formula at every tier (trigram, bigram, unigram), and
    completely unseen states fall through to ``_LOG_FLOOR_2``."""
    l1, l2, l3 = tagger._lambda1, tagger._lambda2, tagger._lambda3
    inv_total = _safe_inverse(tagger._num_tag_tokens)

    def direct(prev2, prev1, current):
        bigram_dist = tagger._tag_bigrams.get(prev1)
        trigram_dist = tagger._tag_trigrams.get((prev2, prev1))
        p = l1 * (tagger._tag_unigrams[current] * inv_total)
        if bigram_dist is not None:
            p += l2 * bigram_dist.get(current, 0) * _safe_inverse(bigram_dist.N())
        if trigram_dist is not None:
            p += l3 * trigram_dist.get(current, 0) * _safe_inverse(trigram_dist.N())
        return _safe_log2(p)

    def cached(prev2, prev1, current):
        v = tagger._trans_logp_trigram.get((prev2, prev1), {}).get(current)
        if v is not None:
            return v
        v = tagger._trans_logp_bigram.get(prev1, {}).get(current)
        if v is not None:
            return v
        return tagger._trans_logp_unigram.get(current, _LOG_FLOOR_2)

    # Case 1: observed trigram (cache hits at the trigram tier).
    sampled_any = False
    for prev_pair, dist in tagger._tag_trigrams.items():
        prev2, prev1 = prev_pair
        for current in dist:
            assert cached(prev2, prev1, current) == direct(prev2, prev1, current)
            sampled_any = True
    assert sampled_any, "fixture should expose observed trigrams"

    # Case 2: observed bigram with a fabricated prev2 that was never
    # paired with prev1; cache falls through to the bigram tier.
    bogus_prev2 = ("BOGUS_PREV2", False)
    for prev1, dist in tagger._tag_bigrams.items():
        if (bogus_prev2, prev1) in tagger._tag_trigrams:
            continue
        for current in dist:
            assert cached(bogus_prev2, prev1, current) == direct(
                bogus_prev2, prev1, current
            )

    # Case 3: observed unigram with fabricated history; cache falls
    # through to the unigram tier.
    bogus_prev1 = ("BOGUS_PREV1", False)
    assert bogus_prev1 not in tagger._tag_bigrams.conditions()
    for state in tagger._tag_unigrams:
        assert cached(bogus_prev2, bogus_prev1, state) == direct(
            bogus_prev2, bogus_prev1, state
        )

    # Case 4: completely unseen state. Cache returns the floor; the
    # direct formula reaches the same floor since p collapses to 0.
    bogus_state = ("NEVER_SEEN", False)
    assert cached(bogus_prev2, bogus_prev1, bogus_state) == _LOG_FLOOR_2
    assert direct(bogus_prev2, bogus_prev1, bogus_state) == _LOG_FLOOR_2


# ---------------------------------------------------------------------
# Public API contracts
# ---------------------------------------------------------------------


def test_two_taggers_trained_on_same_data_are_deterministic():
    """Catches non-determinism from iteration over sets or unordered
    dicts in training or decode, which would make trained taggers
    unreproducible across runs."""
    a = TnT()
    a.train(_TRAIN)
    b = TnT()
    b.train(_TRAIN)
    for sent in _TRAIN:
        words = [w for w, _ in sent]
        assert a.tag(words) == b.tag(words)


def test_tag_sents_matches_per_sentence_tag(tagger):
    """``tag_sents()`` (inherited from ``TaggerI``) must produce the
    same result as iterating ``tag()`` per sentence."""
    sents = [[w for w, _ in s] for s in _TRAIN[:3]]
    assert tagger.tag_sents(sents) == [tagger.tag(s) for s in sents]


def test_trained_tagger_round_trips_through_pickle(tagger):
    """A trained ``TnT`` instance must survive a ``pickle`` round trip
    with identical decode behavior. NLTK users routinely serialize
    trained taggers, so this is a real contract."""
    import pickle

    restored = pickle.loads(pickle.dumps(tagger))
    for sent in _TRAIN:
        words = [w for w, _ in sent]
        assert restored.tag(words) == tagger.tag(words)


# ---------------------------------------------------------------------
# Input shape guards
# ---------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["the cat sat", b"the cat sat"])
def test_tag_rejects_string_or_bytes_input(tagger, bad):
    """``tag()`` accepts any iterable, but ``str`` would iterate over
    characters and ``bytes`` would iterate over ints. Catch the common
    "forgot to tokenize" mistake at the boundary."""
    with pytest.raises(TypeError, match="list of tokens"):
        tagger.tag(bad)


@pytest.mark.parametrize("bad", ["the cat sat", b"the cat sat"])
def test_tagdata_rejects_string_or_bytes_input(tagger, bad):
    """``tagdata()`` expects a list of tokenized sentences. A bare
    ``str`` or ``bytes`` would otherwise be interpreted as a sequence
    of one-character or one-int sentences."""
    with pytest.raises(TypeError, match="list of tokenized sentences"):
        tagger.tagdata(bad)


def test_external_unk_returning_extra_tags_raises_clear_error():
    """A misbehaving external ``unk`` tagger that returns more than one
    tagged token for a single input word must raise a clear
    ``ValueError`` instead of an unrelated ``too many values to unpack``
    from tuple destructuring."""

    class ExtraUnk:
        def train(self, _data):
            pass

        def tag(self, toks):
            return [(toks[0], "X"), ("extra", "X")]

    t = TnT(unk=ExtraUnk())
    t.train(_TRAIN)
    with pytest.raises(ValueError, match="returned 2 tags"):
        t.tag(["xyzzy"])


def test_external_unk_returning_no_tags_raises_clear_error():
    """An external ``unk`` tagger that returns an empty list for a
    single-word input must also raise a clear ``ValueError``."""

    class EmptyUnk:
        def train(self, _data):
            pass

        def tag(self, _toks):
            return []

    t = TnT(unk=EmptyUnk())
    t.train(_TRAIN)
    with pytest.raises(ValueError, match="returned 0 tags"):
        t.tag(["xyzzy"])


def test_external_unk_called_for_each_occurrence_not_cached():
    """Repeated OOV occurrences must invoke the external ``unk`` tagger
    each time. The unk path is intentionally uncached because user-
    supplied taggers may carry state or have side effects, so caching
    a single-token output could silently change behavior."""

    class CountingUnk:
        def __init__(self):
            self.calls = 0

        def train(self, _data):
            pass

        def tag(self, toks):
            self.calls += 1
            return [(toks[0], "UNK")]

    unk = CountingUnk()
    t = TnT(unk=unk)
    t.train(_TRAIN)
    t.tag(["xyzzy"])
    t.tag(["xyzzy"])
    t.tag(["xyzzy"])
    assert unk.calls == 3


def test_candidate_cache_stable_across_repeated_sentence(tagger):
    """The candidate-tags cache should not grow after a repeated pass
    over the same sentence. The first pass may add entries; the second
    pass should be pure cache hits for the same ``(word, c_i)`` keys."""
    sent = ["xyzzy", "asdfgh", "the", "cat"]
    tagger.tag(sent)
    size_after_first = len(tagger._candidate_tags_cache)
    tagger.tag(sent)
    size_after_second = len(tagger._candidate_tags_cache)
    assert size_after_first == size_after_second
