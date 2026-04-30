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
)

# --- Shared fixtures, corpora, helpers, and unk mocks ---

_TRAIN = [
    [("the", "DT"), ("cat", "NN"), ("ran", "VBD"), (".", ".")],
    [("the", "DT"), ("running", "VBG"), ("dog", "NN"), ("barked", "VBD"), (".", ".")],
    [("Dianne", "NNP"), ("loves", "VBZ"), ("to", "TO"), ("hug", "VB"), (".", ".")],
    [("Pappy", "NNP"), ("is", "VBZ"), ("very", "RB"), ("loyal", "JJ"), (".", ".")],
]

_AMBIGUITY_TRAIN = [
    [("the", "DT"), ("dogs", "NNS"), (".", ".")],
    [("the", "DT"), ("dogs", "VBZ"), (".", ".")],
    [("the", "DT"), ("fish", "NN"), ("swims", "VBZ"), (".", ".")],
    [("the", "DT"), ("fish", "NN"), ("swims", "NNS"), (".", ".")],
    [("dogs", "NNS"), ("run", "VBP"), ("fast", "RB"), (".", ".")],
    [("the", "DT"), ("run", "NN"), ("ended", "VBD"), (".", ".")],
    [("dogs", "NNS"), ("bark", "VBP"), ("loudly", "RB"), (".", ".")],
    [("a", "DT"), ("bark", "NN"), ("echoed", "VBD"), (".", ".")],
]

_OOV_WORDS = ["xyzzy", "friendo", "diogenes", "phlogiston"]
_OOV_HEAVY_SENT = ["doodad", "watched", "friendo", "playing", "happily", "."]
_LONG_AMBIGUOUS_SENT = ["dogs", "run", "bark"] * 334 + ["."]  # 1003 tokens

_MODEL_STATE_FIELDS = (
    "_lambda1",
    "_lambda2",
    "_lambda3",
    "_num_tag_tokens",
    "_log2_beam_threshold",
    "_tag_prior_probs",
    "_theta",
    "_trans_logp_unigram",
    "_trans_logp_bigram",
    "_trans_logp_trigram",
)


class _CountingUnk:
    """External ``unk`` mock that records ``train()`` calls."""

    def __init__(self):
        self.train_calls = 0

    def train(self, _data):
        self.train_calls += 1

    def tag(self, toks):
        return [(w, "NN") for w in toks]


class _AlternatingUnk:
    """Returns NN, then JJ, then NN, ... — used to detect cache reuse."""

    def __init__(self):
        self.flip = False

    def train(self, _data):
        pass

    def tag(self, toks):
        self.flip = not self.flip
        return [(w, "NN" if self.flip else "JJ") for w in toks]


class _ConstantTagUnk:
    def __init__(self, tag="NONESUCH"):
        self._constant = tag

    def train(self, _data):
        pass

    def tag(self, toks):
        return [(w, self._constant) for w in toks]


class _ExtraTagsUnk:
    def train(self, _data):
        pass

    def tag(self, toks):
        return [(toks[0], "X"), ("extra", "X")]


class _EmptyOutputUnk:
    def train(self, _data):
        pass

    def tag(self, _toks):
        return []


def _trained_tagger(train_data=None, **kwargs):
    if train_data is None:
        train_data = _TRAIN
    t = TnT(**kwargs)
    t.train(train_data)
    return t


def _trained_tags(train):
    return {tag for sent in train for _, tag in sent}


def _words(sent):
    return [word for word, _ in sent]


def _assert_tag_output(words, out):
    assert len(out) == len(words)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in out)
    assert [w for w, _ in out] == words
    assert all(isinstance(tag, str) for _, tag in out)


def _assert_decode_stable(tagger, words, repeats=2):
    known, unknown = tagger.known, tagger.unknown
    try:
        first = tagger.tag(words)
        for _ in range(repeats - 1):
            assert tagger.tag(words) == first
        return first
    finally:
        tagger.known, tagger.unknown = known, unknown


def _state(tagger, word, tag):
    return (tag, tagger._use_capitalization and bool(word) and word[0].isupper())


def _cfd_snapshot(cfd):
    return {condition: dict(dist) for condition, dist in cfd.items()}


def _suffix_snapshot(tagger):
    return {
        cap: _cfd_snapshot(suffix_trie)
        for cap, suffix_trie in tagger._suffix_trie_by_cap.items()
    }


def _model_state(tagger):
    return {field: getattr(tagger, field) for field in _MODEL_STATE_FIELDS}


def _assert_model_state_equal(a, b):
    assert a._beam_threshold == b._beam_threshold
    assert a._use_capitalization == b._use_capitalization
    assert a._unk_trained == b._unk_trained
    assert dict(a._tag_unigrams) == dict(b._tag_unigrams)
    assert _cfd_snapshot(a._tag_bigrams) == _cfd_snapshot(b._tag_bigrams)
    assert _cfd_snapshot(a._tag_trigrams) == _cfd_snapshot(b._tag_trigrams)
    assert _cfd_snapshot(a._word_tag_freqs) == _cfd_snapshot(b._word_tag_freqs)
    assert _suffix_snapshot(a) == _suffix_snapshot(b)
    assert _model_state(a) == _model_state(b)


def _decode_mutation_snapshot(tagger):
    return (
        dict(tagger._tag_unigrams),
        _cfd_snapshot(tagger._tag_bigrams),
        _cfd_snapshot(tagger._tag_trigrams),
        _cfd_snapshot(tagger._word_tag_freqs),
        _suffix_snapshot(tagger),
    )


def _direct_logp(tagger, prev2, prev1, current):
    """Reference deleted-interpolation transition log-prob from the
    trained n-gram counts (the formula the cache should match)."""
    l1, l2, l3 = tagger._lambda1, tagger._lambda2, tagger._lambda3
    inv_total = _safe_inverse(tagger._num_tag_tokens)
    bigram_dist = tagger._tag_bigrams.get(prev1)
    trigram_dist = tagger._tag_trigrams.get((prev2, prev1))
    p = l1 * (tagger._tag_unigrams[current] * inv_total)
    if bigram_dist is not None:
        p += l2 * bigram_dist.get(current, 0) * _safe_inverse(bigram_dist.N())
    if trigram_dist is not None:
        p += l3 * trigram_dist.get(current, 0) * _safe_inverse(trigram_dist.N())
    return _safe_log2(p)


def _cached_logp(tagger, prev2, prev1, current):
    """Tier-by-tier lookup against the precomputed transition cache."""
    v = tagger._trans_logp_trigram.get((prev2, prev1), {}).get(current)
    if v is not None:
        return v
    v = tagger._trans_logp_bigram.get(prev1, {}).get(current)
    if v is not None:
        return v
    return tagger._trans_logp_unigram.get(current, _LOG_FLOOR_2)


def _assert_transition_match(tagger, prev2, prev1, current):
    assert _cached_logp(tagger, prev2, prev1, current) == _direct_logp(
        tagger, prev2, prev1, current
    )


def _assert_observed_trigrams_match(tagger):
    seen = False
    for (prev2, prev1), dist in tagger._tag_trigrams.items():
        for current in dist:
            _assert_transition_match(tagger, prev2, prev1, current)
            seen = True
    assert seen, "fixture should expose observed trigrams"


def _assert_bigram_fallbacks_match(tagger, bogus_prev2):
    for prev1, dist in tagger._tag_bigrams.items():
        if (bogus_prev2, prev1) in tagger._tag_trigrams:
            continue
        for current in dist:
            _assert_transition_match(tagger, bogus_prev2, prev1, current)


def _assert_unigram_fallbacks_match(tagger, bogus_prev2, bogus_prev1):
    assert bogus_prev1 not in tagger._tag_bigrams.conditions()
    for state in tagger._tag_unigrams:
        _assert_transition_match(tagger, bogus_prev2, bogus_prev1, state)


_BOGUS_PREV2 = ("BOGUS_PREV2", False)
_BOGUS_PREV1 = ("BOGUS_PREV1", False)
_BOGUS_STATE = ("NEVER_SEEN", False)


def _assert_transition_cache_matches_formula(tagger):
    _assert_observed_trigrams_match(tagger)
    _assert_bigram_fallbacks_match(tagger, _BOGUS_PREV2)
    _assert_unigram_fallbacks_match(tagger, _BOGUS_PREV2, _BOGUS_PREV1)

    assert (
        _cached_logp(tagger, _BOGUS_PREV2, _BOGUS_PREV1, _BOGUS_STATE) == _LOG_FLOOR_2
    )
    assert (
        _direct_logp(tagger, _BOGUS_PREV2, _BOGUS_PREV1, _BOGUS_STATE) == _LOG_FLOOR_2
    )


@pytest.fixture
def tagger():
    return _trained_tagger()


@pytest.fixture(params=[False, True], ids=["C_False", "C_True"])
def reordered_taggers(request):
    return (
        _trained_tagger(C=request.param),
        _trained_tagger(list(reversed(_TRAIN)), C=request.param),
    )


# --- Constructor and training semantics ---


@pytest.mark.parametrize(
    "bad",
    [0, -1, 0.5, math.nan, math.inf, -math.inf, True, False, "1000", None, [1000]],
)
def test_invalid_n_raises_value_error(bad):
    with pytest.raises(ValueError):
        TnT(N=bad)


def test_empty_training_zeroes_lambdas():
    t = _trained_tagger([])
    assert (t._lambda1, t._lambda2, t._lambda3) == (0.0, 0.0, 0.0)


def test_lambdas_are_non_negative_and_sum_to_one(tagger):
    lambdas = (tagger._lambda1, tagger._lambda2, tagger._lambda3)
    assert all(x >= 0 for x in lambdas)
    assert math.isclose(sum(lambdas), 1.0, abs_tol=1e-12)


def test_repeated_train_rebuilds_state():
    t = _trained_tagger(_AMBIGUITY_TRAIN)
    t.tag(["doodad"])
    assert t._candidate_tags_cache

    t.train(_TRAIN)

    assert not t._candidate_tags_cache
    _assert_model_state_equal(t, _trained_tagger(_TRAIN))


@pytest.mark.parametrize(
    ("trained_flag", "expected_calls"),
    [(False, 1), (True, 0)],
    ids=["default", "trained_flag"],
)
def test_external_unk_is_trained_only_when_needed(trained_flag, expected_calls):
    cu = _CountingUnk()
    t = _trained_tagger(unk=cu, Trained=trained_flag)
    assert cu.train_calls == expected_calls
    t.train(_TRAIN)
    assert cu.train_calls == expected_calls


# --- Boundary counts and read-only decode state ---


def test_eos_follows_sentence_final_punctuation(tagger):
    """EOS is folded into unigram/bigram/trigram counts and attributed
    to the actual final-tag history from training, not a hardcoded
    predecessor."""
    expected_unigram = sum(1 for sent in _TRAIN if sent)
    assert tagger._tag_unigrams[_EOS] == expected_unigram

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
    t = _trained_tagger([[], [("the", "DT"), ("cat", "NN"), (".", ".")], []])
    assert _EOS not in t._tag_bigrams[_BOS]
    assert _EOS not in t._tag_trigrams[(_BOS, _BOS)]


def test_decode_does_not_mutate_trained_state(tagger):
    before = _decode_mutation_snapshot(tagger)

    for word in _OOV_WORDS:
        tagger.tag([word])

    assert _decode_mutation_snapshot(tagger) == before


# --- Unknown-word and external-unk handling ---


def test_unknown_word_falls_back_to_unk_when_priors_empty():
    assert TnT().tag(["xyzzy"]) == [("xyzzy", "Unk")]


def test_external_unk_returning_unseen_tag_survives_decode():
    """An external ``unk`` may return a tag never seen during training;
    the transition cache falls through to the model floor and the path
    stays alive."""
    t = _trained_tagger(unk=_ConstantTagUnk("NONESUCH"))
    assert t.tag(["xyzzy"]) == [("xyzzy", "NONESUCH")]


def test_external_unk_bypasses_decode_cache():
    """Stateful ``unk`` taggers must be invoked on every call; otherwise
    caching collapses their varying output into a single result."""
    t = _trained_tagger(unk=_AlternatingUnk())
    seen = [t.tag(["xyzzy"])[0][1] for _ in range(3)]
    assert seen == ["NN", "JJ", "NN"]


@pytest.mark.parametrize(
    ("unk_class", "expected_n"),
    [(_ExtraTagsUnk, 2), (_EmptyOutputUnk, 0)],
    ids=["extra_tags", "no_tags"],
)
def test_external_unk_returning_wrong_count_raises_clear_error(unk_class, expected_n):
    """A misbehaving external ``unk`` must raise ``ValueError`` with the
    actual count, not a cryptic ``too many values to unpack``."""
    t = _trained_tagger(unk=unk_class())
    with pytest.raises(ValueError, match=f"returned {expected_n} tags for 1 word"):
        t.tag(["xyzzy"])


# --- Transition probabilities and cache correctness ---


def test_transition_logp_cache_matches_direct_formula(tagger):
    _assert_transition_cache_matches_formula(tagger)


# --- Decode shape, pruning, and stability ---


def test_known_words_decode_to_their_only_seen_tag(tagger):
    for sent in _TRAIN:
        assert tagger.tag(_words(sent)) == sent


def test_threshold_pruning_does_not_empty_beam_under_ambiguity():
    t = _trained_tagger(_AMBIGUITY_TRAIN, N=2)
    words = ["the", "dogs", "."]
    _assert_tag_output(words, t.tag(words))


def test_decode_is_repeat_stable_under_near_tie_beam():
    """Strict ``>`` keeps the first encountered path when scores tie."""
    t = _trained_tagger(_AMBIGUITY_TRAIN, N=1000)
    _assert_decode_stable(t, ["the", "fish", "swims", "."], repeats=10)


@pytest.mark.parametrize(
    "train",
    [pytest.param(_TRAIN, id="full"), pytest.param(_TRAIN[:1], id="one_sent")],
)
def test_decode_handles_oov_heavy_sentence_with_shared_suffixes(train):
    """OOV-heavy input should be well-formed, trained-tagset bounded,
    and repeat-stable even with tiny training data."""
    t = _trained_tagger(train)
    out = _assert_decode_stable(t, _OOV_HEAVY_SENT)

    _assert_tag_output(_OOV_HEAVY_SENT, out)
    assert all(tag in _trained_tags(train) for _, tag in out)


def test_decode_handles_long_ambiguous_sentence():
    """Long ambiguous input must decode iteratively and repeatably. The
    length is above Python's default recursion limit, so a recursive
    decoder would fail here."""
    t = _trained_tagger(_AMBIGUITY_TRAIN, N=1000)
    out = _assert_decode_stable(t, _LONG_AMBIGUOUS_SENT)
    _assert_tag_output(_LONG_AMBIGUOUS_SENT, out)


def test_candidate_cache_stable_across_repeated_sentence(tagger):
    """The candidate-tags cache should not grow on a repeat pass over
    the same sentence."""
    sent = _OOV_WORDS[:2] + ["the", "cat"]
    tagger.tag(sent)
    size_after_first = len(tagger._candidate_tags_cache)
    tagger.tag(sent)
    assert len(tagger._candidate_tags_cache) == size_after_first


# --- Segmentation and input-shape guards ---


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
    out = tagger.tag(words, segment=segment)
    assert len(out) == expected_len
    if expected_len > 0:
        _assert_tag_output(words, out)


def test_segment_true_matches_segment_false_on_single_sentence(tagger):
    words = ["the", "cat", "ran", "."]
    assert tagger.tag(words) == tagger.tag(words, segment=True)


def test_tagdata_segment_kwarg_forwards_to_tag(tagger):
    inputs = [
        ["the", "cat", ".", "the", "dog", "."],
        ["beagles", "are", "happy", "to", "rest", "."],
    ]
    assert tagger.tagdata(inputs, segment=True) == [
        tagger.tag(s, segment=True) for s in inputs
    ]


@pytest.mark.parametrize(
    ("method", "match"),
    [("tag", "list of tokens"), ("tagdata", "list of tokenized sentences")],
)
@pytest.mark.parametrize("bad", ["the cat sat", b"the cat sat"])
def test_string_or_bytes_input_is_rejected(tagger, bad, method, match):
    """``str`` would iterate over characters and ``bytes`` over ints; each
    method catches the common 'forgot to tokenize' mistake at its own
    boundary."""
    with pytest.raises(TypeError, match=match):
        getattr(tagger, method)(bad)


# --- Serialization and reproducibility ---


def test_trained_tagger_round_trips_through_pickle(tagger):
    import pickle

    restored = pickle.loads(pickle.dumps(tagger))
    _assert_model_state_equal(restored, tagger)

    for sent in _TRAIN:
        assert restored.tag(_words(sent)) == tagger.tag(_words(sent))


def test_train_order_invariance_at_model_level(reordered_taggers):
    """Reordering training sentences must not change the trained model."""
    _assert_model_state_equal(*reordered_taggers)


def test_decode_is_bit_stable_across_train_data_reorderings(reordered_taggers):
    """Decoded output is bit-stable across equivalent training-data orderings."""
    a, b = reordered_taggers
    for sent in _TRAIN:
        assert a.tag(_words(sent)) == b.tag(_words(sent))
