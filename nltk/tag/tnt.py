# Natural Language Toolkit: TnT Tagger
#
# Copyright (C) 2001-2026 NLTK Project
# Author: Sam Huston <sjh900@gmail.com>
#         John Winstead <https://github.com/jhnwnstd>
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Implementation of 'TnT - A Statistical Part of Speech Tagger'
by Thorsten Brants

https://aclanthology.org/A00-1031.pdf
"""

from math import log2

from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.tag.api import TaggerI

# Used in place of log2(p) when p underflows; log2(1e-300) ~= -996.58
# is still well above the negative-inf result of log2(0).
_LOG_FLOOR_2 = log2(1e-300)

# Sentinel tags used at sentence boundaries.
_BOS = ("BOS", False)
_EOS = ("EOS", False)

_SENT_MARKS = (".", "!", "?", ";")

# Returned by transition-cache misses so inner-loop ``.get`` calls never
# need an extra ``None`` check before the next lookup.
_EMPTY_DICT = {}


def _safe_log2(p):
    """``log2(p)`` for a non-zero probability, ``_LOG_FLOOR_2`` otherwise."""
    return log2(p) if p > 1e-300 else _LOG_FLOOR_2


def _safe_inverse(n):
    """``1/n`` for a non-zero ``n``, ``0.0`` otherwise so callers can use
    multiplication without an inline guard."""
    return (1.0 / n) if n else 0.0


class TnT(TaggerI):
    """
    TnT statistical POS tagger.

    TnT implements the second order hidden Markov model tagger described
    by Brants (2000). Hidden states are part of speech tags, observed
    symbols are word tokens, and each tag transition depends on the two
    previous tag states.

    Important usage notes

    * Use sentence delimited input.

      TnT estimates and scores beginning of sentence and end of sentence
      transitions. It works best when training and tagging data are already
      split into sentences.

      Training input should be a list of sentences. Each sentence should
      be a list of ``(word, tag)`` tuples.

      ``tag()`` expects one tokenized sentence.
      ``tagdata()`` expects a list of tokenized sentences.

      For simple punctuation based segmentation of token streams, see
      ``basic_sent_chop()``. It splits on the sentence final punctuation
      used by Brants, namely ``.``, ``!``, ``?``, and ``;``.

    * Unknown words use Brants's suffix model.

      Words not observed during training are scored from suffix
      distributions, as described in Brants (2000), section 2.3. The model
      builds capitalization specific suffix tries from infrequent training
      words, smooths suffix tag probabilities by successive abstraction,
      and applies Bayesian inversion to obtain emission like scores.

      A user supplied tagger may be passed with ``unk``. When supplied,
      it overrides the built in suffix model for unknown words.

    * Known words use a tag dictionary.

      For a known word, candidate tags are restricted to the tags assigned
      to that word in the training data. If a word was never seen with a
      tag during training, that tag is not considered for that word at
      decode time.

    Model

      For a sentence of length ``T``, the decoder chooses the tag sequence
      that maximizes

          prod_i P(t_i | t_{i-1}, t_{i-2}) * P(w_i | t_i)

      and then scores an explicit end of sentence transition

          P(EOS | t_{T-1}, t_T)

      Decoding uses Viterbi search in log space over a trellis of
      ``(t_{i-1}, t_i)`` state pairs.

          argmax_{t_1..t_T} sum_i [
              log P(t_i | t_{i-1}, t_{i-2})
              + log P(w_i | t_i)
          ] + log P(EOS | t_{T-1}, t_T)

    Transition smoothing

      The transition probability is a deleted interpolation of unigram,
      bigram, and trigram tag models.

          P(t_i | t_{i-1}, t_{i-2})
              = l1 * P(t_i)
              + l2 * P(t_i | t_{i-1})
              + l3 * P(t_i | t_{i-1}, t_{i-2})

      The interpolation weights ``l1``, ``l2``, and ``l3`` are estimated
      from the training data with the deleted interpolation procedure in
      Brants (2000), section 2.2.

    Emissions

      For known words, ``P(w_i | t_i)`` is the relative frequency of the
      word and tag pair in the training lexicon.

      For unknown words, the emission like score is the Bayes inverted
      suffix score described above.

    Beam search

      The decoder uses beam pruning to limit memory and runtime. After
      each Viterbi step, states whose log probability is worse than the
      current best state by more than ``log2(N)`` are discarded. The
      default threshold ``N=1000`` follows the value reported by Brants.

    Capitalization

      When ``C=True``, capitalization is included in the tag state. This
      is equivalent to using separate tag states for capitalized and
      uncapitalized tokens. It can help when capitalization is informative,
      but its effect depends on the language, corpus, and tagset.
    """

    def __init__(self, unk=None, Trained=False, N=1000, C=False):
        """
        Construct a TnT statistical tagger. The tagger must be trained
        before it can be used to tag input.

        :param unk: instance of a POS tagger, conforms to TaggerI.
                    When supplied, overrides the built-in suffix model
                    on the unknown-word path.
        :type  unk: TaggerI
        :param Trained: Indication that the POS tagger is trained or not.
                        Set True to skip training the optional ``unk``
                        tagger on the next train() call.
        :type  Trained: bool
        :param N: Beam search pruning threshold. After each Viterbi
                step any state whose log-probability is worse than
                the best by more than a factor of N is discarded.
                Must be a positive integer; 1000 is a good default.
        :type  N: int
        :param C: Capitalization flag. When True, tags are differentiated
                by whether the source word is capitalized. This rarely
                improves accuracy in practice.
        :type  C: bool
        """

        # Restricting N to a positive integer keeps the public contract
        # narrow. The explicit bool check matters because ``bool`` is a
        # subclass of ``int`` and would otherwise pass the type check.
        if isinstance(N, bool) or not isinstance(N, int) or N < 1:
            raise ValueError(f"N must be a positive integer, got {N!r}")

        self._beam_threshold = N
        self._use_capitalization = C
        self._unk = unk
        self._unk_trained = Trained

        self._tag_unigrams = FreqDist()
        self._tag_bigrams = ConditionalFreqDist()
        self._tag_trigrams = ConditionalFreqDist()
        self._word_tag_freqs = ConditionalFreqDist()

        self._lambda1 = 0.0
        self._lambda2 = 0.0
        self._lambda3 = 0.0
        self._num_tag_tokens = 0
        self._log2_beam_threshold = 0.0

        # Unknown-word decoding uses a capitalization split suffix model, a
        # raw tag prior, and theta for successive abstraction smoothing.
        self._suffix_trie_by_cap = {
            False: ConditionalFreqDist(),
            True: ConditionalFreqDist(),
        }
        self._tag_prior_probs = {}
        self._theta = 0.0

        # Read by ``_expand_states`` instead of recomputing the deleted
        # interpolation and ``log2`` per beam expansion.
        self._trans_logp_unigram = {}
        self._trans_logp_bigram = {}
        self._trans_logp_trigram = {}

        # The cache depends entirely on trained model state, so it starts
        # empty here and is cleared whenever train() rebuilds the model.
        self._candidate_tags_cache = {}

        self.unknown = 0
        self.known = 0

    def train(self, data):
        """
        Trains the tagger on a list of tagged sentences. Each call
        rebuilds the model from scratch on the supplied data. The
        n-gram counts, word-tag lexicon, suffix model, and
        deleted-interpolation weights are all replaced, and the decode
        cache is cleared.

        The optional external unknown-word tagger (``unk``) is trained
        on the supplied data only the first time ``train()`` is called.
        Subsequent calls leave it alone, since retraining it on each
        new training set is rarely what callers want.

        :param data: list of lists of (word, tag) tuples
        :type  data: list[list[tuple[str, str]]]
        """

        # These structures accumulate corpus statistics, so retraining must
        # rebuild them from scratch rather than layer new counts on top.
        self._candidate_tags_cache.clear()
        self._tag_unigrams = FreqDist()
        self._tag_bigrams = ConditionalFreqDist()
        self._tag_trigrams = ConditionalFreqDist()
        self._word_tag_freqs = ConditionalFreqDist()

        unk = self._unk
        if unk is not None and not self._unk_trained:
            unk.train(data)

        word_tag_freqs = self._word_tag_freqs
        tag_unigrams = self._tag_unigrams
        tag_bigrams = self._tag_bigrams
        tag_trigrams = self._tag_trigrams
        cap_on = self._use_capitalization

        for sent in data:
            state_i_minus_2 = _BOS
            state_i_minus_1 = _BOS
            sent_has_tokens = False

            for word, tag in sent:
                sent_has_tokens = True
                c_i = cap_on and bool(word) and word[0].isupper()
                state_i = (tag, c_i)

                word_tag_freqs[word][tag] += 1
                tag_unigrams[state_i] += 1
                tag_bigrams[state_i_minus_1][state_i] += 1
                tag_trigrams[(state_i_minus_2, state_i_minus_1)][state_i] += 1

                state_i_minus_2, state_i_minus_1 = state_i_minus_1, state_i

            # EOS is treated as an ordinary next state in the n-gram model,
            # but empty sentences are skipped so BOS does not acquire EOS as
            # a spurious successor.
            if sent_has_tokens:
                tag_unigrams[_EOS] += 1
                tag_bigrams[state_i_minus_1][_EOS] += 1
                tag_trigrams[(state_i_minus_2, state_i_minus_1)][_EOS] += 1

        self._compute_lambda()

        # This total intentionally includes EOS because the unigram model
        # and deleted interpolation are estimated over the same event stream.
        self._num_tag_tokens = tag_unigrams.N()
        self._log2_beam_threshold = log2(self._beam_threshold)

        self._build_transition_logp_cache()
        self._build_suffix_model()

        self._unk_trained = True

    def tag(self, tokens, segment=False):
        """
        Tag a single sentence. Delegates the actual decode to
        `_tagword`, then pairs each chosen tag with its input token.

        When `segment` is True, the input may contain mid-sequence
        sentence punctuation [.!?;]. The decoder splits on those
        tokens and re-seeds the BOS state for each segment.
        The default is False because most NLTK callers pre-segment,
        and auto-splitting on `.` would mis-handle abbreviations
        like "Mr." in unsegmented input.

        :param tokens: words to tag
        :type tokens: list[str]
        :param segment: split on [.!?;] and decode each segment with a
                        fresh BOS state
        :type segment: bool
        :return: list of `(word, tag)` tuples
        """
        if segment:
            return self._tag_segmented(tokens)
        if not (sent := list(tokens)):
            return []
        return self._pair_decoded(sent, self._tagword(sent))

    def tagdata(self, data, segment=False):
        """
        Tags a list of sentences. Each input sentence is a list of words;
        each output sentence is a list of (word, tag) tuples.

        :param data: list of list of words
        :type  data: list[list[str]]
        :param segment: forwarded to ``tag``. Pass True to auto-split
                        each input on internal [.!?;] punctuation.
        :type segment: bool
        :return: list of list of (word, tag) tuples
        """
        return [self.tag(sent, segment=segment) for sent in data]

    def _compute_lambda(self):
        """
        Computes the deleted-interpolation weights l1, l2, l3 from the
        observed tag n-grams. Tied maxima split the trigram count evenly
        among the winning lambdas. Branches with a zero denominator
        contribute zero.

        For each trigram (t1, t2, t3) with positive count we compare

            c1 = (f(t3) - 1) / (N - 1)
            c2 = (f(t2, t3) - 1) / (f(t2) - 1)
            c3 = (f(t1, t2, t3) - 1) / (f(t1, t2) - 1)
        """

        tag_unigrams = self._tag_unigrams
        tag_bigrams = self._tag_bigrams
        tag_trigrams = self._tag_trigrams
        unigram_n_minus_1 = tag_unigrams.N() - 1

        lambda1_mass = 0.0
        lambda2_mass = 0.0
        lambda3_mass = 0.0

        for state_i_minus_2, state_i_minus_1 in tag_trigrams.conditions():
            trigram_dist = tag_trigrams[(state_i_minus_2, state_i_minus_1)]
            bigram_dist = tag_bigrams[state_i_minus_1]

            trigram_n_minus_1 = trigram_dist.N() - 1
            bigram_n_minus_1 = bigram_dist.N() - 1

            for state_i, count in trigram_dist.items():
                # Subtracting one leaves the current event out, so each score
                # asks which model order would best predict this tag if this
                # occurrence were held out.
                c1 = (
                    (tag_unigrams[state_i] - 1) / unigram_n_minus_1
                    if unigram_n_minus_1
                    else 0.0
                )
                c2 = (
                    (bigram_dist[state_i] - 1) / bigram_n_minus_1
                    if bigram_n_minus_1
                    else 0.0
                )
                c3 = (count - 1) / trigram_n_minus_1 if trigram_n_minus_1 else 0.0

                # The trigram's count is credited to the model order with the
                # strongest held out estimate. Splitting ties evenly avoids
                # introducing an arbitrary preference between orders.
                maxc = max(c1, c2, c3)
                w1 = c1 == maxc
                w2 = c2 == maxc
                w3 = c3 == maxc
                share = count / (w1 + w2 + w3)

                if w1:
                    lambda1_mass += share
                if w2:
                    lambda2_mass += share
                if w3:
                    lambda3_mass += share

        # Normalization turns the accumulated winning mass into mixture
        # weights. Keeping the zero case explicit prevents a degenerate
        # training run from dividing by zero or reusing stale weights.
        total_mass = lambda1_mass + lambda2_mass + lambda3_mass
        if total_mass > 0:
            self._lambda1 = lambda1_mass / total_mass
            self._lambda2 = lambda2_mass / total_mass
            self._lambda3 = lambda3_mass / total_mass
        else:
            self._lambda1 = 0.0
            self._lambda2 = 0.0
            self._lambda3 = 0.0

    def _build_transition_logp_cache(self):
        """
        Precompute transition log probabilities for Viterbi expansion.

        The decoder scores each transition with the same deleted
        interpolation used by the model.

            P(t_i | t_{i-2}, t_{i-1})
              = l1 P(t_i)
              + l2 P(t_i | t_{i-1})
              + l3 P(t_i | t_{i-2}, t_{i-1})

        Computing this value inside the beam loop repeats the same work
        many times, so training builds caches for the observed unigram,
        bigram, and trigram contexts. During decoding, ``_expand_states``
        first tries the trigram cache, then the bigram cache, then the
        unigram cache.

        The caches store log2 probabilities.

            ``_trans_logp_unigram[state]``
            ``_trans_logp_bigram[prev1][state]``
            ``_trans_logp_trigram[(prev2, prev1)][state]``

        Bigram cache entries include the unigram and bigram interpolation
        terms. Trigram cache entries include all three terms. The
        parenthesization in the probability calculation is kept stable so
        cached values match the previous per-call computation.
        """
        tag_unigrams = self._tag_unigrams
        tag_bigrams = self._tag_bigrams
        tag_trigrams = self._tag_trigrams

        lambda1, lambda2, lambda3 = self._lambda1, self._lambda2, self._lambda3
        inv_total_N = _safe_inverse(self._num_tag_tokens)

        # Computed once and reused below so the bigram and trigram caches
        # don't recompute the unigram contribution per (history, current).
        unigram_part = {
            state: lambda1 * (count * inv_total_N)
            for state, count in tag_unigrams.items()
        }
        self._trans_logp_unigram = {
            state: _safe_log2(p) for state, p in unigram_part.items()
        }

        bigram_logp = {}
        for prev1 in tag_bigrams.conditions():
            bigram_dist = tag_bigrams[prev1]
            inv_bigram_N = _safe_inverse(bigram_dist.N())

            bigram_logp[prev1] = {
                current: _safe_log2(
                    unigram_part.get(current, 0.0) + lambda2 * count * inv_bigram_N
                )
                for current, count in bigram_dist.items()
            }
        self._trans_logp_bigram = bigram_logp

        trigram_logp = {}
        for prev_pair in tag_trigrams.conditions():
            _, prev1 = prev_pair

            trigram_dist = tag_trigrams[prev_pair]
            bigram_dist = tag_bigrams[prev1]

            inv_trigram_N = _safe_inverse(trigram_dist.N())
            inv_bigram_N = _safe_inverse(bigram_dist.N())
            bigram_get = bigram_dist.get

            trigram_logp[prev_pair] = {
                current: _safe_log2(
                    unigram_part.get(current, 0.0)
                    + lambda2 * bigram_get(current, 0) * inv_bigram_N
                    + lambda3 * count * inv_trigram_N
                )
                for current, count in trigram_dist.items()
            }
        self._trans_logp_trigram = trigram_logp

    def _build_suffix_model(self):
        """
        Build the suffix model used for unseen words.

        The model stores three decode-time values: capitalization-split
        suffix tries, the successive-abstraction weight theta, and raw
        tag priors collapsed across capitalization.

        EOS is excluded from the priors because it is a sequence marker,
        not a lexical tag. Suffix counts come only from lexicon words
        with total count at most 10, following Brants's infrequent-word
        threshold for unknown-word modeling.
        """
        tag_unigrams = self._tag_unigrams
        word_tag_freqs = self._word_tag_freqs

        tag_counts = {}
        for (tag, _), count in tag_unigrams.items():
            # The suffix model predicts lexical tags, not boundary markers.
            # Check the raw tag so EOS is excluded from all capitalization states.
            if tag == _EOS[0]:
                continue
            tag_counts[tag] = tag_counts.get(tag, 0) + count

        total = sum(tag_counts.values())
        if total > 0:
            tag_prior_probs = {tag: count / total for tag, count in tag_counts.items()}
        else:
            tag_prior_probs = {}

        # Theta controls how strongly the suffix recursion is pulled back
        # toward the less specific estimate at each abstraction step.
        priors = list(tag_prior_probs.values())
        n_priors = len(priors)
        if n_priors > 1:
            mean = sum(priors) / n_priors
            theta = (
                sum((prior - mean) ** 2 for prior in priors) / (n_priors - 1)
            ) ** 0.5
        else:
            theta = 0.0

        # Each capitalization bucket stores every suffix length up to 10,
        # so decode can walk from the shortest matched ending to the longest.
        suffix_trie_by_cap = {
            False: ConditionalFreqDist(),
            True: ConditionalFreqDist(),
        }

        for word in word_tag_freqs.conditions():
            tag_freqs = word_tag_freqs[word]
            if not word or tag_freqs.N() > 10:
                continue

            suffix_trie = suffix_trie_by_cap[word[0].isupper()]
            max_suffix_len = min(len(word), 10)

            for m in range(1, max_suffix_len + 1):
                suffix_dist = suffix_trie[word[-m:]]
                for tag, count in tag_freqs.items():
                    suffix_dist[tag] += count

        self._tag_prior_probs = tag_prior_probs
        self._theta = theta
        self._suffix_trie_by_cap = suffix_trie_by_cap

    def _unknown_tag_scores(self, word):
        """
        Score candidate tags for an unknown word using Brants's suffix
        model.

        The intuition is that a word's last few characters predict its
        tag well, since `-able` words tend to be adjectives, `-ing`
        words tend to be participles, and so on. Starting from the
        unigram tag prior, we walk one suffix character at a time up
        to the longest suffix we saw during training (capped at 10
        characters), blending each suffix's tag distribution into the
        running estimate via the successive abstraction recursion

            P(t | l_{n-i+1}...l_n) = (P_hat + theta * P_prev) / (1 + theta)

        If the word's tail is unfamiliar, the recursion never gets past
        the prior, which is the back-off case.

        :return: Bayes-inverted scores. Each raw tag maps to a quantity
                proportional to P(suffix | t). The P(suffix) constant
                drops out because it does not depend on the tag, which
                preserves the argmax without computing it. Tags with
                zero prior are omitted.
        """
        tag_priors = self._tag_prior_probs
        if not tag_priors:
            return {}

        is_capitalized = bool(word) and word[0].isupper()
        suffix_trie = self._suffix_trie_by_cap[is_capitalized]
        max_suffix_len = min(len(word), 10)

        # The trie contains every suffix length up to the cutoff. Once the
        # longest matching suffix is found, all shorter suffixes are also
        # available for successive abstraction.
        longest = 0
        for m in range(max_suffix_len, 0, -1):
            if word[-m:] in suffix_trie:
                longest = m
                break

        # No matched suffix means the estimate stays at the unigram prior.
        # After Bayes inversion, every tag then receives the same score.
        if longest == 0:
            return {tag: 1.0 for tag, prior in tag_priors.items() if prior > 0}

        theta = self._theta

        # With theta equal to zero there is no smoothing, so the estimate
        # is the empirical distribution of the longest matched suffix.
        # Tags absent from that bucket would have score zero, so they are
        # dropped from the result instead of being emitted as floor-only
        # candidates that the beam would prune anyway.
        if theta == 0.0:
            suffix_dist = suffix_trie[word[-longest:]]
            inv_suffix_N = 1.0 / suffix_dist.N()
            return {
                tag: (count * inv_suffix_N) / tag_priors[tag]
                for tag, count in suffix_dist.items()
                if tag_priors.get(tag, 0) > 0
            }

        # Dense successive abstraction updates every tag at every suffix
        # length. For tags absent from the current suffix bucket, that update
        # is the same shared shrinkage. Factor that shared term into one
        # scalar, and keep only tag specific corrections in delta.
        denom = 1.0 + theta
        miss_scale = theta / denom

        global_scale = 1.0
        delta = {}

        for i in range(1, longest + 1):
            suffix_dist = suffix_trie[word[-i:]]
            inv_suffix_N = 1.0 / suffix_dist.N()

            # Apply the shared shrinkage for all tags, then add the suffix
            # evidence only for tags observed in this bucket.
            global_scale *= miss_scale
            corr_scale = inv_suffix_N / (denom * global_scale)

            for tag, count in suffix_dist.items():
                delta[tag] = delta.get(tag, 0.0) + count * corr_scale

        # In the factored form, Bayes inversion becomes
        #   P(t | suffix) / P(t) = global_scale * (1 + delta[t] / P(t)).
        # Untouched tags share one score. Touched tags get a correction
        # relative to their unigram prior.
        result = {}
        for tag, prior in tag_priors.items():
            if prior <= 0:
                continue
            extra = delta.get(tag)
            if extra is None:
                result[tag] = global_scale
            else:
                result[tag] = global_scale * (1.0 + extra / prior)

        return result

    def _tag_segmented(self, tokens):
        """
        Tag ``tokens`` as one or more sentences split on ``[.!?;]``.

        Each sentence-final punctuation token stays with the segment it
        closes, and a trailing fragment without sentence-final punctuation
        is still tagged as its own segment.
        """
        tagged = []
        segment = []

        sent_marks = _SENT_MARKS
        tagword = self._tagword
        pair_decoded = self._pair_decoded
        extend = tagged.extend

        for token in tokens:
            segment.append(token)
            if token in sent_marks:
                extend(pair_decoded(segment, tagword(segment)))
                segment.clear()

        if segment:
            extend(pair_decoded(segment, tagword(segment)))

        return tagged

    def _tagword(self, sent):
        """
        Tag one sentence with second-order Viterbi decoding.

        The lattice state is the last two tag states, so paths that share
        ``(state_{i-1}, state_i)`` are merged immediately. Known words draw
        candidates from the lexicon. Unknown words are scored either by the
        external ``unk`` tagger or by the suffix model. After each word, the
        beam keeps only states whose score is within ``log2(N)`` of the best
        surviving path. The decode then scores an explicit EOS transition
        and walks backpointers to recover the best state sequence.

        :param sent: words to tag
        :type sent: list[str]
        :return: list shaped ``[BOS, BOS, state_0, ..., state_{T-1}]``
                where each state is a ``(tag, capitalization)`` pair.
        """
        if not sent:
            return [_BOS, _BOS]

        T = len(sent)

        # Local bindings keep the hot loop on plain locals rather than
        # repeated attribute lookups.
        word_tag_freqs = self._word_tag_freqs
        tag_unigrams = self._tag_unigrams
        trans_logp_unigram_get = self._trans_logp_unigram.get
        log2_beam_threshold = self._log2_beam_threshold
        cap_on = self._use_capitalization
        unk = self._unk
        cache = self._candidate_tags_cache
        unknown_tag_scores = self._unknown_tag_scores
        expand_states = self._expand_states

        # Each level keeps only the best path reaching a given
        # ``(state_{i-1}, state_i)`` key. The backpointer stores
        # ``state_{i-2}`` so the best path can be reconstructed at the end.
        states = {(_BOS, _BOS): (0.0, _BOS)}
        state_history = [states]

        for word in sent:
            c_i = cap_on and bool(word) and word[0].isupper()
            tag_freqs = word_tag_freqs.get(word)

            if tag_freqs is not None:
                self.known += 1
            else:
                self.unknown += 1

            # External unknown-word taggers are treated as potentially
            # stateful. The built-in known-word and suffix-model paths are
            # pure given ``(word, c_i)`` and the trained model, so they cache.
            if tag_freqs is None and unk is not None:
                [(_word, tag)] = unk.tag([word])
                state_i = (tag, c_i)
                unigram_logp = trans_logp_unigram_get(state_i, _LOG_FLOOR_2)
                candidate_tags = ((state_i, 0.0, unigram_logp),)
            else:
                cache_key = (word, c_i)
                candidate_tags = cache.get(cache_key)

                if candidate_tags is None:
                    if tag_freqs is not None:
                        # Known words only consider tags actually seen with
                        # that surface form. The lexical term is P(word | tag).
                        entries = []
                        for tag, tag_count in tag_freqs.items():
                            state_i = (tag, c_i)
                            unigram_state_i = tag_unigrams[state_i]
                            unigram_logp = trans_logp_unigram_get(state_i, _LOG_FLOOR_2)
                            entries.append(
                                (
                                    state_i,
                                    log2(tag_count / unigram_state_i),
                                    unigram_logp,
                                )
                            )
                        candidate_tags = tuple(entries)
                    else:
                        # Unknown words use the suffix model as their lexical
                        # score. Bayes inversion turns the suffix posterior into
                        # the emission-like quantity used by the decoder.
                        suffix_scores = unknown_tag_scores(word)

                        if not suffix_scores:
                            # An untrained tagger has no suffix priors, so the
                            # only safe fallback is a literal ``Unk`` state.
                            state_i = ("Unk", c_i)
                            unigram_logp = trans_logp_unigram_get(state_i, _LOG_FLOOR_2)
                            candidate_tags = ((state_i, 0.0, unigram_logp),)
                        else:
                            entries = []
                            for tag, score in suffix_scores.items():
                                state_i = (tag, c_i)
                                unigram_logp = trans_logp_unigram_get(
                                    state_i, _LOG_FLOOR_2
                                )
                                entries.append(
                                    (state_i, _safe_log2(score), unigram_logp)
                                )
                            candidate_tags = tuple(entries)

                    cache[cache_key] = candidate_tags

            new_states, best_logp = expand_states(states, candidate_tags)

            # Threshold pruning keeps the beam relative to the best current
            # path, which is the pruning rule described in the paper.
            cutoff = best_logp - log2_beam_threshold
            states = {k: v for k, v in new_states.items() if v[0] >= cutoff}
            state_history.append(states)

        # Scored inline rather than via the transition cache to preserve
        # the previous decoder's floating-point evaluation order; per-call
        # cost is small relative to the inner Viterbi loop.
        tag_bigrams = self._tag_bigrams
        tag_trigrams = self._tag_trigrams
        lambda1, lambda2, lambda3 = self._lambda1, self._lambda2, self._lambda3
        inv_num_tag_tokens = _safe_inverse(self._num_tag_tokens)
        p_eos_unigram = tag_unigrams[_EOS] * inv_num_tag_tokens

        best_final_key = next(iter(states))
        best_final_logp = float("-inf")

        for predecessor_key, (prefix_logp, _) in states.items():
            state_i_minus_1 = predecessor_key[1]

            # ``ConditionalFreqDist.__getitem__`` would create empty entries
            # for unseen keys, mutating the trained model during decode. Use
            # ``.get`` so an unseen history just contributes zero.
            bigram_dist = tag_bigrams.get(state_i_minus_1)
            if bigram_dist is None:
                p_eos_bigram = 0.0
            else:
                bigram_N = bigram_dist.N()
                p_eos_bigram = (bigram_dist[_EOS] / bigram_N) if bigram_N else 0.0

            trigram_dist = tag_trigrams.get(predecessor_key)
            if trigram_dist is None:
                p_eos_trigram = 0.0
            else:
                trigram_N = trigram_dist.N()
                p_eos_trigram = (trigram_dist[_EOS] / trigram_N) if trigram_N else 0.0

            p_eos_given_history = (
                lambda1 * p_eos_unigram
                + lambda2 * p_eos_bigram
                + lambda3 * p_eos_trigram
            )
            final_logp = prefix_logp + _safe_log2(p_eos_given_history)
            if final_logp > best_final_logp:
                best_final_logp = final_logp
                best_final_key = predecessor_key

        # Walking the stored ``state_{i-2}`` backpointers recovers the best
        # full state sequence from the best final state pair.
        states_reversed = [best_final_key[1]]
        if T >= 2:
            states_reversed.append(best_final_key[0])

        current_key = best_final_key
        for level in range(T, 2, -1):
            backpointer = state_history[level][current_key][1]
            states_reversed.append(backpointer)
            current_key = (backpointer, current_key[0])

        states_reversed.reverse()
        return [_BOS, _BOS] + states_reversed

    def _expand_states(self, states, candidate_tags):
        """
        Take one Viterbi step. For every predecessor state, score each
        candidate `state_i` from `candidate_tags` and accumulate the
        results into a new state dict keyed by
        `(state_i_minus_1, state_i)`. When two predecessors land on the
        same key, keep the higher-scoring one and discard the other. The
        second-order Markov assumption means everything after this point
        depends only on the last two states, so the discarded path can
        never beat the kept one.

        `candidate_tags` is a sequence of `(state_i, log_emit,
        unigram_logp)` triples. `log_emit` is the lexical log-probability
        for known words, the Bayes-inverted suffix score for unknown
        words, or zero for an external unknown-word tagger.

        `unigram_logp` is the precomputed unigram-tier transition
        log-probability for `state_i`, used as the fallback when the
        trigram and bigram caches both miss. Precomputing it once per
        candidate removes a hot lookup from the inner loop on high-OOV
        corpora.

        :return: ``(new_states, best_logp)``. ``new_states`` maps
                 `(state_i_minus_1, state_i)` to
                 `(logp, state_i_minus_2)`, where `state_i_minus_2` is
                 the backpointer used to reconstruct the best path after
                 the final EOS step. `best_logp` is the maximum `logp`
                 across `new_states`, returned so the caller can apply
                 threshold pruning without a second pass.
        """
        trans_logp_trigram = self._trans_logp_trigram
        trans_logp_bigram = self._trans_logp_bigram

        new_states = {}
        new_states_get = new_states.get
        best_logp = float("-inf")

        for predecessor_key, (prefix_logp, _) in states.items():
            state_i_minus_2, state_i_minus_1 = predecessor_key

            # Per-history caches are constant across the candidate loop;
            # resolving them once and using ``_EMPTY_DICT`` for misses
            # lets the inner ``.get`` calls run without ``None`` branches.
            trigram_logp = trans_logp_trigram.get(predecessor_key, _EMPTY_DICT)
            bigram_logp = trans_logp_bigram.get(state_i_minus_1, _EMPTY_DICT)
            trigram_logp_get = trigram_logp.get
            bigram_logp_get = bigram_logp.get

            for state_i, log_emit, unigram_logp in candidate_tags:
                trans_logp = trigram_logp_get(state_i)
                if trans_logp is None:
                    trans_logp = bigram_logp_get(state_i)
                    if trans_logp is None:
                        # The candidate tuple carries the unigram fallback,
                        # including the floor for states never observed as
                        # unigrams, so the inner loop avoids a third dict lookup.
                        trans_logp = unigram_logp

                # Parens match the previous ``step_logp + prefix_logp``
                # order so the sum is bit-identical to the old decoder.
                path_logp = prefix_logp + (trans_logp + log_emit)
                next_state = (state_i_minus_1, state_i)

                # Once the last two states match, only the better prefix matters.
                # All future transitions depend on this key alone.
                prev_best = new_states_get(next_state)
                if prev_best is None or path_logp > prev_best[0]:
                    new_states[next_state] = (path_logp, state_i_minus_2)
                    if path_logp > best_logp:
                        best_logp = path_logp

        return new_states, best_logp

    @staticmethod
    def _pair_decoded(words, states):
        """Convert `_tagword` output into ``(word, tag)`` pairs by dropping
        the two BOS entries and the capitalization flag from each state."""
        return [(word, states[i + 2][0]) for i, word in enumerate(words)]


########################################
# helper function -- basic sentence tokenizer
########################################


def basic_sent_chop(data, raw=True):
    """
    Basic method for tokenizing input into sentences
    for this tagger:

    :param data: list of tokens (words or (word, tag) tuples)
    :type data: str or tuple(str, str)
    :param raw: boolean flag marking the input data
                as a list of words or a list of tagged words
    :type raw: bool
    :return: list of sentences
             sentences are a list of tokens
             tokens are the same as the input

    Function takes a list of tokens and separates the tokens into lists
    where each list represents a sentence fragment
    This function can separate both tagged and raw sequences into
    basic sentences.

    Sentence markers are the set of [.!?;]

    This is a simple method which enhances the performance of the TnT
    tagger. Better sentence tokenization will further enhance the results.
    """

    new_data = []
    curr_sent = []

    if raw:
        for word in data:
            curr_sent.append(word)
            if word in _SENT_MARKS:
                new_data.append(curr_sent)
                curr_sent = []
    else:
        for word, tag in data:
            curr_sent.append((word, tag))
            if word in _SENT_MARKS:
                new_data.append(curr_sent)
                curr_sent = []

    if curr_sent:
        new_data.append(curr_sent)

    return new_data


def demo():
    from nltk.corpus import brown

    sents = list(brown.tagged_sents())
    test = list(brown.sents())

    tagger = TnT()
    tagger.train(sents[200:1000])

    tagged_data = tagger.tagdata(test[100:120])

    for j in range(len(tagged_data)):
        s = tagged_data[j]
        t = sents[j + 100]
        for i in range(len(s)):
            print(s[i], "--", t[i])
        print()


def demo2():
    from nltk.corpus import treebank

    d = list(treebank.tagged_sents())

    t = TnT(N=1000, C=False)
    s = TnT(N=1000, C=True)
    t.train(d[(11) * 100 :])
    s.train(d[(11) * 100 :])

    for i in range(10):
        tacc = t.accuracy(d[i * 100 : ((i + 1) * 100)])
        tp_un = t.unknown / (t.known + t.unknown)
        tp_kn = t.known / (t.known + t.unknown)
        t.unknown = 0
        t.known = 0

        print("Capitalization off:")
        print("Accuracy:", tacc)
        print("Percentage known:", tp_kn)
        print("Percentage unknown:", tp_un)

        sacc = s.accuracy(d[i * 100 : ((i + 1) * 100)])
        sp_un = s.unknown / (s.known + s.unknown)
        sp_kn = s.known / (s.known + s.unknown)
        s.unknown = 0
        s.known = 0

        print("Capitalization on:")
        print("Accuracy:", sacc)
        print("Percentage known:", sp_kn)
        print("Percentage unknown:", sp_un)


def demo3():
    from nltk.corpus import brown, treebank

    d = list(treebank.tagged_sents())
    e = list(brown.tagged_sents())

    d = d[:1000]
    e = e[:1000]

    d10 = int(len(d) * 0.1)
    e10 = int(len(e) * 0.1)

    tallacc = 0
    sallacc = 0
    tknown = 0
    sknown = 0

    for i in range(10):
        t = TnT(N=1000, C=False)
        s = TnT(N=1000, C=False)

        dtest = d[(i * d10) : ((i + 1) * d10)]
        etest = e[(i * e10) : ((i + 1) * e10)]

        dtrain = d[: (i * d10)] + d[((i + 1) * d10) :]
        etrain = e[: (i * e10)] + e[((i + 1) * e10) :]

        t.train(dtrain)
        s.train(etrain)

        tacc = t.accuracy(dtest)
        tp_un = t.unknown / (t.known + t.unknown)
        tp_kn = t.known / (t.known + t.unknown)
        tknown += tp_kn
        t.unknown = 0
        t.known = 0

        sacc = s.accuracy(etest)
        sp_un = s.unknown / (s.known + s.unknown)
        sp_kn = s.known / (s.known + s.unknown)
        sknown += sp_kn
        s.unknown = 0
        s.known = 0

        tallacc += tacc
        sallacc += sacc

    print("brown   : overall accuracy:", 10 * tallacc)
    print("        : words known:", 10 * tknown)
    print("treebank: overall accuracy:", 10 * sallacc)
    print("        : words known:", 10 * sknown)
