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

# Returned in place of log2(p) when p is zero; log2(1e-300) ~= -996.58.
_LOG_FLOOR_2 = log2(1e-300)

# Sentinel tags used at sentence boundaries.
_BOS = ("BOS", False)
_EOS = ("EOS", False)

_SENT_MARKS = (".", "!", "?", ";")


class TnT(TaggerI):
    """
    TnT - Statistical POS tagger

    IMPORTANT NOTES:

    * HANDLES UNSEEN WORDS VIA BRANTS' SUFFIX MODEL

      - Unknown words are tagged using the suffix-based distribution
        described in Brants (2000) section 2.3, smoothed by successive
        abstraction and scored via Bayesian inversion.
      - An external POS tagger may still be supplied via the ``unk``
        parameter to override the suffix model, see __init__ function.

    * SHOULD BE USED WITH SENTENCE-DELIMITED INPUT

      - Due to the nature of this tagger, it works best when
        trained over sentence delimited input.
      - However it still produces good results if the training
        data and testing data are separated on all punctuation eg: [,.?!]
      - Input for training is expected to be a list of sentences
        where each sentence is a list of (word, tag) tuples
      - Input for tag function is a single sentence
        Input for tagdata function is a list of sentences
        Output is of a similar form

    * Function provided to process text that is unsegmented

      - Please see basic_sent_chop()


    TnT uses a second order Markov model to produce tags for
    a sequence of input, specifically:

      argmax [Proj(P(t_i|t_i-1,t_i-2)P(w_i|t_i))] P(t_T+1 | t_T)

    IE: the maximum projection of a set of probabilities

    The set of possible tags for a given word is derived
    from the training data. It is the set of all tags
    that exact word has been assigned.

    To speed up and get more precision, we can use log addition
    to instead multiplication, specifically:

      argmax [Sigma(log(P(t_i|t_i-1,t_i-2))+log(P(w_i|t_i)))] +
             log(P(t_T+1|t_T))

    The probability of a tag for a given word is the linear
    interpolation of 3 markov models; a zero-order, first-order,
    and a second order model.

      P(t_i| t_i-1, t_i-2) = l1*P(t_i) + l2*P(t_i| t_i-1) +
                             l3*P(t_i| t_i-1, t_i-2)

    A beam search is used to limit the memory usage of the algorithm.
    The beam is controlled by a pruning threshold N after each step,
    states whose probability is worse than the best
    by more than a factor of N are discarded.

    It is possible to differentiate the tags which are assigned to
    capitalized words. However this does not result in a significant
    gain in the accuracy of the results.
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
                1000 is a good default.
        :type  N: int
        :param C: Capitalization flag. When True, tags are differentiated
                by whether the source word is capitalized. This rarely
                improves accuracy in practice.
        :type  C: bool
        """

        # Beam thresholding uses log2(N), so N must be finite and at least 1.
        # The explicit bool check matters because bool is a subclass of int.
        if (
            isinstance(N, bool)
            or not isinstance(N, (int, float))
            or not (1 <= N < float("inf"))
        ):
            raise ValueError(f"N must be a finite number >= 1, got {N!r}")

        self._tag_unigrams = FreqDist()
        self._tag_bigrams = ConditionalFreqDist()
        self._tag_trigrams = ConditionalFreqDist()
        self._word_tag_freqs = ConditionalFreqDist()
        self._lambda1 = 0.0
        self._lambda2 = 0.0
        self._lambda3 = 0.0
        self._beam_threshold = N
        self._use_capitalization = C
        self._unk_trained = Trained

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

        self._unk = unk

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

        if self._unk is not None and not self._unk_trained:
            self._unk.train(data)

        word_tag_freqs = self._word_tag_freqs
        tag_unigrams = self._tag_unigrams
        tag_bigrams = self._tag_bigrams
        tag_trigrams = self._tag_trigrams
        cap_on = self._use_capitalization

        for sent in data:
            state_i_minus_2 = _BOS
            state_i_minus_1 = _BOS
            for word, tag in sent:
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
            if sent:
                tag_unigrams[_EOS] += 1
                tag_bigrams[state_i_minus_1][_EOS] += 1
                tag_trigrams[(state_i_minus_2, state_i_minus_1)][_EOS] += 1

        self._compute_lambda()

        # This total intentionally includes EOS because the unigram model
        # and deleted interpolation are estimated over the same event stream.
        self._num_tag_tokens = self._tag_unigrams.N()
        self._log2_beam_threshold = log2(self._beam_threshold)

        self._build_suffix_model()

        self._unk_trained = True

    def _build_suffix_model(self):
        """
        Build the suffix-based language model TnT uses for unseen words.
        Populates two capitalization-split suffix tries, the smoothing
        weight `theta`, and a unigram tag prior. These are the three
        pieces that `_unknown_tag_scores` reads at decode time.

        The priors exclude the EOS pseudo-tag (a sequence marker, not a
        lexical tag) and sum across capitalization. Suffix statistics
        come only from lexicon words with count of 10 or fewer, the
        "infrequent words" threshold from Brants (2000) section 2.3,
        on the reasoning that frequent words tell us little about what
        unseen words might look like.
        """
        tag_counts = {}
        for (tag, _), count in self._tag_unigrams.items():
            if tag == "EOS":
                continue
            tag_counts[tag] = tag_counts.get(tag, 0) + count
        total = sum(tag_counts.values())
        if total > 0:
            self._tag_prior_probs = {tag: c / total for tag, c in tag_counts.items()}
        else:
            self._tag_prior_probs = {}

        # Theta controls how strongly the suffix recursion is pulled back
        # toward the less specific estimate at each abstraction step.
        priors = list(self._tag_prior_probs.values())
        n = len(priors)
        if n > 1:
            mean = sum(priors) / n
            self._theta = (sum((p - mean) ** 2 for p in priors) / (n - 1)) ** 0.5
        else:
            self._theta = 0.0

        # Each capitalization bucket stores every suffix length up to 10,
        # so decode can walk from the shortest matched ending to the longest.
        self._suffix_trie_by_cap = {
            False: ConditionalFreqDist(),
            True: ConditionalFreqDist(),
        }
        for word in self._word_tag_freqs.conditions():
            tag_freqs = self._word_tag_freqs[word]
            if tag_freqs.N() > 10 or not word:
                continue
            suffix_trie = self._suffix_trie_by_cap[word[0].isupper()]
            for m in range(1, min(len(word), 10) + 1):
                suffix_dist = suffix_trie[word[-m:]]
                for tag, count in tag_freqs.items():
                    suffix_dist[tag] += count

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

        lambda1_mass = 0.0
        lambda2_mass = 0.0
        lambda3_mass = 0.0

        tag_bigrams = self._tag_bigrams
        tag_unigrams = self._tag_unigrams
        unigram_n_minus_1 = tag_unigrams.N() - 1

        for state_i_minus_2, state_i_minus_1 in self._tag_trigrams.conditions():
            trigram_dist = self._tag_trigrams[(state_i_minus_2, state_i_minus_1)]
            trigram_n_minus_1 = trigram_dist.N() - 1
            bigram_dist = tag_bigrams[state_i_minus_1]
            bigram_n_minus_1 = bigram_dist.N() - 1

            for state_i, count in trigram_dist.items():
                # Subtracting one leaves the current event out, so each score
                # asks which model order would best predict this tag if this
                # occurrence were held out.
                c3 = (count - 1) / trigram_n_minus_1 if trigram_n_minus_1 else 0
                c2 = (
                    (bigram_dist[state_i] - 1) / bigram_n_minus_1
                    if bigram_n_minus_1
                    else 0
                )
                c1 = (
                    (tag_unigrams[state_i] - 1) / unigram_n_minus_1
                    if unigram_n_minus_1
                    else 0
                )

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
        theta = self._theta

        # The trie contains every suffix length up to the cutoff. Once the
        # longest matching suffix is found, all shorter suffixes are also
        # available for successive abstraction.
        longest = 0
        for m in range(min(len(word), 10), 0, -1):
            if word[-m:] in suffix_trie:
                longest = m
                break

        # No matched suffix means the estimate stays at the unigram prior.
        # After Bayes inversion, every tag then receives the same score.
        if longest == 0:
            return {tag: 1.0 for tag, prior in tag_priors.items() if prior > 0}

        # With theta equal to zero there is no smoothing, so the estimate is
        # just the empirical distribution of the longest matched suffix.
        if theta == 0.0:
            suffix_dist = suffix_trie[word[-longest:]]
            inv_suffix_N = 1.0 / suffix_dist.N()
            return {
                tag: (suffix_dist[tag] * inv_suffix_N) / prior
                for tag, prior in tag_priors.items()
                if prior > 0
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

    def _expand_states(self, states, candidate_tags):
        """
        Takes one Viterbi step. For every predecessor state we score
        each candidate `state_i` from `candidate_tags` and accumulate
        the results into a new state dict keyed by
        `(state_i_minus_1, state_i)`. When two predecessors land on
        the same key we keep the higher-scoring one and discard the
        other. The second-order Markov assumption means everything
        after this point depends only on the last two states, so the
        discarded path can never beat the kept one.

        `candidate_tags` is a sequence of `(state_i, p_state_i,
        log_emit)` triples. Transitions use the same deleted
        interpolation as on the known-word path, floored before log2
        to avoid blowing up on an all-zero context.

        :return: ``(new_states, best_logp)``. ``new_states`` maps
                 `(state_i_minus_1, state_i)` to
                 `(logp, state_i_minus_2)`, where `state_i_minus_2` is
                 the backpointer used to reconstruct the best path
                 after the final EOS step. `best_logp` is the maximum
                 `logp` across `new_states`, returned so the caller
                 can apply threshold pruning without a second pass.
        """
        lambda1, lambda2, lambda3 = self._lambda1, self._lambda2, self._lambda3
        tag_bigrams = self._tag_bigrams
        tag_trigrams = self._tag_trigrams
        new_states = {}
        best_logp = float("-inf")
        for (state_i_minus_2, state_i_minus_1), (prefix_logp, _) in states.items():
            bigram_dist = tag_bigrams[state_i_minus_1]
            trigram_dist = tag_trigrams[(state_i_minus_2, state_i_minus_1)]
            bigram_N = bigram_dist.N()
            trigram_N = trigram_dist.N()
            inv_bigram_N = (1.0 / bigram_N) if bigram_N else 0.0
            inv_trigram_N = (1.0 / trigram_N) if trigram_N else 0.0
            for state_i, p_state_i, log_emit in candidate_tags:
                p_state_i_given_history = (
                    lambda1 * p_state_i
                    + lambda2 * bigram_dist[state_i] * inv_bigram_N
                    + lambda3 * trigram_dist[state_i] * inv_trigram_N
                )
                step_logp = (
                    log2(p_state_i_given_history)
                    if p_state_i_given_history > 1e-300
                    else _LOG_FLOOR_2
                ) + log_emit
                path_logp = prefix_logp + step_logp
                if path_logp > best_logp:
                    best_logp = path_logp
                next_state = (state_i_minus_1, state_i)
                prev_best = new_states.get(next_state)
                # Once the last two states match, only the better prefix matters.
                # All future transitions depend on this key alone.
                if prev_best is None or path_logp > prev_best[0]:
                    new_states[next_state] = (path_logp, state_i_minus_2)
        return new_states, best_logp

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

        for tok in tokens:
            segment.append(tok)
            if tok in sent_marks:
                extend(pair_decoded(segment, tagword(segment)))
                segment.clear()

        if segment:
            extend(pair_decoded(segment, tagword(segment)))

        return tagged

    @staticmethod
    def _pair_decoded(words, states):
        """Convert `_tagword` output into ``(word, tag)`` pairs by dropping
        the two BOS entries and the capitalization flag from each state."""
        return [(word, states[i + 2][0]) for i, word in enumerate(words)]

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

        # Local bindings keep the hot loop on plain locals rather than
        # repeated attribute lookups.
        word_tag_freqs = self._word_tag_freqs
        tag_bigrams = self._tag_bigrams
        tag_trigrams = self._tag_trigrams
        tag_unigrams = self._tag_unigrams
        inv_num_tag_tokens = (
            (1.0 / self._num_tag_tokens) if self._num_tag_tokens else 0.0
        )
        lambda1, lambda2, lambda3 = self._lambda1, self._lambda2, self._lambda3
        log2_beam_threshold = self._log2_beam_threshold
        cap_on = self._use_capitalization
        unk = self._unk
        cache = self._candidate_tags_cache
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
                p_state_i = tag_unigrams[state_i] * inv_num_tag_tokens
                candidate_tags = ((state_i, p_state_i, 0.0),)
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
                            p_state_i = unigram_state_i * inv_num_tag_tokens
                            entries.append(
                                (
                                    state_i,
                                    p_state_i,
                                    log2(tag_count / unigram_state_i),
                                )
                            )
                        candidate_tags = tuple(entries)
                    else:
                        # Unknown words use the suffix model as their lexical
                        # score. Bayes inversion turns the suffix posterior into
                        # the emission-like quantity used by the decoder.
                        suffix_scores = self._unknown_tag_scores(word)

                        if not suffix_scores:
                            # An untrained tagger has no suffix priors, so the
                            # only safe fallback is a literal ``Unk`` state.
                            state_i = ("Unk", c_i)
                            p_state_i = tag_unigrams[state_i] * inv_num_tag_tokens
                            candidate_tags = ((state_i, p_state_i, 0.0),)
                        else:
                            entries = []
                            for tag, score in suffix_scores.items():
                                state_i = (tag, c_i)
                                p_state_i = tag_unigrams[state_i] * inv_num_tag_tokens
                                log_emit = (
                                    log2(score) if score > 1e-300 else _LOG_FLOOR_2
                                )
                                entries.append((state_i, p_state_i, log_emit))
                            candidate_tags = tuple(entries)

                    cache[cache_key] = candidate_tags

            new_states, best_logp = expand_states(states, candidate_tags)

            # Threshold pruning keeps the beam relative to the best current
            # path, which is the pruning rule described in the paper.
            cutoff = best_logp - log2_beam_threshold
            states = {k: v for k, v in new_states.items() if v[0] >= cutoff}
            state_history.append(states)

        # EOS is scored by the same interpolated transition model as every
        # other step. The best final key is the state pair that maximizes
        # the complete sentence score including the boundary transition.
        p_eos_unigram = tag_unigrams[_EOS] * inv_num_tag_tokens
        best_final_key = next(iter(states))
        best_final_logp = float("-inf")

        for (state_i_minus_2, state_i_minus_1), (prefix_logp, _) in states.items():
            bigram_dist = tag_bigrams[state_i_minus_1]
            trigram_dist = tag_trigrams[(state_i_minus_2, state_i_minus_1)]
            bigram_N = bigram_dist.N()
            trigram_N = trigram_dist.N()
            p_eos_bigram = (bigram_dist[_EOS] / bigram_N) if bigram_N else 0
            p_eos_trigram = (trigram_dist[_EOS] / trigram_N) if trigram_N else 0
            p_eos_given_history = (
                lambda1 * p_eos_unigram
                + lambda2 * p_eos_bigram
                + lambda3 * p_eos_trigram
            )
            eos_logp = (
                log2(p_eos_given_history)
                if p_eos_given_history > 1e-300
                else _LOG_FLOOR_2
            )
            final_logp = prefix_logp + eos_logp
            if final_logp > best_final_logp:
                best_final_logp = final_logp
                best_final_key = (state_i_minus_2, state_i_minus_1)

        # Walking the stored ``state_{i-2}`` backpointers recovers the best
        # full state sequence from the best final state pair.
        T = len(sent)
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
        print("Accuracy over known words:", (tacc / tp_kn))

        sacc = s.accuracy(d[i * 100 : ((i + 1) * 100)])
        sp_un = s.unknown / (s.known + s.unknown)
        sp_kn = s.known / (s.known + s.unknown)
        s.unknown = 0
        s.known = 0

        print("Capitalization on:")
        print("Accuracy:", sacc)
        print("Percentage known:", sp_kn)
        print("Percentage unknown:", sp_un)
        print("Accuracy over known words:", (sacc / sp_kn))


def demo3():
    from nltk.corpus import brown, treebank

    d = list(treebank.tagged_sents())
    e = list(brown.tagged_sents())

    d = d[:1000]
    e = e[:1000]

    d10 = int(len(d) * 0.1)
    e10 = int(len(e) * 0.1)

    tknacc = 0
    sknacc = 0
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

        tknacc += tacc / tp_kn
        sknacc += sacc / sp_kn
        tallacc += tacc
        sallacc += sacc

    print("brown: acc over words known:", 10 * tknacc)
    print("     : overall accuracy:", 10 * tallacc)
    print("     : words known:", 10 * tknown)
    print("treebank: acc over words known:", 10 * sknacc)
    print("        : overall accuracy:", 10 * sallacc)
    print("        : words known:", 10 * sknown)
