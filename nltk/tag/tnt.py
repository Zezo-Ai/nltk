# Natural Language Toolkit: TnT Tagger
#
# Copyright (C) 2001-2026 NLTK Project
# Author: Sam Huston <sjh900@gmail.com>
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Implementation of 'TnT - A Statisical Part of Speech Tagger'
by Thorsten Brants

https://aclanthology.org/A00-1031.pdf
"""

from math import log2

from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.tag.api import TaggerI

# Returned in place of log2(p) when p is zero; log2(1e-300) ~= -996.58.
_LOG_FLOOR_2 = log2(1e-300)


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
        Construct a TnT statistical tagger. Tagger must be trained
        before being used to tag input.

        :param unk: instance of a POS tagger, conforms to TaggerI
        :type  unk: TaggerI
        :param Trained: Indication that the POS tagger is trained or not
        :type  Trained: bool
        :param N: Beam search pruning threshold
        :type  N: int
        :param C: Capitalization flag
        :type  C: bool

        Initializer, creates frequency distributions to be used
        for tagging

        _lx values represent the portion of the tri/bi/uni taggers
        to be used to calculate the probability

        N is the beam search pruning threshold: after scoring, any state
        whose probability is worse than the best by more than a factor of
        N is discarded. A good value for this is 1000.

        C is a boolean value which specifies to use or
        not use the Capitalization of the word as additional
        information for tagging.
        NOTE: using capitalization may not increase the accuracy
        of the tagger
        """

        self._uni = FreqDist()
        self._bi = ConditionalFreqDist()
        self._tri = ConditionalFreqDist()
        self._wd = ConditionalFreqDist()
        self._l1 = 0.0
        self._l2 = 0.0
        self._l3 = 0.0
        self._N = N
        self._C = C
        self._T = Trained

        # cached after train() for the decode hot path
        self._uni_N = 0
        self._log2_N = 0.0

        # Suffix model state for unknown words. Two tries split by word
        # capitalization, plus the smoothing weight and tag priors used
        # by the recursion at decode time.
        self._suffix = {False: ConditionalFreqDist(), True: ConditionalFreqDist()}
        self._tag_priors = {}
        self._suffix_theta = 0.0

        self._unk = unk

        # statistical tools (ignore or delete me)
        self.unknown = 0
        self.known = 0

    def train(self, data):
        """
        Uses a set of tagged data to train the tagger.
        If an unknown word tagger is specified,
        it is trained on the same data.

        :param data: List of lists of (word, tag) tuples
        :type data: tuple(str)
        """

        # Ensure that local C flag is initialized before use
        C = False

        if self._unk is not None and not self._T:
            self._unk.train(data)

        for sent in data:
            history = [("BOS", False), ("BOS", False)]
            for w, t in sent:
                # if capitalization is requested,
                # and the word begins with a capital
                # set local flag C to True
                if self._C and w[0].isupper():
                    C = True

                self._wd[w][t] += 1
                self._uni[(t, C)] += 1
                self._bi[history[1]][(t, C)] += 1
                self._tri[tuple(history)][(t, C)] += 1

                history.append((t, C))
                history.pop(0)

                # set local flag C to false for the next word
                C = False

            # Record EOS as a pseudo-tag in the n-gram counts. Empty sentences are skipped
            # because their history[-1] is still BOS and would absorb the count.
            if sent:
                eos = ("EOS", False)
                self._uni[eos] += 1
                self._bi[history[-1]][eos] += 1
                self._tri[tuple(history)][eos] += 1

        # compute lambda values from the trained frequency distributions
        self._compute_lambda()

        # cache constants used in the decode hot path
        self._uni_N = self._uni.N()
        self._log2_N = log2(self._N)

        # Populate the suffix model used by the unknown-word path.
        self._build_suffix_model()

        # Prevents repeat train() calls from retraining the optional unk tagger.
        self._T = True

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
        # Tag priors over raw tags (excluding EOS), P(t) = f(t) / sum_t' f(t')
        tag_counts = {}
        for (t, _C), count in self._uni.items():
            if t == "EOS":
                continue
            tag_counts[t] = tag_counts.get(t, 0) + count
        total = sum(tag_counts.values())
        if total > 0:
            self._tag_priors = {t: c / total for t, c in tag_counts.items()}
        else:
            self._tag_priors = {}

        # theta is the standard deviation of the unconditioned tag priors.
        # This is the sample standard deviation per the paper's formula.
        priors = list(self._tag_priors.values())
        n = len(priors)
        if n > 1:
            mean = sum(priors) / n
            self._suffix_theta = (sum((p - mean) ** 2 for p in priors) / (n - 1)) ** 0.5
        else:
            self._suffix_theta = 0.0

        # Build the two suffix tries from infrequent lexicon words only.
        self._suffix = {False: ConditionalFreqDist(), True: ConditionalFreqDist()}
        for word in self._wd.conditions():
            if self._wd[word].N() > 10:
                continue
            cap = word[0].isupper()
            trie = self._suffix[cap]
            max_m = min(len(word), 10)
            for m in range(1, max_m + 1):
                suffix = word[-m:]
                for t, count in self._wd[word].items():
                    trie[suffix][t] += count

    def _compute_lambda(self):
        """
        creates lambda values based upon training data

        NOTE: no need to explicitly reference C,
        it is contained within the tag variable :: tag == (tag,C)

        for each tag trigram (t1, t2, t3)
        depending on the maximum value of
        - f(t1,t2,t3)-1 / f(t1,t2)-1
        - f(t2,t3)-1 / f(t2)-1
        - f(t3)-1 / N-1

        increment l3,l2, or l1 by f(t1,t2,t3)

        ISSUES -- Resolutions:
        if 2 values are equal, increment both lambda values
        by (f(t1,t2,t3) / 2)
        """

        # temporary lambda variables
        tl1 = 0.0
        tl2 = 0.0
        tl3 = 0.0

        # for each t1,t2 in system
        for history in self._tri.conditions():
            h1, h2 = history

            # for each t3 given t1,t2 in system
            # (NOTE: tag actually represents (tag,C))
            # However no effect within this function
            for tag in self._tri[history].keys():
                # safe_div provides a safe floating point division
                # it returns 0 if the denominator is 0
                c3 = self._safe_div(
                    (self._tri[history][tag] - 1), (self._tri[history].N() - 1)
                )
                c2 = self._safe_div((self._bi[h2][tag] - 1), (self._bi[h2].N() - 1))
                c1 = self._safe_div((self._uni[tag] - 1), (self._uni.N() - 1))

                # if c1 is the maximum value:
                if (c1 > c3) and (c1 > c2):
                    tl1 += self._tri[history][tag]

                # if c2 is the maximum value
                elif (c2 > c3) and (c2 > c1):
                    tl2 += self._tri[history][tag]

                # if c3 is the maximum value
                elif (c3 > c2) and (c3 > c1):
                    tl3 += self._tri[history][tag]

                # if c3, and c2 are equal and larger than c1
                elif (c3 == c2) and (c3 > c1):
                    tl2 += self._tri[history][tag] / 2.0
                    tl3 += self._tri[history][tag] / 2.0

                # if c1, and c2 are equal and larger than c3
                elif (c2 == c1) and (c1 > c3):
                    tl1 += self._tri[history][tag] / 2.0
                    tl2 += self._tri[history][tag] / 2.0

                # if c1, and c3 are equal and larger than c2
                elif (c1 == c3) and (c1 > c2):
                    tl1 += self._tri[history][tag] / 2.0
                    tl3 += self._tri[history][tag] / 2.0

                # if all three are equal
                elif (c1 == c2) and (c2 == c3):
                    tl1 += self._tri[history][tag] / 3.0
                    tl2 += self._tri[history][tag] / 3.0
                    tl3 += self._tri[history][tag] / 3.0

                # otherwise there might be a problem
                # eg: all values = 0
                else:
                    pass

        # Lambda normalisation:
        # ensures that l1+l2+l3 = 1
        self._l1 = tl1 / (tl1 + tl2 + tl3)
        self._l2 = tl2 / (tl1 + tl2 + tl3)
        self._l3 = tl3 / (tl1 + tl2 + tl3)

    def _safe_div(self, v1, v2):
        """
        Safe floating point division function, does not allow division by 0
        returns 0 if the denominator is 0
        """
        if v2 == 0:
            return 0
        else:
            return v1 / v2

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
        if not self._tag_priors:
            return {}

        cap = word[0].isupper()
        trie = self._suffix[cap]
        theta = self._suffix_theta

        # Find the longest known suffix by scanning from 10 down to 1.
        # If none is found, the loop below leaves P at the unigram prior.
        longest = 0
        for m in range(min(len(word), 10), 0, -1):
            if word[-m:] in trie:
                longest = m
                break

        # Successive abstraction from the prior up to the longest suffix.
        P = dict(self._tag_priors)
        denom = 1.0 + theta
        for i in range(1, longest + 1):
            suffix_dist = trie[word[-i:]]
            suffix_N = suffix_dist.N()
            if suffix_N == 0:
                continue
            inv_N = 1.0 / suffix_N
            P = {
                t: (suffix_dist[t] * inv_N + theta * P[t]) / denom
                for t in self._tag_priors
            }

        # Apply Bayesian inversion so the result ranks by P(suffix | t),
        # which is proportional to P(t | suffix) / P(t).
        return {t: P[t] / prior for t, prior in self._tag_priors.items() if prior > 0}

    def _expand_states(self, states, tag_info):
        """
        Take one Viterbi step. For every predecessor state we score
        each candidate `t_curr` from `tag_info` and accumulate the
        results into a new state dict keyed by `(t_prev, t_curr)`.
        When two predecessors land on the same key we keep the
        higher-scoring one and discard the other. The second-order
        Markov assumption means everything after this point depends
        only on the last two tags, so the discarded path can never
        beat the kept one.

        `tag_info` is a list of `(tC, p_uni, log_emit)` triples for
        the candidate `t_curr` values. Transitions use the same
        deleted interpolation as on the known-word path, floored
        before log2 to avoid blowing up on an all-zero context.

        :return: dict mapping `(t_prev, t_curr)` to `(logp, t_prev2)`.
                 `t_prev2` is the backpointer used to reconstruct the
                 best path after the final EOS step.
        """
        l1, l2, l3 = self._l1, self._l2, self._l3
        bi, tri = self._bi, self._tri
        new_states = {}
        for (t_prev2, t_prev), (curr_logp, _) in states.items():
            bi_dist = bi[t_prev]
            tri_dist = tri[(t_prev2, t_prev)]
            bi_N = bi_dist.N()
            tri_N = tri_dist.N()
            inv_bi = (1.0 / bi_N) if bi_N else 0.0
            inv_tri = (1.0 / tri_N) if tri_N else 0.0
            for tC, p_uni, log_emit in tag_info:
                p = l1 * p_uni + l2 * bi_dist[tC] * inv_bi + l3 * tri_dist[tC] * inv_tri
                lp = (log2(p) if p > 1e-300 else _LOG_FLOOR_2) + log_emit
                total_logp = curr_logp + lp
                new_key = (t_prev, tC)
                existing = new_states.get(new_key)
                if existing is None or total_logp > existing[0]:
                    new_states[new_key] = (total_logp, t_prev2)
        return new_states

    def tagdata(self, data):
        """
        Tags each sentence in a list of sentences

        :param data:list of list of words
        :type data: [[string,],]
        :return: list of list of (word, tag) tuples

        Invokes tag(sent) function for each sentence
        compiles the results into a list of tagged sentences
        each tagged sentence is a list of (word, tag) tuples
        """
        res = []
        for sent in data:
            res1 = self.tag(sent)
            res.append(res1)
        return res

    def tag(self, tokens):
        """
        Tag a single sentence. Delegates the actual decode to
        `_tagword`, then pairs each chosen tag with its input token.

        :param tokens: words to tag
        :type tokens: list[str]
        :return: list of `(word, tag)` tuples
        """
        sent = list(tokens)
        tags = self._tagword(sent)
        res = []
        for i in range(len(sent)):
            # unpack and discard the C flags
            t, C = tags[i + 2]
            res.append((sent[i], t))
        return res

    def _tagword(self, sent):
        """
        Tag a sentence by Viterbi decoding with second-order state
        merging. The per-word work (scoring, expansion, merging) lives
        in `_expand_states`. We threshold-prune after each word, then
        score the EOS transition over the surviving states and walk
        backpointers to reconstruct the best path.

        :param sent: words to tag
        :type sent: list[str]
        :return: history list `[BOS, BOS, t_0, ..., t_{T-1}]` of
                 `(tag, capitalization)` pairs.
        """
        BOS = ("BOS", False)
        EOS = ("EOS", False)

        # local bindings avoid repeated attribute lookups in the hot loop
        wd = self._wd
        bi = self._bi
        tri = self._tri
        uni = self._uni
        uni_N = self._uni_N
        l1, l2, l3 = self._l1, self._l2, self._l3
        log2_N = self._log2_N
        cap_on = self._C

        # Viterbi states per level. Each level maps (tag_{i-1}, tag_i)
        # to (logp, backpointer_tag_{i-2}).
        states = {(BOS, BOS): (0.0, BOS)}
        history = [states]

        for word in sent:
            C = cap_on and word[0].isupper()

            if word in wd:
                self.known += 1
                wd_word = wd[word]
                # Per-tag constants depend only on the candidate tag.
                tag_info = []
                for t, wc in wd_word.items():
                    tC = (t, C)
                    uni_tC = uni[tC]
                    p_uni = (uni_tC / uni_N) if uni_N else 0
                    tag_info.append((tC, p_uni, log2(wc / uni_tC)))
            else:
                self.unknown += 1
                if self._unk is not None:
                    # External unk tagger overrides the suffix model.
                    [(_w, t)] = self._unk.tag([word])
                    tC = (t, C)
                    p_uni = (uni[tC] / uni_N) if uni_N else 0
                    tag_info = [(tC, p_uni, 0.0)]
                else:
                    # Score this unknown word against every candidate
                    # tag using the Bayes-inverted suffix posterior.
                    tag_scores = self._unknown_tag_scores(word)
                    if not tag_scores:
                        # Fall back to a literal "Unk" tag when there is
                        # no prior to score against, for example when
                        # the tagger has not been trained.
                        tC = ("Unk", C)
                        p_uni = (uni[tC] / uni_N) if uni_N else 0
                        tag_info = [(tC, p_uni, 0.0)]
                    else:
                        tag_info = []
                        for t, score in tag_scores.items():
                            tC = (t, C)
                            p_uni = (uni[tC] / uni_N) if uni_N else 0
                            log_score = log2(score) if score > 1e-300 else _LOG_FLOOR_2
                            tag_info.append((tC, p_uni, log_score))

            new_states = self._expand_states(states, tag_info)

            # Threshold prune: drop states worse than the best by more than log2(N).
            best_logp = max(s[0] for s in new_states.values())
            cutoff = best_logp - log2_N
            new_states = {k: v for k, v in new_states.items() if v[0] >= cutoff}

            states = new_states
            history.append(states)

        # Score the EOS transition with the same deleted interpolation
        # used for every other tag, then pick the best final state.
        # `states` is non-empty by construction, so seeding the search
        # with its first key keeps the type concrete.
        p_uni_eos = (uni[EOS] / uni_N) if uni_N else 0
        best_final_key = next(iter(states))
        best_final_logp = float("-inf")
        for (t_prev2, t_prev), (curr_logp, _) in states.items():
            bi_dist = bi[t_prev]
            tri_dist = tri[(t_prev2, t_prev)]
            bi_N = bi_dist.N()
            tri_N = tri_dist.N()
            p_bi = (bi_dist[EOS] / bi_N) if bi_N else 0
            p_tri = (tri_dist[EOS] / tri_N) if tri_N else 0
            p = l1 * p_uni_eos + l2 * p_bi + l3 * p_tri
            eos_lp = log2(p) if p > 1e-300 else _LOG_FLOOR_2
            final_logp = curr_logp + eos_lp
            if final_logp > best_final_logp:
                best_final_logp = final_logp
                best_final_key = (t_prev2, t_prev)

        # Reconstruct the best path by walking backpointers, collecting
        # tags in reverse. At level L the key is (tag_{L-2}, tag_{L-1})
        # and the stored backpointer is tag_{L-3}, so each iteration
        # extends one tag toward the start of the sentence.
        T = len(sent)
        if T == 0:
            return [BOS, BOS]

        tags_reversed = [best_final_key[1]]
        if T >= 2:
            tags_reversed.append(best_final_key[0])
        current_key = best_final_key
        for level in range(T, 2, -1):
            backpointer = history[level][current_key][1]
            tags_reversed.append(backpointer)
            current_key = (backpointer, current_key[0])
        tags_reversed.reverse()
        return [BOS, BOS] + tags_reversed


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

    Sentence markers are the set of [,.!?]

    This is a simple method which enhances the performance of the TnT
    tagger. Better sentence tokenization will further enhance the results.
    """

    new_data = []
    curr_sent = []
    sent_mark = [".", "!", "?", ";"]

    if raw:
        for word in data:
            if word in sent_mark:
                curr_sent.append(word)
                new_data.append(curr_sent)
                curr_sent = []
            else:
                curr_sent.append(word)

    else:
        for word, tag in data:
            if word in sent_mark:
                curr_sent.append((word, tag))
                new_data.append(curr_sent)
                curr_sent = []
            else:
                curr_sent.append((word, tag))
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
        sknacc += sacc / tp_kn
        tallacc += tacc
        sallacc += sacc

        # print(i+1, (tacc / tp_kn), i+1, (sacc / tp_kn), i+1, tacc, i+1, sacc)

    print("brown: acc over words known:", 10 * tknacc)
    print("     : overall accuracy:", 10 * tallacc)
    print("     : words known:", 10 * tknown)
    print("treebank: acc over words known:", 10 * sknacc)
    print("        : overall accuracy:", 10 * sallacc)
    print("        : words known:", 10 * sknown)
