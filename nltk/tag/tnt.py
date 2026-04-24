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
from operator import itemgetter

from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.tag.api import TaggerI

# Returned in place of log2(p) when p is zero; log2(1e-300) ~= -996.58.
_LOG_FLOOR_2 = log2(1e-300)


class TnT(TaggerI):
    """
    TnT - Statistical POS tagger

    IMPORTANT NOTES:

    * DOES NOT AUTOMATICALLY DEAL WITH UNSEEN WORDS

      - It is possible to provide an untrained POS tagger to
        create tags for unknown words, see __init__ function

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

        # Prevents repeat train() calls from retraining the optional unk tagger.
        self._T = True

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
        Tags a single sentence

        :param tokens: list of words
        :type tokens: [string,]

        :return: [(word, tag),]

        Calls recursive function '_tagword'
        to produce a list of tags

        Associates the sequence of returned tags
        with the correct words in the input sequence

        returns a list of (word, tag) tuples
        """

        # seed BOS in the same (tag, C) form stored during training
        current_state = [([("BOS", False), ("BOS", False)], 0.0)]

        sent = list(tokens)

        tags = self._tagword(sent, current_state)

        res = []
        for i in range(len(sent)):
            # unpack and discard the C flags
            t, C = tags[i + 2]
            res.append((sent[i], t))

        return res

    def _tagword(self, sent, current_states):
        """
        :param sent : List of words remaining in the sentence
        :type sent  : [word,]
        :param current_states : List of possible tag combinations for
                                the sentence so far, and the log probability
                                associated with each tag combination
        :type current_states  : [([tag, ], logprob), ]

        Iteratively tags each word in the sentence, maintaining a beam of
        candidate tag sequences and their log probabilities. After the last
        word, scores the EOS transition and returns the best path.

        Uses formula specified above to calculate the probability
        of a particular tag.
        """

        # local bindings avoid repeated attribute lookups in the hot loop
        wd = self._wd
        bi = self._bi
        tri = self._tri
        uni = self._uni
        uni_N = self._uni_N
        l1, l2, l3 = self._l1, self._l2, self._l3
        log2_N = self._log2_N
        cap_on = self._C

        for word in sent:
            new_states = []
            C = cap_on and word[0].isupper()

            # if word is known, expand each state by each tag seen with it
            if word in wd:
                self.known += 1
                wd_word = wd[word]

                # Per-tag constants depend on the candidate tag, not on the history being extended.
                tag_info = []
                for t, wc in wd_word.items():
                    tC = (t, C)
                    uni_tC = uni[tC]
                    p_uni = (uni_tC / uni_N) if uni_N else 0
                    tag_info.append((tC, p_uni, log2(wc / uni_tC)))

                # Per-history hoists fetch the bi/tri dists and their sizes once per history iteration.
                for history, curr_logp in current_states:
                    bi_dist = bi[history[-1]]
                    tri_dist = tri[(history[-2], history[-1])]
                    bi_N = bi_dist.N()
                    tri_N = tri_dist.N()
                    inv_bi = (1.0 / bi_N) if bi_N else 0.0
                    inv_tri = (1.0 / tri_N) if tri_N else 0.0
                    for tC, p_uni, log_p_wd in tag_info:
                        p = (
                            l1 * p_uni
                            + l2 * bi_dist[tC] * inv_bi
                            + l3 * tri_dist[tC] * inv_tri
                        )
                        # Floor p so log2() is safe when all three n-gram components are zero for this (tag | context).
                        lp = (log2(p) if p > 1e-300 else _LOG_FLOOR_2) + log_p_wd
                        new_states.append((history + [tC], curr_logp + lp))

            else:
                self.unknown += 1

                # external tagger if provided, otherwise emit the literal tag "Unk"
                if self._unk is None:
                    tag = ("Unk", C)
                else:
                    [(_w, t)] = self._unk.tag([word])
                    tag = (t, C)

                for history, _ in current_states:
                    history.append(tag)
                new_states = current_states

            # Sort by log prob and threshold-prune
            # drop states worse than the best by more than log2(N).
            new_states.sort(reverse=True, key=itemgetter(1))
            cutoff = new_states[0][1] - log2_N
            current_states = [s for s in new_states if s[1] >= cutoff]

        # Score the EOS transition with the same deleted interpolation used for every tag, then return the best path.
        eos = ("EOS", False)
        p_uni_eos = (uni[eos] / uni_N) if uni_N else 0
        best_h = current_states[0][0]
        best_logp = float("-inf")
        for history, logp in current_states:
            bi_dist = bi[history[-1]]
            tri_dist = tri[(history[-2], history[-1])]
            bi_N = bi_dist.N()
            tri_N = tri_dist.N()
            p_bi = (bi_dist[eos] / bi_N) if bi_N else 0
            p_tri = (tri_dist[eos] / tri_N) if tri_N else 0
            p = l1 * p_uni_eos + l2 * p_bi + l3 * p_tri
            logp += log2(p) if p > 1e-300 else _LOG_FLOOR_2
            if logp > best_logp:
                best_h, best_logp = history, logp
        return best_h


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
    # Brants (2000) §2.1 lists [.!?;] as the sentence boundary set.
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
