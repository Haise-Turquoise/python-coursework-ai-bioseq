"""Implementation of Problem 3, Assignment 1, CS486, Spring 2026.

Please replace 'pass  #(TODO) complete this function' with your own
implementation. Please do not change other parts of the code, including the
function signatures; however, feel free to add imports.
"""
import model
import heapq
import math


def decode(lm: model.LanguageModel, k: int = 2, n: int = 3) -> str:
    """Lowest-cost-first search with frontier size limit.

    Args:
        lm: A language model.
        k: the frontier size limit. Default is 2.
        n: the length of the desired sentence. Default is 3.

    Returns:
        The decoded text, where words are separated by a single space. The output
        must contain exactly n tokens from lm.vocab(); do not append EOS.

    Hints:
        1. lm.vocab() gives you the vocabulary.
        2. lm.prob(w: str, c: List[str]) gives you the probability of word
           w given previous context c.
    """
    vocab = lm.vocab()
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    pathPool = [(0.0, (), [])]  # (acc_cost, tie_key, word sequence)

    while pathPool:
        acc_cost, tie_key, words = heapq.heappop(pathPool)

        if len(words) == n:
            return ' '.join(words)

        for w in vocab:
            p = lm.prob(w, words)
            if p > 0:
                new_key = tie_key + (word_to_idx[w],)
                heapq.heappush(pathPool, (acc_cost + (-math.log(p)), new_key, words + [w]))

        # keep top-k lowest cost
        if len(pathPool) > k:
            pathPool = heapq.nsmallest(k, pathPool)
            heapq.heapify(pathPool)  # trim to k

    return ''


if __name__ == '__main__':
    m = model.LanguageModel(0)
    if decode(m, 2, 3) == 'c d e' and decode(m, 1, 3) == 'c d e':
        print('success!')
    else:
        print('There might be something wrong with your implementation; please double check.')
