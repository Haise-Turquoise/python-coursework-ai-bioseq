import numpy as np
from typing import List



class LanguageModel:
    """Seeded bigram language model for Assignment 1, Problem 3.

    The model scores exactly the tokens returned by vocab(). It does not expose
    an EOS token; the coding task decodes exactly n vocabulary tokens.
    """

    def __init__(self, index: int = 0):
        rng = np.random.RandomState(index)
        self._vocab = ['a', 'b', 'c', 'd', 'e']
        self._probs = rng.rand(len(self._vocab), len(self._vocab))
        self._probs = self._probs / np.sum(self._probs, axis=1)[:, None]
        self._prob_vocab = rng.rand(len(self._vocab))
        self._prob_vocab = self._prob_vocab / np.sum(self._prob_vocab)

    def vocab(self) -> List[str]:
        """Returns the vocabulary of the language model."""
        return self._vocab

    def prob(self, w: str, c: List[str]) -> float:
        """Returns the probability of word w given previous context c."""
        if len(c) == 0:
            return self._prob_vocab[self._vocab.index(w)]
        return self._probs[self._vocab.index(c[-1]), self._vocab.index(w)]
