from collections import Counter
import math


def count_ngram(text: str, n_gram: int) -> Counter:
    n_gram_counter = Counter()
    for start_index in range(len(text) - n_gram + 1):
        n_gram_text = text[start_index: start_index + n_gram]
        n_gram_counter[n_gram_text] += 1

    return n_gram_counter


class NGramStat:
    def __init__(self, counter: Counter):
        iterator = iter(counter.keys())
        first_key = next(iterator)
        self._ngram_count = len(first_key)
        if self._ngram_count != 1:
            n_min_one_gram = Counter()
            for ngram in counter:
                n_min_one_gram[ngram[:-1]] += counter[ngram]

            self._ngram_proba = {ngram: counter[ngram] / n_min_one_gram[ngram[:-1]]
                                 for ngram in counter.keys()}
        else:
            denum = math.fsum(counter.values())
            self._ngram_proba = {ngram: counter[ngram] / denum for ngram in counter.keys()}

        self._ngram_ranks = {rank: gram for rank, (gram, _) in enumerate(sorted(
            self._ngram_proba.items(), reverse=True, key=lambda x: x[1]))}

        self._ngram_counts = counter.copy()

    def get_ngram_by_rank(self, rank: int):
        return self._ngram_ranks[rank]

    def get_ngram_proba(self, ngram: str):
        return self._ngram_proba.get(ngram, 0)

    def get_ngram_count(self, ngram: str) -> int:
        return self._ngram_counts.get(ngram, 1)

    def get_ngram_log_proba(self, ngram: str):
        if ngram in self._ngram_proba:
            return math.log(self.get_ngram_proba(ngram))

        return float("-inf")

    def num_ngram(self):
        return self._ngram_count

    def __iter__(self):
        return iter(self._ngram_ranks.keys())
