import random
from typing import Tuple
import math
import copy

import numpy as np
from tqdm import notebook

from language_model import NGramStat, count_ngram
from utils import LetterPermutation


def log_score_function(encoded_text: str, ngram_stat: NGramStat, mapping: dict):
    decoded_text = "".join(map(lambda x: mapping[x], encoded_text))
    new_ngram_counter = count_ngram(decoded_text, ngram_stat.num_ngram())

    log_score = 0

    for ngram in new_ngram_counter:
        log_score += new_ngram_counter.get(ngram, 1) * \
            math.log(max(1e-8, ngram_stat.get_ngram_count(ngram)))

    return log_score


def mcmc_decryption(encoded_text: str, src_ngram_stat: NGramStat, src_vocab: list, chiper_vocab: list,
                    true_symmetric_key: LetterPermutation,
                    num_iters: int, scaling: float = 1) -> Tuple[str, LetterPermutation]:

    init_symmetric_key = LetterPermutation(set(src_vocab), set(chiper_vocab))
    prev_permutations = np.arange(len(init_symmetric_key))

    assert init_symmetric_key.get_encode_mapping() != true_symmetric_key.get_encode_mapping(), "Init key is same"

    proposal_permutations = prev_permutations.copy()

    new_decode_mapping = init_symmetric_key.get_decode_mapping()

    log_prev_score = log_score_function(encoded_text, src_ngram_stat, new_decode_mapping)

    progress = notebook.trange(num_iters, leave=True)

    for iter in progress:
        progress.set_description(f"Log score: {log_prev_score:.2}")
        index1 = random.randint(0, len(prev_permutations) - 1)
        index2 = random.randint(0, len(prev_permutations) - 1)
        np.copyto(proposal_permutations, prev_permutations)
        proposal_permutations[index1], proposal_permutations[index2] = proposal_permutations[index2], proposal_permutations[index1]

        new_decode_mapping = init_symmetric_key.get_decode_mapping(proposal_permutations)

        log_score_proposal = log_score_function(encoded_text, src_ngram_stat, new_decode_mapping)

        log_rand_var = math.log(max(1e-8, random.random()))

        # accept new proposal
        if log_rand_var < scaling * (log_score_proposal - log_prev_score):
            log_prev_score = log_score_proposal
            np.copyto(prev_permutations, proposal_permutations)

    init_symmetric_key.permute(prev_permutations)
    new_decode_mapping = init_symmetric_key.get_decode_mapping()

    return "".join(map(lambda x: new_decode_mapping[x], encoded_text)), init_symmetric_key
