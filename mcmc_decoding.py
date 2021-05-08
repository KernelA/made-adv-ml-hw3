import random
import math
from collections import namedtuple
import warnings

import numpy as np
from tqdm import notebook

from language_model import NGramStat, count_ngram
from utils import LetterPermutation

MCMCDecoding = namedtuple("MCMCDecoding", ["log_score", "decoded_text", "permutation"])


def log_score_function(encoded_text: str, ngram_stat: NGramStat, mapping: dict):
    decoded_text = "".join(map(lambda x: mapping[x], encoded_text))
    new_ngram_counter = count_ngram(decoded_text, ngram_stat.num_ngram())

    log_score = 0

    for ngram in new_ngram_counter:
        log_score += new_ngram_counter.get(ngram, 1) * \
            math.log(max(1e-8, ngram_stat.get_ngram_count(ngram)))

    return log_score


def mcmc_decryption(encoded_text: str, src_ngram_stat: NGramStat, src_vocab: list,
                    chiper_vocab: list,
                    num_iters: int, scaling: float = 1,
                    generator: random.Random = None,
                    show_progress: bool = True) -> MCMCDecoding:

    if generator is None:
        internal_generator = random.Random(22)
        warnings.warn("Internal random generator with fixed seed will be used. Results will be all same. \
                      Please specify cutsom generator")
    else:
        internal_generator = generator

    init_symmetric_key = LetterPermutation(set(src_vocab), set(chiper_vocab))
    prev_permutations = np.arange(len(init_symmetric_key))

    proposal_permutations = prev_permutations.copy()

    new_decode_mapping = init_symmetric_key.get_decode_mapping()

    log_prev_score = log_score_function(encoded_text, src_ngram_stat, new_decode_mapping)

    if show_progress:
        progress = notebook.trange(num_iters, leave=True)
    else:
        progress = range(num_iters)

    best_log_score = log_prev_score
    best_perm = prev_permutations.copy()

    for iter in progress:
        if show_progress:
            progress.set_postfix_str(f"Log score: {float(log_prev_score):.2}")
        index1 = internal_generator.randint(0, len(prev_permutations) - 1)
        index2 = internal_generator.randint(0, len(prev_permutations) - 1)
        np.copyto(proposal_permutations, prev_permutations)
        proposal_permutations[index1], proposal_permutations[index2] = proposal_permutations[index2], proposal_permutations[index1]

        new_decode_mapping = init_symmetric_key.get_decode_mapping(proposal_permutations)

        log_score_proposal = log_score_function(encoded_text, src_ngram_stat, new_decode_mapping)

        log_rand_var = math.log(max(1e-8, internal_generator.random()))

        # accept new proposal
        if log_rand_var < scaling * (log_score_proposal - log_prev_score):
            log_prev_score = log_score_proposal
            np.copyto(prev_permutations, proposal_permutations)

        if best_log_score < log_score_proposal:
            best_log_score = log_score_proposal
            np.copyto(best_perm, proposal_permutations)

    init_symmetric_key.permute(best_perm)
    new_decode_mapping = init_symmetric_key.get_decode_mapping()

    decoding_res = MCMCDecoding(best_log_score, "".join(
        map(lambda x: new_decode_mapping[x], encoded_text)), init_symmetric_key)

    return decoding_res
