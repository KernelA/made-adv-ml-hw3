import pathlib
from typing import List
import itertools
import random
from multiprocessing import Pool

from tqdm import tqdm
import pandas as pd

from constants import RUSSIAN_LETTERS
from utils import read_text, preprocess_text, LetterPermutation
from language_model import count_ngram, NGramStat
from mcmc_decoding import mcmc_decryption

random.seed(20)


def accuracy(true_perm: LetterPermutation, pred_perm: LetterPermutation) -> float:
    true_mapping = set(true_perm.get_decode_mapping().items())
    pred_mapping = set(pred_perm.get_decode_mapping().items())

    return len(true_mapping & pred_mapping) / len(true_mapping)


def run_stat(encoded_text: str, ngram_stat: NGramStat, true_permutation: LetterPermutation,
             num_iter: int, scaling: float, num_runs: int, exp_num: int) -> pd.DataFrame:
    seed = random.getrandbits(32)
    generator = random.Random(seed)

    accuracy_data = []

    for i in range(num_runs):
        decryption_result = mcmc_decryption(encoded_text, ngram_stat, true_permutation.src_vocab(
        ), true_permutation.chiper_vocab(), num_iter, scaling=scaling, generator=generator, show_progress=False)
        accuracy_data.append({"exp_id": exp_num, "num_run": i, "n_gram": ngram_stat.num_ngram(),
                              "num_iter": num_iter, "scaling": scaling, "text_length": len(encoded_text),
                              "accuracy": accuracy(true_permutation, decryption_result.permutation)})

    return pd.DataFrame.from_records(accuracy_data)


def ngram_stat(ref_text: str, true_key: LetterPermutation, test_text: List[str], num_iter: List[int], scaling: List[float],
               ngram: int, num_runs: int) -> pd.DataFrame:
    ngram_stat = NGramStat(count_ngram(ref_text, ngram))

    encode_mapping = true_key.get_encode_mapping()
    encoded_texts = ["".join(map(lambda x: encode_mapping[x], text)) for text in test_text]

    num_params = len(num_iter) * len(scaling)

    params = list(zip(*itertools.product(num_iter, scaling)))

    final_metric = None

    for text_num, encoded_text in tqdm(enumerate(encoded_texts), total=len(test_text)):
        with Pool() as workers:
            results = workers.starmap(run_stat, zip(
                itertools.repeat(encoded_text, num_params),
                itertools.repeat(ngram_stat, num_params),
                itertools.repeat(true_key, num_params),
                params[0],
                params[1],
                itertools.repeat(num_runs, num_params),
                map(lambda x: num_params * text_num + x, range(num_params))
            ))

        if final_metric is None:
            final_metric = results[0]
            range_indices = range(1, len(results))
        else:
            range_indices = range(len(results))

        for i in range_indices:
            final_metric = final_metric.append(results[i])

    return final_metric


if __name__ == "__main__":
    path_to_zip = pathlib.Path("data", "corpora.zip")
    texts = read_text(path_to_zip, ignore_files=["WarAndPeaceEng.txt"])

    test_text = texts["AnnaKarenina.txt"].splitlines()
    reference_corpus = preprocess_text(texts["WarAndPeace.txt"])

    test_text = [preprocess_text(line) for line in test_text]

    test_text = list(filter(lambda x: len(x) > 0, test_text))

    original_vocab = set(RUSSIAN_LETTERS)
    original_vocab.add(" ")

    chiper_vocab = original_vocab.copy()

    true_key = LetterPermutation(original_vocab, chiper_vocab)
    perm = list(range(len(original_vocab)))
    random.shuffle(perm)
    true_key.permute(perm)

    test_pieces = []

    num_test_text = 10

    for i in range(num_test_text):
        num_lines = random.randint(1, 10)
        test_pieces.append(" ".join(random.sample(test_text, k=num_lines)))

    num_iters = [2_000, 5_000, 10_000]
    scaling = [0.5, 1, 10]
    num_runs = 5

    bigram_metrics = ngram_stat(reference_corpus, true_key, test_pieces,
                                num_iters, scaling, ngram=2, num_runs=num_runs)

    trigram_metrics = ngram_stat(reference_corpus, true_key, test_pieces,
                                 num_iters, scaling, ngram=3, num_runs=num_runs)

    frourgram_metrics = ngram_stat(reference_corpus, true_key, test_pieces,
                                   num_iters, scaling, ngram=4, num_runs=num_runs)

    bigram_metrics = bigram_metrics.append(trigram_metrics)
    bigram_metrics = bigram_metrics.append(frourgram_metrics)

    path_to_metrics = pathlib.Path("data", "metrics.csv.gz")

    bigram_metrics.to_csv(path_to_metrics, encoding="utf-8", index=False)
