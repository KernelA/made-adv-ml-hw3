from typing import List
import math
from collections import namedtuple

import networkx as nx
from tqdm import notebook

from language_model import NGramStat, count_ngram
from utils import get_emoji_vocab, preprocess_text, LetterPermutation

BeamSearchHyphotesis = namedtuple("BeamSearchHyphotesis", ["phrase", "log_likelihood"])


def build_graph(ngrams: List[str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    n_gram = len(ngrams[0])
    for num_gram in range(len(ngrams)):
        gram = ngrams[num_gram]
        assert len(gram) == n_gram

        for j in range(len(gram)):
            text_pos = num_gram + j
            if j + 1 < len(gram):
                graph.add_edge((text_pos, gram[j]), (text_pos + 1, gram[j + 1]))
            shift = 0
            for next_gram in range(num_gram + 1, min(len(ngrams), num_gram + j + 1 + 1)):
                assert j - shift >= 0
                graph.add_edge((text_pos, gram[j]), (text_pos + 1, ngrams[next_gram]
                                                     [j - shift]))
                shift += 1
            shift = 0
            num_prev_ngrams = len(gram) - j - 2

            prev_ngram = num_gram - 1
            for shift in range(num_prev_ngrams):
                if prev_ngram < 0:
                    break
                graph.add_edge((text_pos, gram[j]), (text_pos + 1, ngrams[prev_ngram]
                                                     [j + 2 + shift]))
                prev_ngram -= 1

    return graph


def decode_ngrams(encoded_text: str, source_freq: NGramStat, n_gram: int) -> List[str]:
    target_freq = NGramStat(count_ngram(encoded_text, n_gram))
    inv_mapping = dict()
    ngrams = []

    for target_rank in target_freq:
        inv_mapping[target_freq.get_ngram_by_rank(
            target_rank)] = source_freq.get_ngram_by_rank(target_rank)

    for i in range(len(encoded_text) - n_gram + 1):
        ngrams.append(inv_mapping[encoded_text[i: i + n_gram]])

    return ngrams


def beam_search(graph: nx.DiGraph, width: int, ngram_stat: NGramStat) -> List[BeamSearchHyphotesis]:
    end_index = None
    start_node = None
    for node in graph.nodes:
        if not tuple(graph.predecessors(node)):
            start_node = node
            break

    for node in nx.dfs_preorder_nodes(graph, start_node):
        end_index = node[0]

    progress = notebook.tqdm(total=end_index, desc="Beam search", leave=True)

    hyphotesis = []
    nodes = set([(start_node,)])
    hyphotesis = {(start_node,): 0}

    while True:
        update = False
        while nodes:
            current_path = nodes.pop()
            last_node = current_path[-1]

            if not update:
                progress.update(n=1)
                update = True

            has_neigh = False

            for neigh_node in graph.neighbors(last_node):
                has_neigh = True
                log_likelihood = hyphotesis[current_path]

                new_path = current_path + (neigh_node,)
                if len(new_path) % ngram_stat.num_ngram() == 0:
                    ngram = "".join(map(lambda x: x[1], new_path[-ngram_stat.num_ngram():]))
                    new_log_likelihood = ngram_stat.get_ngram_log_proba(ngram)
                    if math.isfinite(new_log_likelihood):
                        hyphotesis[current_path + (neigh_node,)
                                   ] = log_likelihood + new_log_likelihood
                else:
                    hyphotesis[current_path + (neigh_node,)] = log_likelihood

            if not has_neigh:
                sorted_hyp = sorted(hyphotesis.items(), key=lambda x: x[1], reverse=True)
                result = []
                for text, log_likelihood in sorted_hyp:
                    union_text = map(lambda x: x[1], text)
                    union_text = "".join(union_text)
                    result.append(BeamSearchHyphotesis(union_text, log_likelihood))
                progress.close()
                return result

            hyphotesis.pop(current_path)

        if not hyphotesis:
            raise ValueError("Empty list of hyphotesis")

        if len(hyphotesis) > width:
            hyphotesis = dict(sorted(hyphotesis.items(), key=lambda x: x[1], reverse=True)[:width])

        for path in hyphotesis:
            nodes.add(path)


def decode_text(encoded_text: str, source_freq: NGramStat, n_gram: int, beam_search_width: int):
    decoded_ngrams = decode_ngrams(encoded_text, source_freq, n_gram)
    parse_graph = build_graph(decoded_ngrams)

    return beam_search(parse_graph, width=beam_search_width, ngram_stat=source_freq)


def encode_text(text: str, permutation: LetterPermutation, n_gram: int):
    encode_mapping = permutation.get_encode_mapping()
    print("Encoded text length:", len(preprocess_text(text)))
    processed_text = preprocess_text(text)
    encoded_text = "".join(map(lambda x: encode_mapping[x], processed_text))
    reminder = len(encoded_text) % n_gram
    if reminder != 0:
        encoded_text += "".join([encode_mapping[" "]] * reminder)
    return encoded_text, processed_text
