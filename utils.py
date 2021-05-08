import zipfile
from unicodedata import normalize
from typing import List, Sequence
import string

import emojis

from constants import RUSSIAN_LETTERS


class LetterPermutation:
    def __init__(self, src_vocab: frozenset, chiper_vocab: frozenset):
        self._src_vocab = sorted(src_vocab)
        self._chiper_vocab = sorted(chiper_vocab)
        assert len(self._src_vocab) == len(
            self._chiper_vocab), f"Number of characters must be same. But {len(self._src_vocab)} != {len(self._chiper_vocab)}"

    def get_decode_mapping(self, permutations: Sequence[int] = None) -> dict:
        if permutations is None:
            return {chiper_letter: src_letter for chiper_letter, src_letter in zip(self._chiper_vocab, self._src_vocab)}
        else:
            assert len(set(permutations)) == len(self._chiper_vocab)
            return {self._chiper_vocab[permutations[i]]: src_letter for i, src_letter in enumerate(self._src_vocab)}

    def get_encode_mapping(self, permutations: Sequence[int] = None) -> dict:
        return {src_letter: chiper_letter for chiper_letter, src_letter in self.get_decode_mapping(permutations).items()}

    def permute(self, permutations: Sequence[int]):
        assert len(set(permutations)) == len(
            self._chiper_vocab), "Permutation is not a valid by length"
        self._chiper_vocab = [self._chiper_vocab[permutations[i]] for i in range(len(permutations))]

    def __len__(self) -> int:
        return len(self._src_vocab)

    def src_vocab(self) -> List[str]:
        return self._src_vocab.copy()

    def chiper_vocab(self) -> List[str]:
        return self._chiper_vocab.copy()


def read_text(path_to_zip, ignore_files: List[str] = None, encoding: str = "utf-8") -> dict:
    texts = dict()
    if ignore_files is None:
        ignore_files = []

    with zipfile.ZipFile(path_to_zip, "r") as zip_file:
        for item in zip_file.infolist():
            if not item.is_dir() and item.filename not in ignore_files:
                with zip_file.open(item, "r") as file:
                    text = file.read().decode(encoding)
                    texts[item.filename] = text
    return texts


def get_emoji_vocab() -> set:
    vocab = set()
    for emojii__sym in filter(lambda x: len(x) == 1, emojis.db.get_emoji_aliases().values()):
        if emojis.db.get_emoji_by_code(emojii__sym).unicode_version == "6.0":
            vocab.add(emojii__sym)
    return vocab


def preprocess_text(text: str):
    new_text = normalize("NFKC", text)
    new_text = new_text.lower().replace("\n", " ")
    new_text = "".join(filter(
        lambda x: x in string.whitespace or x in RUSSIAN_LETTERS, new_text))
    return " ".join(new_text.split())
