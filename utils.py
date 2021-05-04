import zipfile
from unicodedata import normalize
from typing import List
import string

import emojis

RUSSIAN_LETTERS = frozenset("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")


def read_text(path_to_zip, ignore_files: List[str] = None) -> dict:
    texts = dict()
    if ignore_files is None:
        ignore_files = []

    with zipfile.ZipFile(path_to_zip, "r") as zip:
        for item in zip.infolist():
            if not item.is_dir() and item.filename not in ignore_files:
                with zip.open(item, "r") as file:
                    text = file.read().decode("utf-8")
                    texts[item.filename] = preprocess_text(text)
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
