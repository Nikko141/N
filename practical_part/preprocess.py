

import re
import os
from collections import Counter


INPUT_PATH = "corpus.txt"
OUTPUT_PATH = "clean_corpus.txt"
STATS_PATH = "corpus_stats.txt"


KEEP_CHARS = "а-яёөүңһәіїґА-ЯЁӨҮҢҺӘІЇҐa-zA-Z"


def clean_text(text: str) -> str:
    text = text.lower()

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)


    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = re.sub(fr"[^{KEEP_CHARS}\s\.\!\?]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[\.!?])\s+", text)
    return [s.strip(" .!?") for s in raw if s.strip()]


def tokenize(sentence: str) -> list[str]:
    tokens = re.findall(fr"[{KEEP_CHARS}]+", sentence)
    return [t for t in tokens if len(t) > 1]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(here, INPUT_PATH)
    out_path = os.path.join(here, OUTPUT_PATH)
    stats_path = os.path.join(here, STATS_PATH)

    with open(in_path, "r", encoding="utf-8") as f:
        raw = f.read()

    raw_chars = len(raw)
    cleaned = clean_text(raw)
    sentences = split_sentences(cleaned)

    seen = set()
    tokenized_sentences: list[list[str]] = []
    for s in sentences:
        toks = tokenize(s)
        if len(toks) < 3:                    
            continue
        key = " ".join(toks)
        if key in seen:                      
            continue
        seen.add(key)
        tokenized_sentences.append(toks)

    with open(out_path, "w", encoding="utf-8") as f:
        for toks in tokenized_sentences:
            f.write(" ".join(toks) + "\n")


    all_tokens = [t for s in tokenized_sentences for t in s]
    vocab = Counter(all_tokens)

    stats = (
        f"Статистика корпуса\n"
        f"-------------------\n"
        f"Символов в исходном файле : {raw_chars:,}\n"
        f"Предложений после чистки  : {len(tokenized_sentences):,}\n"
        f"Всего токенов             : {len(all_tokens):,}\n"
        f"Уникальных токенов        : {len(vocab):,}\n"
        f"Средняя длина предложения : {len(all_tokens)/max(len(tokenized_sentences),1):.2f} токенов\n\n"
        f"Топ-20 самых частотных слов:\n"
    )
    for word, freq in vocab.most_common(20):
        stats += f"    {word:<20s} {freq}\n"

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(stats)

    print(stats)
    print(f"[OK] Чистый корпус сохранён в {out_path}")
    print(f"[OK] Статистика сохранена в  {stats_path}")


if __name__ == "__main__":
    main()
