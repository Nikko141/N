"""
analyze.py
----------
Семантическая близость: для выбранных строительных терминов
выводит топ-10 ближайших слов из словаря модели.

Особенность FastText: умеет давать вектор даже для слов, которых нет в словаре,
за счёт символьных n-грамм (subword embeddings). Поэтому в анализ
включаются и in-vocabulary термины (курулуш, имарат), и OOV-термины
(фундамент, бетон) — для последних модель строит вектор из n-грамм.
"""

import os
from gensim.models import FastText


MODEL_PATH = "fasttext_model.bin"
REPORT_PATH = "report_similarity.txt"

# Строительные термины для анализа
TERMS = [
    "курулуш",     # строительство
    "имарат",      # здание
    "жол",         # дорога
    "көпүрө",      # мост
    "долбоор",     # проект
    "жай",         # помещение / жильё
    "фундамент",   # OOV / редкое
    "бетон",       # OOV / редкое
    "цемент",      # OOV / редкое
    "арматура",    # OOV / редкое
    "тоннель",     # OOV
    "оңдоо",       # ремонт
]

TOP_N = 10


def in_vocab(model, word):
    """True, если слово реально есть в словаре, False — если вектор синтезируется из n-грамм."""
    return word in model.wv.key_to_index


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    model = FastText.load(os.path.join(here, MODEL_PATH))

    lines = []
    lines.append("Семантическая близость строительных терминов")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Размер словаря: {len(model.wv)}")
    lines.append("")

    for term in TERMS:
        vocab_status = "в словаре" if in_vocab(model, term) else "OOV (вектор синтезирован из n-грамм)"
        header = f"[{term}]   — {vocab_status}"
        lines.append(header)
        lines.append("-" * len(header))

        # most_similar работает и для OOV слов
        try:
            sims = model.wv.most_similar(term, topn=TOP_N)
            for word, score in sims:
                lines.append(f"    {word:<25s} {score:.4f}")
        except KeyError as e:
            lines.append(f"    ! Не удалось вычислить соседей: {e}")
        lines.append("")

    text = "\n".join(lines)
    out_path = os.path.join(here, REPORT_PATH)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n[OK] Отчёт сохранён в {out_path}")


if __name__ == "__main__":
    main()
