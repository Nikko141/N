"""
morphology.py
-------------
Проверка морфологических вариантов одного термина.
Для каждой группы (например, курулуш, курулушчу, курулушчулар, курулуштук)
выводится попарная косинусная близость векторов.

FastText, благодаря символьным n-граммам (min_n=3, max_n=6),
должен выдавать высокую близость для слов одного корня, даже если
часть из них отсутствует в словаре корпуса.
"""

import os
import numpy as np
from gensim.models import FastText
from itertools import combinations


MODEL_PATH = "fasttext_model.bin"
REPORT_PATH = "report_morphology.txt"

# Группы морфоформ строительных терминов
MORPH_GROUPS = {
    "курулуш":   ["курулуш", "курулушу", "курулушчу", "курулушчулар",
                  "курулуштук", "куруу", "курулган", "курулуп"],
    "имарат":    ["имарат", "имараттын", "имаратка", "имаратта", "имараттар",
                  "имараттарды", "имараттагы"],
    "жол":       ["жол", "жолу", "жолго", "жолдор", "жолдордун",
                  "жолдогу", "жолдошум"],   # последнее — отвлекающее, должно быть дальше
    "долбоор":   ["долбоор", "долбоору", "долбоорду", "долбоорлору",
                  "долбоордун", "долбоорго"],
    "фундамент": ["фундамент", "фундаменти", "фундаментке", "фундаменттер",
                  "фундаментсиз"],
    "бетон":     ["бетон", "бетондун", "бетонго", "бетондуу", "бетондор"],
}


def in_vocab(model, w):
    return w in model.wv.key_to_index


def pairwise_similarities(model, words):
    """Возвращает список (w1, w2, cosine) для всех пар."""
    pairs = []
    for w1, w2 in combinations(words, 2):
        sim = float(model.wv.similarity(w1, w2))
        pairs.append((w1, w2, sim))
    return pairs


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    model = FastText.load(os.path.join(here, MODEL_PATH))

    lines = []
    lines.append("Морфологическая близость словоформ")
    lines.append("=" * 60)
    lines.append("")

    summary_table = []   
    for root, forms in MORPH_GROUPS.items():
        lines.append(f"Группа «{root}»  ({len(forms)} форм)")
        lines.append("-" * 50)

        for w in forms:
            mark = " " if in_vocab(model, w) else "*"
            lines.append(f"   {mark} {w}")
        lines.append("   (* — OOV, вектор синтезирован из n-грамм)")
        lines.append("")

        pairs = pairwise_similarities(model, forms)
        for w1, w2, s in pairs:
            lines.append(f"      sim({w1:<18s}, {w2:<18s}) = {s:.4f}")

        mean_sim = float(np.mean([s for _, _, s in pairs])) if pairs else 0.0
        lines.append("")
        lines.append(f"   >>> Средняя внутригрупповая близость: {mean_sim:.4f}")
        lines.append("")
        lines.append("")

        summary_table.append((root, len(forms), mean_sim))

    lines.append("Сводная таблица")
    lines.append("=" * 60)
    lines.append(f"   {'Корень':<14s} {'Форм':>6s} {'Средняя близость':>22s}")
    for root, n, sim in summary_table:
        lines.append(f"   {root:<14s} {n:>6d} {sim:>22.4f}")

    text = "\n".join(lines)
    out_path = os.path.join(here, REPORT_PATH)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n[OK] Отчёт сохранён в {out_path}")


if __name__ == "__main__":
    main()
