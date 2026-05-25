import os
import random
import numpy as np
from gensim.models import FastText
from itertools import combinations


MODEL_PATH = "fasttext_model.bin"
REPORT_PATH = "report_metrics.txt"

MORPH_GROUPS = {
    "курулуш":   ["курулуш", "курулушу", "курулушчу", "курулушчулар",
                  "курулуштук", "куруу", "курулган", "курулуп"],
    "имарат":    ["имарат", "имараттын", "имаратка", "имаратта",
                  "имараттар", "имараттарды", "имараттагы"],
    "жол":       ["жол", "жолу", "жолго", "жолдор", "жолдордун", "жолдогу"],
    "долбоор":   ["долбоор", "долбоору", "долбоорду", "долбоорлору",
                  "долбоордун", "долбоорго"],
    "фундамент": ["фундамент", "фундаменти", "фундаментке",
                  "фундаменттер", "фундаментсиз"],
    "бетон":     ["бетон", "бетондун", "бетонго", "бетондуу", "бетондор"],
}

ROOT_SUBSTRINGS = {
    "курулуш":   ["курулуш", "куруу", "курул"],
    "имарат":    ["имарат"],
    "жол":       ["жол"],     
    "долбоор":   ["долбоор"],
    "фундамент": ["фундамент"],
    "бетон":     ["бетон"],
}


def within_group_pairs(model, words):
    return [float(model.wv.similarity(w1, w2))
            for w1, w2 in combinations(words, 2)]


def between_group_pairs(model, groups):
    sims = []
    roots = list(groups.keys())
    for r1, r2 in combinations(roots, 2):
        for w1 in groups[r1]:
            for w2 in groups[r2]:
                sims.append(float(model.wv.similarity(w1, w2)))
    return sims


def anisotropy(model, n_samples=2000, seed=42):
    rng = random.Random(seed)
    keys = list(model.wv.key_to_index.keys())
    sims = []
    for _ in range(n_samples):
        a, b = rng.sample(keys, 2)
        sims.append(float(model.wv.similarity(a, b)))
    return float(np.mean(sims)), float(np.std(sims))


def neighbor_purity(model, term, root_substrings, k=10):
    sims = model.wv.most_similar(term, topn=k)
    matched = 0
    for w, _ in sims:
        if any(sub in w for sub in root_substrings):
            matched += 1
    return matched / k, [(w, s, any(sub in w for sub in root_substrings))
                          for w, s in sims]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    model = FastText.load(os.path.join(here, MODEL_PATH))

    lines = []
    lines.append("Авторские метрики качества модели FastText")
    lines.append("=" * 60)
    lines.append("")

    
    within_all = []
    within_per_group = {}
    for root, forms in MORPH_GROUPS.items():
        sims = within_group_pairs(model, forms)
        within_per_group[root] = float(np.mean(sims))
        within_all.extend(sims)

    between = between_group_pairs(model, MORPH_GROUPS)

    mean_within = float(np.mean(within_all))
    mean_between = float(np.mean(between))
    mcs = mean_within - mean_between
    mcr = mean_within / mean_between if mean_between != 0 else float("inf")

    lines.append("1. Morphological Cohesion Score (MCS)")
    lines.append("-" * 40)
    lines.append(f"   Внутригрупповая близость (mean): {mean_within:.4f}")
    lines.append(f"   Межгрупповая близость   (mean): {mean_between:.4f}")
    lines.append(f"   MCS = {mean_within:.4f} - {mean_between:.4f} = {mcs:.4f}")
    lines.append(f"   MCR = {mean_within:.4f} / {mean_between:.4f} = {mcr:.4f}")
    lines.append("")
    lines.append("   Близость внутри каждой группы:")
    for root, sim in within_per_group.items():
        lines.append(f"      {root:<14s} {sim:.4f}")
    lines.append("")

    a_mean, a_std = anisotropy(model)
    lines.append("2. Анизотропия векторного пространства")
    lines.append("-" * 40)
    lines.append(f"   Средняя cosine случайных пар: {a_mean:.4f} (± {a_std:.4f})")
    lines.append("   (значения, заметно превышающие 0, говорят о сжатости")
    lines.append("    пространства; ранжирование соседей остаётся осмысленным)")
    lines.append("")

    lines.append("3. Neighbor Purity @10  (доля топ-10 соседей с общим корнем)")
    lines.append("-" * 40)
    purities = []
    for term, subs in ROOT_SUBSTRINGS.items():
        purity, neighbors = neighbor_purity(model, term, subs, k=10)
        purities.append(purity)
        lines.append(f"   [{term}]   purity = {purity:.2f}")
        for w, s, hit in neighbors:
            tag = "+" if hit else " "
            lines.append(f"      {tag} {w:<25s} {s:.4f}")
        lines.append("")

    lines.append(f"   Средняя Purity@10 по всем терминам: {np.mean(purities):.3f}")
    lines.append("")

    lines.append("Итоговая таблица")
    lines.append("=" * 60)
    lines.append(f"   MCS                 : {mcs:.4f}")
    lines.append(f"   MCR                 : {mcr:.4f}")
    lines.append(f"   Anisotropy (mean)   : {a_mean:.4f}")
    lines.append(f"   Mean Purity@10      : {float(np.mean(purities)):.4f}")

    text = "\n".join(lines)
    with open(os.path.join(here, REPORT_PATH), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n[OK] Отчёт сохранён в {REPORT_PATH}")


if __name__ == "__main__":
    main()
