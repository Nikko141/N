import os
import time
from gensim.models import FastText


INPUT_PATH = "clean_corpus.txt"
MODEL_PATH = "fasttext_model.bin"
LOG_PATH = "train_log.txt"

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 20
MIN_N = 3
MAX_N = 6
WORKERS = 4
BUCKET = 50_000


def load_corpus(path):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if toks:
                sentences.append(toks)
    return sentences


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(here, INPUT_PATH)
    model_path = os.path.join(here, MODEL_PATH)
    log_path = os.path.join(here, LOG_PATH)

    sentences = load_corpus(in_path)
    print("Loaded sentences:", len(sentences))

    start = time.time()
    model = FastText(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        epochs=EPOCHS,
        min_n=MIN_N,
        max_n=MAX_N,
        sg=1,
        bucket=BUCKET,
        workers=WORKERS,
        seed=42,
    )
    elapsed = time.time() - start

    model.save(model_path)
    vocab_size = len(model.wv)

    lines = [
        "FastText training log",
        "---------------------",
        f"corpus       : {in_path}",
        f"sentences    : {len(sentences)}",
        f"vocab size   : {vocab_size}",
        f"vector_size  : {VECTOR_SIZE}",
        f"window       : {WINDOW}",
        f"min_count    : {MIN_COUNT}",
        f"epochs       : {EPOCHS}",
        f"min_n/max_n  : {MIN_N}/{MAX_N}",
        f"sg           : 1 (skip-gram)",
        f"bucket       : {BUCKET}",
        f"train time   : {elapsed:.2f} s",
        f"model file   : {model_path}",
    ]
    info = "\n".join(lines)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(info)
    print(info)


if __name__ == "__main__":
    main()
