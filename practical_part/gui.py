import os
import sys
import io
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from itertools import combinations

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


st.set_page_config(
    page_title="FastText: кыргызский корпус",
    page_icon="📚",
    layout="wide",
)

st.title("📚 FastText на кыргызском корпусе строительной тематики")
st.caption("Демонстрация практической части дипломной работы")


st.sidebar.header("Пайплайн")

def run_script(name: str):
    with st.spinner(f"Выполняется {name} ..."):
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, name)],
            capture_output=True, text=True, encoding="utf-8",
        )
    if proc.returncode == 0:
        st.success(f"{name} — успешно")
    else:
        st.error(f"{name} — ошибка")
    with st.expander(f"Вывод {name}", expanded=False):
        st.code(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""), language="text")

if st.sidebar.button("1. Предобработка"):
    run_script("preprocess.py")
if st.sidebar.button("2. Обучить FastText"):
    run_script("train.py")
if st.sidebar.button("3. Анализ близости"):
    run_script("analyze.py")
if st.sidebar.button("4. Морфология"):
    run_script("morphology.py")
if st.sidebar.button("5. Метрики"):
    run_script("metrics.py")

st.sidebar.markdown("---")


@st.cache_resource
def load_model():
    from gensim.models import FastText
    path = os.path.join(HERE, "fasttext_model.bin")
    if not os.path.exists(path):
        return None
    return FastText.load(path)


model = load_model()

if model is None:
    st.warning("Модель fasttext_model.bin не найдена. Нажмите «2. Обучить FastText» в левой панели.")
    st.stop()


tab_overview, tab_search, tab_morph, tab_metrics, tab_corpus = st.tabs(
    ["📊 Обзор", "🔎 Поиск соседей", "🧬 Морфоформы", "📐 Метрики", "📃 Корпус"]
)

with tab_overview:
    st.subheader("Параметры модели")
    info = {
        "Размер словаря":          len(model.wv),
        "Размерность векторов":    model.vector_size,
        "Окно контекста":          model.window,
        "Минимальная частота":     model.min_count,
        "Эпох обучения":           model.epochs,
        "min_n / max_n (n-граммы)": f"{model.wv.min_n} / {model.wv.max_n}",
        "Архитектура":             "skip-gram" if model.sg == 1 else "CBOW",
    }
    st.table(pd.DataFrame(info.items(), columns=["Параметр", "Значение"]))

    st.subheader("Статистика корпуса")
    stats_path = os.path.join(HERE, "corpus_stats.txt")
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    else:
        st.info("corpus_stats.txt не найден — запустите preprocess.py.")


with tab_search:
    st.subheader("Топ-N семантически близких слов")
    col1, col2 = st.columns([3, 1])
    with col1:
        term = st.text_input("Введите слово:", value="курулуш")
    with col2:
        topn = st.slider("Топ-N", 5, 30, 10)

    if term:
        in_vocab = term in model.wv.key_to_index
        st.write(f"Статус: **{'в словаре' if in_vocab else 'OOV — вектор синтезирован из n-грамм'}**")

        try:
            sims = model.wv.most_similar(term, topn=topn)
            df = pd.DataFrame(sims, columns=["Слово", "Cosine"])
            df.index = df.index + 1

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.dataframe(df, use_container_width=True, height=420)

            with col_b:
                fig, ax = plt.subplots(figsize=(5, max(4, topn * 0.3)))
                ax.barh(df["Слово"][::-1], df["Cosine"][::-1], color="#4c72b0")
                ax.set_xlabel("Cosine similarity")
                ax.set_xlim(min(df["Cosine"]) * 0.99, 1.0)
                ax.set_title(f"Топ-{topn} соседей для «{term}»")
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Ошибка: {e}")

with tab_morph:
    st.subheader("Попарная косинусная близость словоформ")

    presets = {
        "курулуш":   "курулуш, курулушу, курулушчу, курулушчулар, курулуштук, куруу, курулган, курулуп",
        "имарат":    "имарат, имараттын, имаратка, имаратта, имараттар, имараттарды, имараттагы",
        "жол":       "жол, жолу, жолго, жолдор, жолдордун, жолдогу",
        "долбоор":   "долбоор, долбоору, долбоорду, долбоорлору, долбоордун, долбоорго",
        "фундамент": "фундамент, фундаменти, фундаментке, фундаменттер, фундаментсиз",
        "бетон":     "бетон, бетондун, бетонго, бетондуу, бетондор",
    }
    preset = st.selectbox("Готовая группа", list(presets.keys()))
    words_str = st.text_area(
        "Слова через запятую:",
        value=presets[preset],
        height=80,
    )
    words = [w.strip() for w in words_str.split(",") if w.strip()]

    if len(words) >= 2:
        n = len(words)
        mat = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    mat[i, j] = float(model.wv.similarity(words[i], words[j]))

        df = pd.DataFrame(mat, index=words, columns=words).round(4)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("Матрица близости:")
            st.dataframe(df.style.background_gradient(cmap="Blues", vmin=0.9, vmax=1.0),
                         use_container_width=True)

        with col2:
            fig, ax = plt.subplots(figsize=(max(5, n * 0.6), max(4, n * 0.5)))
            im = ax.imshow(mat, cmap="Blues", vmin=0.9, vmax=1.0)
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(words, rotation=45, ha="right")
            ax.set_yticklabels(words)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            color="black" if mat[i,j] < 0.98 else "white", fontsize=7)
            plt.colorbar(im, ax=ax, label="cosine")
            plt.tight_layout()
            st.pyplot(fig)

        sims = [float(model.wv.similarity(a, b)) for a, b in combinations(words, 2)]
        st.metric("Средняя внутригрупповая близость", f"{np.mean(sims):.4f}")


with tab_metrics:
    st.subheader("Авторские метрики качества")
    report_path = os.path.join(HERE, "report_metrics.txt")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            text = f.read()

        def grab(prefix):
            for line in text.splitlines():
                if prefix in line:
                    try:
                        return float(line.split(":")[1].strip().split()[0])
                    except Exception:
                        return None
            return None

        mcs = grab("MCS  ")
        mcr = grab("MCR  ")
        aniso = grab("Anisotropy (mean)")
        purity = grab("Mean Purity@10")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MCS",  f"{mcs:.4f}"  if mcs  is not None else "—")
        c2.metric("MCR",  f"{mcr:.4f}"  if mcr  is not None else "—")
        c3.metric("Anisotropy", f"{aniso:.4f}" if aniso is not None else "—")
        c4.metric("Purity@10",  f"{purity:.4f}" if purity is not None else "—")

        with st.expander("Полный отчёт по метрикам", expanded=False):
            st.code(text, language="text")
    else:
        st.info("report_metrics.txt не найден — запустите metrics.py.")

with tab_corpus:
    st.subheader("Очищенный корпус (clean_corpus.txt)")
    clean_path = os.path.join(HERE, "clean_corpus.txt")
    if os.path.exists(clean_path):
        with open(clean_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        st.write(f"Всего предложений: **{len(lines)}**")
        n = st.slider("Сколько предложений показать", 5, min(200, len(lines)), 30)
        st.text("\n".join(lines[:n]))
    else:
        st.info("clean_corpus.txt не найден — запустите preprocess.py.")
