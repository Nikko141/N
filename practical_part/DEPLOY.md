# Деплой на Streamlit Community Cloud

## Шаг 1. Залить проект на GitHub

1. Создай новый репозиторий на github.com (например, `kg-fasttext-thesis`). Можно публичный — бесплатный план Streamlit Cloud требует публичных репо.
2. Загрузи туда **всю папку `practical_part`** (через GitHub Desktop, web-интерфейс или git push). Минимум должны быть:

   ```
   gui.py
   preprocess.py
   train.py
   analyze.py
   morphology.py
   metrics.py
   requirements.txt
   runtime.txt
   corpus.txt
   clean_corpus.txt
   fasttext_model.bin                       (~21 МБ)
   fasttext_model.bin.wv.vectors_ngrams.npy (если присутствует)
   corpus_stats.txt
   train_log.txt
   report_similarity.txt
   report_morphology.txt
   report_metrics.txt
   ```

   Файл `fasttext_model.bin` весит около 21 МБ — GitHub принимает файлы до 100 МБ без проблем, обычный git push сработает. Если вдруг будет ошибка про размер — используй Git LFS.

## Шаг 2. Деплой на Streamlit Cloud

1. Зайди на https://share.streamlit.io и авторизуйся через GitHub.
2. Нажми **"New app"** (или "Deploy an app").
3. Заполни форму:
   - **Repository**: `твой-логин/kg-fasttext-thesis`
   - **Branch**: `main` (или `master`)
   - **Main file path**: **`gui.py`**   ← вот тот файл, что ты спрашивал
   - **App URL** (опционально): что-то вроде `kg-fasttext` — получится `kg-fasttext.streamlit.app`
4. Жми **"Deploy!"**. Через 2–5 минут установятся зависимости из `requirements.txt`, и приложение запустится.

## Что нужно подправить в коде для облака

GUI содержит кнопки «Предобработка / Обучить / …», которые запускают другие скрипты как отдельные процессы. На Streamlit Cloud это технически работает, но файловая система там **временная** — после рестарта контейнера новые файлы пропадут. Поэтому при деплое:

- модель `fasttext_model.bin` обязательно должна быть в репозитории — она прочитается прямо оттуда;
- все `.txt`-отчёты (`corpus_stats.txt`, `report_*.txt`) тоже желательно положить в репо — иначе на первом запуске вкладки «Обзор» / «Метрики» покажут сообщения «файл не найден», пока пользователь не нажмёт соответствующую кнопку.

Если хочется, чтобы кнопки в облаке не работали (а только демонстрация результата) — можно убрать кнопки из левой панели; основная функциональность (поиск соседей, морфоформы, метрики) от этого не пострадает.

## Резюме

| Локально (Windows) | Streamlit Cloud |
|---|---|
| `run_gui.bat` (двойной клик) | поле `Main file path` = **`gui.py`** |
| `streamlit run gui.py` | `requirements.txt` ставит зависимости автоматически |

Главный файл для деплоя — **`gui.py`**.
