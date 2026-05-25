# Деплой на Streamlit Community Cloud

## Что обязательно должно быть в репозитории

```
gui.py                                   ← главный файл
preprocess.py train.py analyze.py morphology.py metrics.py
requirements.txt                         ← список пакетов (Python)
.python-version                          ← фиксирует Python 3.11
corpus.txt
clean_corpus.txt
fasttext_model.bin                       (~21 МБ)
fasttext_model.bin.wv.vectors_ngrams.npy (если есть)
corpus_stats.txt
report_similarity.txt
report_morphology.txt
report_metrics.txt
```

## Если предыдущий деплой упал с ошибкой

1. Зайти в Streamlit Cloud → найти приложение → **Manage app → Settings → Delete app**.
2. Дождаться, пока приложение полностью удалится (1–2 минуты).
3. Запушить новые `requirements.txt` и `.python-version` в репозиторий.
4. Снова **New app** и задеплоить.

(Можно и без удаления — нажать **Reboot app** в правом нижнем углу логов, тогда облако перечитает обновлённые файлы.)

## Деплой пошагово

1. https://share.streamlit.io → **New app**.
2. Поля:
   - **Repository**: твой-логин/название-репо
   - **Branch**: `main`
   - **Main file path**: `gui.py`
3. Жми **Deploy!**

При первом запуске Streamlit прочитает `.python-version`, поставит Python 3.11, затем установит пакеты из `requirements.txt`. Обычно занимает 3–5 минут.

## Почему предыдущий деплой не сработал

- В облако попал Python 3.14 (последний доступный). Под него `gensim 4.4.0` не собирается — несовместимые изменения в C-API CPython (`PyLongObject->ob_digit`, `_PyLong_AsByteArray`).
- В `requirements.txt` оказался полный дамп `pip freeze`, включая Django, asgiref, starlette и т.п. — лишние пакеты, конфликтующие версии.
- `runtime.txt` для Streamlit Cloud **не работает** (это формат Heroku). Правильный способ зафиксировать Python — файл `.python-version`.

## Текущая конфигурация

`.python-version`:
```
3.11
```

`requirements.txt` (только то, что реально используется):
```
streamlit==1.39.0
gensim==4.3.3
numpy==1.26.4
scipy==1.13.1
pandas==2.2.3
matplotlib==3.9.2
```

Эти версии проверенно совместимы между собой и с Python 3.11. Главное:
- `numpy<2.0` — gensim 4.3 не работает с numpy 2.x
- `gensim 4.3.3` — стабильная и совпадает с тем, на чём обучалась модель локально (важно при загрузке `fasttext_model.bin`)
- `scipy<1.14` — gensim 4.3 ломается на новых scipy

## Альтернатива: запретить кнопки в облаке

Кнопки «Предобработка / Обучить / …» в облаке работают, но запись файлов не сохранится между перезапусками контейнера. Если хочешь, чтобы облачная версия была только демонстрационной, можно убрать блок с кнопками из `gui.py` (строки `if st.sidebar.button(...)` — 5 штук). Локальная версия в Windows от этого не пострадает, если оставить локальный `gui.py` нетронутым.
