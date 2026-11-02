# Sentiment API

FastAPI-сервис для анализа тональности текста на базе предобученной модели
`distilbert/distilbert-base-uncased-finetuned-sst-2-english` (Hugging Face).

## Запуск локально

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload