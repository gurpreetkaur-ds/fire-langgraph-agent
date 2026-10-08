FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# SQLite lives in /srv/app/data (mount a volume there); use EVAL_DATABASE_URL for Postgres etc.
RUN useradd --create-home --uid 10001 appuser && mkdir -p data && chown -R appuser /srv/app
USER appuser

EXPOSE 8000
# Migrations run automatically at startup (EVAL_AUTO_MIGRATE=true); `alembic upgrade head` also works standalone.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
