FROM python:3.13-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /app/staticfiles && chown app:app /app/staticfiles
COPY --chown=app:app manage.py ./
COPY --chown=app:app config/ config/
COPY --chown=app:app api/ api/
COPY --chown=app:app evaluations/cases.json evaluations/cases.json
COPY --chown=app:app sample_docs/ sample_docs/
COPY --chown=app:app deploy/gunicorn.conf.py deploy/gunicorn.conf.py
USER app
EXPOSE 8000
CMD ["gunicorn", "--config", "deploy/gunicorn.conf.py", "config.wsgi:application"]
