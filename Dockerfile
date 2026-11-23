# The SchemaGraph API (uvicorn schemagraph.api.main:app). Built and started by docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Dependencies first, so editing code does not reinstall them. The package source has to be present for
# pip to resolve it, so install editable and let the real source be copied over afterwards.
COPY pyproject.toml ./
COPY schemagraph ./schemagraph
RUN pip install -e ".[api]"

COPY config.yml ./config.yml
COPY docker/api-entrypoint.sh /usr/local/bin/api-entrypoint
RUN chmod +x /usr/local/bin/api-entrypoint \
    && useradd --system --create-home app \
    && mkdir -p data/cache \
    && chown -R app:app /app/data

USER app
EXPOSE 8000
ENTRYPOINT ["api-entrypoint"]
CMD ["uvicorn", "schemagraph.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
