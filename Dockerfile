# The SchemaGraph API (uvicorn schemagraph.api.main:app). Built and started by docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Dependencies first, so editing code does not reinstall them: this layer depends on pyproject.toml alone.
# pip needs a package to install, so an empty stand-in is installed (editable, which points at /app) and
# the real source is copied over it afterwards. Only a change to pyproject.toml reruns the install.
COPY pyproject.toml ./
RUN mkdir schemagraph && touch schemagraph/__init__.py && pip install -e ".[api]"
COPY schemagraph ./schemagraph

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
