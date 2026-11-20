FROM python:3.13.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --require-hashes -r requirements.txt
RUN useradd --create-home --uid 10001 dashboard
COPY --chown=dashboard:dashboard app.py dashboard.py ./
COPY --chown=dashboard:dashboard src ./src
COPY --chown=dashboard:dashboard config ./config
COPY --chown=dashboard:dashboard .streamlit/config.toml ./.streamlit/config.toml
USER dashboard
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
