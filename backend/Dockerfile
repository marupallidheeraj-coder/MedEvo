# Works on Render (Docker) and Hugging Face Spaces (Docker, set PORT=7860)
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser
ENV PORT=8000 PYTHONUNBUFFERED=1
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'], timeout=3)" || exit 1
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
