# Build from the repository root: docker build -f ui/Dockerfile -t sleepwalker-ui .
# An existing inference-capable image can be supplied as BASE_IMAGE.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}
WORKDIR /app
COPY ui/requirements.txt /app/ui/requirements.txt
COPY sleepwalker/server/requirements-plotly.txt /app/sleepwalker/server/requirements-plotly.txt
RUN python -m pip install --no-cache-dir -r ui/requirements.txt
COPY sleepwalker /app/sleepwalker
COPY ui /app/ui
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
EXPOSE 6006
ENTRYPOINT ["python", "-m", "ui"]
CMD ["--host", "0.0.0.0", "--port", "6006", "--storage", "/tmp/sleepwalker-ui"]
