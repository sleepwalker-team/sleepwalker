# Build from the repository root: docker build -f ui/Dockerfile -t sleepwalker-ui .
# For GPU inference, supply a CUDA-capable image as BASE_IMAGE.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}
WORKDIR /app
COPY pyproject.toml README.md LICENSE /app/
COPY LICENSES /app/LICENSES
COPY src /app/src
COPY ui /app/ui
RUN python -m pip install --no-cache-dir . -r ui/requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
EXPOSE 6006
ENTRYPOINT ["python", "-m", "ui"]
CMD ["--host", "0.0.0.0", "--port", "6006", "--storage", "/tmp/sleepwalker-ui"]
