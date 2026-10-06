FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    FASTEMBED_CACHE_PATH=/models \
    HOME=/tmp

WORKDIR /app

COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /models

COPY --chown=1000:0 src ./src
COPY --chown=1000:0 data/raw_pdfs ./data/raw_pdfs

# OpenShift runs the container with a random UID in group 0:
# make everything the app writes to group-writable.
RUN chgrp -R 0 /app /models && chmod -R g=u /app /models

USER 1000

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]