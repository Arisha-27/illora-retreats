# Dockerfile (place in backend repo root next to unified_backend.py)
FROM python:3.11-slim

# create app directory
WORKDIR /app

# copy only needed files first (for caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy source
COPY . /app

# uvicorn command
# Note: use PORT env provided by Cloud Run (default 8080), override in code if needed
ENV PORT=8080
EXPOSE 8080

# Run with gunicorn or uvicorn. Here we use uvicorn as module
CMD ["uvicorn", "final_compilation:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
