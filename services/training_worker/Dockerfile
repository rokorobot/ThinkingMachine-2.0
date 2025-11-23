FROM python:3.9-slim

WORKDIR /app

COPY services/training_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy libs
COPY libs /app/libs

# Copy service code
COPY services/training_worker /app/services/training_worker

# Set python path
ENV PYTHONPATH=/app

CMD ["python", "-m", "services.training_worker.worker"]
