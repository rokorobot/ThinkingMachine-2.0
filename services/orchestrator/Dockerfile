FROM python:3.9-slim

WORKDIR /app

COPY services/orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy libs
COPY libs /app/libs

# Copy service code
COPY services/orchestrator /app/services/orchestrator

# Set python path
ENV PYTHONPATH=/app

CMD ["python", "-m", "services.orchestrator.worker"]
