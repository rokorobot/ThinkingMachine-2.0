FROM python:3.9-slim

WORKDIR /app

COPY services/core/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy common module
COPY services/common /app/services/common

# Copy core service code
COPY services/core /app/services/core

# Set python path to include /app so imports work
ENV PYTHONPATH=/app

CMD ["uvicorn", "services.core.main:app", "--host", "0.0.0.0", "--port", "8000"]
