FROM python:3.9-slim

WORKDIR /app

COPY services/meta/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy common module
COPY services/common /app/services/common

# Copy meta service code
COPY services/meta /app/services/meta

# Set python path
ENV PYTHONPATH=/app

CMD ["python", "-m", "services.meta.worker"]
