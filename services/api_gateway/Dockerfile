FROM python:3.9-slim

WORKDIR /app

COPY services/api_gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy libs
COPY libs /app/libs

# Copy service code
COPY services/api_gateway /app/services/api_gateway

# Set python path
ENV PYTHONPATH=/app

CMD ["uvicorn", "services.api_gateway.app:app", "--host", "0.0.0.0", "--port", "8080"]
