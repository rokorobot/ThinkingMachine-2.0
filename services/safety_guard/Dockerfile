FROM python:3.9-slim

WORKDIR /app

COPY services/safety_guard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy libs
COPY libs /app/libs

# Copy service code
COPY services/safety_guard /app/services/safety_guard

# Set python path
ENV PYTHONPATH=/app

CMD ["uvicorn", "services.safety_guard.app:app", "--host", "0.0.0.0", "--port", "8001"]
