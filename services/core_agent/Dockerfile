FROM python:3.9-slim

WORKDIR /app

COPY services/core_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy libs
COPY libs /app/libs

# Copy service code
COPY services/core_agent /app/services/core_agent

# Set python path
ENV PYTHONPATH=/app

CMD ["uvicorn", "services.core_agent.app:app", "--host", "0.0.0.0", "--port", "8000"]
