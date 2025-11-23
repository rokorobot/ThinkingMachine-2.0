FROM python:3.9-slim

WORKDIR /app

COPY services/meta_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy libs
COPY libs /app/libs

# Copy service code
COPY services/meta_agent /app/services/meta_agent

# Set python path
ENV PYTHONPATH=/app

CMD ["python", "-m", "services.meta_agent.worker"]
