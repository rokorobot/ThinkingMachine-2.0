FROM python:3.9-slim

WORKDIR /app

COPY services/monitor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy libs
COPY libs /app/libs

# Copy service code
COPY services/monitor /app/services/monitor

# Set python path
ENV PYTHONPATH=/app

CMD ["streamlit", "run", "services/monitor/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
