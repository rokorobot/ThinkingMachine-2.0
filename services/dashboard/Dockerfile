FROM python:3.9-slim

WORKDIR /app

COPY services/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy common module
COPY services/common /app/services/common

# Copy dashboard service code
COPY services/dashboard /app/services/dashboard

# Set python path
ENV PYTHONPATH=/app

CMD ["streamlit", "run", "services/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
