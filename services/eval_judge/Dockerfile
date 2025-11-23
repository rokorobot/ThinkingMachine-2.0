FROM python:3.9-slim

WORKDIR /app

COPY services/eval_judge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy libs
COPY libs /app/libs

# Copy service code
COPY services/eval_judge /app/services/eval_judge

# Set python path
ENV PYTHONPATH=/app

CMD ["python", "-m", "services.eval_judge.worker"]
