FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0+cpu

RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    numpy==2.0.0 \
    pandas==2.2.2 \
    scikit-learn==1.4.2 \
    lightgbm==4.3.0 \
    joblib==1.4.2

RUN pip install --no-cache-dir --timeout 300 --retries 10 -r requirements.txt

COPY app/ ./app/
COPY artifacts/ ./artifacts/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]