FROM python:3.11-slim

WORKDIR /app

# System deps for numpy/scipy/gensim
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY pyproject.toml .

RUN pip install --upgrade pip setuptools wheel

RUN pip install -r requirements.txt

COPY . .

RUN pip install -e .

EXPOSE 5000

CMD ["python", "backend/run.py"]