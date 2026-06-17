FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p checkpoints

CMD ["python", "train.py", "--iterations", "100", "--search-time", "1"]
