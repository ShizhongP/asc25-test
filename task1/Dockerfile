# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install numpy

CMD ["python", "worker.py"]