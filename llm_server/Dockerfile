FROM python:3.11-slim
# FROM sanicframework/sanic:3.10-latest

RUN apt-get update && apt-get install -y \
	build-essential \
	libopenblas-dev \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /home/russellb/developments/DeepLearningNLP/sanicllmdocker

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

CMD ["python3", "server.py"]

