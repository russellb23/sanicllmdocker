FROM sanicframework/sanic:3.10-latest

WORKDIR /home/russellb/developments/DeepLearningNLP/sanicllmdocker

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8001

CMD ["python3", "server.py"]

