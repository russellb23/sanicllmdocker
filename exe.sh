#!/bin/zsh

SERVER="./llm_server"
FAISS="./faissidx"

# BUild and run docker for server
echo "Building server..."
cd $SERVER
echo `docker build -t llm-gpu-image . \
	&& docker run --name server -v /opt/llamacpp:/models \
	-p 8001:8001 -d llm-gpu-image`

sleep 30

echo "Building faiss index ..."
cd ../$FAISS
echo `docker build -t sanic-image . \
	&& docker run --name faiss \
	-v /home/russellb/developments/DeepLearningNLP/sanicllmdocker:/app \
	-p 8002:8002 -d sanic-image`

sleep 180

echo "Creating network ..."
echo `docker network create llm-net && \
	docker network connect llm-net server && \
	docker network connect llm-net faiss`

echo "Sending curl request for testing..."
echo `curl -X POST http://localhost:8002/faiss \
	-H "Content-Type: application/json" \
	-d '{"query": "What are the secondary structural components of a SH3 domain?"}'`
