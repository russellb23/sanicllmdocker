#!/bin/zsh

SERVER="./llm_server"
FAISS="./faiss_index"
TIKA="./tika_server"

# Stop and remove containers before starting
echo "Stoppping if containers are in use ..."
echo `docker container stop tika` 
echo `docker container rm tika` 
echo `docker container stop server` 
echo `docker container rm server` 
echo `docker container stop faiss` 
echo `docker container rm faiss`

echo "Stopping network if in use already ..."
echo `docker network rm llm-net`

# BUild and run docker for server
echo "Building server..."
cd $SERVER
echo `docker build -t llm-gpu-image . \
	&& docker run --name server -v /opt/llamacpp:/models \
	-p 8001:8001 -d llm-gpu-image`
cd ../

sleep 10

# Start TIKA server
echo "Starting tika server . . ."
# cd $TIKA
# echo `docker build -t tikaserver . \
# 	&& docker run -d --name tika -p 9998:9998 tikaserver`
echo `docker pull apache/tika:latest \
	&& docker run -d --name tika -p 9998:9998 apache/tika:latest`
# cd ../
sleep 10

# echo "Extracting PDF content ..."
# cd $PDF
# echo `docker build -t pdf-image . \
# 	&& docker run --name pdf \
# 	-v /home/russellb/developments/DeepLearningNLP/sanicllmdocker/:/app \
# 	-p 8003:8003 -d pdf-image`

# sleep 300

echo "Extracting pdf contents and Building faiss index ..."
cd $FAISS
echo `docker build -t sanic-image . \
	&& docker run --name faiss \
	-v /home/russellb/developments/DeepLearningNLP/sanicllmdocker/:/app \
	-p 8002:8002 -d sanic-image`

sleep 10

echo "Creating network ..."
echo `docker network create llm-net && \
	docker network connect llm-net server && \
	docker network connect llm-net faiss && \
	docker network connect llm-net tika`

sleep 10

echo "Sending curl request for testing..."
echo `curl -X POST http://localhost:8002/faiss \
	-H "Content-Type: application/json" \
	-d '{"query": "What are the secondary structural components of a SH3 domain?"}'`
