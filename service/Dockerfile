FROM python:3.7

COPY microservice/requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir /app
WORKDIR /app

COPY microservice/labels.json .
COPY models/baseline.pkl .
COPY microservice/server.py .

ENTRYPOINT ["python3", "server.py"]
