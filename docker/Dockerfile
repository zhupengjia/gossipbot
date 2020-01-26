FROM debian:buster-slim
MAINTAINER Pengjia Zhu <zhupengjia@gmail.com>

EXPOSE 5000/tcp

ENV WORKDIR /opt/chatbot
ENV BUILD_VERSION b1.0

RUN apt-get update &&\
    apt-get install -y --no-install-recommends python3-numpy python3-joblib python3-flask python3-pip python3-setuptools python3-regex python3-requests python3-boto3 cython3 python3 python3-sklearn&&\
    rm -rf /var/lib/apt/lists/*

# python packages
RUN pip3 install --no-cache torch transformers tqdm

WORKDIR $WORKDIR

ENV BOT_VERSION a0.1

ADD gossipbot gossipbot

ADD interact.py interact.py

CMD python3 interact.py -m ${MODEL_PATH} -c ${CONFIG_PATH} --backend ${BOT_BACKEND} --port 5000 --device "cpu"
#CMD tail -f /dev/null
