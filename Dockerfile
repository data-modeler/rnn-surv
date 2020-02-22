FROM tensorflow/tensorflow:latest-gpu-py3

ADD . /rnnsurv/

WORKDIR /rnnsurv/

RUN pip install --upgrade pip
RUN pip install -r docker-requirements.txt
RUN pip install ai-benchmark

WORKDIR /
