FROM ubuntu:jammy

RUN apt update -y
RUN apt install python3-pip -y

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git curl -y

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN rm requirements.txt

RUN python3 -m pip install -q git+https://github.com/tensorflow/docs

WORKDIR /src

CMD ["/bin/bash"]