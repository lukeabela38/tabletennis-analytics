FROM ubuntu:jammy

RUN apt update -y
RUN apt install python3-pip -y

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN rm requirements.txt

RUN python3 -m pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

WORKDIR /src

EXPOSE 8000

CMD ["/bin/bash"]