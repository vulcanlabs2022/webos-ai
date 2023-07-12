FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y build-essential python3.10 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN python3.10 -m pip install faiss-cpu
RUN python3.10 -m pip install --no-deps sentence-transformers -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN python3.10 -m pip install -r requirements.txt --ignore-installed -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN python3.10 -m pip install InstructorEmbedding==1.0.1 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN python3.10 -m pip install scikit-learn==1.2.2 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN python3.10 -m pip install pypdf==3.9.1 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN python3.10 -m pip install docx2txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN python3.10 -m pip install llama-cpp-python==0.1.70 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
COPY . .
ENTRYPOINT [ "python3.10"]