# webos-ai
0. 购买服务器CUDA版本需要为11.7

1.模型准备
分别在/data/目录下，从huggingface下载下面3个模型
1.1 git clone https://huggingface.co/decapoda-research/llama-7b-hf 

1.2 git clone https://huggingface.co/chainyo/alpaca-lora-7b

1.3 git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2

2.docker安装 
https://docs.docker.com/engine/install/centos/

3.docker GPU支持

3.1 distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

3.2 sudo yum clean expire-cache

3.3 sudo yum install -y nvidia-container-toolkit

3.4 sudo nvidia-ctk runtime configure --runtime=docker

3.5 sudo systemctl start docker