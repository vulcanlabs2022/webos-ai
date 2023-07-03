# webos-ai
0. 分别在/data/目录下，从huggingface下载下面2个模型
   1. 拉取新的模型文件git clone https://huggingface.co/vulcanlabs/3b_v1.0
   2. 拉取新的模型文件git clone https://huggingface.co/hkunlp/instructor-base


[//]: # (1. 服务器CUDA版本需要为11.7)

[//]: # ()
[//]: # (2. 更新流程)

[//]: # (分别在/data/目录下，从huggingface下载下面1个模型)

[//]: # (   1. 删除/data/llama-7b-hf模型参数文件。更新为拉取新的模型文件git clone https://huggingface.co/yahma/llama-7b-hf)

[//]: # (   2. 删除/data/alpaca-lora-7b模型参数文件。更新为本仓库下alpaca-lora-7b目录)

[//]: # (   3. all-mpnet-base-v2保持不变)


[//]: # (   2. git clone https://huggingface.co/chainyo/alpaca-lora-7b)

[//]: # ()
[//]: # (   3.  git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2)


[//]: # (2. docker安装 )

[//]: # (https://docs.docker.com/engine/install/centos/)

[//]: # ()
[//]: # (3. docker GPU支持)

[//]: # ()
[//]: # (   参考官方文档 https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide)

[//]: # ()
[//]: # (   1. distribution=$&#40;. /etc/os-release;echo $ID$VERSION_ID&#41; \)

[//]: # (   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo)

[//]: # ()
[//]: # (   2. sudo yum clean expire-cache)

[//]: # ()
[//]: # (   3. sudo yum install -y nvidia-container-toolkit)

[//]: # ()
[//]: # (   4. sudo nvidia-ctk runtime configure --runtime=docker)

[//]: # ()
[//]: # (   5. sudo systemctl start docker)