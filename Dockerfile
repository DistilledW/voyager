# 使用 NVIDIA 官方 CUDA 12.1 基础镜像
FROM hub.rat.dev/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

WORKDIR /workspace/

# 安装基础依赖（包含colmap） 
RUN apt-get update && apt-get install -y \
    git \
    wget \
    cmake \
    ninja-build \
    build-essential \
    libboost-filesystem-dev \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    zip unzip curl vim 

# 安装 gcc/g++ 11.4.0
RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-11 g++-11 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 --slave /usr/bin/g++ g++ /usr/bin/g++-11

# 安装 Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/opt/miniconda/bin:$PATH"

# 安装 COLMAP(3.9.1)
RUN git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    git checkout 3.9.1 && \
    mkdir build && cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=86 && ninja && ninja install && \
    cd ../.. && rm -rf colmap


# 设置 Python 环境
RUN conda init bash 
RUN conda create -n 3dgs python=3.12 -y
RUN conda run -n 3dgs pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
RUN conda run -n 3dgs pip install plyfile tqdm joblib exif scikit-learn timm==0.4.5 opencv-python==4.9.0.80 gradio_imageslider gradio==4.29.0 matplotlib
# 设置工作目录为项目目录
WORKDIR /workspace

# 常用软件 
RUN apt-get update && apt-get install -y tree 
#     && apt-get install -y net-tools \
#     && apt-get install -y nginx \
#     && apt-get install -y redis-server \
#     && apt-get install -y python-pip python-dev build-essential \
#     && apt-get install -y mysql-server \
#     && apt install -y mysql-client \
#     && apt install -y libmysqlclient-dev
