FROM nvidia/cuda:11.2.2-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV MXNET_CUDNN_AUTOTUNE_DEFAULT=0
ENV MXNET_CUDNN_LIB_CHECKING=0

RUN apt-get -y update && apt-get install -y \
    software-properties-common \
    build-essential \
    checkinstall \
    cmake \
    make \
    pkg-config \
    yasm \
    git \
    vim \
    curl \
    wget \
    sudo \
    nano \
    apt-transport-https \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    dbus-x11 \
    iputils-ping \
    python3-dev \
    python3-pip \
    python3-setuptools \
    unzip

# some image/media dependencies
RUN apt-get -y update && apt-get install -y \
    libjpeg8-dev \
    libpng-dev \
    libtiff5-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-22-dev \
    libxine2-dev \
    libavfilter-dev  \
    libavutil-dev

RUN apt-get -y update && apt-get install -y ffmpeg  

# Build decord with Nvidia decoder
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/local/cuda/lib64/libnvcuvid.so
RUN git clone --recursive https://github.com/dmlc/decord
RUN cd decord && mkdir build && cd build && cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release && make -j2 && cd ../python && python3 setup.py install

# Set up files and install python libraries
RUN mkdir /face
ADD . /face
WORKDIR /face
RUN apt-get install -y libcairo2-dev
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install cython
RUN pip3 install -r requirements.txt

# Fallback - if pip install above does not work, use this unversioned command
# RUN pip3 install cython numpy decord tqdm scikit-image mxnet-cu112 cython insightface keras tensorflow numpy

# Build c extensions for CNN
WORKDIR /face/face/rcnn/cython
RUN python3 setup.py build_ext --inplace
WORKDIR /face/face/rcnn/pycocotools
RUN python3 setup.py build_ext --inplace

# Run analysis
WORKDIR /face
CMD bash pipeline.sh

