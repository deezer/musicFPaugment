FROM python:3.9

ENV CUDA_VERSION 12.0
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
# NOTE: Avoid tensorboard CUPTI error
#       @see https://stackoverflow.com/questions/56860180/tensorflow-cuda-cupti-error-cupti-could-not-be-loaded-or-symbol-could-not-be
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib

ENV NVIDIA_VISIBLE_DEVICES 3
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=12.0 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<526"
ENV CUDNN_VERSION 8.9.3
ENV NCCL_VERSION 2.12.12

LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gnupg2 \
    curl \
    ca-certificates && \
    # https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get purge --autoremove -y curl && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    cuda-cudart-$CUDA_VERSION\
    cuda-compat-$CUDA_VERSION && \
    ln -s cuda-$CUDA_VERSION /usr/local/cuda && \
    echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
    apt-get install -y --no-install-recommends \
    cuda-libraries-$CUDA_VERSION \
    cuda-nvtx-$CUDA_VERSION\
    libcublas-dev-$CUDA_VERSION \
    libcudnn8 \
    libnccl2 && \
    apt-mark hold libnccl2 && \
    apt-mark hold libcudnn8 && \
    apt-mark hold libcublas-dev-$CUDA_VERSION && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y sox ffmpeg postgresql postgresql-contrib libgoogle-perftools4 && \
    mkdir -p /var/probes && touch /var/probes/ready && \
    chmod 777 /var/cache && \
    pip install --upgrade poetry


WORKDIR /workspace/

# COPY pyproject.toml ./
# COPY poetry.lock ./

# Install dependencies
# RUN poetry install --no-root

# COPY .flake8 .

EXPOSE 8503
