# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

ARG LAZYGIT_VERSION=0.41.0

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        libsndfile1 \
        curl \
        ca-certificates \
        bash-completion \
        less \
        tree \
        vim \
        make \
    && rm -rf /var/lib/apt/lists/*

# Install lazygit manually to ensure availability on slim images
RUN curl -L "https://github.com/jesseduffield/lazygit/releases/download/v${LAZYGIT_VERSION}/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz" \
        | tar -xz -C /tmp \
    && mv /tmp/lazygit /usr/local/bin/lazygit \
    && chmod +x /usr/local/bin/lazygit

# Python dependencies (CPU-only PyTorch stack + data science basics)
RUN pip install --upgrade pip \
    && pip install \
        torch==2.2.0+cpu \
        torchvision==0.17.0+cpu \
        torchaudio==2.2.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install \
        numpy \
        scipy \
        pandas \
        soundfile \
        librosa \
        matplotlib \
        seaborn \
        scikit-learn \
        tqdm \
        jupyterlab \
        ipykernel

CMD ["bash"]
