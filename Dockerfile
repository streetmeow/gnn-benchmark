# Dockerfile

# 1. 베이스 이미지: 시스템 드라이버(12.4)와 호환되는 CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 2. 시스템 패키지 및 Python 3.10 설치
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    vim \
    htop \
    numactl \
    && rm -rf /var/lib/apt/lists/*

# 3. python3 -> python 심볼릭 링크
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 4. 작업 디렉토리 설정
WORKDIR /app

# 5. [핵심] 로컬 패키지 간섭 방지 (중요!)
ENV PYTHONNOUSERSITE=1

# --- 여기서부터 라이브러리 설치 ---

# 6. NumPy 고정 (2.0 이슈 방지)
RUN pip install --no-cache-dir "numpy==1.26.4"

# 7. PyTorch 2.2.0 + CUDA 12.1 설치
RUN pip install --no-cache-dir \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    torchaudio==2.2.0+cu121 \
    torchmetrics \
    --index-url https://download.pytorch.org/whl/cu121

# 8. PyG 확장 모듈 설치 (PyTorch 버전과 완벽 일치)
RUN pip install --no-cache-dir \
    pyg_lib torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# 9. 나머지 Requirements 설치
# (requirements.txt 파일을 복사하지 않고 직접 명시하여 이미지에 고정)
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    pyyaml==6.0.1 \
    tqdm==4.66.4 \
    wandb==0.17.0 \
    ogb==1.3.6 \
    torch_geometric==2.5.2 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    scikit-learn \
    matplotlib \
    seaborn \
    umap-learn \
    thop \
    torchinfo

# 10. 기본 실행 명령 (쉘)
CMD ["/bin/bash"]