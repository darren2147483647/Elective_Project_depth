# 使用支援 CUDA 12.4 的 PyTorch 官方映像
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# 設定非互動模式，避免安裝時卡住
ENV DEBIAN_FRONTEND=noninteractive

# 設定工作目錄
WORKDIR /app

# 安裝 OpenCV 必要的系統函式庫 (這是 Linux 環境跑 cv2 必備的)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 複製需求到容器中
COPY requirements.txt .

# 安裝 Python 套件
RUN pip install --no-cache-dir --upgrade -r requirements.txt