# Dockerfile pre termovízny spracovací systém
# Použitie: docker build -t thermal-ai . && docker run -it --gpus all -v $(pwd)/data:/app/data thermal-ai

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Zabraň interaktívnym promptom
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Inštalácia systémových závislostí
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libimage-exiftool-perl \
    fonts-dejavu-core \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Nastavenie pracovného adresára
WORKDIR /app

# Kopírovanie requirements a inštalácia Python balíčkov
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Inštalácia Unsloth (voliteľné, pre fine-tuning)
# RUN pip3 install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Kopírovanie zdrojového kódu
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY README.md .

# Vytvorenie adresárov pre dáta
RUN mkdir -p data/raw data/processed data/output models/finetuned

# Nastavenie Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose port pre Ollama (ak beží v kontajneri)
EXPOSE 11434

# Entrypoint
ENTRYPOINT ["python3", "-m", "src.pipeline"]

# Default príkaz - help
CMD ["--help"]