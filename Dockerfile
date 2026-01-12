# Use RunPod's pre-built PyTorch image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Chatterbox TTS without deps to avoid PyTorch version conflicts
RUN pip install --no-cache-dir --no-deps chatterbox-tts

# Install dependencies (Incluindo torchaudio e numpy que são essenciais para o handler)
RUN pip install --no-cache-dir \
    conformer \
    s3tokenizer \
    librosa \
    resemble-perth \
    huggingface_hub \
    safetensors \
    transformers \
    diffusers \
    einops \
    soundfile \
    scipy \
    numpy \
    torchaudio \
    omegaconf \
    pyloudnorm \
    runpod

# Copy handler
COPY handler.py /app/handler.py

# Pre-download model during build (Isso economiza ~2GB de download em cada boot do worker)
# 
RUN python -c "from chatterbox.tts import ChatterboxTTS; print('Pre-loading model...'); ChatterboxTTS.from_pretrained(device='cpu')"

# Start handler
CMD ["python", "-u", "/app/handler.py"]
