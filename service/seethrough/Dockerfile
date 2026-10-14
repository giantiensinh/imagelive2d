# syntax=docker/dockerfile:1
# Reproducible alternative to startup.sh — build once, run anywhere with an NVIDIA GPU.
# Host needs a recent NVIDIA driver + nvidia-container-toolkit.
#
#   docker build -t seethrough-svc service/seethrough
#   docker run --gpus all -p 8000:8000 -v hf-cache:/root/.cache/huggingface seethrough-svc
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libgl1 libglib2.0-0 python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone See-through source (pinned commit)
WORKDIR /opt
RUN git clone https://github.com/shitagaki-lab/see-through.git /opt/see-through \
    && cd /opt/see-through \
    && git checkout 58a1cb11d13f85acec9bbddb8cd4b6487843d4cf
WORKDIR /opt/see-through
RUN ln -sf common/assets assets

# Create virtualenv
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --- Layer 1: PyTorch CUDA (heavy, ~4 GB) — cached independently so requirements changes don't re-download torch ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
      --index-url https://download.pytorch.org/whl/cu128

# --- Layer 2: See-through requirements + service deps (lighter) ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt && \
    pip install fastapi "uvicorn[standard]"

# --- App code (changes often, keep last) ---
COPY app.py pipeline.py model_manager.py /opt/
COPY patches /opt/patches
RUN python /opt/patches/guard_empty_head_crop.py common/utils/inference_utils.py \
    && python /opt/patches/guard_missing_part_pngs.py common/utils/io_utils.py \
    && python /opt/patches/low_vram_cpu_load.py common/utils/inference_utils.py

RUN useradd --create-home --uid 1001 app && mkdir /models && chown app:app /models
USER app
WORKDIR /opt
ENV SEE_THROUGH_DIR=/opt/see-through
HEALTHCHECK --interval=15s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
