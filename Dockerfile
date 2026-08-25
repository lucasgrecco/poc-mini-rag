FROM python:3.13-slim
# libatomic1: the ROCm torch wheels link against libatomic, which python:*-slim does
# not ship. Without it `import torch` dies with `ImportError: libatomic.so.1`. The
# CUDA wheels do not need it, so this only shows up under GPU=rocm.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libatomic1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install uv
WORKDIR /app
