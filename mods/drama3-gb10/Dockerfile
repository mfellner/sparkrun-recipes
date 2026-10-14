# SPDX-License-Identifier: MIT (packaging only; Fish code has separate terms).
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:450d11555d20ac8ebbbc13ebf17589c2bd42869171a90179ce7098b4a5e64c6a
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev git build-essential ffmpeg libsndfile1 ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/drama-env
ENV PATH=/opt/drama-env/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/opt/fish-speech:/opt/drama \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    FISH_DRAMA3_FP8_EXPERTS=1 FISH_DRAMA3_CUDA_GRAPH=0 \
    OMP_NUM_THREADS=4
RUN pip install --no-cache-dir torch==2.13.0 triton==3.7.1 --index-url https://download.pytorch.org/whl/cu130 && pip install --no-cache-dir --no-deps torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
COPY requirements-runtime.txt requirements-lock.txt /opt/drama/
# Override obsolete audiotools protobuf and unavailable Torch/Torchaudio pairing;
# TensorBoard 2.18 retains protobuf-5-compatible generated code.
RUN pip install --no-cache-dir uv==0.12.9 && uv pip install --python /opt/drama-env/bin/python --no-deps -r /opt/drama/requirements-lock.txt
RUN git init /opt/fish-speech && cd /opt/fish-speech && git remote add origin https://github.com/fishaudio/fish-speech.git && git fetch --depth 1 origin 1ff3c3ed2f80d3106a83909dd25b83129c3d816f && git checkout FETCH_HEAD && test "$(git rev-parse HEAD)" = 1ff3c3ed2f80d3106a83909dd25b83129c3d816f && rm -rf .git
COPY contract.py probe.py server.py run.sh NOTICE ADAPTER_LICENSE.txt FISH_LICENSE.txt model-manifest.json Dockerfile manifest.sha256 default-reference.wav default-reference.txt reference-provenance.json /opt/drama/
RUN userdel ubuntu && useradd -m -u 1000 drama && mkdir -p /cache/huggingface && chown drama:drama /cache/huggingface && chmod 755 /opt/drama/run.sh
USER drama
WORKDIR /cache/huggingface
LABEL org.opencontainers.image.source="https://github.com/mfellner/sparkrun-recipes" \
      org.opencontainers.image.title="Drama 3 native GB10 runtime" \
      org.opencontainers.image.version="drama3-gb10-1" \
      fish.source.revision="1ff3c3ed2f80d3106a83909dd25b83129c3d816f" \
      fish.model.revision="dd64a85cd82e9521f7202610ede97b367fc83946"
ENTRYPOINT ["/opt/drama/run.sh"]
