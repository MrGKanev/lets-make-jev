# SGLang's own image carries the torch/CUDA/flashinfer combination it was built against;
# do not layer it on a TensorFlow image. Pin the tag after the first successful run.
ARG SGLANG_IMAGE=lmsysorg/sglang:latest
FROM ${SGLANG_IMAGE}

WORKDIR /app
COPY requirements-bench.txt .
RUN python3 -m pip install --no-cache-dir -r requirements-bench.txt

COPY . .

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
CMD ["bash", "run.sh"]
