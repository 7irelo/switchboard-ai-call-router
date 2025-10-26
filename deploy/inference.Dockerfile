FROM python:3.12-slim
RUN pip install --no-cache-dir uv
WORKDIR /repo
COPY pyproject.toml uv.lock README.md ./
COPY proto ./proto
COPY scripts/gen_proto.py ./scripts/gen_proto.py
COPY src/python ./src/python
RUN uv sync --frozen --no-install-project --extra otel --extra onnx && .venv/bin/python scripts/gen_proto.py
ENV PYTHONPATH=/repo/src/python
ENTRYPOINT ["/repo/.venv/bin/python", "-m", "switchboard_inference.server"]
