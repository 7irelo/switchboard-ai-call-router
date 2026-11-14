# Local text-to-speech: Kokoro-82M (Apache 2.0) on CPU, in half precision: next to no quality lost against full
# precision, and the fastest of the int8, fp16 and fp32 models on a laptop CPU.
FROM python:3.12-slim
RUN pip install --no-cache-dir kokoro-onnx==0.4.9 scipy==1.15.3 websockets==15.0.1
ADD --checksum=sha256:c1610a859f3bdea01107e73e50100685af38fff88f5cd8e5c56df109ec880204 \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx /models/kokoro-v1.0.fp16.onnx
ADD --checksum=sha256:bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin /models/voices-v1.0.bin
COPY server.py /app/server.py
WORKDIR /app
EXPOSE 8880
HEALTHCHECK --interval=10s --timeout=3s --retries=30 CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8880), 1)"
CMD ["python", "server.py"]
