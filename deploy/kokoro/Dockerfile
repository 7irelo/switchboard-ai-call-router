# Local text-to-speech: Kokoro-82M (Apache 2.0) on CPU, quantised to int8.
FROM python:3.12-slim
RUN pip install --no-cache-dir kokoro-onnx==0.4.9 scipy==1.15.3 websockets==15.0.1
ADD --checksum=sha256:6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx /models/kokoro-v1.0.int8.onnx
ADD --checksum=sha256:bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin /models/voices-v1.0.bin
COPY server.py /app/server.py
WORKDIR /app
EXPOSE 8880
HEALTHCHECK --interval=10s --timeout=3s --retries=30 CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8880), 1)"
CMD ["python", "server.py"]
