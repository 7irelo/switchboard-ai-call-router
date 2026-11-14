# Local text-to-speech: Piper (MIT) on CPU, with its Lessac voice at medium quality:
# about 12x real time on a laptop CPU, against 3x for high, and a phone line's 8 kHz hides the difference.
FROM python:3.12-slim
RUN pip install --no-cache-dir piper-tts==1.8.0 scipy==1.15.3 websockets==15.0.1
ADD --checksum=sha256:5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx /models/en_US-lessac-medium.onnx
ADD --checksum=sha256:efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0 \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json /models/en_US-lessac-medium.onnx.json
COPY server.py /app/server.py
WORKDIR /app
EXPOSE 8880
HEALTHCHECK --interval=10s --timeout=3s --retries=30 CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8880), 1)"
CMD ["python", "server.py"]
