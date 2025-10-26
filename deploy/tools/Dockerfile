# Offline speech tooling: a robotic but real voice for pipeline and Whisper runs.
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng sox && rm -rf /var/lib/apt/lists/*
WORKDIR /work
