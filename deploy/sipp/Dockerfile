FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends sip-tester python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /sipp
COPY uas.xml uac.xml gen_pcap.py ./
# Audio is generated here, so no binary audio lives in git.
RUN python3 gen_pcap.py speech.pcap 8
