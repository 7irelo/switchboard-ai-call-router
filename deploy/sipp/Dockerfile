FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends sip-tester python3 espeak-ng sox && rm -rf /var/lib/apt/lists/*
WORKDIR /sipp
COPY uas.xml uac.xml gen_pcap.py ./
# Audio is generated here, so no binary audio lives in git.
RUN python3 gen_pcap.py speech.pcap 8
RUN espeak-ng -v en-gb -s 140 -w spoken.wav "my router keeps restarting by itself and i need help" \
    && sox spoken.wav -r 8000 -c 1 -e mu-law -t raw spoken.ulaw \
    && python3 -c "print(round(__import__('os').path.getsize('spoken.ulaw') / 8))" > spoken.ms \
    && python3 gen_pcap.py real_speech.pcap 4 spoken.ulaw
