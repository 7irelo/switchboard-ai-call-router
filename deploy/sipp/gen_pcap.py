"""Writes a pcap of G.711 RTP for SIPp to play: alternating noise bursts, loud enough to count as speech."""

import random
import struct
import sys

out, seconds = sys.argv[1], int(sys.argv[2])
rng = random.Random(1)
SRC, DST = bytes([10, 0, 0, 1]), bytes([10, 0, 0, 2])


def checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\0"
    total = sum(struct.unpack(f"!{len(data) // 2}H", data))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


with open(out, "wb") as f:
    f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
    for n in range(seconds * 50):
        # Low mu-law byte values are loud.
        payload = bytes(rng.randint(0x10, 0x40) for _ in range(160))
        rtp = struct.pack("!BBHII", 0x80, 0, n & 0xFFFF, n * 160, 0x53424F58) + payload
        udp = struct.pack("!HHHH", 6000, 6000, 8 + len(rtp), 0) + rtp
        header = struct.pack("!BBHHHBBH", 0x45, 0, 20 + len(udp), n & 0xFFFF, 0, 64, 17, 0) + SRC + DST
        header = header[:10] + struct.pack("!H", checksum(header)) + header[12:]
        frame = b"\x02" * 6 + b"\x04" * 6 + b"\x08\x00" + header + udp
        usec = n * 20000
        f.write(struct.pack("<IIII", usec // 1000000, usec % 1000000, len(frame), len(frame)) + frame)
