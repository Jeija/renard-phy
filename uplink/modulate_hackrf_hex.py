#!/usr/bin/env python3
# Transmit given HEX uplink with HackRF One.
# Usage: modulate_hackrf_hex.py [HEXSTRING]

import scipy.signal
import numpy as np
import subprocess
import binascii
import struct
import sys
import os

SAMPLERATE = 2000000
BAUDRATE = 100

OUTFILE = "/tmp/hackrf_transmission.raw"

HRFTRANS_CMD = "/usr/bin/hackrf_transfer"
HRF_CARRIER_FREQ = "868125700"

if (len(sys.argv) != 2):
	print("Usage: modulate_hackrf_hex.py [HEXSTRING]")
	sys.exit(0)

hexstring = sys.argv[1]

# convert hexadecimal representation to binary string
binstring = "".join(format(c, "08b") for c in binascii.unhexlify(hexstring))

# add "1"s at beginning and end of bitstring for ramp-up / ramp-down
binstring = "1" + binstring + "11"

# create unfiltered baseband signal
baseband = []
phase = True
for bit in binstring:
	baseband = baseband + [1 if phase else -1]
	phase = phase if bit == "1" else not phase

baseband.extend([0] * int(0.5 * BAUDRATE))

# write complete signal to hackrf RAW I/Q file:
# * Repeat bits int(SAMPLERATE / BAUDRATE) times to achieve sample rate
# * Hack: Set I and Q to always be the same value by multiplying by "2"
print("Quantizing...")
iqsignal = np.int8(np.real(baseband) * 127)
rawsignal = [s for s in iqsignal for _ in range(2 * int(SAMPLERATE / BAUDRATE))]

print("Writing output...")
with open(OUTFILE, "wb") as output:
	output.write(struct.pack("%sb" % len(rawsignal), *rawsignal))

hftrans = [HRFTRANS_CMD] + ["-t", OUTFILE, "-s", str(SAMPLERATE), "-f", str(HRF_CARRIER_FREQ), "-x", "47", "-a", "1"]
print(" ".join(hftrans))
subprocess.call(hftrans, stderr = subprocess.STDOUT)
