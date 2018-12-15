#!/usr/bin/env python3
# Generate uplink frame with renard and transmit uplink with HackRF One.
# Usage: modulate_hackrf.py [RENARD CLI ARGS]
# All command line arguments to modulate_hackrf.py will just be forwarded to "renard ulencode".

import scipy.signal
import numpy as np
import subprocess
import binascii
import struct
import sys
import os

SAMPLERATE = 2000000
BAUDRATE = 100
PAUSE_DURATION = 0.5

OUTFILE = "/tmp/hackrf_transmission.raw"

HRFTRANS_CMD = "/usr/bin/hackrf_transfer"
HRF_CARRIER_FREQ = "868125700"

# Use renard to obtain a hexadecimal representation of the sigfox frames to send
# Always run renard in frame-building mode (ulencode)
RENARD_BIN = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, "renard", "build", "renard")
RENARD_CMD = [RENARD_BIN] + ["ulencode"] + sys.argv[1:]

try:
	FRAMES_HEX = subprocess.check_output(RENARD_CMD, stderr = subprocess.STDOUT)
except subprocess.CalledProcessError as e:
	print(RENARD_BIN + " subprocess failed. Error:")
	print(e.output.decode("UTF-8"))
	sys.exit(0)

FRAMES_HEX = FRAMES_HEX.decode("UTF-8").split("\n")

# total signal / baseband are signals where each value represents one symbol value
total_signal = []

for hexstring in FRAMES_HEX:
	if (len(hexstring) == 0):
		continue

	print("Adding frame: " + hexstring)

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

	total_signal.extend(baseband)

	# add pause after the transmission
	# the pause duration seems to be irrelevant
	total_signal.extend([0] * int(PAUSE_DURATION * BAUDRATE))

# write complete signal to hackrf RAW I/Q file:
# * Repeat bits int(SAMPLERATE / BAUDRATE) times to achieve sample rate
# * Hack: Set I and Q to always be the same value by multiplying by "2"
print("Quantizing...")
iqsignal = np.int8(np.real(total_signal) * 127)
rawsignal = [s for s in iqsignal for _ in range(2 * int(SAMPLERATE / BAUDRATE))]

print("Writing output...")
with open(OUTFILE, "wb") as output:
	output.write(struct.pack("%sb" % len(rawsignal), *rawsignal))

hftrans = [HRFTRANS_CMD] + ["-t", OUTFILE, "-s", str(SAMPLERATE), "-f", str(HRF_CARRIER_FREQ), "-x", "47", "-a", "1"]
print(" ".join(hftrans))
subprocess.call(hftrans, stderr = subprocess.STDOUT)
