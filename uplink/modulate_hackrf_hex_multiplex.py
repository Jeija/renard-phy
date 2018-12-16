#!/usr/bin/env python3
# Transmit given HEX uplink with HackRF One.
# Usage: modulate_hackrf_hex_multiplex.py [HEXSTRING1] [HEXSTRING2] ...

import scipy.signal
import numpy as np
import subprocess
import binascii
import struct
import sys
import os

SAMPLERATE = 2000000
BAUDRATE = 100
MPX_FREQ_OFFSET = 50000

OUTFILE = "/tmp/hackrf_transmission_REPLACE.raw"

HRFTRANS_CMD = "/usr/bin/hackrf_transfer"
HRF_CARRIER_FREQ = "868125700"

if (len(sys.argv) < 2):
	print("Usage: modulate_hackrf_hex.py [HEXSTRING]")
	sys.exit(0)

print("This may take some time. Sorry :(")

inputs = sys.argv[1:]
for arg in range(len(inputs)):
	hexstring = sys.argv[1 + arg]

	print("Adding frame: " + hexstring)

	# convert hexadecimal representation to binary string
	# add "1"s at beginning and end of bitstring for ramp-up / ramp-down
	binstring = "".join(format(c, "08b") for c in binascii.unhexlify(hexstring))
	binstring = "1" + binstring + "11"

	# create unfiltered baseband signal
	max_phase = 1 / len(inputs)
	symbol_duration = int(SAMPLERATE / BAUDRATE)

	# superimpose all TX signals, extend duration if necessary
	baseband = np.zeros(symbol_duration * len(binstring), dtype = np.complex_)
	phase = max_phase
	for bit in range(len(binstring)):
		for symbol_time in range(symbol_duration):
			phase = np.exp(1j * 2 * np.pi * arg * MPX_FREQ_OFFSET / SAMPLERATE) * phase
			phase = phase * max_phase / np.abs(phase)

			time = bit * symbol_duration + symbol_time
			if len(baseband) > time:
				baseband[time] = baseband[time] + phase
			else:
				np.append(baseband, phase)

		# differential modulation: 180° phase shift if "1"
		phase = phase if binstring[bit] == "1" else -phase

np.append(baseband, np.zeros(int(0.5 * BAUDRATE)))

# write complete signal to hackrf RAW I/Q file
print("Quantizing...")
iqsignal = np.int8(np.real(baseband) * 127)
rawsignal = iqsignal.repeat(2)

print("Writing output...")
filename = OUTFILE.replace("REPLACE", "-".join(inputs))

with open(filename, "wb") as output:
	output.write(struct.pack("%sb" % len(rawsignal), *rawsignal))

hftrans = [HRFTRANS_CMD] + ["-t", filename, "-s", str(SAMPLERATE), "-f", str(HRF_CARRIER_FREQ), "-x", "47", "-a", "1"]

print("Execute the following command to transmit: ")
print(" ".join(hftrans))
