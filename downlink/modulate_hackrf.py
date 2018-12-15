#!/usr/bin/env python3
# Usage: downlink_hackrf_transmit.py <UPLINK_FREQUENCY> <renard arguments>
# * First argument is frequency of first corresponding uplink frame (in Hz).
#   This frequency is used to calculate the downlink frequency by adding a constant of 1.395 MHz.
# * All additional arguments are simply passed on to renard.

import scipy.signal
import numpy as np
import subprocess
import binascii
import struct
import sys
import os

SAMPLERATE = 2000000
BAUDRATE = 600

GFSK_BANDWIDTH = 1600
SILENCE_DURATION = 0.2

GAUSSFILT_SIZE = 1.2
GAUSSFILT_SIGMA = 0.00015 * SAMPLERATE

OUTFILE = "/tmp/hackrf_transmission.raw"

HRFTRANS_CMD = "/usr/bin/hackrf_transfer"
UPLINK_FREQ = int(sys.argv[1])
HRF_CARRIER_FREQ = UPLINK_FREQ + 1395000

# Use renard to obtain a hexadecimal representation of the sigfox frames to send
# Always run renard in frame-building mode (dlencode)
RENARD_BIN = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, "renard", "build", "renard")
RENARD_CMD = [RENARD_BIN] + ["dlencode"] + sys.argv[2:]

try:
	FRAME_HEX = subprocess.check_output(RENARD_CMD, stderr = subprocess.STDOUT)
except subprocess.CalledProcessError as e:
	print(RENARD_BIN + " subprocess failed. Error:")
	print(e.output.decode("UTF-8"))
	sys.exit(0)

hexstring = FRAME_HEX.decode("UTF-8").rstrip()
print("Calculating signal for frame: " + hexstring)

# convert hexadecimal representation to binary string
binstring = "".join(format(c, "08b") for c in binascii.unhexlify(hexstring))

# create unfiltered baseband signal including ramp-up / ramp-down
# (one value in 'baseband_symbols' represents one symbol)
baseband_symbols = []
baseband_symbols = baseband_symbols + [0, 0]
for bit in binstring:
	baseband_symbols = baseband_symbols + [1 if bit == '1' else -1]
baseband_symbols = baseband_symbols + [0, 0]

# create baseband frequency-over-sample signal
# offset signal by a frequency of -GFSK_BANDWIDTH, so that HackRF One's DC component doesn't disturb signal
# receiver seems to tolerate negative offsets better than positive offsets
baseband_freq = [((s - 2) * GFSK_BANDWIDTH / 2) for s in baseband_symbols for _ in range(int(SAMPLERATE / BAUDRATE))]

# filter baseband frequency-over-sample signal with gaussian pulse
gaussfilt = scipy.signal.gaussian(int(SAMPLERATE / BAUDRATE * GAUSSFILT_SIZE), GAUSSFILT_SIGMA, False)
gaussfilt = gaussfilt / sum(gaussfilt)
baseband_freq = scipy.signal.convolve(baseband_freq, gaussfilt)
baseband_freq = baseband_freq[int(len(gaussfilt) / 2):-int(len(gaussfilt) / 2)]

# create "complex" baseband signal
# 'baseband' is actually a signal consisting of 8-bit signed integer
# I / Q values, alternating between I and Q
baseband = []
phase = 1
for instFreq in baseband_freq:
	if (len(baseband) % 5000 == 0):
		print(str(len(baseband)) + " / " + str(len(baseband_freq) * 2))
	phase = phase / np.abs(phase)
	phase = phase * np.exp(1.0j * 2 * np.pi * instFreq / SAMPLERATE)
	baseband.append(np.int8(np.real(phase) * 127))
	baseband.append(np.int8(np.imag(phase) * 127))

# Add a bit of "silence" after transmission
baseband = baseband  + [np.int8(0)] * int(SILENCE_DURATION * SAMPLERATE * 2)

print("Writing output...")
with open(OUTFILE, "wb") as output:
	output.write(struct.pack("%sb" % len(baseband), *baseband))

hftrans = [HRFTRANS_CMD] + ["-t", OUTFILE, "-s", str(SAMPLERATE), "-f", str(HRF_CARRIER_FREQ), "-x", "40"]
print(" ".join(hftrans))
subprocess.call(hftrans, stderr = subprocess.STDOUT)
