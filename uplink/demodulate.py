#!/usr/bin/env python3

import scipy, scipy.signal, scipy.io, scipy.io.wavfile
import numpy as np
import argparse
import sys

###########################
#    Parse CLI argumets   #
###########################
parser = argparse.ArgumentParser()
parser.add_argument("wavfile", help="WAV file containing the uplink recording to be demodulated")
parser.add_argument("-p", "--plot", action="store_true", default = False, help = "Use matplotlib to display the baseband signal and where in that baseband signal a preamble was detected")
parser.add_argument("-d", "--decode", action="store_true", default = False, help = "Use renard to decode content of uplink frame")
args = parser.parse_args()

###########################
#       Definitions       #
###########################
MIN_FREQ_RES = 10

# Downmixing and filtering
LPF_FREQ = 500

# Frame synchronization (may also occur inverted due to usage of DBPSK!)
# Using DBPSK, this corresponds to
#                    1  1  0   1   0  1  0   1   0  1  0   1   0  1  0   1   0  1  0   1   0
UPLINK_PREAMBLE = [1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1]
#                  3xH       2xL    2xH    2xL     2xH    2xL   2xH    2xL    2xH    2xL    1xH

SILENCE_BEFORE_PREAMBLE = 0.2

# General definitions
UPLINK_BAUDRATE = 100

# Precision by which the preamble will be located (in signal samples)
XCORR_PREAMBLE_PRECISION = 10

###########################
#    Utility functions    #
###########################
def nextpow2(limit):
	n = 1
	while n < limit:
		n = n * 2
	return int(n / 2)

# frequencyAdjustmentFactor:
# Proportion by which the "VCO" frequency will be adjusted from given frequency offset estimate in
# "share of frequency offset corrected per second"
#
# phaseAdjustmentFactor:
# Proportion by which the "VCO" phase will be adjusted from given phase offset estimate in
# "share of phase offset corrected per second"
class CostasLoop(object):
	def __init__(self, sampleRate, frequencyAdjustmentFactor = 1e6, phaseAdjustmentFactor = 1000, instFreqInit = 0):
		# instAngFreq is the angular frequency, not the physical one
		self.instAngFreq = 2 * np.pi * instFreqInit
		self.sampleRate = sampleRate
		self.mixerPhasor = 1
		self.phaseError = 0
		self.phaseErrorSmoothing = 0.9
		self.alpha = 2 * np.pi * phaseAdjustmentFactor / self.sampleRate
		self.beta = 2 * np.pi * frequencyAdjustmentFactor / self.sampleRate

	def step(self, value):
		self.mixerPhasor *= np.exp(-1.0j * self.instAngFreq * 1 / self.sampleRate) / abs(self.mixerPhasor)
		baseband = value * self.mixerPhasor
		currentPhaseError = np.imag(baseband) * np.real(baseband)

		# Simple exponential smoothing for phase error
		self.phaseError = self.phaseError * self.phaseErrorSmoothing + currentPhaseError * (1 - self.phaseErrorSmoothing)

		# Completely fix phase offset
		self.mixerPhasor *= np.exp(-1.0j * self.phaseError * self.alpha)

		# Adjust frequency slightly
		self.instAngFreq += self.phaseError * self.beta

		return baseband

###########################
#     Recording Input     #
###########################
[Fs, IQ] = scipy.io.wavfile.read(args.wavfile)

# IQ offset / imbalance correction
I_nooffset = IQ[:, 0] - sum(IQ[:, 0]) / len(IQ[:, 0])
Q_nooffset = IQ[:, 1] - sum(IQ[:, 1]) / len(IQ[:, 1])
Q_nooffset *= sum(abs(Q_nooffset)) / sum(abs(I_nooffset))
signal = I_nooffset + 1j * Q_nooffset

# Norm all values
signal /= np.mean(abs(signal))

############################
#   Frequency Estimation   #
############################
# Get very approximate frequency estimation
# Costas loop takes care of locking onto actual frequency
nfft = nextpow2(Fs / MIN_FREQ_RES)

print("Estimating Frequency using " + str(nfft) + "-length FFT")
f, spec = scipy.signal.welch(signal, Fs, noverlap = 0, nperseg = nfft / 4, nfft = nfft, return_onesided = False)
sigFreq = f[np.argmax(spec)]

print("Signal Frequency: " + "{0:.2f}".format(sigFreq) + " Hz")

###########################
#       Mixing + LPF      #
###########################
print("Mixing to Baseband with Costas Loop...")
baseband = []
pll = CostasLoop(Fs, instFreqInit = sigFreq)
for i in range(len(signal)):
	baseband.append(pll.step(signal[i]))

print("Low-pass Filtering...")
lpf_b, lpf_a = scipy.signal.butter(4, LPF_FREQ / Fs)
baseband = scipy.signal.lfilter(lpf_b, lpf_a, baseband)

print("Normalizing...")
baseband /= np.mean(abs(baseband))

############################
#    Preamble detection    #
############################
print("Finding Preamble(s)...")
samples_per_symbol = Fs / UPLINK_BAUDRATE
preamble = [0] * int(SILENCE_BEFORE_PREAMBLE * Fs) + [s for s in UPLINK_PREAMBLE for _ in range(int(samples_per_symbol))]

# cross-correlate both baseband signal value and baseband signal power
# this makes it possible to detect if the preamble is actually preceeded by silence
# preamble detection is then based on the sum of both cross-correlations
baseband_pwr = [x**2 for x in baseband]
baseband_pwr -= np.mean(baseband_pwr)
baseband_pwr /= np.mean(abs(baseband_pwr))

preamble_pwr = [x**2 for x in preamble]
preamble_pwr -= np.mean(preamble_pwr)
preamble_pwr /= np.mean(abs(baseband_pwr))

xcorr_abs = abs(scipy.signal.correlate(np.real(baseband), preamble, mode="valid"))
xcorr_pwr = scipy.signal.correlate(baseband_pwr, preamble_pwr, mode="valid")
xcorr = xcorr_abs + xcorr_pwr
xcorrmax = np.max(xcorr)

# TODO: fix this somehow, make configurable
preamble_offsets, _ = scipy.signal.find_peaks(xcorr, distance = samples_per_symbol * 10, height = 0.8 * xcorrmax, prominence = 0.5 * xcorrmax)

if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(np.real(baseband))
	for preamble_offset in preamble_offsets:
		plt.plot(preamble_offset + range(len(preamble)), preamble)
	plt.show()

print("Found " + str(len(preamble_offsets)) + " Preambles!")

for count, preamble_offset in enumerate(preamble_offsets):
	data_offset = int(preamble_offset + SILENCE_BEFORE_PREAMBLE * Fs)

	# Calculate all symbols / differentials in audio file
	# Display scatter plot of differentials at bestoffset
	differentials = []
	samplingInstants = np.arange(data_offset + 3 * samples_per_symbol / 2, len(baseband), int(round(samples_per_symbol)))
	for j in samplingInstants:
		last_symbol = baseband[int(round(j - samples_per_symbol))]
		this_symbol = baseband[int(round(j))]
		differentials.append(np.conj(last_symbol) * this_symbol)

	import matplotlib.pyplot as plt
	plt.plot(np.real(baseband))
	plt.plot(samplingInstants, np.real(baseband)[samplingInstants.astype(int)], "x")
	plt.show()

	avgPower = np.mean(np.abs(differentials))

	# Output decoded data
	bitstring = ""
	for d in differentials:
		if d.real < -avgPower / 2:
			bitstring += "0"
		elif d.real > avgPower / 2:
			bitstring += "1"
		else:
			bitstring += "x"

	bitstring = bitstring.split("x", 1)[0]
	print("Frame " + str(count) + ": " + bitstring)

	# TODO: Use frame type to determine length of uplink
	hexstring = str(hex(int(bitstring[1:int((len(bitstring) - 1) / 4) * 4 + 1], 2)))
	print("Payload: " + hexstring[7:])
