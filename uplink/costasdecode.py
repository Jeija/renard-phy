#!/usr/bin/env python3

import matplotlib.pyplot as plt
import scipy, scipy.signal, scipy.io, scipy.io.wavfile
import numpy as np
import sys


###########################
#       Definitions       #
###########################
FILENAME = sys.argv[1]

# Instantaneous Signal Frequency Estimation
MIN_FREQ_RES = 10
TIME_PER_SEGMENT = 0.5

# Downmixing and filtering
LPF_FREQ = 150

# Frame synchronization (may also occur inverted due to usage of DBPSK!)
# Using DBPSK, this corresponds to
#                          1  1  0   1   0  1  0   1   0  1  0   1   0  1  0   1   0  1  0   1   0
SIGFOX_PREAMBLE_SYMBOLS = [1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1]
#                         3xH       2xL    2xH    2xL     2xH    2xL   2xH    2xL    2xH    2xL    1xH

# General definitions
SIGFOX_BAUDRATE = 100

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
[Fs, IQ] = scipy.io.wavfile.read(FILENAME)

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
# Estimate signal frequency over time by analyzing whole recording
# Choose a length of FFT high enough so that frequency resolution is at least MIN_FREQ_RES [Hz]
# Also, cannot call scipy.signal.welch on whole signal (needs to much memory), thus segment signal before
#nfft = nextpow2(Fs / MIN_FREQ_RES)
#powerSpectrum = np.zeros(nfft)
#sigFreq = 0
#samples_per_segment = int(round(TIME_PER_SEGMENT * Fs))
#instFrequency = []

#for offset in range(0, len(signal), samples_per_segment):
#	print("Samples: " + str(max(0, offset)) + " - " + str(offset + samples_per_segment))
#	print("nfft: " + str(nfft))
#	print("nperseg: " + str(nfft / 4))
#	f, spec = scipy.signal.welch(signal[offset:(offset + samples_per_segment)], Fs, noverlap = 0, nperseg = nfft / 4, nfft = nfft, return_onesided = False)
#	powerSpectrum = np.sum([spec, powerSpectrum], axis = 0)
#	sigFreq = abs(f[np.argmax(powerSpectrum)])
#	instFrequency.extend([sigFreq] * samples_per_segment)
#	print("Estimating Signal Frequency (Guess: " + "{0:.2f}".format(sigFreq) + " Hz)...")
#	plt.plot(f, spec)
#	plt.pause(1)
#	plt.cla()
#
#print("Signal Frequency: " + "{0:.2f}".format(sigFreq) + " Hz")
#

# Get very approximate frequency estimation
# Costas loop takes care of locking onto actual frequency
nfft = nextpow2(Fs / MIN_FREQ_RES)
print("Estimating frequency using " + str(nfft) + "-length FFT")
f, spec = scipy.signal.welch(signal, Fs, noverlap = 0, nperseg = nfft / 4, nfft = nfft, return_onesided = False)
sigFreq = f[np.argmax(spec)]
#spec = np.fft.fft(signal, n = nfft)
#sigFreq = np.fft.fftfreq(nfft, 1 / Fs)[np.argmax(spec)]
print("Signal Frequency: " + "{0:.2f}".format(sigFreq) + " Hz")

###########################
#       Mixing + LPF      #
###########################
#mixer = [1]
#print("#signal: " + str(len(signal)) + ", #instFrequency: " + str(len(instFrequency)))
#for i in range(1, len(signal)):
#	mixer.append(mixer[-1] / abs(mixer[-1]) * np.exp(-1.0j * 2.0 * np.pi * instFrequency[i] / Fs))
#baseband_all = signal * mixer
#lpf_b, lpf_a = scipy.signal.butter(4, LPF_FREQ / Fs)
#baseband = scipy.signal.lfilter(lpf_b, lpf_a, baseband_all)

pll = CostasLoop(Fs, instFreqInit = sigFreq)
baseband = []
mixer = []
for i in range(len(signal)):
	baseband.append(pll.step(signal[i]))
	mixer.append(pll.mixerPhasor)

#plt.plot(range(len(signal)), np.real(signal))
#plt.plot(range(len(mixer)), np.real(mixer))
#plt.plot(range(len(baseband)), np.real(baseband))
#plt.show()

############################
#    Preamble detection    #
############################
print("Preamble detection...")
samples_per_symbol = Fs / SIGFOX_BAUDRATE

#ANIMSTEP = 500
#ANIMSAMPLES_PER_STEP = 600
#for i in range(0, len(baseband), ANIMSTEP):
#	points = baseband[i:i + ANIMSAMPLES_PER_STEP]
#	axes = plt.gca()
#	axes.set_xlim([-5, 5])
#	axes.set_ylim([-5, 5])
#	plt.scatter(np.real(points), np.imag(points))
#	plt.pause(0.001)
#	plt.cla()

# For preamble, turn baseband into either 0, 1 or -1 so that power of signal is irrelevant
avgPower = np.mean(np.abs(baseband))
discreteBaseband = []
for p in baseband:
	if p > avgPower / 2:
		discreteBaseband.append(1)
	elif p < avgPower / 2:
		discreteBaseband.append(-1)
	else:
		discreteBaseband.append(0)

xcorr = []
for offset in range(0, len(discreteBaseband) - int(len(SIGFOX_PREAMBLE_SYMBOLS) * samples_per_symbol), XCORR_PREAMBLE_PRECISION):
	correlation = 0

	for i in range(len(SIGFOX_PREAMBLE_SYMBOLS)):
		correlation += SIGFOX_PREAMBLE_SYMBOLS[i] * np.conj(discreteBaseband[offset + int(round(i * samples_per_symbol))])

	xcorr.append(correlation)

#plt.plot(np.arange(len(xcorr)), xcorr)
#plt.show()

# Correlation has to be at least len(SIGFOX_PREAMBLE_SYMBOLS) to be valid
# Find all non-coherent sections that signify the preamble
possiblePreamblePos = [(1 if abs(p) >= len(SIGFOX_PREAMBLE_SYMBOLS) else 0) for p in xcorr]

startOffsets = []
for sample in range(len(possiblePreamblePos)):
	if possiblePreamblePos[sample]:
		# If there already is a startOffset somewhere within 2 * samples_per_symbol,
		# adjust start position of that
		existsAlready = False
		for i, startOffset in enumerate(startOffsets):
			if abs(startOffset["offset"] - sample) < 2 * samples_per_symbol:
				startOffset["offset"] = (startOffset["offset"] * startOffset["count"] + sample) / (startOffset["count"] + 1)
				startOffset["count"] = startOffset["count"] + 1
				startOffsets[i] = startOffset
				existsAlready = True
				break

		if not existsAlready:
			startOffsets.append({
				"offset" : sample,
				"count" : 1
			})

#plt.plot(np.arange(len(possiblePreamblePos)), possiblePreamblePos)
#plt.show()

print("Found " + str(len(startOffsets)) + " preambles")


dataOutFile = open(FILENAME + ".txt", "w") 
for count, startOffset in enumerate(startOffsets):
	startOffsetSample = int(round(startOffset["offset"])) * XCORR_PREAMBLE_PRECISION

	# Calculate all symbols / differentials in audio file
	# Display scatter plot of differentials at bestoffset
	differentials = []
	symbols = []
	for j in np.arange(startOffsetSample + samples_per_symbol, len(baseband), int(round(samples_per_symbol))):
		last_symbol = baseband[int(round(j - samples_per_symbol))]
		this_symbol = baseband[int(round(j))]
		symbols.append(last_symbol)
		differentials.append(np.conj(last_symbol) * this_symbol)

	#plt.scatter(np.real(differentials), np.imag(differentials))
	#plt.show()

	avgPower = np.mean(np.abs(differentials))

	# Output decoded data
	dataString = ""
	for d in differentials:
		if d.real < -avgPower / 2:
			dataString += "0"
		elif d.real > avgPower / 2:
			dataString += "1"
		else:
			dataString += "x"

	dataString = dataString.split("x", 1)[0]
	dataOutFile.write(dataString + "\n")
	print(dataString)
	hexstring = str(hex(int(dataString, 2)))
	print("Payload: " + hexstring[8:])

dataOutFile.close()
