#!/usr/bin/env python3

import matplotlib.pyplot as plt
import scipy.io.wavfile
import scipy.signal
import numpy as np
import sys

from util import *

# Constants for RCZ 1 (Europe, Middle East, Africa)
# Baudrate (symbols / second) of Sigfox transmission
SIGFOX_BAUDRATE = 100

# Bandwidth (Hz) of Sigfox transmission
SIGFOX_BANDWIDTH = 100

# Minimal duration (seconds) of Sigfox transmission
SIGFOX_MINIMAL_DURATION = 0.8

# Some preamble symbols (F.SYNC) that are used for frame synchronization
# Since Sigfox uses DBPSK, this corresponds to data bits "1010 1010"
SIGFOX_PREAMBLE_SYMBOLS = [1, 1, -1, -1, 1, 1, -1, -1, 1]

class ComplexSignal:
	def __init__(self, signal, sampleRate):
		self.signal = signal
		self.sampleRate = sampleRate

	def spectrogram(self):
		plt.specgram(self.signal, NFFT = 1024, Fs = self.sampleRate)

	def mix(self, frequency):
		time = np.arange(len(self.signal)) / self.sampleRate
		mixer = np.exp(-1.0j * 2.0 * np.pi * frequency * time)
		return ComplexSignal(self.signal * mixer, self.sampleRate)

	def lpf(self, cutoff):
		lpf_b, lpf_a = scipy.signal.butter(5, cutoff / self.sampleRate)
		return ComplexSignal(scipy.signal.lfilter(lpf_b, lpf_a, self.signal), self.sampleRate)

	def decimate(self, factor):
		return ComplexSignal(scipy.signal.decimate(self.signal, factor, ftype = "fir"), self.sampleRate / factor)

	def toPower(self):
		return ComplexSignal(abs(self.signal)**2, self.sampleRate)

	def animateIQ(self, start, duration):
		ANIMSTEP = 50
		ANIMSAMPLES_PER_STEP = 60
		for i in range(0, duration, ANIMSTEP):
			points = self.signal[start + i:start + i + ANIMSAMPLES_PER_STEP]
			axes = plt.gca()
			axes.set_xlim([-1, 1])
			axes.set_ylim([-1, 1])
			plt.scatter(np.real(points), np.imag(points))
			plt.pause(0.001)
			plt.cla()

	# start / size in samples
	# resolution: integer, 1 = normal, otherwise zero-pad with len(signal) * (resolution - 1) zeros
	def getFFT(self, start, size, resolution = 1, window = "rect"):
		timedomain = self.signal[start:start + size]
		if window == "hamming":
			timedomain = timedomain * np.hamming(len(timedomain))
		return np.fft.fft(timedomain, size * resolution)

	# size in samples
	def getFFTFreq(self, size, resolution = 1):
		return np.fft.fftfreq(size * resolution, 1 / self.sampleRate / resolution)

	# start / size in samples
	def plotFFT(self, start, size):
		spec = self.getFFT(start, size)
		freq = self.getFFTFreq(size)
		plt.plot(freq, spec.real, freq, spec.imag)

	# start / size in samples, bandWidth in Hz
	def getBandPower(self, start, size, bandWidth):
		spec = self.getFFT(start, size)
		bandWidthBins = bandWidth / self.sampleRate * size
		power = []
		for startBin in np.arange(0, size, bandWidthBins):
			bandPower = 0
			for binOffset in range(round(bandWidthBins)):
				bandPower += abs(spec[int(round(startBin + binOffset))])**2
			power.append(bandPower)
		#print("getBandPower: " + str(bandWidthBins) + " bins per band, " + str(len(power)) + " bands total.")
		return power

	# start / size in samples, bandWidth in Hz
	def plotBandPower(self, start, size, bandWidth):
		power = self.getBandPower(start, size, bandWidth)
		freq = np.fft.fftfreq(round(self.sampleRate / bandWidth), 1 / self.sampleRate)
		plt.plot(freq, 10 * np.log10(power))

	def plotAmplitude(self):
		plt.plot(np.arange(len(self.signal)) / self.sampleRate, abs(self.signal))

	def plotShow(self):
		plt.show()

class Intermediate(ComplexSignal):
	def __init__(self, filename):
		[sampleRate, IQ] = scipy.io.wavfile.read(filename)

		# IQ offset / imbalance correction
		I_nooffset = IQ[:, 0] - sum(IQ[:, 0]) / len(IQ[:, 0])
		Q_nooffset = IQ[:, 1] - sum(IQ[:, 1]) / len(IQ[:, 1])
		Q_nooffset *= sum(abs(Q_nooffset)) / sum(abs(I_nooffset))
		signal = I_nooffset + 1j * Q_nooffset

		# Norm all values
		signal /= np.mean(abs(signal))

		ComplexSignal.__init__(self, signal, sampleRate)

	# Returns coarse estimates of start index (samples) / frequency (Hz)
	# of possible Sigfox transmissions
	# Parameters: duration of search interval (determines start index accuary),
	# burst detection channel width and minimal SNR (minimal power increase)
	# in dB
	def estimateSigfoxBursts(self, searchIntervalTime, channelWidth, minimalSNR):
		bursts = []

		SNR_linear = 10 ** (minimalSNR / 10)
		searchIntervalSize = nextpow2(searchIntervalTime * self.sampleRate)
		newPower = None

		for i in range(int(np.floor(len(self.signal) / searchIntervalSize))):
			#axes = plt.gca()
			#axes.set_xlim([-30000, 30000])
			#axes.set_ylim([20, 80])
			#self.plotBandPower(i * searchIntervalSize, searchIntervalSize, channelWidth)
			#plt.pause(0.001)
			#plt.cla()

			oldPower = newPower
			newPower = self.getBandPower(i * searchIntervalSize, searchIntervalSize, SIGFOX_BANDWIDTH)
			if (not oldPower is None):
				for band in range(len(oldPower)):
					if (newPower[band] > oldPower[band] * SNR_linear and newPower[band] > np.mean(newPower) * SNR_linear):
						bursts.append({ "frequency" : band * channelWidth, "start" : i * searchIntervalSize })

		# TODO: Remove entries that most likely correspond to the same burst

		return bursts

	def locateSigfoxBursts(self, minimalSNR):
		estimates = self.estimateSigfoxBursts(0.25, SIGFOX_BANDWIDTH / 2, minimalSNR)
		bursts = []
		SNR_linear = 10 ** (minimalSNR / 10)

		for estimate in estimates:
			# Step 1: Find more precise frequency estimate by analyzing >= SIGFOX_MINIMAL_DURATION of signal
			minimalSignalSamples = nextpow2(SIGFOX_MINIMAL_DURATION * self.sampleRate)

			# Choose a length of FFT high enough so that frequency resolution is at least MIN_FREQ_RES [Hz]
			MIN_FREQ_RES = 0.3
			nfft = nextpow2(self.sampleRate / MIN_FREQ_RES)

			transmissionPart = self.signal[estimate["start"]:estimate["start"] + minimalSignalSamples]
			f, powerSpectrum = scipy.signal.welch(transmissionPart, self.sampleRate, nperseg = nfft / 256, nfft = nfft, return_onesided = False)
			sigFreq = f[np.argmax(powerSpectrum)]

			# Step 2: Find more precise start sample by analyzing signal power at specific frequency over time
			baseband = self.mix(sigFreq).lpf(SIGFOX_BANDWIDTH)

			# Calculate average power of the noise floor assuming noise floor is below average power
			power = baseband.toPower()
			meanPower = np.mean(power.signal)
			noiseFloor = [p for p in power.signal if p < meanPower]
			noiseFloorPower = np.mean(noiseFloor)

			# Calculate average power of the transmission assuming transmission power is above average power
			transmission = [p for p in power.signal if p > meanPower]
			transmissionPower = np.mean(transmission)

			# Find first index at which power of baseband signal is above the middle of
			# noiseFloorPower and transmissionPower (arbitrarily defined)
			for startIndex in range(len(power.signal)):
				if power.signal[startIndex] > (noiseFloorPower + transmissionPower) / 2:
					break

			# Step 3: Detect beginning of sigfox preamble in baseband signal around startIndex
			# Search for valid preamble in a search area of up to 6 symbols before / after given startIndex
			searchArea = int(6 * baseband.sampleRate / SIGFOX_BAUDRATE)
			preambleOffset = locatePreamble(baseband.signal, baseband.sampleRate, SIGFOX_PREAMBLE_SYMBOLS, SIGFOX_BAUDRATE, startIndex, searchArea)

			bursts.append({ "baseband" : baseband, "frequency" : sigFreq, "start" : preambleOffset })

		return bursts

	def decodeSigfoxBursts(self, minimalSNR):
		bursts = self.locateSigfoxBursts(minimalSNR)

		for burst in bursts:
			print("* Burst @" + str(burst["frequency"]) + " Hz, start offset " + str(burst["start"] / self.sampleRate) + " s")
			#burst["baseband"].animateIQ(burst["start"], 8000)

			samples_per_symbol = self.sampleRate / SIGFOX_BAUDRATE

			differentials = []
			symbols = []
			# TODO: this doesn't stop decoding after transmission is finished
			for symbolCount in range(1, round((len(burst["baseband"].signal) - burst["start"]) / samples_per_symbol)):
				last_symbol = burst["baseband"].signal[burst["start"] + round((symbolCount - 1) * samples_per_symbol)]
				this_symbol = burst["baseband"].signal[burst["start"] + round(symbolCount * samples_per_symbol)]
				symbols.append(last_symbol)
				differentials.append(np.conj(last_symbol) * this_symbol)

			#plt.scatter(np.real(differentials), np.imag(differentials))
			#plt.show()

			for d in differentials:
				sys.stdout.write("0" if d.real < 0 else "1")
				sys.stdout.flush()
			print()
