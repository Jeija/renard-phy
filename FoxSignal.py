#!/usr/bin/env python3

import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np

from util import *

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

	# start / size in samples
	def plotFFT(self, start, size):
		spec = self.getFFT(start, size)
		freq = np.fft.fftfreq(size, 1 / self.sampleRate)
		plt.plot(freq, spec.real, freq, spec.imag)

	# start / size in samples
	def getFFT(self, start, size):
		return np.fft.fft(self.signal[start:], size)

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
	def findSigfoxBursts(self, searchIntervalTime, channelWidth, minimalSNR):
		SNR_linear = 10 ** (minimalSNR / 10)

		searchIntervalSize = nextpow2(searchIntervalTime * self.sampleRate)
		newPower = None
		for i in range(int(np.floor(len(self.signal) / searchIntervalSize))):
			axes = plt.gca()
			axes.set_xlim([-30000, 30000])
			axes.set_ylim([20, 80])
			self.plotBandPower(i * searchIntervalSize, searchIntervalSize, channelWidth)
			plt.pause(0.001)
			plt.cla()

			oldPower = newPower
			newPower = self.getBandPower(i * searchIntervalSize, searchIntervalSize, 100)
			if (not oldPower is None):
				for band in range(len(oldPower)):
					if (newPower[band] > oldPower[band] * SNR_linear and newPower[band] > np.mean(newPower) * SNR_linear):
						print("Burst in band " + str(band))
