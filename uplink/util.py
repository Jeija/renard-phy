#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def nextpow2(limit):
	n = 1
	while n < limit:
		n = n * 2
	return int(n / 2)

# offset: estimated position of preamble in signal (in samples)
# searchArea: number of samples to search before / after offset for beginning of preamble
# precision: Skip `precision` samples when looking for greatest correlation, a larger number makes return value less precise
def locatePreamble(signal, sampleRate, preamble, baudRate, startOffset, searchArea, precision = 10):
	bestCorrelation = 0
	bestCorrelationOffset = 0
	samples_per_symbol = sampleRate / baudRate

	for offset in range(startOffset - searchArea, startOffset + searchArea, precision):
		correlation = 0
		symbols = []

		for i in range(len(preamble)):
			correlation += preamble[i] * np.conj(signal[offset + int(round(i * samples_per_symbol))])
			symbols.append(preamble[i] * np.conj(signal[offset + int(round(i * samples_per_symbol))]))

		#axes = plt.gca()
		#axes.set_xlim([-1, 1])
		#axes.set_ylim([-1, 1])
		#plt.scatter(np.real(symbols), np.imag(symbols))
		#plt.pause(0.001)
		#plt.cla()

		if abs(correlation) > bestCorrelation:
			bestCorrelation = abs(correlation)
			bestCorrelationOffset = offset

	return bestCorrelationOffset
