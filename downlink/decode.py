#!/usr/bin/env python3

import matplotlib.pyplot as plt
import scipy, scipy.signal, scipy.io, scipy.io.wavfile
import numpy as np
import sys

###########################
#       Definitions       #
###########################
FILENAME = sys.argv[1]

# General definitions
DOWNLINK_BAUDRATE = 600
MIN_FREQ_RES = 10
LPF_FREQ = 4000
DOWNLINK_PREAMBLE = [-1] + [-1, 1] * 45 + [1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1]

# Demodulated signal is upsampled to this sample rate to allow for better preamble detection
DEMOD_SAMPRATE = 300000

###########################
#    Utility functions    #
###########################
def nextpow2(limit):
	n = 1
	while n < limit:
		n = n * 2
	return int(n / 2)

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
# Get very approximate frequency estimation
nfft = nextpow2(Fs / MIN_FREQ_RES)
print("Estimating frequency using " + str(nfft) + "-length FFT")
f, spec = scipy.signal.welch(signal, Fs, noverlap = 0, nperseg = nfft / 4, nfft = nfft, return_onesided = False)

# Calculate spectral centroid and use frequency value as estimated signal frequency
sigFreq = sum(f * abs(spec)) / sum(abs(spec))

############################
#      Mix to baseband     #
############################
mixer = np.exp(-1.0j * 2.0 * np.pi * sigFreq / Fs * np.arange(0, len(signal)))
baseband = signal * mixer
lpf_b, lpf_a = scipy.signal.butter(4, LPF_FREQ / Fs)
baseband = scipy.signal.lfilter(lpf_b, lpf_a, baseband)

phase = np.unwrap(np.angle(baseband))
freq = np.diff(phase)
demod = freq - np.mean(freq)

# Upsample signal - simply repeat samples instead of using scipy.signal.resample (much faster)
print("Upsampling demodulated signal...")
#demod = scipy.signal.resample(demod, int(len(demod) * DEMOD_SAMPRATE / Fs))
demod = [demod[int(i * Fs / DEMOD_SAMPRATE)] for i in range(int(len(demod) * DEMOD_SAMPRATE / Fs))]

############################
#       Find preamble      #
############################
print("Finding preamble...")
preamble = [s for s in DOWNLINK_PREAMBLE for _ in range(int(DEMOD_SAMPRATE / DOWNLINK_BAUDRATE))]
xcorr = scipy.signal.correlate(demod, preamble, mode="valid")
preamble_offset = np.argmax(xcorr)

############################
#   DC Offset Correction   #
############################
# Estimate DC offset by calculating mean value of signal during preamble
demod = demod - np.mean(demod[preamble_offset:preamble_offset + len(preamble)])

plt.plot(demod)
plt.plot([0] * preamble_offset + preamble)# + ([1] * int(DEMOD_SAMPRATE / DOWNLINK_BAUDRATE) + [-1] * int(DEMOD_SAMPRATE / DOWNLINK_BAUDRATE)) * 60)
plt.show()

############################
#       Turn into bits     #
############################
samples_per_symbol = DEMOD_SAMPRATE / DOWNLINK_BAUDRATE

bits = []
for i in (int(preamble_offset + samples_per_symbol / 2) + np.arange(0, samples_per_symbol * (len(DOWNLINK_PREAMBLE) + 120), samples_per_symbol)):
	if demod[int(i)] < 0:
		bits = bits + [0]
	else:
		bits = bits + [1]

bitstring = "".join([str(b) for b in bits])
print(bitstring)
hexstring = str(hex(int(bitstring, 2)))
print("Payload: " + hexstring[int(2 + len(DOWNLINK_PREAMBLE) / 4):])
