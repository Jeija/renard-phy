#!/usr/bin/env python3

import scipy, scipy.signal, scipy.io, scipy.io.wavfile
import numpy as np
import subprocess
import argparse
import sys
import os
# add support to write our frames to a pcap file
from scapy.all import wrpcap, conf, Packet, StrField
 
###########################
#    Parse CLI argumets   #
###########################
parser = argparse.ArgumentParser()
parser.add_argument("wavfile", help="WAV file containing the uplink recording to be demodulated")
parser.add_argument("-p", "--plot", action="store_true", default = False, help = "Use matplotlib to display the baseband signal and where in that baseband signal a preamble was detected")
parser.add_argument("-d", "--decode", action="store_true", default = False, help = "Use renard to decode content of uplink frame")
parser.add_argument("-f", "--file", help = "Write frame contents to the specified PCAP file")
args = parser.parse_args()

###########################
#       Definitions       #
###########################
# TODO: document config
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

FTYPE_LEN_BITS = 12
CRCLEN_BITS = 16

PACKETLEN_BY_FTYPE = {
	0x06b :  8, 0x6e0 :  8, 0x034 :  8, # class A
	0x08d :  9, 0x0d2 :  9, 0x302 :  9, # class B
	0x35f : 12, 0x598 : 12, 0x5a3 : 12, # class C
	0x611 : 16, 0x6bf : 16, 0x72c : 16, # class D
	0x94c : 20, 0x971 : 20, 0x997 : 20, # class E
	0xf67 : 16                          # OOB
}

# If CLI option is specified, use renard to decode uplink frame
RENARD_BIN = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, "renard", "build", "renard")
RENARD_CMD = [RENARD_BIN] + ["uldecode"]

###########################
#    Utility Functions    #
###########################
def nextpow2(limit):
	n = 1
	while n < limit:
		n = n * 2
	return int(n / 2)

###########################
# Wrapper Class for Scapy #
###########################
class SigfoxUplinkPCAP(Packet):
    name = "SigfoxPacket "
    fields_desc = [StrField("Frame", "")]


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
wavfile_mtime = os.path.getmtime(args.wavfile)

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
#    Preamble Detection    #
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

# TODO: make configurable somehow
preamble_offsets, _ = scipy.signal.find_peaks(xcorr, distance = samples_per_symbol * 10, height = 0.5 * xcorrmax, prominence = 0.5 * xcorrmax)

if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(np.real(baseband))
	for preamble_offset in preamble_offsets:
		plt.plot(preamble_offset + range(len(preamble)), preamble)
	plt.show()

print("Found " + str(len(preamble_offsets)) + " Preambles!")

###########################
#    Sample and Output    #
###########################
def hammingDistance(str1, str2):
	return sum([1 if str1[i] == str2[i] else 0 for i in range(len(str1))])

for count, preamble_offset in enumerate(preamble_offsets):
	data_offset = int(preamble_offset + SILENCE_BEFORE_PREAMBLE * Fs)

	# calculate all symbol differentials in baseband starting from frame offsets
	bitstring = ""
	samplingInstants = np.arange(data_offset + 3 / 2 * samples_per_symbol, len(baseband), int(round(samples_per_symbol)))
	for j in samplingInstants:
		last_symbol = baseband[int(round(j - samples_per_symbol))]
		this_symbol = baseband[int(round(j))]
		differential = np.conj(last_symbol) * this_symbol
		bitstring += "0" if differential < 0 else "1"

	# determine packet length: find frame type with smallest hamming distance
	print("[Frame " + str(count) + "]: Determining Frame Type...")
	ftype = bitstring[len(UPLINK_PREAMBLE) - 1:len(UPLINK_PREAMBLE) + FTYPE_LEN_BITS - 1]

	min_hammdist = FTYPE_LEN_BITS
	min_hammdist_ftype = 0
	for candidate in PACKETLEN_BY_FTYPE:
		if hammingDistance("{:012b}".format(candidate), ftype) < min_hammdist:
			min_hammdist_ftype = ftype

	best_ftype = int(min_hammdist_ftype, 2)

	if not best_ftype in PACKETLEN_BY_FTYPE:
		print("[Frame " + str(count) + "]: Frame type 0x{:03x} not found, ignoring".format(int(ftype, 2)))
		print("[Frame " + str(count) + "]: Raw bitstring is " + bitstring)
		continue

	packetlen = PACKETLEN_BY_FTYPE[best_ftype]
	print("[Frame " + str(count) + "]: Frame Type is 0x{:03x}".format(best_ftype) + ", Packet Length " + str(packetlen) + " bytes")

	# extract frame bits and display corresponding hexadecimal representation
	frame_without_preamble = bitstring[(len(UPLINK_PREAMBLE) - 1):(len(UPLINK_PREAMBLE) + FTYPE_LEN_BITS + packetlen * 8 + CRCLEN_BITS - 1)]
	nibblecount = int((FTYPE_LEN_BITS + packetlen * 8 + CRCLEN_BITS) / 4)
	hexstring = ("{:0" + str(nibblecount) + "x}").format(int(frame_without_preamble, 2))
	print("[Frame " + str(count) + "]: Content: "  + hexstring)

	if args.decode:
		print("[Frame " + str(count) + "]: Decoding with renard:")
		try:
			print(subprocess.check_output(RENARD_CMD + ["-f", hexstring], stderr = subprocess.STDOUT).decode('utf-8'))
		except subprocess.CalledProcessError as e:
			print(RENARD_BIN + " subprocess failed. Error:")
			print(e.output.decode("UTF-8"))
			sys.exit(0)

	if args.file:
		print("[Frame " + str(count) + "]: Writing to PCAP file: " + args.file)
		pkt = SigfoxUplinkPCAP()
		pkt.Frame = hexstring
		pkt.time = wavfile_mtime + data_offset / Fs
		# use linktype 147 which is reserved for private use
		wrpcap(args.file, pkt, append = True, linktype = 147)

