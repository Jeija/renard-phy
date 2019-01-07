#!/usr/bin/env python3
# renard-phy downlink demodulation script

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
parser.add_argument("wavfile", help="WAV file containing the downlink recording to be demodulated")
parser.add_argument("-p", "--plot", action="store_true", default = False, help = "Use matplotlib to display the baseband signal and where in that baseband signal a preamble was detected")
parser.add_argument("-d", "--decode", action="store_true", default = False, help = "Use renard to brute-force-decode payload content of downlink frame")
parser.add_argument("-f", "--file", help = "Write frame contents to the specified PCAP file")
args = parser.parse_args()

###########################
#       Definitions       #
###########################
# Baudrate of the downlink signal
DOWNLINK_BAUDRATE = 600

# Minimum frequency resolution of FFT frequency estimation
MIN_FREQ_RES = 10

# Cutoff frequency for low pass filter applied after mixing
LPF_FREQ = 3500

# Symbols that the downlink preamble consists of (antipodally mapped 0x2aaaaaaaaaaaaaaaaaaaaab227)
DOWNLINK_PREAMBLE = [-1] + [-1, 1] * 45 + [1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1]

# Demodulated signal is upsampled to this sample rate to allow for more accurate preamble detection
DEMOD_SAMPRATE = 300000

# Length of Sigfox downlink frame in bits (excluding preamble)
FRAMELENGTH_WITHOUT_PREAMBLE = 120

# Total length of Sigfox downlink frame, including preamble
FRAMELENGTH = FRAMELENGTH_WITHOUT_PREAMBLE + len(DOWNLINK_PREAMBLE)

# If CLI option is specified, use renard to downlink uplink frame
RENARD_BIN = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, "renard", "build", "renard")
RENARD_CMD = [RENARD_BIN] + ["dldecode"]

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
class SigfoxDownlinkPCAP(Packet):
    name = "SigfoxPacket "
    fields_desc = [StrField("Frame", "")]

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
nfft = nextpow2(Fs / MIN_FREQ_RES)
print("Estimating frequency using " + str(nfft) + "-length FFT")
f, spec = scipy.signal.welch(signal, Fs, noverlap = 0, nperseg = nfft / 4, nfft = nfft, return_onesided = False)

# Calculate spectral centroid and use frequency value as estimated signal frequency
sigFreq = sum(f * abs(spec)) / sum(abs(spec))

############################
#      Mix to Baseband     #
############################
mixer = np.exp(-1.0j * 2.0 * np.pi * sigFreq / Fs * np.arange(0, len(signal)))
baseband = signal * mixer
lpf_b, lpf_a = scipy.signal.butter(4, LPF_FREQ / Fs)
baseband = scipy.signal.lfilter(lpf_b, lpf_a, baseband)

# arctan demodulator: get FM baseband singnal
phase = np.unwrap(np.angle(baseband))
freq = np.diff(phase)
demod = freq - np.mean(freq)

# Upsample baseband signal: Simply repeat samples instead of using scipy.signal.resample (much faster)
#demod = scipy.signal.resample(demod, int(len(demod) * DEMOD_SAMPRATE / Fs))
demod = [demod[int(i * Fs / DEMOD_SAMPRATE)] for i in range(int(len(demod) * DEMOD_SAMPRATE / Fs))]

############################
#       Find Preamble      #
############################
print("Finding preamble...")
samples_per_symbol = DEMOD_SAMPRATE / DOWNLINK_BAUDRATE
preamble = [s for s in DOWNLINK_PREAMBLE for _ in range(int(samples_per_symbol))]
xcorr = scipy.signal.correlate(demod, preamble, mode = "valid")
preamble_offset = np.argmax(xcorr)

############################
#   DC Offset Correction   #
############################
# Estimate DC offset by calculating mean value of signal during preamble
demod = demod - np.mean(demod[preamble_offset:preamble_offset + len(preamble)])

############################
#      Optional: Plot      #
############################
if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(demod)
	plt.plot(preamble_offset + range(len(preamble)), preamble)
	plt.show()

############################
#       Turn into Bits     #
############################
first_sample_offset = int(preamble_offset + samples_per_symbol / 2)

bits = ""
for i in (np.arange(0, FRAMELENGTH) * samples_per_symbol + first_sample_offset):
	bits = bits + ("0" if demod[int(i)] < 0 else "1")

hexstring = str(hex(int(bits, 2)))
hexstring_without_preamble = hexstring[int(2 + len(DOWNLINK_PREAMBLE) / 4):]
print("Frame: " + hexstring_without_preamble)

if args.file:
	print("Writing frame to PCAP file: " + args.file)
	pkt = SigfoxDownlinkPCAP()
	pkt.Frame = hexstring_without_preamble
	pkt.time = wavfile_mtime + preamble_offset / Fs
	# use linktype 147 which is reserved for private use
	wrpcap(args.file, pkt, append = True, linktype = 147)

if args.decode:
	try:
		print(subprocess.check_output(RENARD_CMD + ["-f", hexstring_without_preamble, "-c"], stderr = subprocess.STDOUT).decode('utf-8'))
	except subprocess.CalledProcessError as e:
		print(RENARD_BIN + " subprocess failed. Error:")
		print(e.output.decode("UTF-8"))
		sys.exit(0)
