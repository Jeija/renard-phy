#!/usr/bin/env python3

import sys

import FoxSignal

sig = FoxSignal.Intermediate(sys.argv[1])
bursts = sig.decodeSigfoxBursts(8)
