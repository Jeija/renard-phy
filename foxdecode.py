#!/usr/bin/env python3

import sys

import FoxSignal

sig = FoxSignal.Intermediate(sys.argv[1])
bursts = sig.findSigfoxBursts(0.1, 30, 8)
