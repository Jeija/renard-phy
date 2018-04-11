#!/usr/bin/env python3

def nextpow2(limit):
	n = 1
	while n < limit:
		n = n * 2
	return int(n / 2)
