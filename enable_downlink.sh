#!/bin/bash
# This script switches to a version of librenard that implements the
# downlink (de)scrambling algorithm. This version is not publicly accessible.

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $CWD
rm -rf renard
git clone https://github.com/Jeija/renard-full renard
cd renard
git clone https://github.com/Jeija/librenard-full librenard
make
cd ..
