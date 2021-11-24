#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./encode.py <file|directory|glob> /path/to/output.npz
#  PYTHONPATH=src ./train --dataset /path/to/output.npz

import argparse
import numpy as np
import sys
import tqdm

from ftfy import fix_text

import tflex_utils

parser = argparse.ArgumentParser(
    description='Use FTFY to prepare a dataset for training.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('infile', metavar='PATH', type=str, help='Input file, directory, or glob pattern (utf-8 text).')
parser.add_argument('--outfile', default="-", type=str, help='Output file path, or - for stdout')

def main():
    args = parser.parse_args()
    out = sys.stdout if args.outfile == '-' else open(args.outfile, "w")
    for i, line in tflex_utils.for_each_line(args.infile, message='Fixing'):
      fixed = fix_text(line)
      out.write(fixed)
      if i % 100 == 0:
        out.flush()

if __name__ == '__main__':
    main()
