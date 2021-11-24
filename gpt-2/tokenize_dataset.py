#!/usr/bin/env python3
import argparse
import numpy as np

from tokenizers import Tokenizer, models, pre_tokenizers, decoders

parser = argparse.ArgumentParser(
    description='Pre-encode text files into tokenized training set.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate files with <|endoftext|> separator into chunks of this minimum size')
parser.add_argument('-s', '--step', type=int, default=128*1024, help='Number of lines to encode at a time')
parser.add_argument('-b', '--batch', action='store_true', default=False, help='Use tokenizer.encode_batch')
parser.add_argument('-c', '--compression', action='store_true', default=False, help='Save using compression (via .savez_compressed)')
parser.add_argument('in_text', metavar='PATH', type=str, help='Input file')
parser.add_argument('out_npz', metavar='OUT.npz', type=str, default='', nargs='?', help='Output file path')
args = parser.parse_args()

# Initialize a tokenizer based on BPE
vocab = "./models/%s/encoder.json" % args.model_name
merges = "./models/%s/vocab.bpe" % args.model_name
tokenizer = Tokenizer(models.BPE.from_files(vocab, merges))

# Use the byte level 
add_prefix_spaces = False # Whether to automatically prefix the sequences with a space if none found
tokenizer.with_pre_tokenizer(pre_tokenizers.ByteLevel.new(add_prefix_spaces))
tokenizer.with_decoder(decoders.ByteLevel.new())

# Setup truncation if needed
truncate = False
max_length = 1024

if truncate:
  stride = 0
  strategy = 'longest_first' # Can also be `only_first` or `only_second`
  tokenizer.with_truncation(max_length, stride, strategy)

# Setup padding if needed
padding = False
# Whether to always pad to max_length. If this is false, we will pad to the
# longest sequence in the batch.
pad_to_max_length = False
padding_side = "right" # Can also be "left"
pad_token_id = 0
pad_token_type_id = 0
pad_token = "[PAD]"

if padding:
  tokenizer.with_padding(
    max_length if pad_to_max_length else None,
    padding_side,
    pad_token_id,
    pad_token_type_id,
    pad_token
  )

# http://stackoverflow.com/questions/1624883/alternative-way-to-split-a-list-into-groups-of-n
import itertools
def group(n, iterable, fillvalue=None):
    "group(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

import tflex_utils
import tqdm
import time
start = time.time()
optional_pair_sequence = None
tokens = []
if args.batch:
  with open(args.in_text) as f:
    print('Reading...')
    lines = f.readlines()
    print(repr(lines[0]))
  batches = [x for x in group(args.step, lines, fillvalue='\n')]
  for batch in tqdm.tqdm(batches):
    for encoding in tokenizer.encode_batch([x for x in batch]):
      tokens.extend(encoding.ids)
      elapsed = time.time() - start
      print('%d tokens in %.4fs (%.4f tokens/sec)' % (len(tokens), elapsed, len(tokens)/elapsed))
else:
  for i, line in tflex_utils.for_each_line(args.in_text):
    encoding = tokenizer.encode(line, optional_pair_sequence)
    tokens.extend(encoding.ids)
    if i % args.step == 0:
      elapsed = time.time() - start
      print('%d tokens in %.4fs (%.4f tokens/sec)' % (len(tokens), elapsed, len(tokens)/elapsed))
elapsed = time.time() - start
print('%d tokens in %.4fs (%.4f tokens/sec)' % (len(tokens), elapsed, len(tokens)/elapsed))
if args.out_npz and len(args.out_npz) > 0:
  print('Saving to %s...' % args.out_npz)
  if args.compression:
    np.savez_compressed(args.out_npz, tokens)
  else:
    np.savez(args.out_npz, tokens)

