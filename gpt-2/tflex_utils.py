import tqdm
import sys

def for_each_line(infile, verbose=True, ignore_errors=True, message=None):
    n = 0
    prev = None
    with open(infile) as f:
      while True:
        try:
          for line in f:
            n += 1
            prev = line
            #print(n)
          break
        except UnicodeDecodeError:
          if verbose:
            sys.stderr.write('Error on line %d after %s\n' % (n+1, repr(prev)))
          if not ignore_errors:
            raise
    if message:
      print('%s %d lines...' % (message, n))
    i = 0
    with open(infile) as f:
      while True:
        try:
          n -= i
          for line in tqdm.tqdm(f, total=n) if verbose else f:
            yield i, line
            i += 1
          break
        except UnicodeDecodeError:
          pass

