import os
import sys
import requests
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Pre-encode text files into tokenized training set.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--vocab', action='store_true', help='Download only encoder.json, hparams.json, and vocab.bpe?')
parser.add_argument('models', metavar='MODEL', type=str, default=['117M'], nargs='*', help='Pretrained model name(s)')

def main(args):
  for model in tqdm(args.models):
    subdir = os.path.join('models', model)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\','/') # needed for Windows

    vocab_files = ['encoder.json','hparams.json','vocab.bpe']
    files = vocab_files + (['checkpoint','model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta'] if not args.vocab else [])
    for filename in tqdm(files):

        r = requests.get("https://storage.googleapis.com/gpt-2/" + subdir + "/" + filename, stream=True)

        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

