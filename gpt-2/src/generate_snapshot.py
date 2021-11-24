#!/usr/bin/env python3

import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')]
sys.path += [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))]

import fire
import json
import numpy as np
import tensorflow as tf

import model, sample, encoder

import tflex

@tflex.register_command
def clear_context():
  tflex.reset_context()
  print('')
  print('')
  print('')

def clear_output(wait=False):
  import subprocess, platform
  if platform.system()=="Windows":
      subprocess.Popen("cls", shell=True).communicate()
  else:
      print("\033c", end="")

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def interact_model(
    model_name='117M',
    restore_from=None,
    seed=None,
    nsamples=1,
    step=1,
    length=64,
    prompt="\n",
    clear=None,
    maxlen=-1,
    temperature=1,
    top_k=0,
    top_p=0,
    penalize=0
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :step=1 : Number of tokens to generate at a time
    :length=64 : Window size; use 1024 for maximum size per sample
    :prompt="\\n" : Prompt to start with. The default of "" prompts with an <|endoftext|> token.
    :clear=None : If this string is encountered, clear the context window.
    :maxlen=-1 : if this many tokens are generated without
     encountering --clear, then print it and clear the context window.
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    :penalize=0.0 : Float value controlling "used" penalty. Implements repetition
     reduction (similar to CTRL) if set to a value > 0. A decent setting might be 0.85
     with temperature 0.3 and top_k 40.
    """
    batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length > hparams.n_ctx:
        raise ValueError("Length can't be largeer than n_ctx: %s" % hparams.n_ctx)
    if step > length:
        raise ValueError("Can't get samples longer than length: %s" % length)

    with tflex.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=step,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p, penalize=penalize
        )

        saver = tflex.Saver(reshape=True)
        if restore_from is None:
          restore_from = os.path.join('models', model_name)
        ckpt = tflex.latest_checkpoint(restore_from)
        saver.restore(sess, ckpt)
        saver2 = tf.train.Saver()
        counter = int(ckpt.split('-')[-1].split('.')[0])
        saver2.save(
        sess,
        os.path.join('saved', 'model'),
        global_step=counter)

if __name__ == '__main__':
    fire.Fire(interact_model)

