#!/usr/bin/env python
from os import environ, path
import argparse

from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *

MODELDIR = "pocketsphinx/model"
DATADIR = "pocketsphinx/test/data"

# Create a decoder with certain model
config = Decoder.default_config()
config.set_string('-hmm', path.join(MODELDIR, 'en-us/en-us'))
config.set_string('-lm', path.join(MODELDIR, 'en-us/en-us.lm.bin'))
config.set_string('-dict', path.join(MODELDIR, 'en-us/cmudict-en-us.dict'))

if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--filepath', type=str, help='File path to audio file to process; can be relative or absolute', default='')
  args = arg_parser.parse_args()
  # Decode streaming data.
  decoder = Decoder(config)
  decoder.start_utt()
  stream = open(path.expanduser(args.filepath) or path.join(DATADIR, 'goforward.raw'), 'rb')
  while True:
    buf = stream.read(1024)
    if buf:
      decoder.process_raw(buf, False, False)
    else:
      break
  decoder.end_utt()
  print ('Best hypothesis segments: ', [seg.word for seg in decoder.seg()])
