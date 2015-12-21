#!/usr/bin/env python
from os import environ, path
from multiprocessing import Pool, dummy
import argparse
import wave
import subprocess

from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *

MODELDIR = "pocketsphinx/model"
DATADIR = "pocketsphinx/test/data"

# Create a decoder with certain model
config = Decoder.default_config()
config.set_string('-hmm', path.join(MODELDIR, 'en-us/en-us'))
config.set_string('-lm', path.join(MODELDIR, 'en-us/en-us.lm.bin'))
config.set_string('-dict', path.join(MODELDIR, 'en-us/cmudict-en-us.dict'))
# decoder = Decoder(config)

# Decode streaming data.
# decoder = Decoder(config)
# decoder.start_utt()

def inner_decoder(info):
    decoder = None
    if info:
        decoder = Decoder(config)
        decoder.start_utt()
        decoder.process_raw(info, True, True)
        decoder.end_utt()
    return [seg.word for seg in decoder.seg()] if decoder else []

def stream_decoder(stream, buff_size, offset, ending_offset, cut_off_allow, total_frames, frame_rate, is_last):
    stream_0 = wave.open(stream, 'rb')
    normative_offset = frame_rate * offset
    if normative_offset and cut_off_allow:
        # go back 5 seconds, helps with potential clipping between partitions
        if normative_offset - (cut_off_allow * frame_rate):
            normative_offset -= (cut_off_allow * frame_rate)
    try:
        stream_0.setpos(normative_offset)
    except wave.Error:
        return []
    # stream_pool = Pool(processes=2)
    inner_results = []
    decoder = Decoder(config)
    decoder.start_utt()
    normative_ending_offset = (offset + ending_offset) * frame_rate
    if cut_off_allow and (normative_ending_offset + (cut_off_allow * frame_rate) < total_frames):
        normative_ending_offset += (cut_off_allow * frame_rate)
    frames_to_read = (normative_ending_offset - normative_offset) or int((ending_offset - offset)*frame_rate)
    last_frame_index = frames_to_read + normative_offset
    frames_to_read = total_frames - stream_0.tell() if frames_to_read + stream_0.tell() > total_frames else frames_to_read
    frames_to_read = frames_to_read if frames_to_read > buff_size else buff_size + 1
    cur_frame = normative_offset
    gone_in_yet = False
    while ((cur_frame + buff_size) <= last_frame_index or not gone_in_yet) or is_last:
        # print(frames_to_read)
        gone_in_yet = True
        try:
            buf = stream_0.readframes(buff_size)
            cur_frame = stream_0.tell()
            if buf:
                decoder.process_raw(buf, False, False)
            else:
                break
        except Exception as e:
            import traceback
            print(normative_ending_offset - normative_offset, stream_0.tell(), total_frames)
            print(traceback.format_exc(e))
    # stream_pool.close()
    # stream_pool.join()
    stream_0.close()
    decoder.end_utt()
    return [[seg.word for seg in decoder.seg()], ('start frame', normative_offset), ('end frame', last_frame_index), ('file position', stream_0.tell())]

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('filepath', type=str, help='File path to audio file to process; can be relative or absolute')
    arg_parser.add_argument('--concurrency', default=5, type=int, help='Number of concurrent workers to use.')
    arg_parser.add_argument('--concurrency-type', default='p', choices=['p', 't'], help='type of concurrency the system will use (process, thread)')
    arg_parser.add_argument('--partition-size', default=10, type=int, help='define how long each partition should be, in seconds')
    arg_parser.add_argument('--window-bleed', default=5, type=int, help='how much time we should bleed before or after partition size; may help reduce word cut offs.')
    arg_parser.add_argument('--convert-file', default=True, type=bool, help='does the input file need to be converted for usage? To appropriate wav type. Will create a converted file with UNIX time in the name, too.')
    args = arg_parser.parse_args()
    source_path = path.expanduser(args.filepath)
    import time
    wav_source = source_path + '.' + str(int(time.time())) + '.wav'
    if args.convert_file:
        print(source_path)
        print(wav_source)
        subprocess.call(['ffmpeg', '-i', source_path, '-acodec','pcm_s16le','-ac','1','-ar','16000',wav_source])
    else:
        wav_source = source_path
    results = []
    main_pool = Pool(processes=args.concurrency) if args.concurrency_type == 'p' else dummy.Pool(processes=args.concurrency)
    handler = wave.open(source_path)
    total_frames = handler.getnframes()
    duration = total_frames / handler.getframerate()
    enumerations = duration / args.partition_size
    print(enumerations)
    import sys
    # sys.exit(1)
    for offset in xrange(enumerations):
        offset_start = offset * args.partition_size
        offset_end = ((offset + 1)* args.partition_size) - 1
        is_last = (offset + 1) == enumerations
        results.append(main_pool.apply_async(stream_decoder, (wav_source, 1024, offset_start, args.partition_size, args.window_bleed, total_frames, handler.getframerate(), is_last)))
    print ('Best hypothesis segments: ', [result.get()[0] for result in results if result.get()])
    print([result.get()[1:] for result in results if result.get() and len(result.get()) > 1])
    print('Number of frames: {}, duration: {}'.format(total_frames, duration))
    handler.close()

