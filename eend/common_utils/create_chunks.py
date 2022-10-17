#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022
# Author: Jan Profant <jan.profant@rev.com>
# All Rights Reserved

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np

from eend.common_utils.kaldi_data import KaldiData


def trim_wav(wav_path, output_wav_path, start, end):
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    subprocess.check_call([f'sox {wav_path} {output_wav_path} trim {start} {end - start}'], shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_kaldi_dir', type=Path, help='path to input kaldi directory')
    parser.add_argument('input_wer_dir', type=Path, help='path to input wer results directory')
    parser.add_argument('output_kaldi_dir', type=Path, help='path to out kaldi directory with chunks')
    parser.add_argument('output_wer_dir', type=Path, help='path to output wer results directory with chunks')
    parser.add_argument('--chunk-size', type=float, default=180.00, help='size of chunks in seconds')

    reco2dur, utt2spk = {}, {}
    args = parser.parse_args()

    input_kaldi_data = KaldiData(args.input_kaldi_dir)

    os.makedirs(args.output_kaldi_dir / 'audio', exist_ok=True)

    with \
            (args.output_kaldi_dir / 'wav.scp').open('w') as fwavscp, \
            (args.output_kaldi_dir / 'utt2spk').open('w') as futt2spk, \
            (args.output_kaldi_dir / 'spk2utt').open('w') as fspk2utt, \
            (args.output_kaldi_dir / 'segments').open('w') as fsegments, \
            (args.output_kaldi_dir / 'reco2dur').open('w') as freco2dur:

        for wav in input_kaldi_data.wavs:
            # create new kaldi directory
            dur = input_kaldi_data.reco2dur[wav]
            for chunk_start in np.arange(0, dur, args.chunk_size):
                chunk_end = chunk_start + args.chunk_size if chunk_start + args.chunk_size < dur else dur
                chunk_name = f'{wav}_{int(chunk_start * 100):08d}-{int(chunk_end * 100):08d}'
                chunk_seg_name = f'{chunk_name}_seg'
                output_wav_path = args.output_kaldi_dir / 'audio' / f'{chunk_name}.wav'
                seg_speaker = input_kaldi_data.utt2spk[f'{wav}_seg']

                # trim_wav(input_kaldi_data.wavs[wav], output_wav_path, chunk_start, chunk_end)

                fwavscp.write(f'{chunk_name} {output_wav_path}\n')
                freco2dur.write(f'{chunk_name} {chunk_end - chunk_start:.2f}\n')
                fsegments.write(f'{chunk_seg_name} {chunk_name} 0.00 {chunk_end - chunk_start:.2f}\n')
                futt2spk.write(f'{chunk_seg_name} {seg_speaker}\n')
                fspk2utt.write(f'{seg_speaker} {chunk_seg_name}\n')

            # create new *.aligned.nlp files
            os.makedirs(args.output_wer_dir, exist_ok=True)
            input_aligned_nlp = args.input_wer_dir / f'{wav}.aligned.nlp'
            if not os.path.isfile(input_aligned_nlp):
                continue
            with input_aligned_nlp.open() as finnlp:
                header = finnlp.readline()
                old_line = None
                for chunk_start in np.arange(0, dur, args.chunk_size):
                    chunk_end = chunk_start + args.chunk_size if chunk_start + args.chunk_size < dur else dur
                    chunk_name = f'{wav}_{int(chunk_start * 100):08d}-{int(chunk_end * 100):08d}'
                    with (args.output_wer_dir / f'{chunk_name}.aligned.nlp').open('w') as foutnlp:
                        foutnlp.write(header)
                        if old_line is not None:
                            foutnlp.write(old_line)
                            old_line = None

                        for line in finnlp:
                            try:
                                end = float(line.split('|')[3])
                            except ValueError:
                                foutnlp.write(line)
                                continue
                            if end > chunk_end:
                                old_line = line
                                break
                            else:
                                foutnlp.write(line)


