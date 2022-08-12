#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022
# Author: Jan Profant <jan.profant@rev.com>
# All Rights Reserved

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
import yaml

from pyannote.database.util import load_rttm
import wandb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_shuffle', required=True, choices=['True', 'False'],
                        help='Shuffle time-axis order before input to the network')
    parser.add_argument('--estimate_spk_qty', required=True, type=int)
    parser.add_argument('--estimate_spk_qty_thr', required=True, type=float)
    parser.add_argument('--threshold', required=True, type=float)
    parser.add_argument('--median_window_length', required=True, type=int)
    parser.add_argument('--wder_script', required=True, type=Path,
                        help='path to wder.py script to obtain metrics')
    parser.add_argument('--wder_interpreter', required=True, type=Path)
    parser.add_argument('--wer_results', required=True, type=Path,
                        help='path to wer results directory')
    parser.add_argument('--infer_data_dir', required=True, type=Path,
                        help='path to kaldi directory with dev suite')
    parser.add_argument('--infer_config', required=True, type=Path,
                        help='path to generic inference config')
    parser.add_argument('--models_path', required=True, type=Path)

    args = parser.parse_args()
    args_as_kwargs = vars(args)

    wandb.login(host="http://wandb.speech-rnd.internal", key="local-473ad2cf1f9ed9023faf837048e75943e1bbe7c5")
    wandb.init(project='DIAR-93', config=args)

    config = wandb.config

    tmp_dir = Path(tempfile.mkdtemp(dir=os.path.dirname(os.path.realpath(__file__))))
    print(tmp_dir)
    new_yaml_config = tmp_dir / 'infer_config.yaml'
    with open(new_yaml_config, 'w') as fw:
        with open(args.infer_config) as fr:
            yaml_dict = yaml.safe_load(fr)
            for key in yaml_dict:
                if key in args_as_kwargs:
                    if key == 'time_shuffle':
                        args_as_kwargs[key] = True if args_as_kwargs[key] == 'True' else False
                    if isinstance(args_as_kwargs[key], Path):
                        args_as_kwargs[key] = str(args_as_kwargs[key])
                    yaml_dict[key] = args_as_kwargs[key]
                elif key == 'rttms_dir':
                    yaml_dict[key] = str(tmp_dir / 'rttms')

            yaml.safe_dump(yaml_dict, fw, indent=4, width=120)

    # call inference code
    subprocess.check_call(f'python infer.py -c {new_yaml_config}', shell=True)

    # create .seg files so we can evaluate wder metric
    for rttm in tmp_dir.rglob('rttms/*.rttm'):
        basename = os.path.splitext(os.path.basename(rttm))[0]
        seg_dir = tmp_dir / 'segs' / basename
        os.makedirs(seg_dir, exist_ok=True)
        with (seg_dir / 'show.seg').open('w') as fw:
            annot = load_rttm(rttm)
            if len(annot) > 0:
                assert list(annot.keys()) == [basename], f'Got {annot.keys()} instead of {basename} in file {rttm}'
                for segment, track, label in annot[basename].itertracks(yield_label=True):
                    print(segment, segment.start, segment.end)
                    fw.write(f'{basename} 0 {int(segment.start * 100)} {int((segment.end - segment.start) * 100)} S U U {label}\n')

    subprocess.check_call(f'{args.wder_interpreter} {args.wder_script} '
                          f'-s {tmp_dir / "segs"} '
                          f'-n {args.wer_results} '
                          f'-o {tmp_dir / "wder.json"}',
                          shell=True)
    with (tmp_dir / 'wder.json').open('r') as f:
        json_dict = json.load(f)
        print(json_dict)

    wandb.log({
        'WDER': json_dict['cumulative']['wder'],
        'Speaker Switch WDER': json_dict['cumulative']['0.18344013066968212'],
    })
