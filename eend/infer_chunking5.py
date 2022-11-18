#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

from backend.models import (
    average_checkpoints,
    get_model,
)
from backend.clustering import twoGMMcalib_lin, AHC, l2_norm, cos_similarity
from common_utils.diarization_dataset import KaldiDiarizationDataset
from common_utils.gpu_utils import use_single_gpu
from os.path import join
from pathlib import Path
from scipy.signal import medfilt
from torch.utils.data import DataLoader
from train import _convert
from types import SimpleNamespace
from typing import TextIO
from pyannote.core import Annotation, Segment
import logging
import numpy as np
import os
import random
import torch
import yamlargparse
from collections import defaultdict


def get_infer_dataloader(args: SimpleNamespace) -> DataLoader:
    infer_set = KaldiDiarizationDataset(
        args.infer_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
        subsampling=args.subsampling,
        use_last_samples=True,
        min_length=0,
    )
    infer_loader = DataLoader(
        infer_set,
        batch_size=1,
        collate_fn=_convert,
        num_workers=1,
        shuffle=False,
        worker_init_fn=_init_fn,
    )

    Y, _, _ = infer_set.__getitem__(0)
    assert Y.shape[1] == \
        (args.feature_dim * (1 + 2 * args.context_size)), \
        f"Expected feature dimensionality of \
        {args.feature_dim} but {Y.shape[1]} found."
    return infer_loader


def hard_labels_to_rttm(
    labels: np.ndarray,
    id_file: str,
    rttm_file: TextIO,
    frameshift: float = 10
) -> None:
    """
    Transform NfxNs matrix to an rttm file
    Nf is the number of frames
    Ns is the number of speakers
    The frameshift (in ms) determines how to interpret the frames in the array
    """
    if len(labels.shape) > 1:
        # Remove speakers that do not speak
        non_empty_speakers = np.where(labels.sum(axis=0) != 0)[0]
        labels = labels[:, non_empty_speakers]

    # Add 0's before first frame to use diff
    if len(labels.shape) > 1:
        labels = np.vstack([np.zeros((1, labels.shape[1])), labels])
    else:
        labels = np.vstack([np.zeros(1), labels])
    d = np.diff(labels, axis=0)

    spk_list = []
    ini_list = []
    end_list = []
    if len(labels.shape) > 1:
        n_spks = labels.shape[1]
    else:
        n_spks = 1
    for spk in range(n_spks):
        if n_spks > 1:
            ini_indices = np.where(d[:, spk] == 1)[0]
            end_indices = np.where(d[:, spk] == -1)[0]
        else:
            ini_indices = np.where(d[:] == 1)[0]
            end_indices = np.where(d[:] == -1)[0]
        # Add final mark if needed
        if len(ini_indices) == len(end_indices) + 1:
            end_indices = np.hstack([
                end_indices,
                labels.shape[0] - 1])
        assert len(ini_indices) == len(end_indices), \
            "Quantities of start and end of segments mismatch. \
            Are speaker labels correct?"
        n_segments = len(ini_indices)
        for index in range(n_segments):
            spk_list.append(spk)
            ini_list.append(ini_indices[index])
            end_list.append(end_indices[index])
    for ini, end, spk in sorted(zip(ini_list, end_list, spk_list)):
        rttm_file.write(
            f"SPEAKER {id_file} 1 " +
            f"{round(ini * frameshift / 1000, 3)} " +
            f"{round((end - ini) * frameshift / 1000, 3)} " +
            f"<NA> <NA> spk{spk} <NA> <NA>\n")
        #yield round(ini * frameshift / 1000, 3), round(end * frameshift / 1000, 3), spk
        yield ini, end, spk


def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def postprocess_output(
    probabilities,
    subsampling: int,
    threshold: float,
    median_window_length: int
) -> np.ndarray:
    thresholded = probabilities > threshold
    filtered = np.zeros(thresholded.shape)
    for spk in range(filtered.shape[1]):
        filtered[:, spk] = medfilt(
            thresholded[:, spk],
            kernel_size=median_window_length)
    probs_extended = np.repeat(filtered, subsampling, axis=0)
    return probs_extended


def parse_arguments() -> SimpleNamespace:
    parser = yamlargparse.ArgumentParser(description='EEND inference')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('--context-size', default=0, type=int)
    parser.add_argument('--encoder-units', type=int,
                        help='number of units in the encoder')
    parser.add_argument('--epochs', type=str,
                        help='epochs to average separated by commas \
                        or - for intervals.')
    parser.add_argument('--feature-dim', type=int)
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--frame-shift', type=int)
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--hidden-size', type=int,
                        help='number of units in SA blocks')
    parser.add_argument('--infer-data-dir', help='inference data directory.')
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--log-report-batches-num', default=1, type=float)
    parser.add_argument('--median-window-length', default=11, type=int)
    parser.add_argument('--model-type', default='TransformerEDA',
                        help='Type of model (for now only TransformerEDA)')
    parser.add_argument('--models-path', type=str,
                        help='directory with model(s) to evaluate')
    parser.add_argument('--num-frames', default=-1, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int)
    parser.add_argument('--rttms-dir', type=str,
                        help='output directory for rttm files.')
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--subsampling', default=10, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--ahc-threshold', default=0.0, type=float)
    parser.add_argument('--transformer-encoder-n-heads', type=int)
    parser.add_argument('--transformer-encoder-n-layers', type=int)
    parser.add_argument('--transformer-encoder-dropout', type=float)
    parser.add_argument('--chunk-size', type=int, default=1000, help='chunk size of features pass to the encoder')

    attractor_args = parser.add_argument_group('attractor')
    attractor_args.add_argument(
        '--time-shuffle', action='store_true',
        help='Shuffle time-axis order before input to the network')
    attractor_args.add_argument('--attractor-loss-ratio', default=1.0,
                                type=float, help='weighting parameter')
    attractor_args.add_argument('--attractor-encoder-dropout',
                                default=0.1, type=float)
    attractor_args.add_argument('--attractor-decoder-dropout',
                                default=0.1, type=float)
    attractor_args.add_argument('--estimate-spk-qty', default=-1, type=int)
    attractor_args.add_argument('--estimate-spk-qty-thr',
                                default=-1, type=float)
    attractor_args.add_argument(
        '--detach-attractor-loss', default=False, type=bool,
        help='If True, avoid backpropagation on attractor loss')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    logging.info(args)

    infer_loader = get_infer_dataloader(args)

    if args.gpu >= 1:
        gpuid = use_single_gpu(args.gpu)
        logging.info('GPU device {} is used'.format(gpuid))
        args.device = torch.device("cuda")
    else:
        gpuid = -1
        args.device = torch.device("cpu")

    assert args.estimate_spk_qty_thr != -1 or \
        args.estimate_spk_qty != -1, \
        ("Either 'estimate_spk_qty_thr' or 'estimate_spk_qty' "
         "arguments have to be defined.")
    if args.estimate_spk_qty != -1:
        out_dir = join(args.rttms_dir, f"spkqty{args.estimate_spk_qty}_\
            thr{args.threshold}_median{args.median_window_length}")
    elif args.estimate_spk_qty_thr != -1:
        out_dir = join(args.rttms_dir, f"spkqtythr{args.estimate_spk_qty_thr}_\
            thr{args.threshold}_median{args.median_window_length}")

    model = get_model(args)

    model = average_checkpoints(
        args.device, model, args.models_path, args.epochs)
    model.eval()

    out_dir = join(
        args.rttms_dir,
        f"epochs{args.epochs}",
        f"timeshuffle{args.time_shuffle}",
        (f"spk_qty{args.estimate_spk_qty}_"
            f"spk_qty_thr{args.estimate_spk_qty_thr}"),
        f"detection_thr{args.threshold}",
        f"median{args.median_window_length}",
        "rttms"
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(infer_loader):
        input = torch.stack(batch['xs']).to(args.device)
        name = batch['names'][0]
        print(f'Processing {name}, signal shape {input.shape}.')
        speaker_means = []
        timings_dict = {}
        speaker_map = defaultdict(list)
        speakers = []
        for chunk_idx, chunk_start in enumerate(range(0, input.shape[1], args.chunk_size)):
            #print(f'Processing chunk number {chunk_idx + 1}')
            xs_chunk = input[:, chunk_start:chunk_start + args.chunk_size, :]
            with torch.no_grad():
                y_pred, emb = model.estimate_sequential(xs_chunk, args, return_emb=True)
                y_pred = y_pred[0]
            post_y = postprocess_output(
                y_pred, args.subsampling,
                args.threshold, args.median_window_length)
            #print(emb.shape)
            rttm_filename = join(out_dir, f"{name}_chunk{chunk_idx}.rttm")
            spk_dict = defaultdict(list)
            #timings_dict[chunk_idx] = defaultdict(list)
            with open(rttm_filename, 'w') as rttm_file:
                timings = list(hard_labels_to_rttm(post_y, name, rttm_file))
                timings_dict[chunk_idx] = timings
                speakers_in_chunk = set()
                for start, end, speaker in timings:
                    # FIXME not sure about this, is it supposed to be 10 or args.subsampling?
                    speaker_turn_emb = np.array(emb[0, start // 10:end // 10, :])
                    pooling_mean = np.mean(speaker_turn_emb, axis=0)
                    meansq = np.mean(speaker_turn_emb * speaker_turn_emb, axis=0)
                    pooling_std = np.sqrt(meansq - pooling_mean ** 2 + 1e-10)
                    speaker_emb = np.concatenate((pooling_mean, pooling_std), axis=0)
                    speaker_means.append(speaker_emb)
                    spk_dict[chunk_idx].append(speaker_emb)
                    speakers_in_chunk.add(speaker)
                #     #print(speaker_turn_emb.shape, start, end, speaker)
                #     spk_dict[speaker].append(speaker_turn_emb)
                # for speaker in sorted(spk_dict):
                #     speaker_embs = torch.cat(spk_dict[speaker], dim=1)
                #     #print('speaker emb', speaker_embs.shape)
                #     speaker_mean = np.mean(np.array(speaker_embs[0, :, :]), axis=0)
                #     #print(speaker_mean)
                #     #speaker_emb.append(np.mean(torch.cat(spk_dict[speaker], dim=1))
                #     speaker_means.append(speaker_mean)
                #     speaker_map[chunk_idx].append(speaker)
            #break
            speakers.extend(sorted(speakers_in_chunk))
            os.remove(rttm_filename)
        rttm_filename = join(out_dir, f"{name}.rttm")
        annotation = Annotation(uri=name)
        speaker_means = np.array(speaker_means)
        if speaker_means.size == 0:
            # create empty rttm file
            with open(rttm_filename, 'w') as rttm_file:
                continue
        if np.array(speaker_means).ndim == 1:
            speaker_means = speaker_means[np.newaxis, :]
        #print(speaker_means.shape)
        if len(set(speakers)) == 1:  # there was only 1 speaker found in the chunks
            labels = [0 for x in speaker_means]
        else:
            scr_mx = cos_similarity(l2_norm(speaker_means))
            #print(scr_mx)
            if scr_mx.shape == (1, 1):
                labels = [0]
            else:
                thr = twoGMMcalib_lin(scr_mx.ravel(), niters=10, var_eps=1e-06)
                labels = AHC(scr_mx, thr + args.ahc_threshold)

        #print(labels)
        # create pyannote object for easy writing of rttm
        speaker_offset = 0
        labels_idx = 0
        for chunk_idx in sorted(timings_dict):
            #print(timings_dict[chunk_idx])
            # chunk_map = {}
            # num_speakers_in_chunk = len(speaker_map[chunk_idx])
            # chunk_speakers = speaker_map[chunk_idx]
            # cluster_speakers = labels[speaker_offset:num_speakers_in_chunk]
            # assert len(chunk_speakers) == len(cluster_speakers)
            # for chunk_speaker, cluster_speaker in zip(chunk_speakers, cluster_speakers):
            #     chunk_map[chunk_speaker] = cluster_speaker
            #print(chunk_map)
            for start, end, speaker in timings_dict[chunk_idx]:
                seg_start = chunk_idx * args.chunk_size / 10 + float(start / 100)
                seg_end = chunk_idx * args.chunk_size / 10 + float(end / 100)
                annotation[Segment(seg_start, seg_end)] = labels[labels_idx]
                labels_idx += 1
        #print(annotation)

        with open(rttm_filename, 'w') as rttm_file:
            annotation.write_rttm(rttm_file)
            #hard_labels_to_rttm(post_y, name, rttm_file)