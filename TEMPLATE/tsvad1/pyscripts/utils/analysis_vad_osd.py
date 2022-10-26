#!/usr/bin/env python3

import collections.abc
import humanfriendly
from pathlib import Path
from typing import Union

import argparse
import datetime
import logging
import numpy as np
from espnet2.fileio.npy_scp import NpyScpReader
from espnet2.fileio.rttm import DNCRttmReader
from espnet2.utils.types import str_or_int
from espnet2.utils.types import str2bool
from typeguard import check_argument_types
from scipy import stats


class Stats:
    def __init__(self, class_size=3):
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0
        self.class_distribution = np.zeros(class_size)

    def check_seq(self, pred, label, target_label=0):
        self.true_positive += np.sum(
            np.logical_and(pred == target_label, label == target_label)
        )
        self.false_positive += np.sum(
            np.logical_and(pred == target_label, label != target_label)
        )
        self.true_negative += np.sum(
            np.logical_and(pred != target_label, label != target_label)
        )
        self.false_negative += np.sum(
            np.logical_and(pred != target_label, label == target_label)
        )
        distribution = np.bincount(pred[label == target_label])
        self.class_distribution[: len(distribution)] += distribution

    def analysis_stats(self):
        accuracy = (self.true_positive + self.true_negative) / (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )
        recall = self.true_positive / (self.true_positive + self.false_negative)
        precision = self.true_positive / (self.true_positive + self.false_positive)
        return (
            accuracy,
            recall,
            precision,
            self.class_distribution / np.sum(self.class_distribution),
        )


def process_vad_osd_label_from_sample(
    sample_label, win_length, hop_length, spk_win_length, spk_hop_length
):
    frame_size = (len(sample_label) - win_length) // hop_length + 1
    frame_info = np.zeros((frame_size,), dtype=int)
    for i in range(frame_size):
        frame_info[i] = stats.mode(
            sample_label[i * hop_length : i * hop_length + win_length]
        )[0]

    spk_size = (frame_size - spk_win_length) // spk_hop_length + 1
    spk_info = np.zeros((spk_size,), dtype=int)
    for i in range(spk_size):
        spk_info[i] = stats.mode(
            frame_info[i * spk_hop_length : i * spk_hop_length + spk_win_length]
        )[0]

    return spk_info


def vad_osd_analysis(args):
    rttm_reader = DNCRttmReader(args.gt_label)
    npy_reader = NpyScpReader(args.inference_scp)
    vad_stats = Stats()
    if not args.vad_only:
        osd_stats = Stats()
    for meeting_id in npy_reader.keys():
        rttm_info = rttm_reader[meeting_id]
        org_len = len(rttm_info)
        inference_info = npy_reader[meeting_id]
        if args.vad_only:
            inference_info[inference_info > 1] = 1
        else:
            inference_info[inference_info > 2] = 2

        rttm_info = rttm_info[
            args.frame_shift // 2 :: args.frame_shift * args.subsampling
        ]
        # TODO(jiatong): add arguments
        # rttm_info = process_vad_osd_label_from_sample(rttm_info, 512, 256, 5, 5)
        # if meeting_id == "R8001_M8004":
        #     logging.info(meeting_id)
        #     logging.info("{}".format(" ".join(list(map(lambda x: str(int(x)), list(rttm_info))))))
        #     exit(0)
        logging.info(
            "length gt: {}, length inference: {}, org_len: {}".format(
                len(rttm_info), len(inference_info), org_len
            )
        )
        # time=15
        # rttm_info = rttm_info[375 * time :375 * (time + 1)]
        # inference_info = inference_info[375 * time: 375 * (time + 1)]
        sync_len = min(len(inference_info), len(rttm_info))
        rttm_info = rttm_info[:sync_len].astype(int)
        inference_info = inference_info[:sync_len].astype(int)

        vad_stats.check_seq(inference_info, rttm_info, target_label=0)
        if not args.vad_only:
            osd_stats.check_seq(inference_info, rttm_info, target_label=1)

    with open(args.output, "a", encoding="utf-8") as f:
        f.write("[{}] \n".format(datetime.datetime.now()))
        f.write("source: {}\n".format(args.inference_scp))
        f.write("target: {}\n".format(args.gt_label))
        sil_msg = "Silence Detection. ACC: {}, RECALL: {}, PRECISION: {}, MISS_DISTRIBUTION: {}\n".format(
            *vad_stats.analysis_stats()
        )
        f.write(sil_msg)
        print(sil_msg)
        if not args.vad_only:
            osd_msg = "Overlap Detection. ACC: {}, RECALL: {}, PRECISION: {}, MISS_DISTRIBUTION: {}\n\n".format(
                *osd_stats.analysis_stats()
            )
            f.write(osd_msg)
            print(osd_msg)


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Analysis VAD OSD prediciton statisitics (e.g., accuracy, recall, etc.)"
    )

    parser.add_argument(
        "--inference_scp",
        required=True,
        type=str,
        help="inference scp for vad-osd prediction",
    )
    parser.add_argument(
        "--output", required=True, type=str, help="result output directory"
    )
    parser.add_argument(
        "--gt_label", required=True, type=str, help="ground truth label"
    )
    parser.add_argument(
        "--frame_shift", default=256, type=int, help="frame shift of a frame"
    )
    parser.add_argument(
        "--sampling_rate",
        type=str_or_int,
        default=16000,
        help="sampling rate of the audio",
    )
    parser.add_argument("--subsampling", type=int, default=1, help="subsampling rate")
    parser.add_argument(
        "--vad_only",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    return parser


def main():
    """Convert standard rttm to sample-based result"""
    args = get_parser().parse_args()

    # logging info
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    vad_osd_analysis(args)

    logging.info("Successfully finished analysis VAD/OSD performances.")


if __name__ == "__main__":
    main()
