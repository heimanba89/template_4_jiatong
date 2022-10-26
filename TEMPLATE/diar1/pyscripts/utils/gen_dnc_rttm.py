#!/usr/bin/env python3

import collections.abc
import humanfriendly
from collections import defaultdict
from pathlib import Path
from typing import Union

import argparse
import logging
import numpy as np
import re
import torch
import os
import soundfile
from espnet2.utils.types import str_or_int
from espnet2.utils.types import str2bool
from typeguard import check_argument_types


def default_meeting_spk():
    return {"spk_count": 0, "spk_label": {}, "meeting_duration": 0}


def gen_dnc_rttm(
    rttm: Union[Path, str],
    output_path: Union[Path, str],
    vad_only: bool = False,
    osd_vad_only: bool = False,
) -> None:
    """Convert a ESPnet RTTM to dnc labels (dnc RTTM)

    Note: only support speaker information now
    """
    output_handler = Path(os.path.join(output_path, "dnc_rttm")).open(
        "w", encoding="utf-8"
    )

    assert check_argument_types()
    # structure:
    # meeting_info = {
    #     "meeting_id1": [
    #         (label_type, channel, start, end, spk_id),
    #         (label_type1, channel1, start1, end1, spk_id1),
    #         (label_type2, channel2, start2, end2, spk_id2),
    #     ],
    #     "meeting_id2": ...
    # }
    #
    # meeting_spk = {
    #     "meeting_id1": {
    #         "spk_count": 2,
    #         "spk_label": {
    #             "spk1": 0,
    #             "spk2": 1
    #         }
    #     },
    #     "meeting_id2":{
    #         "spk_count": 3,
    #         ...
    #     }
    # }
    meeting_info = defaultdict(list)
    meeting_spk = defaultdict(default_meeting_spk)
    espnet_rttm_flag = False

    # loading espnet_rttm data
    with Path(rttm).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = re.split(" +", line.rstrip())

            # RTTM format must have exactly 9 fields
            label_type, meeting_id, channel, start, end, _, _, spk_id, _ = sps
            if label_type == "END":
                espnet_rttm_flag = True
                meeting_spk[meeting_id]["meeting_duration"] = end
                output_handler.write(line)  # no need to change it
                continue

            meeting_info[meeting_id].append(
                (label_type, channel, int(start), int(end), spk_id)
            )
            if spk_id not in meeting_spk[meeting_id]["spk_label"].keys():
                meeting_spk[meeting_id]["spk_label"][spk_id] = meeting_spk[meeting_id][
                    "spk_count"
                ]
                meeting_spk[meeting_id]["spk_count"] += 1

    assert (
        espnet_rttm_flag
    ), "not using espnet_rttm structure, please use pyscripts/utils/convert_rttm.py to fix the rttm"

    for meeting_id in meeting_info.keys():
        # TODO(jiatong): add flexible chunksize for long meeting process
        #                now only support whole meeting processing
        meeting_duration = meeting_spk[meeting_id]["meeting_duration"]
        mixture_label = np.zeros(
            (int(meeting_duration))
        )  # very long for long recordings
        for seg_info in meeting_info[meeting_id]:
            label_type, channel, start, end, spk_id = seg_info
            mixture_label[start:end] = (mixture_label[start:end] == 0) * (
                meeting_spk[meeting_id]["spk_label"][spk_id] + 2
            ) + (  # add new speaker information to silence
                mixture_label[start:end] != 0
            ) * 1  # make overlap label

        if vad_only:
            mixture_label, counts = osd_vad_label_reformat(mixture_label, False)
        elif osd_vad_only:
            mixture_label, counts = osd_vad_label_reformat(mixture_label, True)
        else:
            mixture_label, counts = dnc_label_reformat(mixture_label)

        start = 0
        for i in range(len(mixture_label)):
            output_handler.write(
                "{} {} {} {} {} <NA> <NA> {} <NA>\n".format(
                    "SPEAKER",
                    meeting_id,
                    1,
                    start,
                    start + counts[i],
                    int(mixture_label[i]),
                )
                # Note(jiatong): channel use 1 as default
                # (the speaker cannot be assigned to single channel in diarization anyway)
                # for 0 (silence) and 1 (overlap), we also regards them as speaker
            )
            start += counts[i]

    output_handler.close()


def dnc_label_reformat(mixture_label):
    # FIXME(jiatong): currnet reformating is straight-forward but might
    # have efficiency issue. Please fix later
    # Note(jiatong): it is different from the function in espnet2.diar.dnc_aggregation
    unique_label, counts = torch.tensor(mixture_label).unique_consecutive(
        return_counts=True
    )

    unique_label = unique_label.int()
    # dictionary for converting absolute dnc label to relative ones
    abs2real_dict = {}
    order = 2  # 0 for silence, 1 for overlap, 2+ for others single speaker
    for i in range(len(unique_label)):
        if unique_label[i] > 1:  # only update for single speaker cases
            if int(unique_label[i]) not in abs2real_dict:
                # do not have the label, add a new label
                abs2real_dict[int(unique_label[i])] = order
                unique_label[i] = order
                order += 1
            else:
                # already have the label, use existing label
                unique_label[i] = abs2real_dict[int(unique_label[i])]
    return unique_label.numpy(), counts


def osd_vad_label_reformat(mixture_label, use_osd=False):
    unique_label, counts = torch.tensor(mixture_label).unique_consecutive(
        return_counts=True
    )

    unique_label = unique_label.int()
    if use_osd:
        unique_label[unique_label > 1] = 2
    else:
        unique_label[unique_label > 0] = 1
    return unique_label, counts


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert standard rttm file to ESPnet-DNC format"
    )
    parser.add_argument("--rttm", required=True, type=str, help="Path of rttm file")
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="Output directory to storry espnet_rttm",
    )
    parser.add_argument(
        "--vad_only",
        default="false",
        type=str2bool,
        help="only perform VAD training",
    )
    parser.add_argument(
        "--osd_vad_only",
        default="false",
        type=str2bool,
        help="only perform VAD and OSD training",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    return parser


def main():
    """Convert espnet_rttm to dnc_label (dnc_rttm)"""
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

    gen_dnc_rttm(args.rttm, args.output_path, args.vad_only, args.osd_vad_only)

    logging.info("Successfully finished DNC RTTM converting.")


if __name__ == "__main__":
    main()
