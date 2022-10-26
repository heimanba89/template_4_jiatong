#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (Yusuke Fujita)
#           2021 Johns Hopkins University (Jiatong Shi)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from espnet2.fileio.npy_scp import NpyScpReader
import logging
import numpy as np
from scipy.signal import medfilt
import humanfriendly


def get_parser() -> argparse.Namespace:
    """Get argument parser."""

    parser = argparse.ArgumentParser(description="make rttm from decoded result")
    parser.add_argument("diarize_scp")
    parser.add_argument("out_rttm_file")
    parser.add_argument("--frame_shift", default=1280, type=int)
    parser.add_argument("--subsampling", default=1, type=int)
    parser.add_argument("--sampling_rate", default="8000", type=str)
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    return parser


def formating_output(key, start, end, spk_ids, writer_handler, factor=1):
    fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
    for spk_id in spk_ids:
        print(
            fmt.format(
                key,
                start * factor,
                (end - start + 1) * factor,
                key + "_" + str(spk_id),
            ),
            file=writer_handler,
        )


def main():
    """Make rttm based on diarization inference results"""
    args = get_parser().parse_args()
    sampling_rate = humanfriendly.parse_size(args.sampling_rate)
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

    scp_reader = NpyScpReader(args.diarize_scp)
    factor = (args.frame_shift * args.subsampling) / sampling_rate

    with open(args.out_rttm_file, "w") as wf:
        for key in scp_reader.keys():
            data = scp_reader[key]
            index_of_change_point = np.where(data[:-1] != data[1:])[0]
            index_of_change_point = [0] + list(index_of_change_point) + [len(data) - 1]
            print(data)
            print(np.bincount(data))
            print(len(data))

            start_index, end_index, spk_label = [], [], []
            if len(index_of_change_point) > 2:
                for i in range(1, len(index_of_change_point)):
                    start_index.append(index_of_change_point[i - 1] + 1)
                    end_index.append(index_of_change_point[i])
                    spk_label.append(data[index_of_change_point[i - 1] + 1])
                for i in range(len(index_of_change_point) - 1):
                    if spk_label[i] == 1:  # overlap case
                        if i == 0:  # find overlap at the start
                            if max(spk_label) < 2:
                                # no other speakers, regard as one speaker
                                start, end, spk = 0, end_index[i], [spk_label[i]]
                            else:
                                # use next speaker for start overlapping
                                start, end, spk = 0, end_index[i], [spk_label[i + 1]]
                        elif i == len(index_of_change_point) - 2:
                            # approaching the end
                            start, end, spk = (
                                start_index[i],
                                end_index[i],
                                [spk_label[i - 1]],
                            )
                        else:
                            start, end = start_index[i], end_index[i]
                            if spk_label[i - 1] == spk_label[i + 1]:
                                # previous speaker is the same as next speaker
                                spk = [spk_label[i - 1]]
                            else:
                                if spk_label[i - 1] == 0:
                                    # prev spk is sil, use next spk
                                    spk = [spk_label[i + 1]]
                                elif spk_label[i + 1] == 0:
                                    # next spk is sil, use prev spk
                                    spk = [spk_label[i - 1]]
                                else:
                                    # use both spk
                                    spk = [spk_label[i - 1], spk_label[i + 1]]
                    elif spk_label[i] > 1:
                        if i == 0:
                            start, end, spk = 0, end_index[i], [spk_label[i]]
                        else:
                            start, end, spk = (
                                start_index[i],
                                end_index[i],
                                [spk_label[i]],
                            )
                    else:
                        continue

                    formating_output(key, start, end, spk, wf, factor)

    logging.info("Constructed RTTM for {}".format(args.diarize_scp))


if __name__ == "__main__":
    main()
