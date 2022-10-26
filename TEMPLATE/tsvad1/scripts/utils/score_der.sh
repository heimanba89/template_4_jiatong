#!/bin/bash

# Copyright 2022 Tencent AI Lab (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

frame_shift=128
fs=8000
subsampling=1
collar=0.25   # collar for speaker diarization
use_dnc=false # whether to use dnc

scoring_dir=  # scoring directory
infer_scp=    # inference scp from diar_inference
gt_label=     # ground truth rttm

. ./utils/parse_options.sh || exit 1

# if [ $# -lt 3 ]; then
#     echo "Usage: $0 --scoring_dir <scoring_dir> --infer_scp <infer_scp> --gt_label <gt_label>";
#     exit 1;
# fi

echo "Scoring at ${scoring_dir}"
mkdir -p $scoring_dir || exit 1;


if ${use_dnc}; then
    pyscripts/utils/make_rttm_from_dnc.py --frame_shift=${frame_shift} \
        --subsampling=${subsampling} --sampling_rate=${fs} \
        $infer_scp ${scoring_dir}/hyp.rttm

    # pyscripts/utils/analysis_dnc_rttm.py --frame_shift=${frame_shift} \
    #     --subsampling=${subsampling} --sampling_rate=${fs} \
    #     ${infer_scp} ${scoring_dir}/hyp.rttm

    collar=0.25
    md-eval.pl -c ${collar} \
        -r ${gt_label} \
        -s ${scoring_dir}/hyp.rttm \
        > ${scoring_dir}/result_collar${collar} 2>/dev/null || exit 1

    grep OVER ${scoring_dir}/result_collar${collar} \
        | grep -v nooverlap \
        | sort -nrk 7 | tail -n 1

else

    for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
            # TODO(jiatong): add frame_calib for frame mismatching
            pyscripts/utils/make_rttm_from_post.py --median=$med --threshold=$th \
                --frame_shift=${frame_shift} --subsampling=${subsampling} --sampling_rate=${fs} \
                $infer_scp ${scoring_dir}/hyp_${th}_$med.rttm

            md-eval.pl -c ${collar} \
                -r ${gt_label} \
                -s ${scoring_dir}/hyp_${th}_$med.rttm \
                > ${scoring_dir}/result_th${th}_med${med}_collar${collar} 2>/dev/null || exit 1

        done
    done

    grep OVER $1/result_th0.[^_]*_med[^_]*_collar${collar} \
        | grep -v nooverlap \
        | sort -nrk 7 | tail -n 1
fi

