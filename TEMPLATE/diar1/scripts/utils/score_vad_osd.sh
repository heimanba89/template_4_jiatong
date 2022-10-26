#!/bin/bash

# Copyright 2022 Tencent AI Lab (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

collar=0.0
frame_shift=128
fs=8000
subsampling=1
vad_only=false
vad_osd_only=false

scoring_dir=  # scoring directory
infer_scp=    # inference scp from diar_inference
gt_label=     # ground truth rttm

. ./utils/parse_options.sh || exit 1

if [ $# -lt 0 ]; then
    echo "Usage: $0 --scoring_dir <scoring_dir> --infer_scp <infer_scp> --gt_label <gt_label>";
    exit 1;
fi

echo "Scoring at ${scoring_dir}"
mkdir -p $scoring_dir || exit 1;

pyscripts/utils/analysis_vad_osd.py --frame_shift ${frame_shift} \
    --subsampling ${subsampling} \
    --vad_only ${vad_only} \
    --inference_scp ${infer_scp} --output ${scoring_dir}/osd_vad_result --gt_label ${gt_label}





