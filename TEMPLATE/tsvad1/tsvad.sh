#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages
skip_train=false     # Skip training stages
skip_eval=false      # Skip decoding and evaluation stages
skip_upload=true     # Skip packing and uploading stages
skip_upload_hf=true # Skip uploading to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes
nj=32                # The number of parallel jobs.
dumpdir=dump         # Directory to dump features.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands
train_use_memory=false # Directly use memory for training (efficient use for large memory)

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=8k                # Sampling rate.
hop_length=128       # Hop length in sample number
min_wav_duration=0.1 # Minimum duration in second

# tsvad model related
tsvad_tag=    # Suffix to the result dir for tsvad model training.
tsvad_config= # Config for tsvad model training.
tsvad_args=   # Arguments for tsvad model training, e.g., "--max_epoch 10".
             # Note that it will overwrite args in tsvad config.
feats_normalize=global_mvn # Normalizaton layer type.
pretrained_speaker_model= # load pretrained speaker model (e.g., ECAPA_TDNN)
ignore_init_mismatch=false # whether to ignore init_param mismatch


# tsvad related
inference_config= # Config for tsvad model inference
# inference_model=valid.loss.best.pth
inference_model=latest.pth
inference_tag=    # Suffix to the inference dir for tsvad model inference
gt_inference=false  # whether to decode based gt
download_model=   # Download a model from Model Zoo and use it for tsvadization.

# Upload model related
hf_repo=

# scoring related
collar=0.25         # collar for der scoring
frame_shift=128  # frame shift to convert frame-level label into real time
                 # this should be aligned with frontend feature extraction
                 # for EEND-based model, the frame shift is the same as normal stft feature
                 # for EESD-based model, the frame shift is frame_shift * spk_embed_shift

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of development set.
test_sets=       # Names of evaluation sets. Multiple items can be specified.
tsvad_speech_fold_length=800 # fold_length for speech data during tsvad training
                            # Typically, the label also follow the same fold length
lang=noinfo      # The language type of corpus.


help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --valid-set <valid_set_name> --test_sets <test_set_names>

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in inference (default="${inference_nj}").
    --gpu_inference  # Whether to use gpu for inference (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type       # Feature type (only support raw currently).
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --hop_length       # Hop length in sample number (default="${hop_length}")
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").


    # target speaker VAD model related
    --tsvad_tag        # Suffix to the result dir for TSVAD model training (default="${tsvad_tag}").
    --tsvad_config     # Config for TSVAD model training (default="${tsvad_config}").
    --tsvad_args       # Arguments for TSVAD model training, e.g., "--max_epoch 10" (default="${tsvad_args}").
                      # Note that it will overwrite args in tsvad config.
    --feats_normalize # Normalizaton layer type (default="${feats_normalize}").
    --pretrained_speaker_model # load pretrained speaker model (e.g., ECAPA_TDNN) (default="${pretrained_speaker_model}")
    --ignore_init_mismatch     # whether to ignore init_param mismatch (default="${ignore_init_mismatch}")

    # TSVAD related
    --inference_config # Config for tsvad model inference
    --inference_model  # TSVAD model path for inference (default="${inference_model}").
    --inference_tag    # Suffix to the inference dir for tsvad model inference
    --download_model   # Download a model from Model Zoo and use it for TSVAD (default="${download_model}").

    # Scoring related
    --collar      # collar for der scoring
    --frame_shift # frame shift to convert frame-level label into real time

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set               # Name of training set (required).
    --valid_set               # Name of development set (required).
    --test_sets               # Names of evaluation sets (required).
    --tsvad_speech_fold_length # fold_length for speech data during TSVAD training  (default="${tsvad_speech_fold_length}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh || exit 1

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

# . ./path.sh
. ./cmd.sh


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] &&   { log "${help_message}"; log "Error: --valid_set is required"  ; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

data_feats=${dumpdir}/raw

# Set tag for naming of model directory
if [ -z "${tsvad_tag}" ]; then
    if [ -n "${tsvad_config}" ]; then
        tsvad_tag="$(basename "${tsvad_config}" .yaml)_${feats_type}"
    else
        tsvad_tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${tsvad_args}" ]; then
        tsvad_tag+="$(echo "${tsvad_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi

if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
fi

# The directory used for collect-stats mode
tsvad_stats_dir="${expdir}/tsvad_stats_${fs}_${train_set}"
# The directory used for training commands
tsvad_exp="${expdir}/tsvad_${tsvad_tag}"

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

        log "Stage 2: Format wav.scp: data/ -> ${data_feats}"

        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and also it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.
        for dset in "${train_set}" "${valid_set}" ${test_sets}; do
        # for dset in ${test_sets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
            rm -f ${data_feats}${_suf}/${dset}/{wav.scp,reco2file_and_channel}

            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}"  \
                "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"
            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"

            # specifics for TSVAD
            steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                    "${data_feats}${_suf}/${dset}"/utt2spk \
                    "${data_feats}${_suf}/${dset}"/segments \
                    "${data_feats}${_suf}/${dset}"/rttm

            # convert standard rttm file into espnet-format rttm (measure with samples)
            pyscripts/utils/convert_rttm.py \
                --rttm "${data_feats}${_suf}/${dset}"/rttm \
                --wavscp "${data_feats}${_suf}/${dset}"/wav.scp \
                --output_path "${data_feats}${_suf}/${dset}" \
                --sampling_rate "${fs}"

        done
    fi


    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove short data: ${data_feats}/org -> ${data_feats}"

        for dset in "${train_set}" "${valid_set}"; do
        # NOTE: Not applying to test_sets to keep original data

            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            # TSVAD typically accept long recordings, so does not has
            # max length requirements
            <"${data_feats}/org/${dset}/utt2num_samples" \
                awk -v min_length="${_min_length}" \
                    '{ if ($2 > min_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
            <"${data_feats}/org/${dset}/wav.scp" \
                utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                >"${data_feats}/${dset}/wav.scp"

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh "${data_feats}/${dset}"

            # specifics for TSVAD
            steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                    "${data_feats}/${dset}"/utt2spk \
                    "${data_feats}/${dset}"/segments \
                    "${data_feats}/${dset}"/rttm

            # convert standard rttm file into espnet-format rttm (measure with samples)
            pyscripts/utils/convert_rttm.py \
                --rttm "${data_feats}/${dset}"/rttm \
                --wavscp "${data_feats}/${dset}"/wav.scp \
                --output_path "${data_feats}/${dset}" \
                --sampling_rate "${fs}"
            
        done
    fi
else
    log "Skip the data preparation stages"
fi


# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then
    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        _tsvad_train_dir="${data_feats}/${train_set}"
        _tsvad_valid_dir="${data_feats}/${valid_set}"
        log "Stage 4: TSVAD collect stats: train_set=${_tsvad_train_dir}, valid_set=${_tsvad_valid_dir}"

        _opts=
        if [ -n "${tsvad_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.tsvad_train --print_config --optim adam
            _opts+="--config ${tsvad_config} "
        fi

        _feats_type="$(<${_tsvad_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" and "sound_in_memory" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            elif "${train_use_memory}"; then
                _type=memory_sound
            else
                _type=sound
            fi
            _opts+="--frontend_conf fs=${fs} "
            _opts+="--frontend_conf hop_length=${hop_length} "
            _opts+="--label_aggregator_conf hop_length=${hop_length} "
        else
            echo "does not support other feats_type (i.e., ${_feats_type}) now"
        fi


        # TODO(jiatong): add description for espnet_rttm
        rttm_file=espnet_rttm
        rttm_type=espnet_rttm

        # 1. Split the key file
        _logdir="${tsvad_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_tsvad_train_dir}/${_scp} wc -l)" "$(<${_tsvad_valid_dir}/${_scp} wc -l)")

        key_file="${_tsvad_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_tsvad_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${tsvad_stats_dir}/run.sh'. You can resume the process from stage 4 using this script"
        mkdir -p "${tsvad_stats_dir}"; echo "${run_args} --stage 4 \"\$@\"; exit \$?" > "${tsvad_stats_dir}/run.sh"; chmod +x "${tsvad_stats_dir}/run.sh"

        # 3. Submit jobs
        log "TSVAD collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.     

        # shellcheck disable=SC2046,SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.tsvad_train \
                --collect_stats true \
                --use_preprocessor true \
                --batch_size 1 \
                --train_data_path_and_name_and_type "${_tsvad_train_dir}/${_scp},speech,${_type}" \
                --train_data_path_and_name_and_type "${_tsvad_train_dir}/${rttm_file},spk_labels,${rttm_type}" \
                --valid_data_path_and_name_and_type "${_tsvad_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_tsvad_valid_dir}/${rttm_file},spk_labels,${rttm_type}" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${tsvad_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${tsvad_stats_dir}"

    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _tsvad_train_dir="${data_feats}/${train_set}"
        _tsvad_valid_dir="${data_feats}/${valid_set}"
        log "Stage 5: TSVAD Training: train_set=${_tsvad_train_dir}, valid_set=${_tsvad_valid_dir}"

        _opts=
        if [ -n "${tsvad_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.tsvad_train --print_config --optim adam
            _opts+="--config ${tsvad_config} "
        fi

        _feats_type="$(<${_tsvad_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" and "sound_in_memory" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            # elif "${train_use_memory}"; then
            #     _type=memory_sound
            else
                _type=sound
            fi
            _fold_length="$((tsvad_speech_fold_length * 100))"
            _opts+="--frontend_conf fs=${fs} "
            _opts+="--frontend_conf hop_length=${hop_length} "
        else
            echo "does not support other feats_type (i.e., ${_feats_type}) now"
        fi

        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${tsvad_stats_dir}/train/feats_stats.npz "
        elif [ "${feats_normalize}" = None ]; then
            _opts+="--normalize=None "
        fi

        if [ ! -z "${pretrained_speaker_model}" ]; then
            _opts+="--init_param ${pretrained_speaker_model}::speaker_net "
            # _opts+="--init_param ${pretrained_speaker_model}:torchfb:frontend.torchfb "
            _opts+="--ignore_init_mismatch ${ignore_init_mismatch} "
        fi


        rttm_file=espnet_rttm
        rttm_type=espnet_rttm

        _opts+="--train_data_path_and_name_and_type ${_tsvad_train_dir}/${_scp},speech,${_type} "
        _opts+="--train_data_path_and_name_and_type ${_tsvad_train_dir}/${rttm_file},spk_labels,${rttm_type} "
        _opts+="--train_shape_file ${tsvad_stats_dir}/train/speech_shape "
        _opts+="--train_shape_file ${tsvad_stats_dir}/train/spk_labels_shape "

        _opts+="--valid_data_path_and_name_and_type ${_tsvad_valid_dir}/${_scp},speech,${_type} "
        _opts+="--valid_data_path_and_name_and_type ${_tsvad_valid_dir}/${rttm_file},spk_labels,${rttm_type} "
        _opts+="--valid_shape_file ${tsvad_stats_dir}/valid/speech_shape "
        _opts+="--valid_shape_file ${tsvad_stats_dir}/valid/spk_labels_shape "

        log "Generate '${tsvad_exp}/run.sh'. You can resume the process from stage 5 using this script"
        mkdir -p "${tsvad_exp}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${tsvad_exp}/run.sh"; chmod +x "${tsvad_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "TSVAD training started... log: '${tsvad_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${tsvad_exp})"
        else
            jobname="${tsvad_exp}/train.log"
        fi

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${tsvad_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${tsvad_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.tsvad_train \
                --use_preprocessor true \
                --resume true \
                --fold_length "${_fold_length}" \
                --fold_length "${tsvad_speech_fold_length}" \
                --output_dir "${tsvad_exp}" \
                ${_opts} ${tsvad_args}

    fi
else
    log "Skip the training stages"
fi

if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    tsvad_exp="${expdir}/${download_model}"
    mkdir -p "${tsvad_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${tsvad_exp}/config.txt"

    # Get the path of each file
    _tsvad_model_file=$(<"${tsvad_exp}/config.txt" sed -e "s/.*'tsvad_model_file': '\([^']*\)'.*$/\1/")
    _tsvad_train_config=$(<"${tsvad_exp}/config.txt" sed -e "s/.*'tsvad_train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_tsvad_model_file}" "${tsvad_exp}"
    ln -sf "${_tsvad_train_config}" "${tsvad_exp}"
    inference_tsvad_model=$(basename "${_tsvad_model_file}")

fi

if ! "${skip_eval}"; then
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: Target Speaker Voice Activity Detection: training_dir=${tsvad_exp}"

        if ${gpu_inference}; then
            _cmd=${cuda_cmd}
            _ngpu=1
        else
            _cmd=${decode_cmd}
            _ngpu=0
        fi

        log "Generate '${tsvad_exp}/run_tsvad.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${tsvad_exp}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${tsvad_exp}/run_tsvad.sh"; chmod +x "${tsvad_exp}/run_tsvad.sh"
        _opts=

        inference_binary=espnet2.bin.tsvad_inference

        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi

        for dset in "${valid_set}" ${test_sets}; do
        # for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${tsvad_exp}/${inference_tag}_tsvad_${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _scp=wav.scp
            _type=sound

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}


            # 2. Submit inference jobs
            log "TSVAD started... log: '${_logdir}/tsvad_inference.*.log'"
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/tsvad_inference.JOB.log \
                ${python} -m ${inference_binary} \
                    --ngpu "${_ngpu}" \
                    --fs "${fs}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --train_config "${tsvad_exp}"/config.yaml \
                    --model_file "${tsvad_exp}"/"${inference_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    --vad_only ${vad_only} \
                    --osd_vad_only ${osd_vad_only} \
                    ${_opts} || { cat $(grep -l -i error "${_logdir}"/tsvad_inference.*.log) ; exit 1; }

            # 3. Concatenates the output files from each jobs
            for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/tsvad.scp"
            done | LC_ALL=C sort -k1 > "${_dir}/tsvad.scp"

        done
    fi

    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Stage 7: Scoring"
        _cmd=${decode_cmd}

        if "${vad_only}" || "${osd_vad_only}" ; then
            echo  "${vad_only} ${osd_vad_only}"
            for dset in "${valid_set}" ${test_sets}; do
                _data="${data_feats}/${dset}"
                _inf_dir="${tsvad_exp}/${inference_tag}_tsvad_${dset}"
                _dir="${tsvad_exp}/${inference_tag}_tsvad_${dset}/vad_osd_scoring"
                mkdir -p "${_dir}"

                scripts/utils/score_vad_osd.sh \
                    --scoring_dir ${_dir} \
                    --infer_scp ${_inf_dir}/tsvad.scp \
                    --gt_label ${_data}/dnc_rttm \
                    --fs ${fs} --frame_shift ${frame_shift} \
                    --vad_only ${vad_only} --vad_osd_only ${osd_vad_only}
            done

        else
            for dset in "${valid_set}" ${test_sets}; do
                _data="${data_feats}/${dset}"
                _inf_dir="${tsvad_exp}/${inference_tag}_tsvad_${dset}"
                _dir="${tsvad_exp}/${inference_tag}_tsvad_${dset}/scoring"
                mkdir -p "${_dir}"

                scripts/utils/score_der.sh \
                    --scoring_dir ${_dir}  \
                    --infer_scp ${_inf_dir}/tsvad.scp \
                    --gt_label ${_data}/rttm \
                    --collar ${collar} --fs ${fs} --frame_shift ${frame_shift} \
                    --use_dnc ${use_dnc}
            done

            # Show results in Markdown syntax
            scripts/utils/show_tsvad_result.sh "${tsvad_exp}" > "${tsvad_exp}"/RESULTS.md
            cat "${tsvad_exp}"/RESULTS.md
        fi

    fi
else
    log "Skip the evaluation stages"
fi


packed_model="${tsvad_exp}/${tsvad_exp##*/}_${inference_model%.*}.zip"
if [ -z "${download_model}" ]; then
    # Skip pack preparation if using a downloaded model
    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: Pack model: ${packed_model}"

        ${python} -m espnet2.bin.pack tsvad \
            --train_config "${tsvad_exp}"/config.yaml \
            --model_file "${tsvad_exp}"/"${inference_model}" \
            --option "${tsvad_exp}"/RESULTS.md \
            --option "${tsvad_stats_dir}"/train/feats_stats.npz  \
            --option "${tsvad_exp}"/images \
            --outpath "${packed_model}"
    fi
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
