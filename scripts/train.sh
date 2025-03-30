 #! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# set variables
echo "Current working directory: $(pwd)"

DATA_PATH=/home/jupyter-samantha_caasi@dls-bf571/datasets/datasets_for_vsp-llm/ouluvs2/muavic_dataset # path to train dataset dir
OUT_PATH=/home/jupyter-samantha_caasi@dls-bf571/models/VSP-LLM/training_output_ouluvs2  # output path to save 

ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
SRC=${ROOT}/src
LLM_PATH=${ROOT}/checkpoints/Llama-2-7b-hf   # path to llama checkpoint
PRETRAINED_MODEL_PATH=${ROOT}/checkpoints/large_vox_iter5.pt   # path to pretrained avhubert

echo "starting to train ouluvs2 with vsp-llm"

# start training
export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"
fairseq-hydra-train \
    --config-dir ${SRC}/conf \
    --config-name vsp-llm-433h-finetune \
        common.user_dir=${SRC} \
        task.data=${DATA_PATH} \
        task.label_dir=${DATA_PATH} \
        task.llm_ckpt_path=${LLM_PATH} \
        model.w2v_path=${PRETRAINED_MODEL_PATH} \
        model.llm_ckpt_path=${LLM_PATH} \
        hydra.run.dir=${OUT_PATH} \
        distributed_training.distributed_world_size=1 \
        distributed_training.nprocs_per_node=1 \
