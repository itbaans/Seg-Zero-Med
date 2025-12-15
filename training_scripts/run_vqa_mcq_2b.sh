#!/bin/bash

# VQA MCQ Training with Qwen2-VL-2B

# Download pretrained model first if needed:
# mkdir -p pretrained_models
# cd pretrained_models
# git lfs install
# git clone https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct

export CUDA_VISIBLE_DEVICES=0,1

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=pretrained_models/Qwen2-VL-2B-Instruct # replace it with your local file path

RUN_NAME=$(basename "$0" .sh)


python -m verl.trainer.main \
    config=training_scripts/vqa_mcq_2b.yaml \
    data.train_files=data/vqa_mcq_840 \
    data.image_key=image \
    data.val_files=None \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.3 \
    worker.rollout.n=6 \
    worker.reward.compute_score=vqa_mcq \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.experiment_name=${RUN_NAME} \
    trainer.project_name=vqa_mcq_2b \
    trainer.save_checkpoint_path=checkpoints/${RUN_NAME}
