#!/bin/bash

# VQA MCQ Training with Qwen2-VL-2B
# Single 24GB GPU configuration

# Download pretrained model first if needed:
# mkdir -p pretrained_models
# cd pretrained_models
# git lfs install
# git clone https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct

python -m verl.trainer.main \
    data.train_files=data/vqa_mcq_840 \
    +data.image_key=image \
    worker.actor.model.model_path=pretrained_models/Qwen2-VL-2B-Instruct \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.3 \
    worker.rollout.n=6 \
    worker.reward.compute_score=vqa_mcq \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=checkpoints/vqa_mcq_2b \
    trainer.project_name=vqa_mcq_2b
