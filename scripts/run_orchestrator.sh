#!/bin/bash

export WANDB_ENTITY="lamas-aipr"
export WANDB_PROJECT="orchestrator-grpo"

# Activate the environment
# source activate verifier_env

# Default values for command-line arguments
MODEL_NAME=${1:-"Qwen/Qwen2.5-Coder-3B-Instruct"}
LEARNING_RATE=${2:-"1e-6"}
NUM_GENERATIONS=${3:-"8"}
PER_DEVICE_TRAIN_BATCH_SIZE=${4:-"8"}
GRAD_ACCUM_STEPS=${5:-"4"}
NUM_ITERATIONS=${6:-"1"}
MAX_STEPS=${7:-"200"}
BETA=${8:-"0.0"}
NUM_AGENTS=${9:-"2"}

# Detect the number of GPUs on the machine
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
NUM_GPUS_MINUS_1=$((NUM_GPUS - 1))
echo "Detected ${NUM_GPUS} GPUs on the machine"

# Display configuration
echo "Using ${NUM_GPUS_MINUS_1} GPUs for training and 1 GPU for rollout generation with model ${MODEL_NAME}"
echo "Number of Agents: ${NUM_AGENTS}"

# Launch the Orchestrator-Worker GRPO-MR training
# Using trainer "grpo" without --no_turn_reward implies GRPO-MR (Merged Reward)
accelerate launch --config-file configs/zero3.yaml --num-processes ${NUM_GPUS_MINUS_1} \
    verifiers/examples/math_orchestrator.py \
    --model_name ${MODEL_NAME} \
    --num_gpus ${NUM_GPUS} \
    --learning_rate ${LEARNING_RATE} \
    --num_generations ${NUM_GENERATIONS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --grad_accum_steps ${GRAD_ACCUM_STEPS} \
    --num_iterations ${NUM_ITERATIONS} \
    --max_steps ${MAX_STEPS} \
    --beta ${BETA} \
    --trainer "grpo" \
    --num_agents ${NUM_AGENTS} \

