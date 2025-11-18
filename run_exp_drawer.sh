#!/bin/bash

# Base command components

BASE_CMD="python dreamer_distill.py"
CONFIGS="--configs defaults metaworld"
LOGDIR_BASE="/storage/ssd1/richtsai1103/vid2act/log/metaworld/open3/window_close/new_moe/original"
ENCODER_MODE="original_conv"
TASK="metaworld_window_close"
DEVICE="cuda:4"
TEACHER_MODEL_PATH="/storage/ssd1/richtsai1103/vid2act/models/original_teacher/teacher_model.pt"
VAE_MODEL_PATH="/storage/ssd1/richtsai1103/vid2act/models/original_teacher/vae_model.pt"

# BASE_CMD="python dreamer_distill.py"
# CONFIGS="--configs defaults metaworld"
# LOGDIR_BASE="/storage/ssd1/richtsai1103/vid2act/log/metaworld/open3/window_close/new_moe/moe"
# ENCODER_MODE="moe_conv"
# TASK="metaworld_window_close"
# DEVICE="cuda:1"
# TEACHER_MODEL_PATH="/storage/ssd1/richtsai1103/vid2act/models/moe_teacher_new/teacher_model.pt"
# VAE_MODEL_PATH="/storage/ssd1/richtsai1103/vid2act/models/moe_teacher_new/vae_model.pt"

# Fixed random seeds
SEEDS=(0 123 456 789 2024)

echo "Using fixed seeds: ${SEEDS[@]}"
echo ""

# Run the command for each seed with different logdirs
for seed in "${SEEDS[@]}"
do
    echo "========================================="
    echo "Running experiment with seed: $seed"
    echo "========================================="
    
    # Append seed to logdir to keep results separate
    LOGDIR="${LOGDIR_BASE}_seed${seed}"
    
    # Run the command with proper quoting
    $BASE_CMD $CONFIGS --logdir "$LOGDIR" --encoder_mode $ENCODER_MODE --device $DEVICE --teacher_model_path "$TEACHER_MODEL_PATH" --vae_model_path "$VAE_MODEL_PATH" --task "$TASK" --seed $seed

    
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "Seed $seed completed successfully"
    else
        echo "Seed $seed failed with error code $?"
        # Uncomment the next line if you want to stop on first failure
        # exit 1
    fi
    
    echo ""
done

echo "All experiments completed!"
echo "Seeds used: ${SEEDS[@]}"
