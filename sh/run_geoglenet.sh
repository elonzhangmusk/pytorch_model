#!/bin/bash
# ==========================================================
# Train GoogLeNet on your dataset using 4 GPUs
# ==========================================================

# 指定使用的 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置训练参数
BATCH_SIZE=64
LEARNING_RATE=0.003
WARMUP=1

# 启动训练
python train.py \
-net googlenet \
-gpu 0 \
-b $BATCH_SIZE \
-lr $LEARNING_RATE \
-warm $WARMUP

# TensorBoard 提示
echo "✅ Training started. You can visualize with TensorBoard:"
echo "tensorboard --logdir=runs --port=6006 --host=localhost"
