#!/bin/bash
# ==========================================================
# Train Residual Attention Network on CIFAR-100
# ==========================================================

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0

# 启动训练
python train.py \
  -net resnext50 \
  -gpu 0 \
  -b 128 \
  -lr 0.1 \
  -warm 1

# 启动 TensorBoard (可选)
echo "✅ Training started. You can visualize with TensorBoard:"
echo "tensorboard --logdir=runs --port=6006 --host=localhost"
