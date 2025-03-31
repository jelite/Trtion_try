#!/bin/bash

# 경로 설정
KERNEL_PATH="/workspace/Trtion_try/vectorAdd.py"
KERNEL_NAME="add_kernel"
OUT_NAME="add_kernel"
SIGNATURE="*fp32:16, *fp32:16, *fp32:16, i32, 1024"
GRID="1,1,1"                                     # 예시 grid (필요시 수정)
NUM_WARPS=4
NUM_STAGES=3

# Triton AOT 컴파일 실행
python3 compile.py \
  --kernel-name $KERNEL_NAME \
  --signature "$SIGNATURE" \
  --grid "$GRID" \
  --num-warps $NUM_WARPS \
  --num-stages $NUM_STAGES \
  --out-name $OUT_NAME \
  $KERNEL_PATH