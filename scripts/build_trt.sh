#!/usr/bin/env bash
# Compile an ONNX model to a TensorRT FP16 engine via trtexec.
# Usage: build_trt.sh <onnx_file> <engine_file>
# Requires TensorRT (provided by JetPack 5.x on Jetson Orin Nano).

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <onnx_file> <engine_file>" >&2
    exit 1
fi

ONNX="$1"
ENGINE="$2"

trtexec \
    --onnx="${ONNX}" \
    --saveEngine="${ENGINE}" \
    --fp16 \
    --minShapes=input:1x3x32x32 \
    --optShapes=input:1x3x32x32 \
    --maxShapes=input:128x3x32x32

echo "  Built engine: ${ENGINE}"
