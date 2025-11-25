#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./utils/compile_strategy.sh path/to/strategy.mlir [basename] [num_events]
#
# Produces (under ${REPO_ROOT}/out/<basename>/):
#   <basename>.lowered.mlir
#   <basename>.llvm.mlir
#   <basename>.ll
#   <basename>.o
#   lib<basename>.dylib
# and optionally runs the engine.

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 path/to/strategy.mlir [basename] [num_events]" >&2
  exit 1
fi

INPUT="$1"
DEFAULT_BASE="$(basename "$INPUT" .mlir)"
BASENAME="${2:-$DEFAULT_BASE}"
NUM_EVENTS="${3:-10}"

REPO_ROOT=$(git rev-parse --show-toplevel)
MICROTICK_BUILD_DIR="$REPO_ROOT/build"
LLVM_BUILD_DIR="$HOME/Downloads/llvm-project/build"

OUT_DIR="$REPO_ROOT/out/${BASENAME}"
mkdir -p "$OUT_DIR"

MT_OPT="$MICROTICK_BUILD_DIR/microtick-opt/microtick-opt"
MLIR_OPT="$LLVM_BUILD_DIR/bin/mlir-opt"
MLIR_TRANSLATE="$LLVM_BUILD_DIR/bin/mlir-translate"
LLC="$LLVM_BUILD_DIR/bin/llc"
CLANG="clang"

echo "[INFO] Strategy input   : $INPUT"
echo "[INFO] Basename         : $BASENAME"
echo "[INFO] Output directory : $OUT_DIR"
echo "[INFO] Num events       : $NUM_EVENTS"

# 1) MicroTick pipeline
echo "[STEP] MicroTick pipeline -> ${BASENAME}.lowered.mlir"
"$MT_OPT" \
  --microtick-verify \
  --microtick-lower-runtime \
  --microtick-lower-handlers \
  "$INPUT" \
  -o "${OUT_DIR}/${BASENAME}.lowered.mlir"

# 2) MLIR -> LLVM dialect
echo "[STEP] Lower MLIR to LLVM dialect -> ${BASENAME}.llvm.mlir"
"$MLIR_OPT" \
  "${OUT_DIR}/${BASENAME}.lowered.mlir" \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts \
  -o "${OUT_DIR}/${BASENAME}.llvm.mlir"

# 3) LLVM dialect -> LLVM IR
echo "[STEP] Translate MLIR LLVM dialect to LLVM IR -> ${BASENAME}.ll"
"$MLIR_TRANSLATE" \
  --mlir-to-llvmir \
  "${OUT_DIR}/${BASENAME}.llvm.mlir" \
  -o "${OUT_DIR}/${BASENAME}.ll"

# 4) LLVM IR -> object
echo "[STEP] Compile LLVM IR to object file -> ${BASENAME}.o"
"$LLC" \
  -filetype=obj \
  "${OUT_DIR}/${BASENAME}.ll" \
  -o "${OUT_DIR}/${BASENAME}.o"

# 5) Object -> shared library
echo "[STEP] Link object file to shared library -> lib${BASENAME}.dylib"
"$CLANG" \
  -shared \
  -undefined dynamic_lookup \
  "${OUT_DIR}/${BASENAME}.o" \
  -o "${OUT_DIR}/lib${BASENAME}.dylib"

echo "[DONE] Build complete: ${OUT_DIR}/lib${BASENAME}.dylib"

# 6) Run engine, if available
ENGINE_BIN="$MICROTICK_BUILD_DIR/runtime/src/microtick-engine"
echo "[TEST] Running strategy with MicroTick engine (if available)"

if [ -x "$ENGINE_BIN" ]; then
  echo "Running: $ENGINE_BIN ${OUT_DIR}/lib${BASENAME}.dylib ${BASENAME} ${NUM_EVENTS}"
  "$ENGINE_BIN" "${OUT_DIR}/lib${BASENAME}.dylib" "${BASENAME}" "${NUM_EVENTS}"
else
  echo "MicroTick engine binary not found or not executable: $ENGINE_BIN"
  echo "Please build the MicroTick engine to run the strategy."
fi
