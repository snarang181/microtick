#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./tools/compile_strategy.sh path/to/strategy.mlir [basename]
#
# Produces:
#   <basename>.lowered.mlir
#   <basename>.llvm.mlir
#   <basename>.ll
#   <basename>.o
#   lib<basename>.dylib

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 path/to/strategy.mlir [basename]" >&2
  exit 1
fi

INPUT="$1"
# Strip directory and extension to get basename
DEFAULT_BASE="$(basename "$INPUT" .mlir)"
BASENAME="${2:-$DEFAULT_BASE}"

REPO_ROOT=$(git rev-parse --show-toplevel)
MICROTICK_BUILD_DIR="$REPO_ROOT/build"
LLVM_BUILD_DIR="$HOME/Downloads/llvm-project/build"

MT_OPT="$MICROTICK_BUILD_DIR/microtick-opt/microtick-opt"
MLIR_OPT="$LLVM_BUILD_DIR/bin/mlir-opt"
MLIR_TRANSLATE="$LLVM_BUILD_DIR/bin/mlir-translate"
LLC="$LLVM_BUILD_DIR/bin/llc"
CLANG="clang"
STRATEGY_NAME="$BASENAME"

echo "[STEP] MicroTick pipeline -> ${BASENAME}.lowered.mlir"
"$MT_OPT" \
  --microtick-verify \
  --microtick-lower-runtime \
  --microtick-lower-handlers \
    "$INPUT" \
    -o "${BASENAME}.lowered.mlir"

echo "[STEP] Lower MLIR to LLVM dialect -> ${BASENAME}.llvm.mlir"
"$MLIR_OPT" \
    "${BASENAME}.lowered.mlir" \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --convert-cf-to-llvm \
    --reconcile-unrealized-casts \
    -o "${BASENAME}.llvm.mlir"

echo "[STEP] Translate MLIR LLVM dialect to LLVM IR -> ${BASENAME}.ll"
"$MLIR_TRANSLATE" \
    --mlir-to-llvmir \
    "${BASENAME}.llvm.mlir" \
    -o "${BASENAME}.ll"

echo "[STEP] Compile LLVM IR to object file -> ${BASENAME}.o"
"$LLC" \
    -filetype=obj \
    "${BASENAME}.ll" \
    -o "${BASENAME}.o"

echo "[STEP] Link object file to shared library -> lib${BASENAME}.dylib"
"$CLANG" \
    -shared \
    -undefined dynamic_lookup \
    "${BASENAME}.o" \
    -o "lib${BASENAME}.dylib"

echo "Build complete: lib${BASENAME}.dylib"

ENGINE_BIN="$MICROTICK_BUILD_DIR/runtime/src/microtick-engine"
echo "[TEST] Running strategy with MicroTick engine"

if [ -x "$ENGINE_BIN" ]; then
   echo "Running MicroTick engine with strategy lib${BASENAME}.dylib"
  "$ENGINE_BIN"  "./lib${BASENAME}.dylib" "${BASENAME}" 10
else
  echo "MicroTick engine binary not found or not executable: $ENGINE_BIN"
  echo "Please build the MicroTick engine to run the strategy."
fi