# Microtick 🕒

**Microtick** is an experimental [MLIR](https://mlir.llvm.org/)–based DSL for expressing
high-frequency trading (HFT) strategies as IR.

## Lower your IR from initial MLIR 

Let's say ${REPO_ROOT}/testing-ir/strategy_messy.mlir 

```bash
microtick-opt/microtick-opt \
  --microtick-verify \
  --microtick-lower-runtime \
  --microtick-lower-handlers \
  ../testing-ir/strategy_messy.mlir \
  -o ../strategy_messy.lowered.mlir
```

### Lower MLIR native ops

```bash 
mlir-opt \
  ../strategy_messy.lowered.mlir \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts \
  -o ../strategy_messy.llvm.mlir
```

### Go from MLIR->LLVM

```bash
mlir-translate \
  --mlir-to-llvmir \
  ../strategy_messy.llvm.mlir \
  -o ../strategy_messy.ll
```

### LLIR to Obj 

```bash 
llc \
  -filetype=obj \
  ../strategy_messy.ll \
  -o ../strategy_messy.o
```

### Compile to dynamic shared obj 

```bash 
clang -shared -undefined dynamic_lookup \
  ../strategy_messy.o \
  -o ../libstrategy_messy.dylib
```

### Finally, we can run the engine 

```
./runtime/src/microtick-engine ../libstrategy_messy.dylib
```
