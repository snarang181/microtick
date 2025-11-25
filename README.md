# Microtick 🕒

**Microtick** is an experimental [MLIR](https://mlir.llvm.org/)–based DSL for expressing
high-frequency trading (HFT) strategies as IR.

## Building Microtick

### Prerequisites

Microtick is built against a local LLVM/MLIR build tree (not system packages).

You’ll need:

- A local LLVM/MLIR build with the following tools built:
  - `mlir-opt`, `mlir-translate`, `llc`, `clang`
- CMake ≥ 3.20
- Ninja
- A C++17 compiler (Clang recommended)

The examples below assume:

```text
LLVM build:  $HOME/Downloads/llvm-project/build
Microtick:   $REPO_ROOT (this repo)
```
and that your LLVM build was configured with MLIR enabled, e.g. something like:
```bash
cmake -G Ninja \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DCMAKE_BUILD_TYPE=Release \
  ../llvm
ninja
```

**Configure and build Microtick**
```bash
cd ${REPO_ROOT}
mkdir -p build
cd build

cmake -G Ninja \
  -DLLVM_DIR=$HOME/Downloads/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=$HOME/Downloads/llvm-project/build/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Debug \
  ..

# Build the Microtick tools and engine
ninja 

## Verification
# Check that the custom microtick-opt exists and is wired up
./microtick-opt/microtick-opt --help | grep microtick

# Check that the engine binary is built
./runtime/src/microtick-engine || true
# (it will print a usage message complaining about a missing strategy, which is fine)
```

Congrats, you're all set up!

## End-to-End Pipeline & Naming Conventions

MicroTick has a reproducible end-to-end flow:

1. **Tick dialect strategy** (MLIR with `tick.on_book`, `tick.order.send`, etc.)
2. **MicroTick passes** (`microtick-opt`)
   - `--microtick-verify`
   - `--microtick-lower-runtime`
   - `--microtick-lower-handlers`
3. **MLIR → LLVM lowering**
4. **LLVM → shared library** (`lib<strategy>.dylib`)
5. **C++ engine** (`microtick-engine`)  
   - `dlopen` is the shared library
   - `dlsym` is the strategy entrypoint
   - calls it repeatedly and implements the runtime API (`mt_order_send`, `mt_order_cancel`, …)
  
## Inspecting `${REPO_ROOT}/testing-ir/strategy_e2e_demo.mlir`

Let's say we run `./utils/compile_strategy.sh testing-ir/strategy_e2e_demo.mlir strategy_e2e_demo 10`

In the MLIR, we have two send orders followed by a cancel: a passive buy of (99.50 x 10) and an aggressive buy of (101.25 x 50) followed by immediate cancel of the most recebt buy.The surviving orders at the end of the event handler, the send order will be filled at the *limit price*. 

So per market event:

  * $99.50 * 10$: open and gets filled

  * $101.25 * 50$: canceled and gets skipped

Since we invoked 10 market events:
  
  Per Event:

    * Position Change: +10

    * Cash Change: -99.50 x 10 = -995 
    
  After 10 events:

    * Position = 10 x 10 = 100 

    * Cash = 10 x (-995) = -9,950.00
  
  The Engine assumes a market price of 100 for the symbol to calculate Profit and Loss (*PnL*)

  ```cpp
  double marketPrice = 100.0;
double pnl = cash + position * marketPrice;
           = -9950 + 100 * 100
           = -9950 + 10000
           = 50

  ```
  



## Lower your IR from initial MLIR (e2e demo)

We’ll use the demo strategy in:

- `${REPO_ROOT}/testing-ir/strategy_e2e_demo.mlir`

```mlir
// testing-ir/strategy_e2e_demo.mlir
module {
  func.func @strategy_e2e_demo() {
    tick.on_book {
      %p = arith.constant 101.25 : f64
      %q = arith.constant 50 : i64

      tick.risk.check_notional { limit = 1.000000e+06 : f64 }
      tick.risk.check_inventory { limit = 1000 : i64 }

      tick.order.send %p, %q
        symbol("AAPL")
        side("Buy") : f64, i64

      tick.order.cancel
        symbol("AAPL")
        client_order_id("ABC123")
        side("Buy")

      tick.yield
    }
    return
  }
}
```

**TL;DR: one-shot e2e with the helper script**

From `${REPO_ROOT}` 

`./utils/compile_strategy.sh testing-ir/strategy_e2e_demo.mlir strategy_e2e_demo 10`

## Manual Steps (Inspect the IR)

All commands below assume you are in the build directory:

`cd ${REPO_ROOT}/build`

1. **MicroTick pipeline: verify + runtime lowering + handler lowering**

```bash
microtick-opt/microtick-opt \
  --microtick-verify \
  --microtick-lower-runtime \
  --microtick-lower-handlers \
  ../testing-ir/strategy_e2e_demo.mlir \
  -o ../strategy_e2e_demo.lowered.mlir
```

This produces an IR where:

  - `tick.order.send` → `call @mt_order_send(i32, i8, f64, i64)`

  - `tick.order.cancel` → `call @mt_order_cancel(i32, i8)`

  - `tick.on_book` → `func.func private @strategy_e2e_demo_on_book()`

2. **Lower MLIR native ops to LLVM dialect**

```bash
mlir-opt \
  ../strategy_e2e_demo.lowered.mlir \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts \
  -o ../strategy_e2e_demo.llvm.mlir
```

3. **Go from MLIR-LLVM dialect to LLVM-IR
```bash
mlir-translate \
  --mlir-to-llvmir \
  ../strategy_e2e_demo.llvm.mlir \
  -o ../strategy_e2e_demo.ll
```

4. **LLVM IR to Object File**
```bash
llc \
  -filetype=obj \
  ../strategy_e2e_demo.ll \
  -o ../strategy_e2e_demo.o
```

5. **Compile to a dynamic shared lib**
```
clang -shared -undefined dynamic_lookup \
  ../strategy_e2e_demo.o \
  -o ../libstrategy_e2e_demo.dylib
```

6. **Run the engine with the compiled strategy**
```
./runtime/src/microtick-engine \
  ../libstrategy_e2e_demo.dylib \
  strategy_e2e_demo \
  10
```

Here:
  - `../libstrategy_e2e_demo.dylib` is the compiled strategy. 
  - `strategy_e2e_demo` is the **base name** of the strategy.
  - the engine looks up `strategy_e2e_demo_on_book` as the entry point and calls it 10 times. 
  


### Strategy naming convention

To keep things simple, MicroTick assumes the same **base name** is used at all of these levels:

- **MLIR file name (without extension)**  
  Example:
  ```text
  testing-ir/strategy_messy.mlir
  → basename = "strategy_messy"
  
  Top-level MLIR strategy function

  func.func @strategy_messy() {
  tick.on_book {
    // ...
    tick.yield
  }
  return
  } 


