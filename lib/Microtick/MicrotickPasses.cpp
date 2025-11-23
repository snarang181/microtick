//===- MicrotickPasses.cpp - MicroTick passes -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "Microtick/MicrotickPasses.h"
#include "Microtick/MicrotickDialect.h"
#include "Microtick/MicrotickOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace microtick::tick;

namespace {
struct MicrotickVerifyPass : public PassWrapper<MicrotickVerifyPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MicrotickVerifyPass)

  void runOnOperation() override {
    func::FuncOp func    = getOperation();
    bool         hasEror = false;

    // 1. Verify that every `tick.order.send` is dominated by a
    //    `tick.risk_check.notional` in the same block.
    func.walk([&](OnBookOp onBook) {
      Block &block          = onBook.getBody().front();
      bool   hasRiskCheckOp = false;

      for (Operation &op : block) {
        if (auto riskOp = dyn_cast<RiskCheckNotionalOp>(op)) {
          hasRiskCheckOp = true;
          continue;
        }

        if (auto orderSendOp = dyn_cast<OrderSendOp>(op)) {
          if (!hasRiskCheckOp) {
            hasEror = true;
            op.emitError() << "`tick.order.send` must be dominated by a "
                              "`tick.risk_check.notional` in the same block.";
          }
          continue;
        }
      }
    });
    // 2. Risk-before-send in tick.on_trade (same as tick.on_book for now).
    func.walk([&](OnTradeOp onTrade) {
      Block &block          = onTrade.getBody().front();
      bool   hasRiskCheckOp = false;

      for (Operation &op : block) {
        if (auto riskOp = dyn_cast<RiskCheckNotionalOp>(op)) {
          hasRiskCheckOp = true;
          continue;
        }

        if (auto orderSendOp = dyn_cast<OrderSendOp>(op)) {
          if (!hasRiskCheckOp) {
            hasEror = true;
            op.emitError() << "`tick.order.send` must be dominated by a "
                              "`tick.risk_check.notional` in the same block.";
          }
          continue;
        }
      }
    });

    // 3. No direct sends/cancels in tick.on_timer handlers.
    func.walk([&](OnTimerOp onTimer) {
      Block &block = onTimer.getBody().front();

      for (Operation &op : block) {
        if (auto orderSendOp = dyn_cast<OrderSendOp>(op)) {
          hasEror = true;
          op.emitError() << "`tick.order.send` is not allowed in `tick.on_timer` handlers.";
        }
        if (auto orderCancelOp = dyn_cast<OrderCancelOp>(op)) {
          hasEror = true;
          op.emitError() << "`tick.order.cancel` is not allowed in `tick.on_timer` handlers.";
        }
      }
    });
    if (hasEror) {
      signalPassFailure();
    }
  }
  StringRef getArgument() const final { return "microtick-verify"; }
  StringRef getDescription() const final {
    return "Verify MicroTick IR strategy-level invariants (risk-before-send).";
  }
};
} // namespace

std::unique_ptr<mlir::Pass> microtick::tick::createMicrotickVerifyPass() {
  return std::make_unique<MicrotickVerifyPass>();
}

void microtick::tick::registerMicrotickPasses() { PassRegistration<MicrotickVerifyPass>(); }