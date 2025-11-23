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

/// Helper: enforce "risk-before-send" invariants in MicroTick IR
///
/// For each block in the handler body
/// require that any `tick.order.send` is dominated by a
/// `tick.risk_check.notional` in the same block.
template <typename HandlerOp>
static void verifyRiskBeforeSendInHandler(HandlerOp handlerOp, bool &hasError) {
  Block &block             = handlerOp.getBody().front();
  bool   hasNotionalCheck  = false;
  bool   hasInventoryCheck = false;

  for (Operation &op : block) {
    if (llvm::isa<RiskCheckNotionalOp>(op)) {
      hasNotionalCheck = true;
      continue;
    }
    if (llvm::isa<RiskCheckInventoryOp>(op)) {
      hasInventoryCheck = true;
      continue;
    }

    if (auto orderSendOp = dyn_cast<OrderSendOp>(op)) {
      if (!hasNotionalCheck || !hasInventoryCheck) {
        hasError = true;
        op.emitError() << "`tick.order.send` must be dominated by a "
                          "`tick.risk_check.notional` in the same block.";
      }
      continue;
    }
  }
}

/// Helper: disallow direct sends/cancels in tick.on_timer handlers
static void verifyNoOrdersInTimer(OnTimerOp OnTimerOp, bool &hasError) {
  Block &block = OnTimerOp.getBody().front();

  for (Operation &op : block) {
    if (llvm::isa<OrderSendOp>(&op)) {
      hasError = true;
      op.emitError() << "`tick.order.send` is not allowed in `tick.on_timer` handlers.";
    }
    if (llvm::isa<OrderCancelOp>(&op)) {
      hasError = true;
      op.emitError() << "`tick.order.cancel` is not allowed in `tick.on_timer` handlers.";
    }
  }
}
struct MicrotickVerifyPass : public PassWrapper<MicrotickVerifyPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MicrotickVerifyPass)

  void runOnOperation() override {
    func::FuncOp func    = getOperation();
    bool         hasEror = false;

    // 1. Verify that every `tick.order.send` is dominated by a
    //    `tick.risk_check.notional` in the same block.
    func.walk([&](OnBookOp onBookOp) { verifyRiskBeforeSendInHandler(onBookOp, hasEror); });
    func.walk([&](OnTradeOp onTradeOp) { verifyRiskBeforeSendInHandler(onTradeOp, hasEror); });
    // 2. Verify that `tick.on_timer` handlers do not contain direct order sends/cancels
    func.walk([&](OnTimerOp onTimerOp) { verifyNoOrdersInTimer(onTimerOp, hasEror); });
    if (hasEror) {
      signalPassFailure();
    }
  }
  StringRef getArgument() const final { return "microtick-verify"; }
  StringRef getDescription() const final {
    return "Verify MicroTick IR strategy-level invariants (risk-before-send).";
  }
};

/// Helper already used by the verify pass:
/// - RiskCheckNotionalOp
/// - RiskCheckInventoryOp
/// - OnBookOp/OnTradeOp/OnTimerOp
///
/// Hoist all risk check ops to the top of the handler block.
///
/// For now, we only look at the firs block of the handler and:
/// - Collect all risk check ops
/// Move them before the first non-risk-check op. (This is a simple heuristic)

template <typename HandlerOp> static void hoisRiskChecksInHandler(HandlerOp handlerOp) {
  Block &block = handlerOp.getBody().front();

  // Collect risk check ops
  SmallVector<Operation *> riskCheckOps;
  for (Operation &op : block) {
    if (llvm::isa<RiskCheckNotionalOp>(op) || llvm::isa<RiskCheckInventoryOp>(op)) {
      riskCheckOps.push_back(&op);
    }
    if (riskCheckOps.empty())
      return; // No risk checks to hoist

    // Find the insertion point: first non-risk-check op
    Operation *firstNonRiskCheckOp = nullptr;
    for (Operation &op : block) {
      if (!llvm::isa<RiskCheckNotionalOp>(op) &&
          !llvm::isa<RiskCheckInventoryOp>(op)) { // non-risk-check
        firstNonRiskCheckOp = &op;
        break;
      }
    }
    // If all ops are risk checks, nothing to do
    if (!firstNonRiskCheckOp)
      return;

    // MOve each risk op (in order) before the insertion point
    for (Operation *riskOp : riskCheckOps) {
      riskOp->moveBefore(firstNonRiskCheckOp);
    }
  }
}

struct MicrotickCanonicalizeHandlersPass
    : public PassWrapper<MicrotickCanonicalizeHandlersPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MicrotickCanonicalizeHandlersPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    func.walk([&](OnBookOp onBookOp) { hoisRiskChecksInHandler(onBookOp); });
    func.walk([&](OnTradeOp onTradeOp) { hoisRiskChecksInHandler(onTradeOp); });
    // Note: No risk checks in OnTimerOp
  }

  StringRef getArgument() const final { return "microtick-canonicalize-handlers"; }
  StringRef getDescription() const final {
    return "Canonicalize MicroTick handler ops (hoist risk checks).";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> microtick::tick::createMicrotickVerifyPass() {
  return std::make_unique<MicrotickVerifyPass>();
}

std::unique_ptr<mlir::Pass> microtick::tick::createMicrotickCanonicalizeHandlersPass() {
  return std::make_unique<MicrotickCanonicalizeHandlersPass>();
}

void microtick::tick::registerMicrotickPasses() {
  PassRegistration<MicrotickVerifyPass>();
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return microtick::tick::createMicrotickLowerRuntimePass();
  });
  PassRegistration<MicrotickCanonicalizeHandlersPass>();
}