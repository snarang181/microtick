//===- MicrotickLowerRuntime.cpp - Lower MicroTick to runtime calls -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Microtick/MicrotickDialect.h"
#include "Microtick/MicrotickOps.h"
#include "Microtick/MicrotickPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace microtick::tick;

namespace {
/// Ensure a declaration for the MicroTick runtime functions exists.
/// Something like `func @mt_order_send(f64, i64)` exists in the module.
/// Returns the FuncOp for the function. (Creates it if it does not exist.)

static func::FuncOp getOrCreateMtOrderSendFunc(ModuleOp module, OpBuilder &builder) {
  constexpr StringLiteral funcName("mt_order_send");

  // Check if the function is already declared.
  if (auto existingFunc = module.lookupSymbol<func::FuncOp>(funcName))
    return existingFunc;

  auto loc    = module.getLoc();
  auto f64Ty  = builder.getF64Type();
  auto i64Ty  = builder.getI64Type();
  auto fnType = builder.getFunctionType({f64Ty, i64Ty}, {});

  auto func = func::FuncOp::create(loc, funcName, fnType);
  // Runtime functions should be set private.
  func.setPrivate();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  builder.insert(func);
  return func;
}

/// Ensure a declaration for `mt_order_cancel` exists.
static func::FuncOp getOrCreateMtOrderCancelFunc(ModuleOp module, OpBuilder &builder) {
  constexpr StringLiteral funcName("mt_order_cancel");

  // Check if the function is already declared.
  if (auto existingFunc = module.lookupSymbol<func::FuncOp>(funcName))
    return existingFunc;

  auto loc    = module.getLoc();
  auto fnType = builder.getFunctionType({}, {});

  auto func = func::FuncOp::create(loc, funcName, fnType);
  // Runtime functions should be set private.
  func.setPrivate();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  builder.insert(func);
  return func;
}

/// Pattern: tick.order.send %price, %quantity ... -> func.call @mt_order_send(%price, %quantity)
struct LowerOrderSendPattern : public OpRewritePattern<OrderSendOp> {
  using OpRewritePattern<OrderSendOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(OrderSendOp op, PatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(op, "not inside a module");

    OpBuilder builder(module.getContext());
    auto      sendFunc = getOrCreateMtOrderSendFunc(module, builder);

    SmallVector<Value> args;
    args.push_back(op.getPrice());
    args.push_back(op.getQty());

    rewriter.replaceOpWithNewOp<func::CallOp>(op, sendFunc.getSymName(),
                                              sendFunc.getFunctionType().getResults(), args);
    return success();
  }
};

/// Pattern: tick.order.cancel ... -> func.call @mt_order_cancel()
struct LowerOrderCancelPattern : public OpRewritePattern<OrderCancelOp> {
  using OpRewritePattern<OrderCancelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(OrderCancelOp op, PatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(op, "not inside a module");

    OpBuilder          builder(module.getContext());
    auto               cancelFunc = getOrCreateMtOrderCancelFunc(module, builder);
    SmallVector<Value> args; // No arguments for cancel

    rewriter.replaceOpWithNewOp<func::CallOp>(op, cancelFunc.getSymName(),
                                              cancelFunc.getFunctionType().getResults(), args);
    return success();
  }
};

struct MicrotickLowerRuntimePass
    : public PassWrapper<MicrotickLowerRuntimePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MicrotickLowerRuntimePass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerOrderSendPattern, LowerOrderCancelPattern>(&getContext());

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const final { return "microtick-lower-runtime"; }
  StringRef getDescription() const final {
    return "Lower MicroTick operations to runtime API calls.";
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> microtick::tick::createMicrotickLowerRuntimePass() {
  return std::make_unique<MicrotickLowerRuntimePass>();
}
