#include "Microtick/MicrotickDialect.h"
#include "Microtick/MicrotickOps.h"
#include "Microtick/MicrotickPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace microtick::tick;

namespace {
static std::string makeHandlerName(func::FuncOp parentFunc, StringRef suffix) {
  std::string parentName = parentFunc.getSymName().str();
  parentName += suffix.str();
  return parentName;
}

/// Clone the body of a handler region inti a new func.func operation.
/// Drops `tick.yield` ops at the end of the region, and
/// ends the function with a `func.return` op.

template <typename HandlerOp>
static func::FuncOp lowerSingleHandler(HandlerOp handlerOp, ModuleOp module, OpBuilder &builder,
                                       StringRef suffix) {
  auto parentFunc = handlerOp->template getParentOfType<func::FuncOp>();
  if (!parentFunc) {
    handlerOp.emitError("handler op not nested in a func.func");
    return nullptr;
  }

  auto loc      = handlerOp.getLoc();
  auto funcType = builder.getFunctionType({}, {});

  std::string newName = makeHandlerName(parentFunc, suffix);
  auto        newFunc = func::FuncOp::create(loc, newName, funcType);
  newFunc.setPrivate();

  // Insert the new function into the module
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(parentFunc);
  builder.insert(newFunc);

  // Create entry block
  Block *entryBlock  = newFunc.addEntryBlock();
  Block &handerBlock = handlerOp.getBody().front();

  // Splice all ops from handlerBlock into entryBlock
  entryBlock->getOperations().splice(entryBlock->end(), handerBlock.getOperations());

  // Remove tick.yield ops and replace with func.return
  for (auto it = entryBlock->begin(), e = entryBlock->end(); it != e;) {
    Operation &op = *it++; // Increment iterator before potentially erasing
    if (auto yieldOp = dyn_cast<YieldOp>(op)) {
      OpBuilder yb(yieldOp);
      (void) func::ReturnOp::create(yb, yieldOp.getLoc(), ValueRange{});
      yieldOp.erase();
    }
  }
  return newFunc;
}

struct MicrotickLowerHandlersPass
    : public PassWrapper<MicrotickLowerHandlersPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MicrotickLowerHandlersPass)

  void runOnOperation() override {
    ModuleOp  module = getOperation();
    OpBuilder builder(module.getContext());

    // Collect handlers first, then transform to avoid iterator invalidation
    SmallVector<microtick::tick::OnBookOp>  onBookHandlers;
    SmallVector<microtick::tick::OnTradeOp> onTradeHandlers;
    SmallVector<microtick::tick::OnTimerOp> onMarketDataHandlers;

    module.walk([&](microtick::tick::OnBookOp op) { onBookHandlers.push_back(op); });
    module.walk([&](microtick::tick::OnTradeOp op) { onTradeHandlers.push_back(op); });
    module.walk([&](microtick::tick::OnTimerOp op) { onMarketDataHandlers.push_back(op); });

    for (microtick::tick::OnBookOp onBook : onBookHandlers) {
      lowerSingleHandler(onBook, module, builder, "_on_book");
      onBook.erase();
    }

    for (microtick::tick::OnTradeOp onTrade : onTradeHandlers) {
      lowerSingleHandler(onTrade, module, builder, "_on_trade");
      onTrade.erase();
    }

    for (microtick::tick::OnTimerOp onTimer : onMarketDataHandlers) {
      lowerSingleHandler(onTimer, module, builder, "_on_timer");
      onTimer.erase();
    }
  }

  StringRef getArgument() const final { return "microtick-lower-handlers"; }

  StringRef getDescription() const final {
    return "Lower Microtick handler ops into plain func.func entrypoints.";
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> microtick::tick::createMicrotickLowerHandlersPass() {
  return std::make_unique<MicrotickLowerHandlersPass>();
}