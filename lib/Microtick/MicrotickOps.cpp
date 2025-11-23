//===- MicrotickOps.cpp - Microtick Ops Implementation --------------------===//
//
// Part of the MicroTick Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
#include "Microtick/MicrotickOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace microtick::tick;

// This macro block comes from the generated .cpp.inc
#define GET_OP_CLASSES
#include "Microtick/MicrotickOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Tick_OrderCancelOp verifier
//===----------------------------------------------------------------------===//

LogicalResult OrderCancelOp::verify() {
  // side must be "Buy" or "Sell".
  llvm::StringRef sideStr = getSide();
  if (sideStr != "Buy" && sideStr != "Sell") {
    return emitOpError("side attribute must be either 'Buy' or 'Sell'");
  }
  return success();
}