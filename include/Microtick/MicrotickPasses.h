//===- MicrotickPasses.h - MicroTick passes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MICROTICK_MICROTICKPASSES_H
#define MICROTICK_MICROTICKPASSES_H

#include "mlir/Pass/Pass.h"

namespace microtick {
namespace tick {

/// Create a pass that verifies strategy-level invariants for MicroTick IR.
/// As of now, we enforce
/// - In each `tick.on_book` handler, every `tick.order.send` must be
/// dominated in the same block by a `tick.risk_check.notional` operation.

std::unique_ptr<mlir::Pass> createMicrotickVerifyPass();

/// Create a pass that lowers MicroTick IR to a simple runtime API.
/// expressed using the func dialect. This rewrites
/// - tick.order_send -> func.call @mt_order_send(...);
/// - tick.order_cancel -> func.call @mt_order_cancel(...);
std::unique_ptr<mlir::Pass> createMicrotickLowerRuntimePass();

/// Register all MicroTick passes.
void registerMicrotickPasses();

} // namespace tick
} // namespace microtick

#endif // MICROTICK_MICROTICKPASSES_H
