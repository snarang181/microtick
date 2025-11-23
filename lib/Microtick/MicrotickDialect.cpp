//===- MicrotickDialect.cpp - MicroTick 'tick' dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "Microtick/MicrotickDialect.h"
#include "Microtick/MicrotickOps.cpp.inc"
#include "Microtick/MicrotickOps.h"
#include "Microtick/MicrotickOpsDialect.cpp.inc"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace microtick::tick;

void TickDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Microtick/MicrotickOps.cpp.inc"
      >();
}