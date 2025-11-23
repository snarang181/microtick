//===- MicrotickOps.h - MicroTick ops ------------------------*- C++ -*-===//
//
// Part of the MicroTick Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#ifndef MICROTICK_MICROTICKOPS_H
#define MICROTICK_MICROTICKOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Microtick/MicrotickOps.h.inc"

#endif // MICROTICK_MICROTICKOPS_H