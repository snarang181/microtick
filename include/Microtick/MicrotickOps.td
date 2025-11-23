//===- MicrotickOps.td - MicroTick dialect ops -------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MICROTICK_OPS_TD
#define MICROTICK_OPS_TD

include "mlir/IR/OpBase.td"
include "MicrotickDialect.td"

def Tick_PingOp : Op<Tick_Dialect, "ping"> {
    let summary = "MicroTick ping op (no-op, for testing)";
    let description = [{
        The `tick.ping` operation is a no-operation (no-op) used primarily for
        tesing and useful to verify that the MicroTick dialect is correctly set up
        within an MLIR environment.
    }];
    let assemblyFormat = "attr-dict";
}

#endif // MICROTICK_OPS_TD
