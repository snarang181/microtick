//===- MicrotickDialect.td - MicroTick dialect -------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MICROTICK_DIALECT_TD
#define MICROTICK_DIALECT_TD



def Tick_Dialect : Dialect {
    let name = "tick"; 
    let cppNamespace = "::microtick::tick";
}

#endif // MICROTICK_DIALECT_TD