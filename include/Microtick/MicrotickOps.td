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

// tick.order.send %price, %qty symbol side : f64, i64, string, string
def Tick_OrderSendOp : Op<Tick_Dialect, "order.send"> {
    let summary = "Send an order to the exchange";
    let description = [{
        Issues an order with 
            - price : f64
            - quantity : i64
            - symbol : string attribute (e.g., "AAPL")
            - side : string attribute (e.g., "buy" or "sell")
    }];

    let arguments = (ins
        F64:$price,
        I64:$qty,
        StrAttr:$symbol,
        StrAttr:$side
    );
    let results = (outs);
    let assemblyFormat = [{
    $price `,` $qty
      `symbol` `(` $symbol `)`
      `side` `(` $side `)`
      attr-dict
      `:` type($price) `,` type($qty)
  }];

}

// Cancel a previously sent order.
// For now, we just key off symbol + client_order_id + side.
def Tick_OrderCancelOp : Op<Tick_Dialect, "order.cancel"> {
    let summary = "Cancel an existing order";
    let description = [{
        Cancels an existing order identified by:
            - symbol : string attribute (e.g., "AAPL")
            - client_order_id : string attribute (unique per client)
            - side : string attribute (e.g., "buy" or "sell")
        
        This op has no operands or results; it is a logical operation.
    }];

    let arguments = (ins
        StrAttr:$symbol,
        StrAttr:$client_order_id,
        StrAttr:$side
    );

    // We need a custom C++ verifier.
    let hasVerifier = 1;

    let assemblyFormat = [{
    `symbol` `(` $symbol `)`
    `client_order_id` `(` $client_order_id `)`
    `side` `(` $side `)`
    attr-dict
  }];

}

#endif // MICROTICK_OPS_TD
