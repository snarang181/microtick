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
def Tick_OrderSendOp : Op<Tick_Dialect, "order.send", [HasParent<"OnBookOp">]> {
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
def Tick_OrderCancelOp : Op<Tick_Dialect, "order.cancel", [HasParent<"OnBookOp">]> {
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

// Event handler op hat runs when the order book updates.
// Usage:
//  tick.on_book {
//    ... strategy code ...
//}
def Tick_OnBookOp : Op<Tick_Dialect, "on_book", [IsolatedFromAbove]> {
    let summary = "Event handler for order book updates";
    let description = [{
        Represents a strategy callback that runs on each book update.
        The body contains the strategy logic to be executed (sends, cancels, etc.).
        For now, region takes no arguments and has no results.
    }];

    let regions = (region AnyRegion:$body);

    let assemblyFormat = [{
    attr-dict-with-keyword
    $body
  }];

}

// Terminator for the tick.on_book body region.
def Tick_YieldOp : Op<Tick_Dialect, "yield",
                      [Terminator, HasParent<"OnBookOp">]> {
  let summary = "terminator for tick.on_book region";
  let description = [{
    Terminates the body region of a tick.on_book handler. For now, it takes
    no operands and produces no results.
  }];

  let assemblyFormat = "attr-dict";
}

// Check that the notional (price * qty) is within a specified limit.
//
// For now we just have a single attribute: limit : f64

def Tick_RiskCheckNotionalOp : Op<Tick_Dialect, "risk.check_notional", [HasParent<"OnBookOp">]> {
    let summary = "Check that order notional is within limit";
    let description = [{
        Asserts that the subsuequent orders (in this handler) should respect the
        given notional limit (price * quantity <= limit).
        For now, this just encodes the entent; a later pass
        will enforce that every tick.order.send is dominated by 
        at least one tick.risk.check_notional with an appropriate limit.
    }];

    let arguments = (ins
        F64Attr:$limit // e.g. 1_000_000.0 for one million
    );
    // We need a custom C++ verifier to enforce limit > 0.
    let hasVerifier = 1;
    let assemblyFormat = "attr-dict";

}

#endif // MICROTICK_OPS_TD
