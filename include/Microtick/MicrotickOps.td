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

// Trade event hander:
// 
// tick.on_trade {
//   ...
//   tick.yield
// }
def Tick_OnTradeOp : Op<Tick_Dialect, "on_trade", [IsolatedFromAbove]> {
  let summary = "Event handler for trade events";
  let description = [{
    Represents a strategy callback that runs on each trade or print event.
    The body region contains the reactive logic (e.g., reacting to 
    aggressive trades, updation of internal signals, staes, placing/canceling orders).
    For now, region takes no arguments; trade details are assumed to be available via
    some external mechanism (runtime, global state, etc.).
  }];
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    attr-dict-with-keyword
    $body
  }];
}

// Periodic timer callback:
//
// tick.on_timer period_ns = 1_000_000 {
// .... 
//  tick.yield
// }
def Tick_OnTimerOp : Op<Tick_Dialect, "on_timer", [IsolatedFromAbove]> {
  let summary = "Event handler for periodic timer callbacks";
  let description = [{
    Represents a perioidic callback that runs every `period_ns` nanoseconds.
    The body contains risk/housekeeping logic to be executed. We will, for now,
    enforce that no `tick.order.send` ops are allowed in this region.
    Actual order sending should be done in the `tick.on_book` handler. 
    This is intentional to separate risk logic from order sending logic.
  }];

  // period_ns : i64 attribute
  let arguments = (ins
    I64Attr:$period_ns // e.g., 1_000_000 for 1 millisecond
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    attr-dict-with-keyword
    $body
  }];

}

// Check that inventory (current position) is within specified limits.
//
// This is a marker op: it carries an integer limit attribute
// representing the maximum absolute position allowed for the given symbol.
def Tick_RiskCheckInventoryOp : Op<Tick_Dialect, "risk.check_inventory"> {
    let summary = "Check that inventory position is within limit";
    let description = [{
        Asserts that the subsuequent orders (in this handler) should respect the
        given inventory limit (abs(position) <= limit).
        For now, this just encodes the entent; a verify pass
        will enforce that every tick.order.send is