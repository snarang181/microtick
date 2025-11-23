// testing-ir/strategy_with_timer_ok.mlir
module {
  func.func @strategy_ok() {
    // Periodic risk housekeeping, no direct orders here.
    tick.on_timer attributes { period_ns = 1000000000 : i64 } {
      tick.risk.check_notional { limit = 5.000000e+05 : f64 }
      tick.yield
    }

    // Book-driven strategy logic.
    tick.on_book {
      tick.risk.check_notional { limit = 1.000000e+06 : f64 }

      %p = arith.constant 101.25 : f64
      %q = arith.constant 50 : i64

      tick.order.send %p, %q
        symbol("AAPL")
        side("Buy") : f64, i64

      tick.yield
    }

    return
  }
}
