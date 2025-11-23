// testing-ir/strategy_bad.mlir
module {
  func.func @strategy_bad() {
    tick.on_book {
      %p = arith.constant 101.25 : f64
      %q = arith.constant 50 : i64

      tick.order.send %p, %q
        symbol("AAPL")
        side("Buy") : f64, i64

      // Too late / missing risk check
      tick.risk.check_notional { limit = 1.000000e+06 : f64 }

      tick.yield
    }
    return
  }
}