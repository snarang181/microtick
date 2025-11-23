// testing-ir/strategy_missing_inventory.mlir
module {
  func.func @strategy_missing_inventory() {
    tick.on_book {
      // Only notional check.
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
