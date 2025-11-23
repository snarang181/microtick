// testing-ir/strategy_timer_bad.mlir
module {
  func.func @strategy_bad() {
    tick.on_timer attributes { period_ns = 1000000000 : i64 } {
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
