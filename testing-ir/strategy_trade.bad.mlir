// testing-ir/strategy_trade_bad.mlir
module {
  func.func @strategy_trade_bad() {
    tick.on_trade {
      %p = arith.constant 99.5 : f64
      %q = arith.constant 25 : i64

      tick.order.send %p, %q
        symbol("AAPL")
        side("Sell") : f64, i64

      tick.yield
    }
    return
  }
}