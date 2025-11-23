// testing-ir/strategy_trade_ok.mlir
module {
  func.func @strategy_trade_ok() {
    tick.on_trade {
      tick.risk.check_notional { limit = 2.000000e+06 : f64 }

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
