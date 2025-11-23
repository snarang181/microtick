// testing-ir/strategy_messy.mlir
module {
  func.func @strategy_messy() {
    tick.on_book {
      %cst = arith.constant 1.012500e+02 : f64
      tick.risk.check_inventory { limit = 1000 : i64 }
      %c50_i64 = arith.constant 50 : i64
      tick.risk.check_notional { limit = 1.000000e+06 : f64 }

      tick.order.send %cst, %c50_i64
        symbol("AAPL")
        side("Buy") : f64, i64

      tick.yield
    }
    return
  }
}
