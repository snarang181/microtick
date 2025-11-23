// testing-ir/strategy_dual_risk_ok.mlir
module {
  func.func @strategy_dual_risk_ok() {
    tick.on_book {
      // Notional + inventory checks at top.
      tick.risk.check_notional  { limit = 1.000000e+06 : f64 }
      tick.risk.check_inventory { limit = 1000 : i64 }

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
