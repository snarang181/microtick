module {
  func.func @strategy_e2e_demo() {
    tick.on_book {
      // A “passive” buy we actually want to trade.
      %p_passive = arith.constant 99.50 : f64
      %q_passive = arith.constant 10 : i64

      // A “probing” order we will cancel.
      %p_cancel = arith.constant 101.25 : f64
      %q_cancel = arith.constant 50 : i64

      // Static risk checks – required before sends.
      tick.risk.check_notional { limit = 1.000000e+06 : f64 }
      tick.risk.check_inventory { limit = 1000 : i64 }

      // 1) Resting order that will be filled by the engine.
      tick.order.send %p_passive, %q_passive
        symbol("AAPL")
        side("Buy") : f64, i64

      // 2) Probing order that we immediately cancel.
      tick.order.send %p_cancel, %q_cancel
        symbol("AAPL")
        side("Buy") : f64, i64

      tick.order.cancel
        symbol("AAPL")
        client_order_id("ABC123")
        side("Buy")

      tick.yield
    }
    return
  }
}
