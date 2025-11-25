module {
  func.func @strategy_e2e_demo() {
    tick.on_book {
      %p = arith.constant 101.25 : f64
      %q = arith.constant 50 : i64

      tick.risk.check_notional { limit = 1.000000e+06 : f64 }
      tick.risk.check_inventory { limit = 1000 : i64 }

      tick.order.send %p, %q
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
