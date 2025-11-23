module {
  func.func @test_cancel_ok() {
    tick.order.cancel
      symbol("AAPL")
      client_order_id("ABC123")
      side("Buy")
    return
  }
}