// Should fail verification because side is neither "Buy" nor "Sell".
module {
  func.func @test_cancel_bad() {
    tick.order.cancel
      symbol("AAPL")
      client_order_id("ABC123")
      side("Hold")
    return
  }
}