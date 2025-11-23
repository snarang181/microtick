module {
  func.func @test() {
    %p = arith.constant 123.45 : f64
    %q = arith.constant 100 : i64

    tick.order.send %p, %q
      symbol("AAPL")
      side("Buy") : f64, i64

    return
  }
}