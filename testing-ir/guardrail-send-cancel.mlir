// This should not be allowed, the send/cancel should be inside a tick.on_book block
module {
  func.func @weird() {
    %p = arith.constant 101.25 : f64
    %q = arith.constant 50 : i64

    // This “works”, but semantically it's weird
    tick.order.send %p, %q
      symbol("AAPL")
      side("Buy") : f64, i64

    return
  }
}