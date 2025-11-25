#include "microtick_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <vector>

struct Order {
  std::int64_t id;
  std::int32_t symbol_id;
  std::int8_t  side; // +1 for buy, -1 for sell
  double       price;
  std::int64_t qty;  // positive for buy, negative for sell
  bool         open; // true if order is still open
};

struct Fill {
  std::int64_t order_id;
  double       price;
  std::int64_t qty;
};

struct EngineState {
  double             cash        = 0.0; // total cash available
  std::int64_t       position    = 0;   // +ve for long, -ve for short
  std::int64_t       nextOrderId = 1;   // incremental order ID
  std::vector<Order> openOrders;        // list of open orders
  std::vector<Fill>  fills;             // list of fills
};

static EngineState engineState; // global engine state

extern "C" void mt_order_send(std::int32_t symbol_id, std::int8_t side, double price,
                              std::int64_t qty) {
  std::int64_t orderId = engineState.nextOrderId++; // assign unique order ID
  Order        newOrder{orderId, symbol_id, side, price, qty, /*open=*/true};
  engineState.openOrders.push_back(newOrder);

  const char *symbolName = (symbol_id == 0)   ? "AAPL"
                           : (symbol_id == 1) ? "MSFT"
                                              : "UNKNOWN"; //  Simple symbol mapping for now
  const char *sideName   = (side > 0) ? "BUY" : "SELL";

  // Log order and fill
  std::printf("[ENGINE] Order Sent: ID=%lld, Symbol=%s, Side=%s, Price=%.2f, Qty=%lld\n", orderId,
              symbolName, sideName, price, qty);
}

extern "C" void mt_order_cancel(std::int32_t symbol_id, std::int8_t side) {
  const char *symbolName = (symbol_id == 0)   ? "AAPL"
                           : (symbol_id == 1) ? "MSFT"
                                              : "UNKNOWN"; //  Simple symbol mapping for now
  const char *sideName   = (side > 0) ? "BUY" : "SELL";

  // Find the most recent open order for the given symbol and side.
  for (auto it = engineState.openOrders.rbegin(); it != engineState.openOrders.rend(); ++it) {
    if (!it->open)
      continue; // Skip closed orders
    if (it->symbol_id != symbol_id || it->side != side)
      continue; // Not matching
    // Cancel the order
    it->open = false;
    std::printf("[ENGINE] Order Canceled: ID=%lld, Symbol=%s, Side=%s\n", it->id, symbolName,
                sideName);
    return;
  }
  std::printf("[ENGINE] No open order found to cancel for Symbol=%s, Side=%s\n", symbolName,
              sideName);
}

void simulateFills() {
  // Toy model:
  //  - Any order that is still open at the end of the event gets filled at its limit price.
  for (auto &order : engineState.openOrders) {
    if (!order.open)
      continue; // Skip closed orders
    double tradingPrice = order.price;

    // Apply fill
    engineState.cash -= tradingPrice * order.qty * order.side; // Update cash
    engineState.position += order.qty * order.side;            // Update position

    // Log fill
    std::printf("[ENGINE] Order Filled: ID=%lld, Price=%.2f, Qty=%lld\n", order.id, tradingPrice,
                order.qty);
    // Add to fills
    Fill newFill{order.id, tradingPrice, order.qty};
    engineState.fills.push_back(newFill);
    order.open = false; // Mark order as closed
  }
}

using OnBookFn = void (*)();

OnBookFn loadStrategyOnBook(const std::string &libPath, const std::string &symbolName) {
  void *handle = dlopen(libPath.c_str(), RTLD_NOW); // Load the shared library in RTLD_NOW mode
  if (!handle) {
    std::fprintf(stderr, "Failed to load strategy library: %s\n", dlerror());
    std::exit(EXIT_FAILURE);
  }

  dlerror(); // Clear any existing error
  void       *sym         = dlsym(handle, symbolName.c_str());
  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    std::fprintf(stderr, "Failed to load symbol '%s': %s\n", symbolName.c_str(), dlsym_error);
    std::exit(1);
  }

  return reinterpret_cast<OnBookFn>(sym);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "Usage: %s <strategy_shared_library> <strategy_name> [num_events]\n",
                 argv[0]);
    std::fprintf(stderr, "  Example: %s ./libstrategy_messy.dylib strategy_messy 10\n", argv[0]);
    return -1;
  }
  std::string strategyLibPath = argv[1];
  std::string strategyName    = argv[2]; // e.g., "strategy_messy"

  // Build the symbol name for on_book function
  std::string onBookSymbolName = strategyName + "_on_book";
  OnBookFn    onBook           = loadStrategyOnBook(strategyLibPath, onBookSymbolName);

  int numEvents = 1;
  if (argc >= 4) {
    numEvents = std::atoi(argv[3]);
    if (numEvents <= 0) {
      std::fprintf(stderr, "Invalid number of events: %s\n", argv[3]);
      return -1;
    }
  }
  // Simulate market events
  for (int i = 0; i < numEvents; ++i) {
    std::printf("[ENGINE] Processing market event %d\n", i + 1);
    onBook();        // Call the strategy's on_book function -> may send/cancel orders
    simulateFills(); // Simulate fills for any open orders
  }

  // Print final engine state
  std::printf("[ENGINE] Final Cash: %.2f\n", engineState.cash);
  std::printf("[ENGINE] Final Position: %lld\n", engineState.position);
  double marketPrice = 100.0; // Assume a market price for PnL calculation
  double pnl         = engineState.cash + engineState.position * marketPrice;
  std::printf("[ENGINE] Final PnL: %.2f\n", pnl);

  return 0;
}
