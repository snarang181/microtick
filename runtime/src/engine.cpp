#include "microtick_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <vector>

struct Order {
  std::int64_t id;
  double       price;
  std::int64_t qty; // positive for buy, negative for sell
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

extern "C" void mt_order_send(double price, std::int64_t qty) {
  std::int64_t orderId = engineState.nextOrderId++; // assign unique order ID
  Order        newOrder{orderId, price, qty};
  engineState.openOrders.push_back(newOrder);

  // Simple immediate fill logic model.
  Fill newFill{orderId, price, qty};
  engineState.fills.push_back(newFill);

  // Update cash and position
  // buy qty @ price -> pay price * qty cash, increase position by qty
  // sell qty @ price -> receive price * qty cash, decrease position by qty
  engineState.position += qty;
  engineState.cash -= price * static_cast<double>(qty);

  // Log order and fill
  std::printf("[ENGINE] Order Sent: ID=%lld, Price=%.2f, Qty=%lld\n", orderId, price, qty);
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
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <strategy_shared_library>\n", argv[0]);
    return 1;
  }

  std::string strategyLibPath = argv[1];
  OnBookFn    onBook          = loadStrategyOnBook(strategyLibPath, "strategy_messy_on_book");

  // Simple event loop: pretend we have 10 book events
  const int numEvents = 10;
  for (int i = 0; i < numEvents; ++i) {
    std::printf("[ENGINE] Book Event %d\n", i + 1);
    onBook(); // Call the strategy's on_book function
  }

  // Print final engine state
  std::printf("[ENGINE] Final Cash: %.2f\n", engineState.cash);
  std::printf("[ENGINE] Final Position: %lld\n", engineState.position);
  double marketPrice = 100.0; // Assume a market price for PnL calculation
  double pnl         = engineState.cash + engineState.position * marketPrice;
  std::printf("[ENGINE] Final PnL: %.2f\n", pnl);

  return 0;
}
