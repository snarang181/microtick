#pragma once

#include <cstdint>

extern "C" {

/// Called by compiled MicroTick strategy:
/// symbol_id: small integer id for the symbol (For now, AAPL=0, MSFT=1, etc.)
/// side: +1 for buy, -1 for sell
/// price: limit price of the order
/// qty: signed quantity of the order (positive for buy, negative for sell)
void mt_order_send(std::int32_t symbol_id, std::int8_t side, double price, std::int64_t qty);

/// TODO: add more variants like mt_order_cancel etc.
} // extern "C"