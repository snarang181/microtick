#pragma once

#include <cstdint>

extern "C" {

/// Called by compiled MicroTick strategy:
/// price: limit price of the order
/// qty: signed quantity of the order (positive for buy, negative for sell)
void mt_order_send(double price, std::int64_t qty);

/// TODO: add more variants like mt_order_cancel etc.
} // extern "C"