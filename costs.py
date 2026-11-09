"""Estimated API cost per run, from token counts.

Prices in USD per 1M tokens, Gemini API paid tier, standard (non-batch).
Source: https://aistudio.google.com/docs/pricing (checked 2026-11-09).
Update this table when prices change. Output price includes thinking tokens.
"""

from __future__ import annotations

PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    # model:                  (input, output)
    "mock":                   (0.0, 0.0),    # free random stand-in, no API calls
    "gemini-3.8-flash":       (0.75, 3.75),  # rises to (1.50, 7.50) on 2027-01-01
    "gemini-3.6-flash":       (0.75, 3.75),  # rises to (1.50, 7.50) on 2027-01-01
    "gemini-3.5-flash-lite":  (0.30, 2.50),
    "gemini-3.1-flash-lite":  (0.25, 1.50),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """USD cost estimate, or None if the model's price is unknown (or it's a mock run)."""
    prices = PRICES_PER_MILLION.get(model)
    if prices is None:
        return None
    price_in, price_out = prices
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def format_cost(cost: float | None) -> str:
    return "n/a (unknown model price)" if cost is None else f"${cost:.2f}"
