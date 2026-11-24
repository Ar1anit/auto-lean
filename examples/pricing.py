def apply_discount(price_cents: int, discount_pct: int) -> int:
    if price_cents <= 0:
        return 0
    pct = max(0, min(100, discount_pct))
    return price_cents - (price_cents * pct) // 100
