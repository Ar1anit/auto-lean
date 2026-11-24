```lean
/-- Lean model of `apply_discount` from `pricing.py`. Python's `//` by the
positive literal 100 is Lean's `/` on `Int`. -/
def applyDiscount (priceCents discountPct : Int) : Int :=
  if priceCents ≤ 0 then 0
  else
    let pct := max 0 (min 100 discountPct)
    priceCents - (priceCents * pct) / 100

/-- A discounted price is never negative. -/
theorem applyDiscount_nonneg (priceCents discountPct : Int) :
    0 ≤ applyDiscount priceCents discountPct := by sorry

/-- A discount never raises a non-negative price. -/
theorem applyDiscount_le_price (priceCents discountPct : Int) (h : 0 ≤ priceCents) :
    applyDiscount priceCents discountPct ≤ priceCents := by sorry
```
=== next reply ===
```lean
/-- Lean model of `apply_discount` from `pricing.py`. Python's `//` by the
positive literal 100 is Lean's `/` on `Int`. -/
def applyDiscount (priceCents discountPct : Int) : Int :=
  if priceCents ≤ 0 then 0
  else
    let pct := max 0 (min 100 discountPct)
    priceCents - (priceCents * pct) / 100

/-- A discounted price is never negative. -/
theorem applyDiscount_nonneg (priceCents discountPct : Int) :
    0 ≤ applyDiscount priceCents discountPct := by
  simp only [applyDiscount]
  split
  · exact Int.le_refl 0
  · have hp : 0 ≤ priceCents := by omega
    have h0 : 0 ≤ max 0 (min 100 discountPct) := by omega
    have h1 : max 0 (min 100 discountPct) ≤ 100 := by omega
    have h2 : 0 ≤ priceCents * max 0 (min 100 discountPct) := Int.mul_nonneg hp h0
    have h3 : priceCents * max 0 (min 100 discountPct) ≤ priceCents * 100 :=
      Int.mul_le_mul_of_nonneg_left h1 hp
    omega

/-- A discount never raises a non-negative price. -/
theorem applyDiscount_le_price (priceCents discountPct : Int) (h : 0 ≤ priceCents) :
    applyDiscount priceCents discountPct ≤ priceCents := by
  simp only [applyDiscount]
  split
  · omega
  · have h0 : 0 ≤ max 0 (min 100 discountPct) := by omega
    have h2 : 0 ≤ priceCents * max 0 (min 100 discountPct) := Int.mul_nonneg h h0
    omega
```
