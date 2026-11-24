```lean
/-- Lean model of `midpoint` from `midpoint_bug.rs`. Rust's `u32` is modelled
by `UInt32`, whose `+` wraps around like Rust's release-mode arithmetic. -/
def midpoint (lo hi : UInt32) : UInt32 :=
  (lo + hi) / 2

/-- The midpoint of a range is not below its lower end. -/
theorem midpoint_ge_lo (lo hi : UInt32) (h : lo ≤ hi) : lo ≤ midpoint lo hi := by sorry

/-- The midpoint of a range is not above its upper end. -/
theorem midpoint_le_hi (lo hi : UInt32) (h : lo ≤ hi) : midpoint lo hi ≤ hi := by sorry
```
=== next reply ===
```lean
/-- Lean model of `midpoint` from `midpoint_bug.rs`. Rust's `u32` is modelled
by `UInt32`, whose `+` wraps around like Rust's release-mode arithmetic. -/
def midpoint (lo hi : UInt32) : UInt32 :=
  (lo + hi) / 2

/-- The midpoint of a range is not below its lower end. -/
theorem midpoint_ge_lo (lo hi : UInt32) (h : lo ≤ hi) : lo ≤ midpoint lo hi := by
  have hle : lo.toNat ≤ hi.toNat := UInt32.le_iff_toNat_le.mp h
  have hhi : hi.toNat < 2 ^ 32 := hi.toNat_lt
  have two : (2 : UInt32).toNat = 2 := rfl
  rw [UInt32.le_iff_toNat_le]
  simp only [midpoint, UInt32.toNat_div, UInt32.toNat_add, two]
  omega

/-- The midpoint of a range is not above its upper end. -/
theorem midpoint_le_hi (lo hi : UInt32) (h : lo ≤ hi) : midpoint lo hi ≤ hi := by
  have hle : lo.toNat ≤ hi.toNat := UInt32.le_iff_toNat_le.mp h
  have hhi : hi.toNat < 2 ^ 32 := hi.toNat_lt
  have two : (2 : UInt32).toNat = 2 := rfl
  rw [UInt32.le_iff_toNat_le]
  simp only [midpoint, UInt32.toNat_div, UInt32.toNat_add, two]
  omega
```
=== next reply ===
`omega` could not close the lower bound. Trying `grind`, which knows more about `UInt32`.

```lean
/-- Lean model of `midpoint` from `midpoint_bug.rs`. Rust's `u32` is modelled
by `UInt32`, whose `+` wraps around like Rust's release-mode arithmetic. -/
def midpoint (lo hi : UInt32) : UInt32 :=
  (lo + hi) / 2

/-- The midpoint of a range is not below its lower end. -/
theorem midpoint_ge_lo (lo hi : UInt32) (h : lo ≤ hi) : lo ≤ midpoint lo hi := by
  unfold midpoint
  grind

/-- The midpoint of a range is not above its upper end. -/
theorem midpoint_le_hi (lo hi : UInt32) (h : lo ≤ hi) : midpoint lo hi ≤ hi := by
  have hle : lo.toNat ≤ hi.toNat := UInt32.le_iff_toNat_le.mp h
  have hhi : hi.toNat < 2 ^ 32 := hi.toNat_lt
  have two : (2 : UInt32).toNat = 2 := rfl
  rw [UInt32.le_iff_toNat_le]
  simp only [midpoint, UInt32.toNat_div, UInt32.toNat_add, two]
  omega
```
=== next reply ===
Splitting on whether the sum fits in 32 bits isolates the case that resists.

```lean
/-- Lean model of `midpoint` from `midpoint_bug.rs`. Rust's `u32` is modelled
by `UInt32`, whose `+` wraps around like Rust's release-mode arithmetic. -/
def midpoint (lo hi : UInt32) : UInt32 :=
  (lo + hi) / 2

/-- The midpoint of a range is not below its lower end. -/
theorem midpoint_ge_lo (lo hi : UInt32) (h : lo ≤ hi) : lo ≤ midpoint lo hi := by
  have hle : lo.toNat ≤ hi.toNat := UInt32.le_iff_toNat_le.mp h
  have hhi : hi.toNat < 2 ^ 32 := hi.toNat_lt
  have two : (2 : UInt32).toNat = 2 := rfl
  rw [UInt32.le_iff_toNat_le]
  simp only [midpoint, UInt32.toNat_div, UInt32.toNat_add, two]
  by_cases overflow : lo.toNat + hi.toNat < 2 ^ 32
  · omega
  · omega

/-- The midpoint of a range is not above its upper end. -/
theorem midpoint_le_hi (lo hi : UInt32) (h : lo ≤ hi) : midpoint lo hi ≤ hi := by
  have hle : lo.toNat ≤ hi.toNat := UInt32.le_iff_toNat_le.mp h
  have hhi : hi.toNat < 2 ^ 32 := hi.toNat_lt
  have two : (2 : UInt32).toNat = 2 := rfl
  rw [UInt32.le_iff_toNat_le]
  simp only [midpoint, UInt32.toNat_div, UInt32.toNat_add, two]
  omega
```

NOTE: `midpoint_ge_lo` is false as stated. For lo = hi = 2147483648 the sum `lo + hi` wraps around to 0, so `midpoint lo hi = 0 < lo`.
