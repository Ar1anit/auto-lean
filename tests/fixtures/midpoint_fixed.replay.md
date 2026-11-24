```lean
/-- Lean model of `midpoint` from `midpoint_fixed.rs`. Rust's `u32` is modelled
by `UInt32`, whose `+` and `-` wrap around like Rust's release-mode arithmetic. -/
def midpoint (lo hi : UInt32) : UInt32 :=
  lo + (hi - lo) / 2

/-- The midpoint of a range is not below its lower end. -/
theorem midpoint_ge_lo (lo hi : UInt32) (h : lo ≤ hi) : lo ≤ midpoint lo hi := by sorry

/-- The midpoint of a range is not above its upper end. -/
theorem midpoint_le_hi (lo hi : UInt32) (h : lo ≤ hi) : midpoint lo hi ≤ hi := by sorry
```
=== next reply ===
```lean
/-- Lean model of `midpoint` from `midpoint_fixed.rs`. Rust's `u32` is modelled
by `UInt32`, whose `+` and `-` wrap around like Rust's release-mode arithmetic. -/
def midpoint (lo hi : UInt32) : UInt32 :=
  lo + (hi - lo) / 2

/-- The midpoint of a range is not below its lower end. -/
theorem midpoint_ge_lo (lo hi : UInt32) (h : lo ≤ hi) : lo ≤ midpoint lo hi := by
  have hle : lo.toNat ≤ hi.toNat := UInt32.le_iff_toNat_le.mp h
  have hhi : hi.toNat < 2 ^ 32 := hi.toNat_lt
  have two : (2 : UInt32).toNat = 2 := rfl
  rw [UInt32.le_iff_toNat_le]
  simp only [midpoint, UInt32.toNat_add, UInt32.toNat_div, UInt32.toNat_sub_of_le _ _ h, two]
  omega

/-- The midpoint of a range is not above its upper end. -/
theorem midpoint_le_hi (lo hi : UInt32) (h : lo ≤ hi) : midpoint lo hi ≤ hi := by
  have hle : lo.toNat ≤ hi.toNat := UInt32.le_iff_toNat_le.mp h
  have hhi : hi.toNat < 2 ^ 32 := hi.toNat_lt
  have two : (2 : UInt32).toNat = 2 := rfl
  rw [UInt32.le_iff_toNat_le]
  simp only [midpoint, UInt32.toNat_add, UInt32.toNat_div, UInt32.toNat_sub_of_le _ _ h, two]
  omega
```
