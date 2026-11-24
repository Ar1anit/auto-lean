The key fact is how `myRev` distributes over `myAppend`. I state it as a helper and finish the main proof from there.

```lean
/-!
# auto-lean demo (Mode 1)

Two list helpers written from scratch, and a theorem about them whose proof is
left open for `auto-lean` to find:

    auto-lean solve examples/demo.lean

Nothing here goes beyond Lean 4's core library, so the file compiles as it is
with a plain `lean examples/demo.lean`.
-/

/-- Appends two lists by recursion on the first one. -/
def myAppend {α : Type} : List α → List α → List α
  | [], ys => ys
  | x :: xs, ys => x :: myAppend xs ys

/-- Reverses a list the naive way: reverse the tail, then put the head last. -/
def myRev {α : Type} : List α → List α
  | [] => []
  | x :: xs => myAppend (myRev xs) [x]

theorem myRev_myAppend {α : Type} (xs ys : List α) :
    myRev (myAppend xs ys) = myAppend (myRev ys) (myRev xs) := by
  sorry

/-- Reversing a list twice gives the original list back. -/
theorem myRev_myRev {α : Type} (xs : List α) : myRev (myRev xs) = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [myRev, myRev_myAppend, myAppend, ih]
```
=== next reply ===
```lean
/-!
# auto-lean demo (Mode 1)

Two list helpers written from scratch, and a theorem about them whose proof is
left open for `auto-lean` to find:

    auto-lean solve examples/demo.lean

Nothing here goes beyond Lean 4's core library, so the file compiles as it is
with a plain `lean examples/demo.lean`.
-/

/-- Appends two lists by recursion on the first one. -/
def myAppend {α : Type} : List α → List α → List α
  | [], ys => ys
  | x :: xs, ys => x :: myAppend xs ys

/-- Reverses a list the naive way: reverse the tail, then put the head last. -/
def myRev {α : Type} : List α → List α
  | [] => []
  | x :: xs => myAppend (myRev xs) [x]

theorem myAppend_nil {α : Type} (xs : List α) : myAppend xs [] = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [myAppend, ih]

theorem myRev_myAppend {α : Type} (xs ys : List α) :
    myRev (myAppend xs ys) = myAppend (myRev ys) (myRev xs) := by
  induction xs with
  | nil => simp [myAppend, myRev, myAppend_nil]
  | cons x xs ih => simp [myAppend, myRev, ih]

/-- Reversing a list twice gives the original list back. -/
theorem myRev_myRev {α : Type} (xs : List α) : myRev (myRev xs) = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [myRev, myRev_myAppend, myAppend, ih]
```
=== next reply ===
The remaining goal is associativity of `myAppend`, so that becomes a third helper.

```lean
/-!
# auto-lean demo (Mode 1)

Two list helpers written from scratch, and a theorem about them whose proof is
left open for `auto-lean` to find:

    auto-lean solve examples/demo.lean

Nothing here goes beyond Lean 4's core library, so the file compiles as it is
with a plain `lean examples/demo.lean`.
-/

/-- Appends two lists by recursion on the first one. -/
def myAppend {α : Type} : List α → List α → List α
  | [], ys => ys
  | x :: xs, ys => x :: myAppend xs ys

/-- Reverses a list the naive way: reverse the tail, then put the head last. -/
def myRev {α : Type} : List α → List α
  | [] => []
  | x :: xs => myAppend (myRev xs) [x]

theorem myAppend_nil {α : Type} (xs : List α) : myAppend xs [] = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [myAppend, ih]

theorem myAppend_assoc {α : Type} (xs ys zs : List α) :
    myAppend (myAppend xs ys) zs = myAppend xs (myAppend ys zs) := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [myAppend, ih]

theorem myRev_myAppend {α : Type} (xs ys : List α) :
    myRev (myAppend xs ys) = myAppend (myRev ys) (myRev xs) := by
  induction xs with
  | nil => simp [myAppend, myRev, myAppend_nil]
  | cons x xs ih => simp [myAppend, myRev, ih, myAppend_assoc]

/-- Reversing a list twice gives the original list back. -/
theorem myRev_myRev {α : Type} (xs : List α) : myRev (myRev xs) = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [myRev, myRev_myAppend, myAppend, ih]
```
