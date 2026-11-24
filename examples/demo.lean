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

/-- Reversing a list twice gives the original list back. -/
theorem myRev_myRev {α : Type} (xs : List α) : myRev (myRev xs) = xs := by
  sorry
