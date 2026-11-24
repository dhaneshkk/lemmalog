/-
  Formal guarantees for tiered coreference (core Lean 4, no Mathlib).
  Check with `lean Coref.lean`.

  Rust ↔ Lean correspondence
  * `coref::resolve_tiered` (tier 1 then tier 2)  ↔ `tiered`,
      theorems `sieve_links_are_kept`, `tier_two_only_on_abstentions`,
      `no_scorer_is_the_sieve`
  * `coref::gated` (score ≥ min_score, margin ≥ min_margin) ↔ `Accepts`,
      theorems `gate_meets_min_score`, `gate_picks_at_most_one`
  * `coref_eval::evaluate` pair counts          ↔ `pairCount`, theorem `pairs_partition`

  Scores are modelled in per-mille (`Nat`), as `coref` facts store them.
  The Rust margin is best minus the runner-up after sorting; `Accepts`
  states it for every other candidate, which is the same condition.
-/

/-! ### 1. tier 2 sees only what the sieves abstain on -/

inductive Outcome where
  | resolved (antecedent : Nat) (conf : Nat)
  | ambiguous
  | noAntecedent
  deriving DecidableEq

def Outcome.abstains : Outcome → Bool
  | .resolved _ _ => false
  | _ => true

/-- Sieve first; a pronoun the sieves abstain on takes tier 2's gated answer
    if there is one, and keeps the sieve's outcome otherwise. -/
def tiered {α} (sieve : α → Outcome) (tier2 : α → Option (Nat × Nat)) (p : α) : Outcome :=
  match sieve p with
  | .resolved a c => .resolved a c
  | o => match tier2 p with
    | some (a, c) => .resolved a c
    | none => o

theorem sieve_links_are_kept {α} (s : α → Outcome) (t : α → Option (Nat × Nat)) (p : α)
    (a c : Nat) (h : s p = .resolved a c) : tiered s t p = .resolved a c := by
  simp [tiered, h]

theorem tier_two_only_on_abstentions {α} (s : α → Outcome) (t : α → Option (Nat × Nat)) (p : α)
    (h : tiered s t p ≠ s p) : (s p).abstains = true := by
  cases hs : s p with
  | resolved a c =>
    have key := sieve_links_are_kept s t p a c hs
    simp_all
  | ambiguous => rfl
  | noAntecedent => rfl

theorem no_scorer_is_the_sieve {α} (s : α → Outcome) (p : α) :
    tiered s (fun _ => none) p = s p := by
  unfold tiered
  cases hs : s p <;> rfl

/-! ### 2. the gate -/

structure Gate where
  minScore : Nat
  minMargin : Nat

/-- Candidate `i` passes: its score reaches `minScore` and beats every other
    candidate by at least `minMargin`. -/
def Accepts (xs : List Nat) (g : Gate) (i : Nat) : Prop :=
  ∃ h : i < xs.length, g.minScore ≤ xs[i] ∧
    ∀ (j : Nat) (hj : j < xs.length), j ≠ i → xs[j] + g.minMargin ≤ xs[i]

theorem gate_meets_min_score (xs : List Nat) (g : Gate) (i : Nat) (h : Accepts xs g i) :
    ∃ hi : i < xs.length, g.minScore ≤ xs[i] := by
  obtain ⟨hi, hs, _⟩ := h
  exact ⟨hi, hs⟩

/-- With a positive margin, two candidates can never both pass: tier 2
    links a pronoun to one antecedent or to none. -/
theorem gate_picks_at_most_one (xs : List Nat) (g : Gate) (hm : 0 < g.minMargin)
    (i k : Nat) (hi : Accepts xs g i) (hk : Accepts xs g k) : i = k := by
  obtain ⟨hi1, _, hi3⟩ := hi
  obtain ⟨hk1, _, hk3⟩ := hk
  apply Classical.byContradiction
  intro hne
  have a := hi3 k hk1 (fun e => hne e.symm)
  have b := hk3 i hi1 hne
  omega

/-! ### 3. mention-pair counts -/

/-- One mention pair: (same gold cluster, same predicted cluster). -/
def pairCount : List (Bool × Bool) → Bool → Bool → Nat
  | [], _, _ => 0
  | p :: ps, gold, pred => (if p.1 = gold ∧ p.2 = pred then 1 else 0) + pairCount ps gold pred

/-- TP + FN + FP + TN is every pair once, so MCC is computed over all pairs. -/
theorem pairs_partition (ps : List (Bool × Bool)) :
    pairCount ps true true + pairCount ps true false + pairCount ps false true
      + pairCount ps false false = ps.length := by
  induction ps with
  | nil => rfl
  | cons p ps ih =>
    obtain ⟨g, q⟩ := p
    cases g <;> cases q <;> simp [pairCount] <;> omega
