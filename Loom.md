# Loom Language Specification v1.0

### *A Meta-Language for Collaborative Program and Model Development*

---

## 1. Overview

**Loom** is a lightweight, composable language for structured collaboration between humans and AI systems.
It provides a single document format that unifies:

1. **Informal reasoning** — goals, intent, and narrative context.
2. **Formal specification** — typed definitions, constraints, and verifiable logic.

This dual structure allows natural-language conversation to coexist with formal semantics, enabling AI models to reason, generate, and verify in the same document.

---

## 2. Core Principles

1. **Dual Layers**

   * *Informal Layer:* human-readable context, purpose, and rationale.
   * *Formal Layer:* machine-checkable definitions and proofs, written in a structured type system.

2. **Bidirectional Linking**
   The informal layer explains the *why*; the formal layer defines the *how*.
   Identifiers and references (e.g., `K.C`) can be used across both.

3. **Agent Interoperability**
   Loom is designed for systems like Codex, Gemini, ChatGPT, and similar AI models to co-develop shared specifications or programs.

4. **Execution Neutrality**
   Loom does not prescribe a runtime or logic engine.
   Formal blocks can be translated into languages such as Agda, Coq, Lean, or Rust for verification or execution.

---

## 3. File Structure

Each `.loom` or `.loom.md` file has three main sections:

```loom
meta:
informal:
formal:
```

### Example Skeleton

```loom
meta:
  title: "Orbital Resonance Simulation"
  version: "v1.0"
  authors: ["Alice", "Codex", "Gemini"]
  date: "2025-11-05"

informal:
  purpose: |
    Define a simulation interface for multi-body resonance systems.
    The formal section below specifies the stepwise update of state
    and parameters.

  notes: |
    This file serves as a shared specification among agents.  
    Each version is self-contained and mergeable via semantic diff.

formal:
  type Time : Set
  type State : Time → Set
  type Kernel : (t : Time) → State t → Set

  def step :
      (t : Time) → (s : State t) → (k : Kernel t s) →
      Σ[ s' ∈ State (t+1) ] Kernel (t+1) s'

  theorem CoherenceBound :
      ∀ t s k. (0 ≤ k.C ≤ 1) →
      (step t s k).snd.C ∈ [0,1]
```

---

## 4. Syntax and Semantics

| Concept            | Description                                                                     |
| ------------------ | ------------------------------------------------------------------------------- |
| **meta:**          | Document metadata (title, version, participants, date).                         |
| **informal:**      | Narrative and reasoning, written in natural language or Markdown.               |
| **formal:**        | Structured block with typed definitions and rules.                              |
| **Σ[ x ∈ A ] B x** | Dependent pair — represents “a value x of A, and a value of type B(x)”.         |
| **References**     | Identifiers may link between informal and formal sections for coherence.        |
| **Imports**        | `import:` directive allows linking to other Loom documents.                     |
| **Execution**      | Interpreters can read the formal block and emit valid ITT/Agda/Coq definitions. |

---

## 5. Document Semantics

### 5.1. Informal Section

* Defines the problem and design intent.
* Can contain Markdown, diagrams, or pseudocode.
* May reference formal constructs for readability.
  Example:

  > “When `k.C` approaches 1, the system is considered highly coherent.”

### 5.2. Formal Section

* Must define the symbolic logic or type-level structure referenced above.
* Can be verified or type-checked independently.
* Expected to be pure and declarative (no side effects).

### 5.3. Meta Section

* Provides context for toolchains, authorship, and versioning.
* Can include fields like:

  ```yaml
  tools: ["Agda", "Julia", "Rust"]
  source_repo: "https://github.com/example/project"
  ```

---

## 6. Example: Simple Process Specification

```loom
meta:
  title: "Coherence Oscillator"
  authors: ["Alice", "Codex"]
  version: "v0.9"

informal:
  overview: |
    Models a scalar variable `C` that increases with alignment (μ)
    and decays with novelty (ν).  The equation is logistic in form.

  narrative: |
    - μ: growth rate or reinforcement
    - ν: decay or dissipation
    - ξ: time-scale factor
    - The result is bounded: 0 ≤ C ≤ 1

formal:
  record Kernel : Set where
    field
      C  : ℚ
      μ  : ℚ
      ν  : ℚ
      ξ  : ℚ
      β  : ℚ

  def dC_dt (k : Kernel) : ℚ =
      k.ξ * (k.μ*k.C - k.ν*k.C - k.β*k.C*(1 - k.C))

  theorem StabilityBound :
      ∀ k. (0 ≤ k.C ≤ 1) → (0 ≤ k.C + dC_dt(k) ≤ 1)
```

---

## 7. Agent Workflow

| Agent Role                                  | Behavior                                             |
| ------------------------------------------- | ---------------------------------------------------- |
| **Human Author**                            | Writes `informal:` sections, reviews semantics.      |
| **Generator Model (e.g., ChatGPT, Gemini)** | Writes or edits `formal:` sections, based on intent. |
| **Compiler/Checker (e.g., Codex, Lean)**    | Parses and verifies the formal definitions.          |
| **Simulator/Runtime**                       | Uses the verified output for computation.            |

Loom thus provides a full lifecycle:

```
Intent → Specification → Verification → Implementation
```

---

## 8. Implementation Hints (for Codex or LLMs)

* Treat `informal:` as prompt context.
* Treat `formal:` as structured AST for verification or code synthesis.
* Maintain referential integrity between both.
* Support export to `.agda`, `.lean`, `.jl`, `.rs`, or `.py` as needed.

---

## 9. Future Features

* **Loom.ProofGraph**: visualize dependencies between definitions.
* **Loom.Delta**: semantic patching and version tracking (`patch Loom v1.1 → v1.2`).
* **Loom.AgentSync**: record provenance of which model authored which part.
* **Loom.Run**: runtime harness for real-time simulations.

---

## 10. Summary

Loom is a **collaborative specification language** that unifies natural reasoning and formal logic.
It supports AI-human co-development by ensuring that every informal statement can correspond to a verifiable formal structure.
Its minimal syntax and type-driven semantics make it compatible with both large language models and formal proof engines.
