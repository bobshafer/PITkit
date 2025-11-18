# Loom Language Specification v2.0

### *A Meta-Language for Collaborative Program and Model Development*

---

## 1. Overview

**Loom** is a lightweight, composable language for structured collaboration between humans and AI systems.
It provides a single document format that unifies:

1. **Informal reasoning** — goals, intent, and narrative context.
2. **Formal specification** — typed definitions, constraints, and verifiable logic.

This dual structure allows natural-language conversation to coexist with formal semantics, enabling AI models to reason, generate, and verify in the same document.

### Version 2.0 Changes

**Major Change:** Loom now uses **native markdown structure** instead of headers within code blocks.

**Why:** 
- Better human readability with collapsible sections
- Improved LLM parsing (semantic tree vs. flat text)
- Working table of contents in any markdown viewer
- Clear separation: actual code in code blocks, metadata in tables/lists
- Faster information retrieval for both humans and AI

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

5. **Type-Theoretic Foundation**
   The formal layer is based on Intuitionistic Type Theory (ITT), as developed by Per Martin-Löf.
   This provides constructive semantics, dependent types, and proof-carrying code capabilities.

---

## 3. File Structure

### v2.0 Structure (Native Markdown)

Loom files now use **actual markdown headers** for organization:

```markdown
# Document Title

**Version:** v1.0  
**Authors:** Alice, Codex  
**Date:** 2025-11-05

## Overview

Natural language description...

## Section Name

### Subsection

Content with **proper markdown formatting**.

#### Detailed Item

**Property:** Value  
**Another Property:** Value

| Structured | Data |
|------------|------|
| In | Tables |

### Code Examples

```language
// Actual code in proper code blocks
def function():
    return result
```

## Formal Specification

```yaml
type: Time
  domain: Set
  
type: State
  signature: Time → Set
```
```

### Legacy Structure (v1.0)

The older format used headers within text blocks (now deprecated but still parseable):

```loom
meta:
  title: "Orbital Resonance Simulation"
informal:
  purpose: |
    Description here
formal:
  type Time : Set
```

---

## 4. Syntax and Semantics

### Structural Elements

| Element | v2.0 Format | Purpose |
|---------|-------------|---------|
| **Document Metadata** | Native markdown (bold key-value pairs) | Title, version, authors, date |
| **Sections** | `##` headers | Major organizational units |
| **Subsections** | `###` and `####` headers | Hierarchical content |
| **Metadata Fields** | Tables or `**Key:** Value` format | Structured data |
| **Code Examples** | Code blocks with language tag | Actual implementation code |
| **Formal Specs** | YAML or structured blocks | Machine-parseable definitions |

### Semantic Concepts

| Concept | Description |
|---------|-------------|
| **Native Headers** | Use markdown `#`, `##`, `###` for structure (not headers in code blocks) |
| **Metadata** | Document properties shown as `**Key:** Value` or tables |
| **Narrative** | Natural language sections explaining intent and rationale |
| **Formal Definitions** | YAML blocks or structured notation for type definitions |
| **Code Examples** | Real code in language-specific blocks (```c, ```python, etc.) |
| **ITT** | Intuitionistic Type Theory (Per Martin-Löf) — constructive foundation |
| **Σ[ x ∈ A ] B x** | Dependent pair — "a value x of A, and a value of type B(x)" |
| **Cross-references** | Use markdown links: `[Section Name](#section-name)` |
| **Imports** | Reference other Loom documents via links |

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
