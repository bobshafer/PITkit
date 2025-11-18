# [Document Title]

**Version:** v0.1  
**Authors:** [Your Name], [Collaborator or AI Model]  
**Date:** YYYY-MM-DD  
**Tags:** spec, prototype, collaboration, [domain-tags]  
**Tools:** [ChatGPT | Codex | Gemini], [Agda | Lean | Coq]  
**Source Repository:** [optional URL]

## Overview

Brief summary of what this Loom document defines or implements.

For example: "Implements a dynamic model for multi-agent coordination" or "Specification of the XYZ application for ABC domain."

**Key Concepts:**
- [Main concept 1]
- [Main concept 2]
- [Main concept 3]

---

## Table of Contents

- [Purpose](#purpose)
- [Business Context](#business-context)
- [Architecture](#architecture)
- [Components](#components)
- [Formal Specification](#formal-specification)
- [Examples](#examples)
- [Notes](#notes)

---

## Purpose

Describe the problem this document addresses and what goals it achieves.

**What:**
- What system, model, or process is being defined?
- What boundaries does this specification cover?

**Why:**
- What problem does this solve?
- What value does it provide?

**Who:**
- Who are the stakeholders?
- What agents or systems interact with this?

---

## Business Context

Background and motivation for this specification.

### Domain

Describe the business or technical domain this operates within.

### Stakeholders

| Role | Interest | Responsibilities |
|------|----------|------------------|
| [Role 1] | [What they care about] | [What they do] |
| [Role 2] | [What they care about] | [What they do] |

### Constraints

**Technical:**
- [Constraint 1]
- [Constraint 2]

**Business:**
- [Constraint 1]
- [Constraint 2]

---

## Architecture

### High-Level Design

Describe the overall structure and key components.

```
[ASCII diagram or description]

┌─────────────┐
│  Component  │
│      A      │──────▶ [Operation]
└─────────────┘
       │
       ▼
┌─────────────┐
│  Component  │
│      B      │
└─────────────┘
```

### Key Patterns

| Pattern | Usage | Rationale |
|---------|-------|-----------|
| [Pattern Name] | [Where used] | [Why chosen] |

### Data Flow

1. [Step 1 description]
2. [Step 2 description]
3. [Step 3 description]

---

## Components

### Component Name

**Responsibility:** What this component does  
**Type:** [Module | Service | Function | Aggregate | Entity]  
**Interfaces:** How it connects to other components

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| [prop1] | `Type1` | Purpose of this property |
| [prop2] | `Type2` | Purpose of this property |

#### Operations

##### Operation Name

**Signature:** `(param1: Type1) → (param2: Type2) → Result`  
**Preconditions:** What must be true before calling  
**Postconditions:** What will be true after completion  
**Side Effects:** Any state changes or external interactions

```language
// Example implementation or pseudocode
function operationName(param1, param2) {
    // implementation
    return result;
}
```

---

## Formal Specification

### Type Definitions

```yaml
type: TypeName
  domain: Set | Type
  description: What this type represents
  constraints:
    - [constraint 1]
    - [constraint 2]

type: AnotherType
  signature: TypeA → TypeB → TypeC
  description: Functional type description
```

### Relations

```yaml
relation: RelationName
  signature: TypeA → TypeB → Prop
  description: What relationship this captures
  properties:
    - reflexive | symmetric | transitive | [other]
```

### Operations

```yaml
function: functionName
  signature: |
    (param1 : Type1) → (param2 : Type2) →
    Result
  description: What this function computes
  properties:
    - pure | idempotent | associative | [other]
  complexity: O(n) | O(log n) | [other]
```

### Invariants

```yaml
invariant: InvariantName
  statement: |
    ∀ x y. Property(x, y) → Condition holds
  description: Why this must always be true
  enforcement: [hook_name | validation_function]
```

### Theorems

```yaml
theorem: TheoremName
  statement: |
    ∀ params. Precondition →
    Postcondition
  proof_sketch: |
    Brief explanation of why this holds.
    Can reference lemmas or prior theorems.
```

---

## Examples

### Example 1: [Scenario Name]

**Given:**
```yaml
input:
  field1: value1
  field2: value2
```

**Preconditions:**
- [Condition 1]
- [Condition 2]

**Execution:**
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Postconditions:**
```yaml
output:
  field1: new_value1
  field2: new_value2
```

**Result:**
- [Outcome 1]
- [Outcome 2]

### Example 2: [Edge Case Name]

**Scenario:** What happens when [special condition]

**Execution:**
```language
// Code example showing the edge case
```

**Expected Behavior:**
- [Behavior 1]
- [Behavior 2]

---

## Typical Usage Flows

### Flow 1: [Common Operation]

1. User/Agent initiates [action]
2. System performs [operation]
3. Validation checks [criteria]
4. Result is [outcome]

### Flow 2: [Another Common Operation]

1. [Step 1]
2. [Step 2]
3. [Step 3]

---

## Implementation Notes

### Technology Stack

| Layer | Technology | Version | Notes |
|-------|------------|---------|-------|
| [Layer] | [Tech] | [v1.0] | [Why chosen] |

### Configuration

**Required Settings:**
```yaml
setting1: value
setting2: value
```

**Optional Settings:**
```yaml
setting3: value  # Purpose of this setting
setting4: value  # Purpose of this setting
```

### Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| [Library] | [v1.0] | [What it provides] |

---

## Notes

### Design Decisions

**Decision 1: [Topic]**
- **Context:** [What led to this decision]
- **Options Considered:** [Alternatives]
- **Decision:** [What was chosen]
- **Rationale:** [Why]
- **Consequences:** [Trade-offs]

**Decision 2: [Topic]**
- [Same structure as above]

### Future Extensions

- [Planned feature 1]
- [Planned feature 2]
- [Possible enhancement 3]

### Known Issues

- [Issue 1 description and workaround]
- [Issue 2 description and status]

### Related Documents

- [Document 1: Link and description]
- [Document 2: Link and description]

### Glossary

| Term | Definition |
|------|------------|
| [Term 1] | [Definition] |
| [Term 2] | [Definition] |

---

## Metadata Reference

**File Structure:**
```
/path/to/document.loom.md
```

**Change History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| v0.1 | YYYY-MM-DD | [Name] | Initial draft |

**Review Status:** [ ] Draft | [ ] Under Review | [ ] Approved

**Reviewers:**
- [Name 1] - [Date] - [Comments/Status]
- [Name 2] - [Date] - [Comments/Status]
