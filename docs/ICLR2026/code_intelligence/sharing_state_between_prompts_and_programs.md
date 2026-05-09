---
title: >-
  [Paper Note] Sharing State Between Prompts and Programs
description: >-
  [ICLR 2026][shared program state] This paper proposes the *shared program state* abstraction, enabling prompts to directly read and write program variables, manipulate heap objects, and control program flow. The abstraction is realized in the Nightjar system (Python + prompt hybrid programming), achieving a 39.6% reduction in code size while maintaining or improving accuracy (+4–19%).
tags:
  - ICLR 2026
  - shared program state
  - natural language programming
  - prompt-program interoperability
  - Nightjar
  - programming abstractions
date: 2026-05-08
content_hash: 0993f4381e41dd61
---

# Sharing State Between Prompts and Programs

**Conference**: ICLR 2026
**arXiv**: [2512.14805](https://arxiv.org/abs/2512.14805)
**Code**: [https://github.com/psg-mit/nightjarpy](https://github.com/psg-mit/nightjarpy)
**Area**: Programming Languages / LLM Programming
**Keywords**: shared program state, natural language programming, prompt-program interoperability, Nightjar, programming abstractions

## TL;DR
This paper proposes the *shared program state* abstraction, enabling prompts to directly read and write program variables, manipulate heap objects, and control program flow. The abstraction is realized in the Nightjar system (Python + prompt hybrid programming), achieving a 39.6% reduction in code size while maintaining or improving accuracy (+4–19%).

## Background & Motivation

**Background**: LLMs have given rise to natural language programming, in which prompts instruct models to perform tasks. Existing systems (LangChain, DSPy, SGLang, etc.) support prompt-program interoperability but adopt an *isolated program state* design: prompts execute in a separate environment, requiring developers to manually serialize and deserialize data to transfer program state.

**Limitations of Prior Work**: The isolated state design leads to substantial boilerplate code—developers must define schema classes, serialization functions, and deserialization functions to pass data between prompts and programs, increasing development complexity and introducing potential errors.

**Key Challenge**: Prompts inherently need access to program context to make sound decisions (reading variable values, modifying object state, controlling branches/loops), yet existing systems strictly isolate prompt execution from program state, forcing developers to write bridging code by hand.

**Goal**: (a) Define a programming abstraction for shared program state; (b) formalize a schema for the natural function interface; (c) implement the Nightjar system to validate feasibility and benefits.

**Key Insight**: Drawing on the *effects & handlers* paradigm from programming language theory, the paper formalizes prompt operations on program state as effects, with handlers implemented by the host language.

**Core Idea**: Allow prompts to directly access the program variable scope, heap, and control flow—as functions do—eliminating the development overhead of manual state transfer.

## Method

### Overall Architecture
Nightjar treats prompts as first-class code within Python programs. Developers annotate functions with `@nightjar.fn` and write prompts as triple-quoted strings inside the function body. Within a prompt, `<variable>` reads a local variable, `<:variable>` writes to a variable, Python objects can be manipulated directly, and control-flow operations such as `break`/`continue` are supported.

### Key Designs

1. **Shared Scopes**:

    - Function: Prompts can read from and write to Python variables.
    - Mechanism: `<graph>` in a prompt references the `graph` variable in the current scope; `<:response>` in the LLM output binds the value to the `response` variable. The system captures a scope snapshot before prompt execution and updates variables afterward.
    - Design Motivation: Eliminates schema definitions and serialization code required for manually passing data in and out, making prompts a true part of the program.

2. **Shared Heap**:

    - Function: Prompts can manipulate Python objects (modifying attributes, calling methods, in-place updates of mutable objects).
    - Mechanism: The LLM does not operate on the heap directly; instead, it interacts indirectly via reference/dereference effects. The system maintains an object reference table and translates the LLM's operation instructions into actual operations on Python objects.
    - Design Motivation: Enables prompts to modify complex program data structures (e.g., graphs, lists) rather than merely returning serialized new versions.

3. **Shared Control State**:

    - Function: Prompts can trigger control-flow operations such as `break` and `continue`.
    - Mechanism: Prompts reference control-flow constructs in the program via labels. When the LLM emits a break effect, Nightjar's handler executes the corresponding `break` in the host Python program.
    - Design Motivation: Allows prompts to determine when to terminate a loop or skip an iteration based on conversational semantics, avoiding additional conditional code.

4. **Natural Function Interface Schema**:

    - Function: Formalizes the interaction interface between prompts and programs.
    - Mechanism: Based on the effects & handlers paradigm. Effects define the operation types a prompt may perform (read variable, write variable, reference object, break, etc.); handlers define the implementation of these operations in the host language.
    - Design Motivation: Provides a language-agnostic specification so that shared program state can be implemented on any programming language.

### Loss & Training
Nightjar involves no model training; its contribution is at the programming-systems level. The core technical challenge lies in mapping the LLM's natural language output to correct program operations.

## Key Experimental Results

### Main Results (Nightjar vs. Manual Implementation)

| Task | Nightjar Accuracy | Manual Accuracy | Code Reduction | Runtime Overhead |
|------|------------------|----------------|---------------|-----------------|
| Graph manipulation | +4–19% | Baseline | ~40% | 0.4–4.3× |
| Data processing | On par or higher | Baseline | ~40% | Moderate |
| Control-flow tasks | Higher | Baseline | Significant | Slightly higher |

### Ablation Study

| Configuration | Performance | Notes |
|--------------|-------------|-------|
| Full shared state | Best | Scope + heap + control flow |
| Shared scope only | Usable but limited | Cannot modify mutable objects |
| Isolated state (baseline) | Requires extensive boilerplate | Conventional approach |

### Key Findings
- Average code size reduction of **39.6%**, primarily from eliminating schema definitions and serialization/deserialization code.
- Accuracy improvement of **+4–19%**: shared state avoids information loss and formatting errors introduced by manual serialization.
- Runtime overhead of 0.4–4.3×, mainly attributable to reference resolution and the additional communication required for effect handling.

## Highlights & Insights
- **The contribution at the programming-abstraction level** is more significant than the concrete system: shared program state is a new programming paradigm not limited to Python. The natural function interface schema is language-agnostic and can be implemented in any programming system.
- **The application of effects & handlers to LLM programming** is particularly elegant: abstracting prompt operations on program state as effects, with handlers implemented by the host language, represents a refined synthesis of PL theory and practical LLM systems.
- The work reveals a broader trend: computation is increasingly being planned and executed dynamically and adaptively, and LLMs make "runtime programming" a reality.

## Limitations & Future Work
- Runtime overhead (0.4–4.3×) may be unacceptable in latency-sensitive settings.
- LLM operations on complex program objects are susceptible to errors (hallucinated writes with incorrect values).
- Only a Python implementation is currently available; portability to other languages remains to be validated.
- Security concerns: direct prompt manipulation of program state may introduce unintended side effects.

## Related Work & Insights
- **vs. LangChain/DSPy**: These systems adopt isolated state, requiring manual schema definitions and serialization. Nightjar eliminates this burden.
- **vs. AskIt/ANPL**: These systems use LLM-generated functions to replace prompts, partially sharing state but without support for variable writes or control flow.
- **vs. tool use**: Tool use requires developers to define custom functions; Nightjar's shared state requires no additional function definitions from the developer.

## Rating
- Novelty: ⭐⭐⭐⭐ — Shared program state is a novel programming abstraction; the application of effects & handlers is innovative.
- Experimental Thoroughness: ⭐⭐⭐ — Limited number of tasks; large-scale application validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ — PL formal specification and practical system are well integrated.
- Value: ⭐⭐⭐⭐ — Provides meaningful guidance for the design of LLM programming systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning From Design Procedure To Generate CAD Programs for Data Augmentation](../../NeurIPS2025/code_intelligence/learning_from_design_procedure_to_generate_cad_programs_for_data_augmentation.md)
- [\[ICLR 2026\] DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models](dro-instructzero_distributionally_robust_prompt_optimization_for_instruction_fol.md)
- [\[ICLR 2026\] Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](paper2code_automating_code_generation_from_scientific_papers_in_machine_learning.md)
- [\[ICLR 2026\] IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation](imse_intrinsic_mixture_of_spectral_experts_fine-tuning_for_test-time_adaptation.md)
- [\[ICLR 2026\] ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](reasoningbank_scaling_agent_self-evolving_with_reasoning_memory.md)

</div>

<!-- RELATED:END -->
