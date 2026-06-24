---
title: >-
  [Paper Note] Controllable and Reliable Knowledge-Intensive Task-Oriented Conversational Agents with Declarative Genie Worksheets
description: >-
  [ACL 2025][Information Retrieval & RAG][Task-Oriented Dialogue] Genie proposes a programmable framework for knowledge-intensive task-oriented dialogues. It defines LLM agent policies through declarative Worksheet specifications, limiting the LLM to two roles—semantic parsing and response generation—while an algorithmic runtime system enforces the policies. This achieves a realistic task completion rate improvement from $21.8\%$ to $82.8\%$.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Task-Oriented Dialogue"
  - "Declarative Specification"
  - "Controllable Agent"
  - "Knowledge-Intensive Dialogue"
  - "Dialogue State Management"
date: 2026-05-08
content_hash: 4ab795da1fb27080
---

# Controllable and Reliable Knowledge-Intensive Task-Oriented Conversational Agents with Declarative Genie Worksheets

**Conference**: ACL 2025  
**arXiv**: [2407.05674](https://arxiv.org/abs/2407.05674)  
**Code**: [https://github.com/stanford-oval/genie-worksheets](https://github.com/stanford-oval/genie-worksheets)  
**Area**: Information Retrieval  
**Keywords**: Task-Oriented Dialogue, Declarative Specification, Controllable Agent, Knowledge-Intensive Dialogue, Dialogue State Management  

## TL;DR
Genie proposes a programmable framework for knowledge-intensive task-oriented dialogues. It defines LLM agent policies through declarative Worksheet specifications, limiting the LLM to two roles—semantic parsing and response generation—while an algorithmic runtime system enforces the policies. This achieves a realistic task completion rate improvement from $21.8\%$ to $82.8\%$.

## Background & Motivation
**Background**: LLMs can conduct human-like conversations, but face issues like hallucination, failure to follow conditional logic, and difficulties in knowledge integration during practical deployment.

**Limitations of Prior Work**:
   - Dialogue trees require developers to manually exhaust all dialogue paths, which is infeasible due to exponential complexity.
   - Directly using LLM function calling for task-oriented dialogues fails to maintain dialogue state and yields unstable instruction-following.
   - Knowledge queries and task execution are usually decoupled, lacking support for compositionality (e.g., combining restaurant search and reservation).

**Key Challenge**: Developers require precise control over agent behavior, but LLMs are inherently unreliable and fail to follow complex logical instructions.

**Goal**: How to make agents reliably follow developer-defined policies while maintaining the naturalness of LLM conversations.

**Key Insight**: Decouple the LLM and the algorithmic system—let the LLM handle only parsing and generation, while policy execution is delegated to a deterministic runtime.

**Core Idea**: Declarative Worksheet specifications + execution of policies enforced by an algorithmic Runtime + LLM limited to NLU/NLG.

## Method

### Overall Architecture
Developers write declarative Genie Worksheets (defining required fields and actions) $\rightarrow$ Genie Parser (LLM semantically parses user input into formal state updates) $\rightarrow$ Genie Runtime (algorithmically executes policies: checks predicates, fills fields, queries knowledge bases, executes API calls, and generates deterministic Agent Acts) $\rightarrow$ Response Generator (LLM translates formal Agent Acts into natural language).

### Key Designs
1. **Genie Worksheet (Declarative Specification Language)**:

    - **Function**: Developers only need to declare the fields, types, conditional predicates, and corresponding actions to be collected from the user.
    - **Mechanism**: Supports two types of Worksheets—Task Worksheets (defining tasks) and Knowledge Worksheets (declaring knowledge sources)—supporting composition (e.g., field types referencing instances of another Worksheet).
    - **Design Motivation**: Analogous to class definitions, this is much more concise than dialogue trees—developers do not need to map out all dialogue paths.

2. **Genie Parser (LLM Semantic Parsing)**:

    - **Function**: Translates user natural language inputs into Python statements that update the dialogue state.
    - **Mechanism**: Implemented in two stages—CSP (Contextual Semantic Parsing) maps user input to state updates, and KP (Knowledge Parsing) translates natural language queries into formal queries like SUQL.
    - **Design Motivation**: Feeds only the latest Worksheet state and a single turn of dialogue history to the LLM, preventing forgetfulness caused by excessively long contexts.

3. **Genie Runtime (Algorithmic Runtime)**:

    - **Function**: Deterministically executes agent policies to generate formal Agent Acts (Report/Confirm/Say/Propose/Ask).
    - **Mechanism**: Evaluates predicates $\rightarrow$ checks types $\rightarrow$ executes knowledge queries $\rightarrow$ executes actions $\rightarrow$ finds the first unfilled field to Ask the user.
    - **Design Motivation**: Since LLMs cannot reliably follow all developer instructions, an algorithmic system guarantees $100\%$ policy execution.

### Loss & Training
Zero-shot, no fine-tuning required. Relies on LLMs like GPT-4 Turbo/GPT-4o-mini for semantic parsing and response generation, with a few few-shot examples.

## Key Experimental Results

### Main Results (StarV2 benchmark, Agent Act accuracy)

| Method | Bank | Trip | Trivia |
|------|------|------|--------|
| AnyTOD-PROG+SGD (Fine-tuned T5-XXL) | 65.0 | 62.9 | 86.3 |
| GPT-4 Turbo (function calling) | 55.1 | ~50 | ~80 |
| Genie (GPT-4 Turbo) | **82.5** | **83.4** | **92.7** |
| Genie (GPT-4o-mini) | 82.1 | 76.3 | 84.8 |

### User Study (62 participants, 3 real-world tasks)

| Method | Goal Completion Rate |
|------|-----------|
| GPT-4 Turbo + function calling | 21.8% |
| Genie (GPT-4 Turbo) | **82.8%** |

### Key Findings
- Genie allows a weaker model (GPT-4o-mini) to approach the performance of a stronger model (GPT-4 Turbo), and even outperform GPT-4 Turbo without Genie.
- The function calling method degrades severely in multi-turn dialogues—failing to maintain dialogue state, often asking repetitive questions, or forgetting information.
- Decoupling policy execution from the LLM is crucial—the LLM only needs to excel at parsing and generation, rather than understanding complex business logic.
- In real-user studies, the completion rate improved nearly fourfold.

## Highlights & Insights
- The architectural concept of "LLM for NLU/NLG only, leaving policy to a deterministic system" is highly practical, suitable for enterprise-grade Agent deployment.
- Worksheet declarative specification is far more elegant than dialogue trees—developers define "what is needed" rather than "how to converse".
- The compositionality of tasks and knowledge queries (where a field type is another Worksheet) addresses common hybrid needs in real-world scenarios.
- Dialogue states are maintained using formal variables, passing only the most recent state to the LLM, which mitigates long-dialogue forgetting problems.

## Limitations & Future Work
- Developers still need to write Worksheet specifications, which poses a barrier for non-technical users.
- Highly dependent on the quality of the LLM's semantic parsing—parsing errors can cascade and affect the entire system.
- Not applicable to open-domain, unstructured conversation scenarios (specifically designed for task-oriented dialogues).
- Currently, the expressiveness of the Worksheet language is limited; complex business logic might require extensions.

## Related Work & Insights
- **vs LLM Function Calling**: Function calling cannot maintain dialogue state and fails to follow complex policies; Genie resolves this through an algorithmic Runtime.
- **vs RASA/Dialogue Trees**: Dialogue trees require enumerating all paths; Genie Worksheet only declares fields and actions.
- **vs AnyTOD**: AnyTOD requires large amounts of training data for fine-tuning; Genie is zero-shot and significantly outperforms it.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The architectural separation of declarative specification and algorithmic runtime is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation involving benchmarks, user studies with 3 real-world applications, comparisons across multiple LLMs, and detailed analyses.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear architecture diagrams, rich examples, and complete system design.
- **Value**: ⭐⭐⭐⭐⭐ Exceptional practical engineering value, with the completion rate improving from $22\%$ to $83\%$ representing a substantial advancement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reliable Decision Making via Calibration Oriented Retrieval Augmented Generation](../../NeurIPS2025/information_retrieval/reliable_decision_making_via_calibration_oriented_retrieval_augmented_generation.md)
- [\[ACL 2025\] ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](arise_risk_adaptive_search.md)
- [\[ACL 2025\] Empaths at SemEval-2025 Task 11: Retrieval-Augmented Approach to Perceived Emotions Prediction](empaths_at_semeval-2025_task_11_retrieval-augmented_approach_to_perceived_emotio.md)
- [\[ACL 2026\] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](../../ACL2026/information_retrieval/visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)
- [\[ICLR 2026\] RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](../../ICLR2026/information_retrieval/reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)

</div>

<!-- RELATED:END -->
