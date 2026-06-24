---
title: >-
  [Paper Note] Intention Chain-of-Thought Prompting with Dynamic Routing for Code Generation
description: >-
  [AAAI 2026][Reasoning][Intention Chain] This paper proposes RoutingGen — a difficulty-aware code generation framework grounded in the principle of cognitive economy. A Qwen3-8B classifier dynamically routes tasks to either a simple path (few-shot direct generation) or a complex path (Intention CoT = specification constraints + algorithmic intent + complexity analysis), achieving a +45.15% improvement on McEval while reducing average token consumption by 46.37%.
tags:
  - "AAAI 2026"
  - "Reasoning"
  - "Intention Chain"
  - "Dynamic Routing"
  - "Code Generation"
  - "Cognitive Economy"
  - "Difficulty-Awareness"
date: 2026-05-08
content_hash: aed0ee54e60d4a08
---

# Intention Chain-of-Thought Prompting with Dynamic Routing for Code Generation

**Conference**: AAAI 2026
**arXiv**: [2512.14048](https://arxiv.org/abs/2512.14048)  
**Code**: [https://github.com/Guai001/RoutingGen](https://github.com/Guai001/RoutingGen)  
**Area**: LLM Reasoning / Code Generation
**Keywords**: Intention Chain, Dynamic Routing, Code Generation, Cognitive Economy, Difficulty-Awareness

## TL;DR
This paper proposes RoutingGen — a difficulty-aware code generation framework grounded in the principle of cognitive economy. A Qwen3-8B classifier dynamically routes tasks to either a simple path (few-shot direct generation) or a complex path (Intention CoT = specification constraints + algorithmic intent + complexity analysis), achieving a +45.15% improvement on McEval while reducing average token consumption by 46.37%.

## Background & Motivation

**Background**: Chain-of-Thought (CoT) prompting is effective for code generation — it guides LLMs to reason about algorithmic design before writing code. However, applying CoT uniformly to all tasks introduces an "overthinking" problem.

**Limitations of Prior Work**:
- Applying CoT to simple tasks (e.g., string reversal) reduces efficiency and may introduce erroneous reasoning.
- Existing CoT methods lack "intent abstraction" — they focus on syntactic correctness while neglecting core algorithm design and efficiency.
- Treating all tasks identically violates the principle of cognitive economy (only difficult tasks warrant deep reasoning).

**Key Challenge**: CoT is beneficial for hard tasks but wasteful for easy ones — an adaptive mechanism is needed to determine when to activate structured reasoning.

**Goal**: Dynamically select a generation strategy based on problem difficulty: simple problems → direct generation; complex problems → intent-level CoT reasoning followed by generation.

**Key Insight**: The cognitive economy principle — activate structured reasoning only when necessary, so as to conserve cognitive resources.

**Core Idea**: Difficulty-classifier routing + Intention CoT (Specification + Algorithmic Intent + Complexity) = efficient and high-quality code generation.

## Method

### Overall Architecture
An input programming problem is fed to a Qwen3-8B difficulty classifier (Simple/Complex + rationale) → Simple path: few-shot direct generation → Complex path: ICoT (Specification + Idea) → conditional code generation.

### Key Designs

1. **Difficulty-Aware Dynamic Routing**:

    - **Function**: Automatically assesses problem difficulty and selects the appropriate generation strategy.
    - **Mechanism**: The classifier outputs a Simple/Complex label along with a textual rationale. The simple path skips CoT and generates directly; the complex path enters ICoT.
    - **Design Motivation**: Cognitive economy — avoids performing algorithmic analysis on trivially simple tasks such as "print hello world."

2. **Intention Chain-of-Thought (ICoT)**:

    - **Function**: Conducts intent-level reasoning prior to code generation.
    - **Mechanism**: Comprises two components — (1) *Specification*: input/output constraints, boundary conditions, and data types ("what"); (2) *Idea*: core algorithmic logic, time complexity targets, and key data structure choices ("how" and "why").
    - **Design Motivation**: Conventional CoT focuses on surface-level steps ("first read input, then loop..."), whereas ICoT targets algorithmic intent ("use dynamic programming because subproblems overlap"), more closely mirroring how human programmers think.

3. **Conditional Code Generation**:

    - **Function**: Generates code conditioned on ICoT output.
    - **Mechanism**: The Specification and Idea produced by ICoT are injected as context into the code generation prompt.
    - **Design Motivation**: With a clearly articulated algorithmic intent, code generation can focus on implementation rather than design.

### Loss & Training
- Classifier: Qwen3-8B, used without fine-tuning (prompt-based).
- Code generation: Various LLMs (GPT-4o, DeepSeek-Coder, etc.).

## Key Experimental Results

### Main Results

| Benchmark | RoutingGen | Baseline | Gain | Token Savings |
|-----------|-----------|----------|------|---------------|
| HumanEval | **77.10%** | 74.97% | +2.13% | — |
| MBPP | **69.11%** | 56.59% | +12.52% | — |
| McEval | **38.90%** | 26.80% | **+45.15%** | — |
| Avg. Tokens | — | — | — | **−46.37%** |

### Ablation Study

| Configuration | MBPP Pass@1 | Notes |
|---------------|-------------|-------|
| No CoT (direct generation) | 56.59% | Baseline |
| Full CoT (all tasks) | 63.42% | Improves but wastes tokens |
| RoutingGen (Simple → direct) | 65.82% | Simple path avoids overthinking |
| **RoutingGen + ICoT** | **69.11%** | Full system achieves best performance |

### Key Findings
- **Largest gain on McEval (hard tasks)** (+45%): ICoT is most effective on problems where algorithm design is critical.
- **Skipping CoT on the simple path improves performance**: Avoiding overthinking eliminates reasoning-induced errors.
- **46% token reduction**: Simple tasks bypass CoT entirely and proceed directly to generation.
- **"Intent" outperforms "steps"**: Conventional CoT provides procedural steps ("first… then…"), whereas ICoT provides algorithmic intent ("use DP because…"), which offers stronger guidance to LLMs.

## Highlights & Insights
- Transferring the **cognitive economy** principle from psychology to LLMs is both natural and effective — not every problem warrants deep deliberation.
- The **Specification + Idea** structure of ICoT is better suited to code generation than general CoT — it abstracts algorithmic design rather than implementation steps.
- The pattern of dynamic routing combined with dedicated reasoning paths is broadly applicable to other tasks that require varying depths of reasoning.

## Limitations & Future Work
- The difficulty classifier may misclassify problems — routing a complex task to the simple path can result in severe performance degradation.
- The binary (Simple/Complex) categorization may be too coarse — finer-grained difficulty levels are needed.
- ICoT quality depends on the LLM's algorithmic knowledge — it may fail for algorithms underrepresented in training data.

## Related Work & Insights
- **vs. Self-Planning (Jiang et al.)**: Planning-style CoT without routing. RoutingGen adaptively selects the strategy.
- **vs. Scratchpad**: Supports intermediate computation but lacks intent abstraction. ICoT operates at a higher level.
- The routing + dedicated reasoning pattern is transferable to domains such as mathematical reasoning and scientific question answering.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of difficulty routing and intention CoT is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three benchmarks, multiple baselines, ablation studies, and token analysis.
- Writing Quality: ⭐⭐⭐⭐ The introduction of cognitive economy is natural and well-motivated.
- Value: ⭐⭐⭐⭐ Practically valuable for efficient code generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] RPM-MCTS: Knowledge-Retrieval as Process Reward Model with Monte Carlo Tree Search for Code Generation](rpm-mcts_knowledge-retrieval_as_process_reward_model_with_monte_carlo_tree_searc.md)
- [\[ECCV 2024\] Controllable Navigation Instruction Generation with Chain of Thought Prompting](../../ECCV2024/llm_reasoning/controllable_navigation_instruction_generation_with_chain_of_thought_prompting.md)
- [\[ICLR 2026\] String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation](../../ICLR2026/llm_reasoning/string_seed_of_thought_prompting_llms_for_distribution-faithful_and_diverse_gene.md)
- [\[ACL 2026\] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](../../ACL2026/llm_reasoning/learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)
- [\[AAAI 2026\] Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning](deep_hidden_cognition_facilitates_reliable_chain-of-thought_.md)

</div>

<!-- RELATED:END -->
