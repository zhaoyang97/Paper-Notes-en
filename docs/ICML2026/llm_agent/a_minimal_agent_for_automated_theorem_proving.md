---
title: >-
  [Paper Note] A Minimal Agent for Automated Theorem Proving
description: >-
  [ICML 2026][LLM Agent][Theorem Proving] AxProverBase is proposed—a minimalist Lean 4 theorem-proving agent. By relying on only three components—"Compiler Feedback + Self-Managed Notebook + Lightweight Tool Search"—it mat…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Theorem Proving"
  - "Lean 4"
  - "Agent Architecture"
  - "Iterative Refinement"
  - "Self-managed Memory"
date: 2026-05-08
content_hash: 5cc791fadde0216c
---

# A Minimal Agent for Automated Theorem Proving

**Conference**: ICML 2026  
**arXiv**: [2602.24273](https://arxiv.org/abs/2602.24273)  
**Code**: https://github.com/Axiomatic-AI/ax-prover-base  
**Area**: LLM Agent / Formal Mathematics  
**Keywords**: Theorem Proving, Lean 4, Agent Architecture, Iterative Refinement, Self-managed Memory

## TL;DR
AxProverBase is proposed—a minimalist Lean 4 theorem-proving agent. By relying on only three components—"Compiler Feedback + Self-Managed Notebook + Lightweight Tool Search"—it matches or exceeds specialized systems such as Hilbert/Seed-Prover using non-fine-tuned frontier LLMs (Claude Opus), while reducing costs by 100x.

## Background & Motivation

**Background**: Significant breakthroughs have occurred in AI theorem proving recently (AlphaProof, Hilbert, Seed-Prover). However, most rely on large-scale synthetic data fine-tuning or reinforcement learning (RL), leading to extreme complexity and cost. Meanwhile, the formal mathematical capabilities of frontier general-purpose LLMs are improving rapidly, yet it remains difficult to decouple the contributions of system design versus model improvements to final performance.

**Limitations of Prior Work**: (1) Complex architectures are difficult to reproduce; (2) Tight coupling with Lean/Mathlib versions requires retraining for upgrades; (3) GPU clusters or API costs are prohibitive; (4) The individual contributions of iterative feedback, memory, and tool search have not been quantified.

**Key Challenge**: It is widely assumed that strong provers require complex designs, but is this true? Would simplification cause performance to collapse?

**Goal**: To identify the "minimum necessary combination of modules" to achieve competitive performance with a minimalist architecture and provide a clear ablation baseline.

**Key Insight**: Starting from the ReAct framework, the system is decomposed into three replaceable modules: Proposer, Reviewer, and Memory. These are stacked layer-by-layer from the bottom up to quantify marginal gains.

**Core Idea**: Iterative feedback >> Memory >> Tool search. "Compiler feedback + self-reflective notebook" can already rival the most complex systems; tool search is merely supplementary.

## Method

### Overall Architecture

The core loop:

```python
while not_proved and iters < N:
  proposal = Proposer(theorem, file_context, memory)
  feedback = Compiler(proposal)
  if not_proved:
    memory.update(proposal, feedback, reasoning)
```

- **Proposer**: A general LLM (Claude Opus) or a ReAct agent, with optional integration of LeanSearch / Tavily tools.
- **Reviewer**: Dual-layer verification consisting of the Lean 4 compiler and an LLM reviewer to prevent false proofs.
- **Memory**: Three strategies—No Memory, Historical Memory (previous $N$ full attempts), and Self-Managed Notebook (LLM-maintained refined insights).

### Key Designs

1. **Proposer with Constrained Tool Use**:
    - **Function**: Allows the model to retrieve Mathlib theorems and web information while limiting the number of calls to prevent informational noise from dominating the proposal process.
    - **Mechanism**: Before each proposal, at most one round of parallel calls to LeanSearch (vector retrieval of theorems) and Tavily (web search) is allowed. Web search is permitted because the core difficulty often lies in "writing compilable Lean code" rather than logic.
    - **Design Motivation**: Tools are helpful but not decisive; excessive calls bloat the context, which degrades quality.

2. **Self-Managed Context (Memory)**:
    - **Function**: The LLM autonomously maintains a "laboratory notebook" recording key technical insights and ways to avoid past errors.
    - **Mechanism**: After each iteration, the Proposer reflects on the attempt and updates the notebook—retaining important insights while deleting obsolete entries. Subsequent iterations prioritize reading the notebook over the full history. Compared to historical memory, context is reduced by ~50%, costs drop by 20%, and variance is halved.
    - **Design Motivation**: Mimics how mathematicians work—memorizing key points rather than full logs; allows the LLM to judge information value to avoid hard-coded heuristics.

3. **Multi-layer Review to Prevent False Proofs**:
    - **Function**: Blocks the use of `sorry`, `admit`, or metaprogramming tricks to fake "proof completion."
    - **Mechanism**: The first layer uses the Lean compiler to verify that code compiles and lacks `sorry`/`admit`/`suggestion`; the second layer extracts remaining goals to ensure no unclosed subgoals remain; the third layer uses an LLM reviewer to verify the theorem statement has not been tampered with and ensures no circular reasoning due to over-generalization.
    - **Design Motivation**: Acts as the final line of defense for the Lean system's trustworthiness; the multi-layer design is low-cost but significantly enhances security.

### Loss & Training
No training; utilizes off-the-shelf LLM inference directly.

## Key Experimental Results

### Ablation Study (Subset of 100 PutnamBench problems)

| Configuration | Pass@1 (%) | Pass@20 (%) | Avg. Cost | Description |
|------|-----------|-----------|--------|------|
| Single-shot LLM (Claude Opus) | 2.0 | 5.0 | – | Baseline |
| + Iterative Feedback (1 retry) | 8.5 | 18.0 | $0.30/prob | **Largest single improvement** |
| + Historical Memory (5x) | 15.2 | 31.0 | $0.80/prob | Effective but context bloats |
| + Self-Managed Memory (5x) | 16.3 | 33.2 | $0.64/prob | **Optimal trade-off** |
| + Tool Search | 17.8 | 35.5 | $0.72/prob | Marginal gain ~8% |

### Main Results (Full system, 50 iterations)

| Model | Pass@1 | Pass@50 | Relative Cost |
|------|--------|---------|--------|
| Claude Sonnet 4.5 (10k thinking) | 28.5% | 51.3% | 0.8x |
| Claude Opus 4.5 (10k thinking) | 38.2% | 60.7% | 1.0x |
| Claude Opus 4.5 (32k thinking) | 45.1% | **68.3%** | 1.8x |
| Gemini 3 Flash (high) | 9.2% | 25.1% | 0.3x |
| Gemini 3 Pro (high) | 12.5% | 28.7% | 0.6x |

### Main Results (Opus 32k, 50 iterations)

| Benchmark | AxProverBase | Prev. SOTA | Remarks |
|------|-------------|-----------|------|
| PutnamBench (pass@1) | **54.7%** | Hilbert 55.9% | 100x lower cost |
| FATE-M (pass@1) | **98.0%** | REAL-Prover 56.7% | Significant lead |
| FATE-H (pass@1) | **66.0%** | REAL-Prover 0% | First to >60% |
| FATE-X (pass@1) | 24.0% | Seed-Prover 33% | Extremely difficult |
| LeanCat (pass@1) | **59.0%** | Opus single-shot 8.25% | Significant iterative gain |

### Key Findings
- **Iterative Feedback is Decisive**: Simply adding a feedback loop increased pass@1 from 2% to 8.5% (a 4.25x increase), exceeding the cumulative effect of other modifications.
- **Self-Managed Memory Beats Historical Memory**: It offers better performance and higher stability at the same cost, demonstrating that "curated memory > full memory."
- **Model Capabilities are Amplified by the Framework**: Opus with 32k thinking outperformed the 10k version by 7.6 percentage points in pass@50, showing that stronger models gain more from this framework.
- **Tool Search has Limited Value**: In competitive math environments, web search provides almost no help; LeanSearch helps slightly but is not critical.
- **Cross-Domain Generalization**: The minimalist architecture is applicable across domains, from competitive math to abstract algebra (FATE-M) and category theory (LeanCat).

## Highlights & Insights
- **The Power of Minimalism**: Proof generation does not require large-scale training or complex search; "compiler feedback + self-reflection + strong models" can rival the SOTA.
- **Effectiveness of Self-Reflection**: Allowing the LLM to maintain its own notebook outperforms fixed heuristic IR, highlighting the value of "metacognition" in AI systems.
- **Rigorous Ablation Design**: Layer-by-layer stacking from the ground up provides clear quantification of each layer's contribution, offering direction for future improvements.
- **New Perspective on Cost-Performance**: The cost of $12.6/problem compared to the hundreds or thousands of dollars for Hilbert significantly lowers the barrier to entry.

## Limitations & Future Work
- The 24% score on the hardest dataset, FATE-X, indicates a bottleneck in deep mathematical intuition.
- Only one model family (Claude) was evaluated; performance across different architectures was not tested.
- The system is Lean 4 specific; its portability to Coq/Isabelle needs verification.
- Self-managed memory depends on the model's introspective capabilities and may fail with weaker models.
- Future Directions: Enhancing hybrid semantic+symbolic retrieval; integrating specialized geometric/algebraic solvers; adopting a two-stage "sketch-then-formalize" paradigm.

## Related Work & Insights
- **vs. Seed-Prover / Goedel-Prover**: These rely on large-scale synthetic data and RL. This paper demonstrates that general-purpose LLMs can be competitive.
- **vs. AlphaProof**: AlphaProof uses tree search and complex heuristics; this paper uses a linear iterative process that is concise yet competitive.
- **Insights**: The paradigm of "Iterative Feedback + Self-Reflection + Light Tools" can be transferred to other complex reasoning tasks such as program synthesis and scientific verification.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Individual modules are not highly novel, but the experimental conclusion that "minimalism is powerful" is itself insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 benchmarks, multiple models, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear architecture, complete pseudocode, and precise presentation of results.
- Value: ⭐⭐⭐⭐⭐ Lowers the barrier for formal mathematics AI, significantly impacting the open-source community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents](../../AAAI2026/llm_agent/structured_personalization_modeling_constraints_as_matroids_for_data-minimal_llm.md)
- [\[NeurIPS 2025\] VeriMaAS: Automated Multi-Agent Workflows for RTL Design](../../NeurIPS2025/llm_agent/automated_multi-agent_workflows_for_rtl_design.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](../../ACL2026/llm_agent/feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[ICML 2026\] Agent JIT Compilation for Latency-Optimizing Web Agent Planning and Scheduling](agent_jit_compilation_for_latency-optimizing_web_agent_planning_and_scheduling.md)
- [\[NeurIPS 2025\] Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX](../../NeurIPS2025/llm_agent/benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)

</div>

<!-- RELATED:END -->
