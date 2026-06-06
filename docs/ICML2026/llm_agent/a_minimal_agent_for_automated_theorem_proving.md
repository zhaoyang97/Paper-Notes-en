---
title: >-
  [Paper Note] A Minimal Agent for Automated Theorem Proving
description: >-
  [ICML 2026][LLM Agent][Theorem Proving] This paper introduces AxProverBase—a minimal Lean 4 theorem-proving agent that, using only three components ("compiler feedback + self-managed notebook + lightweight tool search")…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Theorem Proving"
  - "Lean 4"
  - "Agent Architecture"
  - "Iterative Refinement"
  - "Self-Managed Memory"
date: 2026-05-08
content_hash: 02df230062257270
---

# A Minimal Agent for Automated Theorem Proving

**Conference**: ICML 2026  
**arXiv**: [2602.24273](https://arxiv.org/abs/2602.24273)  
**Code**: https://github.com/Axiomatic-AI/ax-prover-base  
**Area**: LLM Agent / Formal Mathematics  
**Keywords**: Theorem Proving, Lean 4, Agent Architecture, Iterative Refinement, Self-Managed Memory

## TL;DR
This paper introduces AxProverBase—a minimal Lean 4 theorem-proving agent that, using only three components ("compiler feedback + self-managed notebook + lightweight tool search"), achieves or surpasses specialized systems like Hilbert/Seed-Prover on cutting-edge, untuned LLMs (Claude Opus), while reducing costs by 100x.

## Background & Motivation

**Background**: Recent years have seen breakthroughs in AI theorem proving (AlphaProof, Hilbert, Seed-Prover), but most approaches rely on large-scale synthetic data fine-tuning or RL, resulting in high complexity and cost. Meanwhile, frontier general-purpose LLMs are rapidly improving in formal mathematics, but the contributions of system design versus model advances to final performance remain unclear.

**Limitations of Prior Work**: (1) Complex architectures are hard to reproduce; (2) Tight coupling with Lean/Mathlib versions requires retraining for upgrades; (3) High GPU cluster or API costs; (4) The individual contributions of iterative feedback, memory, and tool search are unquantified.

**Key Challenge**: It is widely assumed that strong provers require complex designs, but is this true? Does simplification necessarily lead to performance collapse?

**Goal**: Identify the "minimal necessary module combination" to achieve competitive performance with a minimalist architecture, and provide clear ablation baselines.

**Key Insight**: Starting from the ReAct framework, the system is decomposed into three interchangeable modules: Proposer, Reviewer, and Memory. The architecture is built bottom-up, quantifying marginal gains at each layer.

**Core Idea**: Iterative feedback >> memory >> tool search; "compiler feedback + self-reflective notebook" alone can rival the most complex systems, with tool search providing only marginal benefit.

## Method

### Overall Architecture

Core loop:

```python
while not_proved and iters < N:
  proposal = Proposer(theorem, file_context, memory)
  feedback = Compiler(proposal)
  if not_proved:
    memory.update(proposal, feedback, reasoning)
```

- **Proposer**: General-purpose LLM (Claude Opus) or ReAct agent, optionally equipped with LeanSearch / Tavily tools.
- **Reviewer**: Dual-layer verification with Lean 4 compiler and LLM reviewer to prevent false proofs.
- **Memory**: Three strategies—no memory, historical memory (full attempts from previous N rounds), self-managed notebook (LLM maintains distilled insights).

### Key Designs

1. **Proposer with Restricted Tool Calls**:

    - **Function**: Allows the model to retrieve Mathlib theorems and web information, but limits the number of calls to prevent information noise from dominating the proposal process.
    - **Mechanism**: Before each proposal, at most one round of LeanSearch (vector theorem retrieval) and Tavily (web search) are called in parallel. Web search is permitted because the core challenge is "writing compilable Lean code" rather than logic itself.
    - **Design Motivation**: Tools are helpful but not decisive; excessive calls inflate context and degrade quality.

2. **Self-Managed Memory (Self-Managed Context)**:

    - **Function**: The LLM autonomously maintains a "lab notebook," recording key technical insights and ways to avoid past mistakes.
    - **Mechanism**: After each iteration, the Proposer reflects on the attempt and updates the notebook—retaining important insights and deleting outdated entries. Subsequent iterations prioritize reading the notebook over the full history. Compared to historical memory, context is reduced by ~50%, cost drops by 20%, and variance is halved.
    - **Design Motivation**: Mimics mathematicians' workflow—remembering key points rather than full logs; lets the LLM judge information value, avoiding hard-coded heuristics.

3. **Multi-Layer Review to Prevent False Proofs**:

    - **Function**: Prevents the use of sorry/admit/metaprogramming tricks to fake "proof completion."
    - **Mechanism**: First, the Lean compiler verifies that the code compiles and contains no sorry/admit/suggestion; second, remaining goals are extracted to ensure no unclosed subgoals; third, the LLM reviewer checks that the theorem statement is unaltered and that there is no circular reasoning due to over-generalization.
    - **Design Motivation**: The final safeguard for Lean system trustworthiness; multi-layer design is low-cost but significantly enhances security.

### Loss & Training
No training; direct inference with off-the-shelf LLMs.

## Key Experimental Results

### Ablation Study (PutnamBench 100-question subset)

| Configuration | Pass@1 (%) | Pass@20 (%) | Avg. Cost | Notes |
|---------------|------------|-------------|-----------|-------|
| Single-shot LLM (Claude Opus) | 2.0 | 5.0 | – | Baseline |
| + Iterative Feedback (1 retry) | 8.5 | 18.0 | $0.30/problem | **Largest single improvement** |
| + Historical Memory (5 rounds) | 15.2 | 31.0 | $0.80/problem | Effective but context bloat |
| + Self-Managed Memory (5 rounds) | 16.3 | 33.2 | $0.64/problem | **Best trade-off** |
| + Tool Search | 17.8 | 35.5 | $0.72/problem | Marginal gain ~8% |

### Model Comparison (Full system, 50 iterations)

| Model | Pass@1 | Pass@50 | Relative Cost |
|-------|--------|---------|--------------|
| Claude Sonnet 4.5 (10k thinking) | 28.5% | 51.3% | 0.8x |
| Claude Opus 4.5 (10k thinking) | 38.2% | 60.7% | 1.0x |
| Claude Opus 4.5 (32k thinking) | 45.1% | **68.3%** | 1.8x |
| Gemini 3 Flash (high) | 9.2% | 25.1% | 0.3x |
| Gemini 3 Pro (high) | 12.5% | 28.7% | 0.6x |

### Main Results (Opus 32k, 50 iterations)

| Benchmark | AxProverBase | SOTA Competitor | Notes |
|-----------|--------------|-----------------|-------|
| PutnamBench (pass@1) | **54.7%** | Hilbert 55.9% | 100x lower cost |
| FATE-M (pass@1) | **98.0%** | REAL-Prover 56.7% | Large margin |
| FATE-H (pass@1) | **66.0%** | REAL-Prover 0% | First >60% |
| FATE-X (pass@1) | 24.0% | Seed-Prover 33% | Extremely challenging |
| LeanCat (pass@1) | **59.0%** | Opus single-shot 8.25% | Significant iterative gain |

### Key Findings
- **Iterative feedback is decisive**: Adding a feedback loop alone raises pass@1 from 2% to 8.5% (4.25x), exceeding the cumulative effect of other changes.
- **Self-managed memory outperforms historical memory**: At equal cost, it delivers better performance and stability, demonstrating the value of "curated memory > full memory."
- **Model capability is amplified by the framework**: Opus 32k thinking outperforms 10k thinking by 7.6 percentage points in pass@50; stronger models gain more from this framework.
- **Tool search has limited value**: In competition math, web search is almost unhelpful; LeanSearch provides slight benefit but is not critical.
- **Cross-domain generalization**: The simple architecture works from competition math to abstract algebra (FATE-M) and category theory (LeanCat).

## Highlights & Insights
- **The power of minimalism**: Proof does not require large-scale training or complex search; "compiler feedback + self-reflection + strong model" can rival SOTA.
- **Effectiveness of self-reflection**: Letting the LLM maintain its own notebook outperforms fixed heuristic retrieval, highlighting the value of "metacognition" in AI systems.
- **Rigorous ablation design**: Bottom-up stacking with clear quantification at each layer provides direction for future improvements.
- **New perspective on cost-performance**: $12.6/problem versus Hilbert's hundreds or thousands of dollars, greatly lowering the barrier for adoption.

## Limitations & Future Work
- Achieving only 24% on the hardest FATE-X set indicates a bottleneck in deep mathematical intuition.
- Only a single model family (Claude) was evaluated; performance of other architectures remains untested.
- Lean 4-specific; transferability to Coq/Isabelle needs validation.
- Self-managed memory relies on the model's introspective ability and may fail with weaker models.
- Future directions: enhance hybrid semantic+symbolic retrieval; integrate specialized geometry/algebra solvers; adopt a two-stage sketch-to-formalization paradigm.

## Related Work & Insights
- **vs Seed-Prover / Goedel-Prover**: The latter rely on large-scale synthetic data + RL; this work shows that general-purpose LLMs alone can be competitive.
- **vs AlphaProof**: AlphaProof uses tree search + complex heuristics; this work uses a linear iterative program, remaining competitive with greater simplicity.
- **Insights**: The paradigm of iterative feedback + self-reflection + lightweight tools can transfer to program synthesis, scientific verification, and other complex reasoning tasks.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Individual modules are not highly novel, but the experimental finding that "minimalism is strong" is itself inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 benchmarks, multiple models, detailed ablation, broad coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear architecture, complete pseudocode, precise results.
- Value: ⭐⭐⭐⭐⭐ Lowers the AI barrier for formal mathematics, with significant impact on the open-source community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents](../../AAAI2026/llm_agent/structured_personalization_modeling_constraints_as_matroids_for_data-minimal_llm.md)
- [\[NeurIPS 2025\] VeriMaAS: Automated Multi-Agent Workflows for RTL Design](../../NeurIPS2025/llm_agent/automated_multi-agent_workflows_for_rtl_design.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](../../ACL2026/llm_agent/feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[NeurIPS 2025\] Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX](../../NeurIPS2025/llm_agent/benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)
- [\[NeurIPS 2025\] Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](../../NeurIPS2025/llm_agent/automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)

</div>

<!-- RELATED:END -->
