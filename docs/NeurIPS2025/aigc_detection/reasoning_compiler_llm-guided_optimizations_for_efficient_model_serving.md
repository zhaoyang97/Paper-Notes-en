---
title: >-
  [Paper Note] Reasoning Compiler: LLM-Guided Optimizations for Efficient Model Serving
description: >-
  [NeurIPS 2025][AIGC Detection][LLM-guided compilation] This paper proposes Reasoning Compiler, which models compiler optimization as a sequential decision-making process…
tags:
  - "NeurIPS 2025"
  - "AIGC Detection"
  - "LLM-guided compilation"
  - "MCTS"
  - "program optimization"
  - "neural compiler"
  - "sampling efficiency"
date: 2026-05-08
content_hash: 0b44a763e148d107
---

# Reasoning Compiler: LLM-Guided Optimizations for Efficient Model Serving

**Conference**: NeurIPS 2025
**arXiv**: [2506.01374](https://arxiv.org/abs/2506.01374)  
**Code**: [https://github.com/he-actlab/REASONING_COMPILER](https://github.com/he-actlab/REASONING_COMPILER)  
**Area**: AIGC Detection
**Keywords**: LLM-guided compilation, MCTS, program optimization, neural compiler, sampling efficiency

## TL;DR
This paper proposes Reasoning Compiler, which models compiler optimization as a sequential decision-making process, employing an LLM as a context-aware proposal engine combined with MCTS to balance exploration and exploitation. The approach achieves an average 5.0× speedup across 5 representative benchmarks and 5 hardware platforms, with 10.8× better sampling efficiency than TVM's evolutionary search.

## Background & Motivation

**Background**: Large-scale model inference is computationally expensive, and compiler optimizations (tiling, fusion, vectorization, etc.) are critical acceleration techniques. Existing neural compilers (TVM, Ansor, etc.) explore the transformation space via evolutionary search or simulated annealing.

**Limitations of Prior Work**:
   - Rule-based compilers rely on hand-crafted heuristics and overfit to specific workloads or hardware
   - Random search methods exhibit **low sampling efficiency**—they fail to exploit contextual dependencies among transformations
   - The space of transformation combinations grows exponentially and is highly interdependent (e.g., the benefit of loop tiling depends on whether fusion has been applied previously)

**Key Challenge**: Compiler transformations exhibit complex non-local interactions; blind search cannot effectively capture these contextual dependencies.

**Key Insight**: LLMs are naturally suited for contextual reasoning—given the current code state, transformation history, and performance feedback, an LLM can reason about which transformation to apply next.

**Core Idea**: LLM-provided context-aware transformation proposals + MCTS-structured search = high-sampling-efficiency compiler optimization.

## Method

### Overall Architecture
Program optimization is modeled as a finite-horizon MDP: $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R} \rangle$. States are program variants, actions are compiler transformations, transitions are deterministic, and rewards are evaluated by a hardware cost model. MCTS performs structured search, with the LLM proposing transformation candidates at each expansion node.

### Key Designs

1. **LLM-Guided Contextual Reasoning**:

    - Function: At each MCTS expansion step, the LLM proposes the next transformation based on full context.
    - Mechanism: The prompt includes the current program $p_i$, source code and performance predictions of parent/grandparent programs, transformation history, and the full set of available transformations $O$. The LLM is instructed to perform CoT reasoning—analyzing synergistic or antagonistic effects among transformations.
    - Design Motivation: Transformation combinations exhibit complex interactions (e.g., tiling followed by vectorization may yield superior results); the LLM can reason about such relationships.

2. **MCTS Structured Search**:

    - Function: Balances exploration and exploitation while reusing common transformation prefixes.
    - Mechanism: Node selection uses UCT: $\text{UCT}(p_i) = \frac{W(p_i)}{N(p_i)} + c\sqrt{\frac{\ln N(p_{i-1})}{N(p_i)}}$; LLM proposals drive node expansion; random rollouts with the cost model provide evaluation; backpropagation updates statistics.
    - Design Motivation: The tree structure of MCTS naturally supports backtracking and reuse of transformation sequences, addressing the inefficiency of purely random search.

3. **Hardware-Aware Cost Model**:

    - Function: Serves as a proxy for real hardware execution, providing fast performance estimation.
    - Mechanism: A learned surrogate model $\hat{f}$ predicts latency for rollout reward computation.
    - Design Motivation: Real compilation and execution is prohibitively slow (on the order of minutes), whereas surrogate model evaluation is on the order of milliseconds.

### Loss & Training
- The LLM is used zero-shot (GPT-4o mini) without fine-tuning.
- MCTS hyperparameters: exploration coefficient $c = \sqrt{2}$, branching factor $B = 2$.
- Transformation validation: LLM outputs are parsed and filtered; invalid transformations fall back to random sampling.

## Key Experimental Results

### Main Results (5 kernels × 5 hardware platforms)

| Method | Avg. Speedup | Avg. Samples | Sampling Efficiency |
|--------|-------------|-------------|-------------------|
| TVM Evolutionary Search | Baseline | 5.8× more | 1.0× |
| MCTS (w/o LLM) | Moderate | Moderate | Moderate |
| **Reasoning Compiler** | **5.0×** | **5.8× fewer** | **10.8×** |

### End-to-End Llama-3-8B

| Method | Speedup | Sample Reduction | Sampling Efficiency |
|--------|---------|-----------------|-------------------|
| TVM | Baseline | Baseline | Baseline |
| Reasoning Compiler | 4.0× | 3.9× fewer | **5.6×** |

### Key Findings
- On FLUX attention, Reasoning Compiler achieves 2× speedup within 36 samples, while TVM requires 600+ samples (16× reduction).
- On Llama-4 MLP, a 12.7× speedup is achieved with 20 samples; TVM fails to reach this level even with 3,000 samples.
- **The advantage is greatest in low-budget settings**—high-quality optimization sequences are discovered within the first few dozen samples.

## Highlights & Insights
- **LLM as proposer, not decision-maker**: The LLM's role is precisely scoped as a "context-aware transformation proposer," while correctness and search structure are guaranteed by MCTS. This LLM-in-the-loop design paradigm is worth adopting broadly.
- **MDP formulation of compiler optimization**: Modeling transformation sequences as an MDP allows the theoretical guarantees of MCTS to apply directly.
- **Cross-hardware generalization**: The same framework is effective on ARM, x86, and Apple Silicon without hardware-specific tuning.

## Limitations & Future Work
- Dependence on LLM API calls may incur high costs for large-scale deployment.
- The code comprehension capability of GPT-4o mini is limited; stronger LLMs may yield further improvements.
- Random transformation sequences during rollout may be insufficiently efficient.
- No comparison is made against other LLM-for-code compiler optimization works.

## Related Work & Insights
- **vs. TVM/Ansor/MetaSchedule**: These methods use evolutionary search or simulated annealing with low sampling efficiency; Reasoning Compiler substantially improves upon this through LLM-guided reasoning.
- **vs. STOKE**: STOKE uses MCMC for superoptimization but also fails to leverage context; the LLM+MCTS combination proposed here is more efficient.
- The work demonstrates the significant potential of LLMs in system-level optimization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The LLM+MCTS compiler optimization framework is pioneering, elegantly combining LLM reasoning with structured search.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 benchmarks × 5 hardware platforms plus end-to-end evaluation—highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Formalization is clear and motivation is well articulated.
- Value: ⭐⭐⭐⭐⭐ Significant impact for both the compiler and model serving communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text](asciibench_evaluating_language-model-based_understanding_of_visually-oriented_te.md)
- [\[AAAI 2026\] ActiShade: Activating Overshadowed Knowledge to Guide Multi-Hop Reasoning in Large Language Models](../../AAAI2026/aigc_detection/actishade_activating_overshadowed_knowledge_to_guide_multi-h.md)
- [\[ICML 2026\] DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection](../../ICML2026/aigc_detection/dgs-net_distillation-guided_gradient_surgery_for_clip_fine-tuning_in_ai-generate.md)
- [\[NeurIPS 2025\] CLAWS: Creativity Detection for LLM-Generated Solutions Using Attention Window of Sections](clawscreativity_detection_for_llm-generated_solutions_using_attention_window_of_.md)
- [\[NeurIPS 2025\] Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code](classical_planning_with_llm-generated_heuristics_challenging_the_state_of_the_ar.md)

</div>

<!-- RELATED:END -->
