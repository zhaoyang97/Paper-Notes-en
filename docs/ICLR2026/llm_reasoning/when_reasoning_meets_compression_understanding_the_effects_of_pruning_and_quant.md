---
title: >-
  [Paper Note] When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models
description: >-
  [ICLR2026][LLM Reasoning][Model Compression] This paper systematically studies the effects of three compression methods—quantization, distillation, and pruning—on Large Reasoning Models (LRMs) through performance benchmarking and mechanistic interpretability analysis. Key findings include: parameter count affects knowledge memorization more than reasoning ability; the last-layer MLP `up_proj` is the most critical component; and current quantization methods over-compress the final layers.
tags:
  - ICLR2026
  - LLM Reasoning
  - Model Compression
  - Reasoning Models
  - Quantization
  - Distillation
  - Pruning
  - Interpretability
  - DeepSeek-R1
date: 2026-05-08
content_hash: f3c216a48565a0b1
---

# When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models

**Conference**: ICLR2026  
**arXiv**: [2504.02010](https://arxiv.org/abs/2504.02010)  
**Code**: [github.com/psunlpgroup/Compression-Effects](https://github.com/psunlpgroup/Compression-Effects)  
**Area**: LLM Reasoning  
**Keywords**: Model Compression, Reasoning Models, Quantization, Distillation, Pruning, Interpretability, DeepSeek-R1

## TL;DR
This paper systematically studies the effects of three compression methods—quantization, distillation, and pruning—on Large Reasoning Models (LRMs) through performance benchmarking and mechanistic interpretability analysis. Key findings include: parameter count affects knowledge memorization more than reasoning ability; the last-layer MLP `up_proj` is the most critical component; and current quantization methods over-compress the final layers.

## Background & Motivation
- Large reasoning models such as DeepSeek-R1 achieve strong performance on complex reasoning tasks but incur high deployment costs.
- Prior compression research faces two bottlenecks:
    - **Evaluation bottleneck**: Existing quantization/pruning evaluations primarily rely on perplexity and simple tasks, with insufficient testing on complex reasoning benchmarks.
    - **Analysis bottleneck**: In-depth interpretability analysis of compression effects is lacking.
- Core question: How is the reasoning capability of LRMs degraded during compression, and which weights are most critical for reasoning?

## Method

### 1. Evaluation Framework
- **Model selection**: DeepSeek-R1 (671B) and its compressed variants:
    - Quantization: Unsloth dynamic quantization (2.51/1.73/1.58-bit), AWQ, GPTQ, GPTAQ, ANY4/ANY3
    - Distillation: R1-Distill-Llama (70B/8B), R1-Distill-Qwen (32B/7B)
    - Pruning: SparseGPT, AlphaPruning (multiple sparsity levels)
- **Evaluation datasets** (in increasing difficulty):
    - AIME 2024 (mathematical reasoning)
    - FOLIO (logical reasoning)
    - Temporal Sequences (temporal reasoning, from BIG-Bench Hard)
    - MuSiQue (multi-hop reasoning, closed-book setting to test knowledge + reasoning)

### 2. Mechanistic Interpretability Analysis
Targeting four core reasoning behaviors: backtracking, uncertainty estimation, example testing, and adding knowledge.

**Direction vector extraction via mean difference**:
For each linear module $m$ at layer $\ell$, the direction vector for behavior $c$ is extracted as:

$$\mathbf{u}_{m\ell}^c = \frac{1}{|\mathcal{D}_+|} \sum_{s_i^c \in \mathcal{D}_+} \bar{\mathbf{a}}_{m\ell}^c(s_i^c) - \frac{1}{|\mathcal{D}_-|} \sum_{s_j \in \mathcal{D}_-} \bar{\mathbf{a}}_{m\ell}(s_j)$$

where $\bar{\mathbf{a}}_{m\ell}^c(s_i^c)$ is the mean activation over the behavior token sequence.

**Importance score computation via attribution patching**:

$$\mathbf{I}_{m\ell}^c \approx \frac{1}{|\mathcal{D}_+|} \left| \sum_{s_i^c \in \mathcal{D}_+} (\tilde{\mathbf{u}}_{m\ell}^c)^\top \frac{\partial}{\partial \mathbf{a}_{m\ell}} \mathcal{L}(s_i^c) \right|$$

A higher $\mathbf{I}_{m\ell}^c$ indicates a stronger causal relationship between the module and reasoning behavior $c$.

**Compression effect decoding**: The impact of compression is tracked by measuring changes in relative importance $\mathbf{RI}_{m\ell}^c$ (importance shift).

## Key Experimental Results

### Overall Performance Comparison

| Model | Params | Compression | AIME 2024 | FOLIO | Temporal | Avg | MuSiQue (EM, F1) |
|------|--------|----------|-----------|-------|----------|-----|-------------------|
| DeepSeek-R1 | 671B | None | 73.3 | 76.4 | 99.6 | 83.1 | (17.0, 27.51) |
| DeepSeek-R1 | 671B | 2.51-bit | 76.7 | 77.8 | 100.0 | **84.8** | (17.0, 24.43) |
| DeepSeek-R1 | 671B | 1.58-bit | 66.7 | 75.4 | 94.0 | 78.7 | (14.0, 22.34) |
| R1-Distill-Llama | 70B | Distillation | 65.6 | 79.8 | 99.9 | 81.8 | (13.3, 21.57) |
| R1-Distill-Qwen | 32B | Distillation | 64.4 | 82.3 | 99.9 | 82.2 | (2.7, 10.95) |
| R1-Distill-Llama | 8B | Distillation | 42.2 | 71.9 | 81.5 | 65.2 | (0.0, 4.43) |
| R1-Distill-Llama | 70B | 50% SparseGPT | 23.3 | 71.6 | 97.6 | 64.2 | (6.7, 13.49) |

### Selective Quantization Validates Component Importance

| Quantized Component | Rank | AIME 2024 | FOLIO | Temporal | Avg |
|----------|------|-----------|-------|----------|-----|
| 32_up (last-layer up_proj) | Global #1 | 20.0 | 63.1 | 63.6 | 48.9 |
| 32_gate | #2 in column | 33.3 | 62.1 | 67.2 | 54.2 |
| 32_v | Last in column | 43.3 | 68.0 | 79.6 | 63.6 |
| Unquantized baseline | - | 42.2 | 71.9 | 81.5 | 65.2 |

Quantizing only `32_up` (0.7% of total weights) causes an average accuracy drop of **16.3%**.

### Effect of Protecting Critical Weights

| Compression | Protected | AIME 2024 | FOLIO | Temporal | Avg |
|----------|---------|-----------|-------|----------|-----|
| 3-bit AWQ | No | 10.0 | 59.6 | 68.4 | 46.0 |
| 3-bit AWQ | Last-layer MLP protected | **16.7** | **67.0** | **74.0** | **52.57** |

Protecting only ~2% of weights at full precision yields an average accuracy gain of **6.57%**, surpassing the previous SOTA quantization method by up to **23.17%**.

### Collapse Point Analysis (SparseGPT at Various Sparsity Levels)

| Sparsity | R1-Distill-Llama-70B AIME | R1-Distill-Llama-70B FOLIO |
|--------|---------------------------|---------------------------|
| 0% | 63.3 | 78.8 |
| 30% | 63.3 | 79.3 |
| 40% | 56.7 | 73.9 |
| 50% | 26.7 | 70.9 |
| 60% | 0.0 | 65.0 |
| 70% | 0.0 | 49.8 |

The collapse point is negatively correlated with task difficulty: AIME collapses at 40–50% sparsity, while FOLIO collapses at 60–70%.

## Three Core Findings

### Finding 1: Parameter Count Affects Knowledge Memorization More Than Reasoning
- Qwen-based models demonstrate stronger reasoning than Llama-based ones, yet score substantially lower on MuSiQue (knowledge-intensive).
- Pruning causes knowledge memorization to collapse earlier than reasoning (MuSiQue collapses at 30–40% sparsity).
- Conclusion: For knowledge-intensive tasks, quantization (which preserves parameter count) is preferable over pruning or distillation.

### Finding 2: The Last-Layer MLP `up_proj` Is the Most Critical Component
- This pattern is consistently observed across both R1-Distill-Llama-8B and R1-Distill-Qwen-7B.
- Distillation is identified as the cause of this component's heightened importance (the original Llama model does not exhibit this property).
- This finding supplements prior work claiming `o_proj` is the most important component.

### Finding 3: Current Quantization Methods Over-Compress the Last Layer and `gate_proj`
- Both AWQ and GPTQ excessively compress the last-layer modules and mid-layer `gate_proj`.
- Protecting only the last-layer MLP modules yields significant performance improvements (+6.57% average).
- This finding generalizes to pruning methods as well.

## Highlights & Insights
1. **First systematic comparison of three compression methods on LRMs**: fills a gap in the LRM compression literature.
2. **Fine-grained interpretability analysis**: importance is analyzed at the level of individual linear modules, going beyond prior layer-level analyses.
3. **High practical value**: protecting only 2% of weights yields substantial gains, providing clear guidance for future compression methods.
4. **Generalizable findings**: core findings hold across both R1 and non-R1 model families.
5. **Integration of theory and practice**: each finding is supported by validation experiments.

## Limitations & Future Work
- Interpretability analysis uses only 120 instances, limiting statistical power.
- Optimal strategies for mixed-precision quantization remain unexplored (only simple last-layer protection is validated).
- Pruning analysis is less comprehensive than quantization and distillation due to the unavailability of high-sparsity models.
- Distillation analysis is limited to SFT-based approaches; RL-stage distillation is not examined.
- Inference latency and deployment efficiency metrics are not reported.

## Related Work & Insights
- Compared to existing compression benchmarks (e.g., EleutherAI harness): this paper employs more challenging reasoning datasets.
- Compared to Venhoff et al.'s layer-level analysis: this paper provides module-level fine-grained analysis.
- Compared to Shao & Wu's claim that `o_proj` is most important: this paper finds `up_proj` to be more critical in distilled models.
- Compared to surveys by Liu et al. and Feng et al.: this paper contributes a unique mechanistic interpretability perspective.
- The finding on last-layer MLP `up_proj` importance can directly guide future quantization and pruning algorithm design.
- The mixed-precision protection strategy is generalizable to broader compression scenarios.
- The knowledge-vs-reasoning decomposition provides a theoretical basis for selecting appropriate compression methods.
- The correlation between collapse points and task difficulty can be used to estimate post-compression capability bounds.

## Rating
- Novelty: ⭐⭐⭐⭐ (Novel combination of systematic study and interpretability analysis, though the underlying methods are not original)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Covers quantization/distillation/pruning across multiple models, benchmarks, and validation experiments)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and concise presentation of findings, though the abundance of tables slightly burdens readability)
- Value: ⭐⭐⭐⭐⭐ (Three core findings are directly actionable for improving compression methods; protecting 2% of weights for a 6.57% gain is highly practical)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](towards_safe_reasoning_in_large_reasoning_models_via_correct-by-construction_gu.md)
- [\[ICLR 2026\] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)
- [\[ICLR 2026\] Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_solution_complexity.md)
- [\[ICLR 2026\] TopoBench: Benchmarking LLMs on Hard Topological Reasoning](topobench_benchmarking_llms_on_hard_topological_reasoning.md)
- [\[ICLR 2026\] Native Reasoning Models: Training Language Models to Reason on Unverifiable Data](native_reasoning_models_training_language_models_to_reason_on_unverifiable_data.md)

</div>

<!-- RELATED:END -->
