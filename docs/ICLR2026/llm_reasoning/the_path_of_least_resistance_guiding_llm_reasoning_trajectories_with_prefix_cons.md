---
title: >-
  [Paper Note] The Path of Least Resistance: Guiding LLM Reasoning Trajectories with Prefix Consensus
description: >-
  [ICLR 2026][LLM Reasoning][Self-consistency decoding] This paper proposes PoLR (Path of Least Resistance), the first inference-time method that exploits prefix consensus in reasoning chains. By clustering short prefixes…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "Self-consistency decoding"
  - "prefix consensus"
  - "reasoning efficiency"
  - "cluster pruning"
  - "inference-time computation"
date: 2026-05-08
content_hash: 6fe1929cc2ca12b7
---

# The Path of Least Resistance: Guiding LLM Reasoning Trajectories with Prefix Consensus

**Conference**: ICLR 2026
**arXiv**: [2601.21494](https://arxiv.org/abs/2601.21494)  
**Code**: To be confirmed  
**Area**: LLM Reasoning
**Keywords**: Self-consistency decoding, prefix consensus, reasoning efficiency, cluster pruning, inference-time computation

## TL;DR
This paper proposes PoLR (Path of Least Resistance), the first inference-time method that exploits prefix consensus in reasoning chains. By clustering short prefixes and expanding only the dominant cluster, PoLR replaces standard Self-Consistency while maintaining or improving accuracy on GSM8K, Math500, AIME, and GPQA, with 40%–60% reduction in token usage and up to 50% lower latency.

## Background & Motivation
Self-Consistency (SC) decoding is a strong baseline for LLM reasoning: multiple reasoning chains are sampled and a majority vote is taken over final answers, substantially outperforming greedy decoding. However, SC incurs significant computational overhead because all $N$ reasoning chains must be fully generated to completion, with a large fraction being redundant.

Existing improvements such as Adaptive Consistency and Early-Stop SC attempt to terminate sampling early once sufficient answer-level agreement is observed, but share a fundamental limitation: answer consistency can only be measured **after complete reasoning chains have been generated**, precluding any use of structural information from early stages of the reasoning process.

Recent work has identified a key phenomenon — **prefix consistency**: the first $L$ tokens of reasoning chains are often highly similar across samples, and this early agreement strongly correlates with the correctness of final answers. Ji et al. (2025) exploit this phenomenon at training time, but require costly fine-tuning.

**Core Idea**: Since reasoning prefixes already encode strong signals about final answers, it is possible **at inference time** to cluster short prefixes, identify the dominant reasoning pattern, and fully expand only the chains within the dominant cluster — safely discarding redundant paths. Analogous to physical systems following the path of least resistance, PoLR allocates computation to the most promising reasoning directions.

## Method

### Overall Architecture
PoLR proceeds in four steps: (1) sample $N$ short prefixes of length $L_p$; (2) encode with TF-IDF and apply hierarchical clustering; (3) select the dominant cluster and expand only $K$ chains ($K \ll N$) into full reasoning sequences; (4) apply majority voting over the expanded chains to obtain the final answer.

### Key Designs

1. **Prefix Sampling and Encoding (Steps 1–2)**

    - **Function**: Sample $N$ reasoning prefixes for input $x$ (setting `max_new_tokens` $= L_p$), then encode each as a sparse TF-IDF bag-of-words vector.
    - **Design Motivation**: Prefixes require only $L_p$ (e.g., 256) tokens — far fewer than complete reasoning chains (thousands of tokens) — making sampling cheap; TF-IDF encoding is lightweight, model-agnostic, and CPU-friendly.
    - **Mechanism**: Agglomerative hierarchical clustering with cosine similarity is applied without requiring a pre-specified number of clusters $m$, yielding interpretable groupings automatically.
    - **Novelty**: Ablations show that neural encoders (e.g., sentence transformers) substantially increase clustering overhead with negligible accuracy gain.

2. **Dominant Cluster Selection and Expansion (Steps 3–4)**

    - **Function**: Select the largest cluster $C^* = \arg\max |C_j|$ as the dominant cluster, and continue autoregressive generation from all $K$ prefixes within it to produce complete reasoning chains.
    - **Design Motivation**: The dominant cluster represents the most frequent reasoning pattern and contains the paths most likely to yield correct answers; small clusters tend to reflect noisy or anomalous reasoning.
    - **Mechanism**: Each chain is obtained via $r_k = M(x \mid p_k)$, i.e., conditional generation from prefix $p_k$, followed by majority voting over the $K$ resulting answers.
    - **Token Efficiency**: $\eta = 1 - \frac{N \cdot \ell_p + K \cdot (\ell_f - \ell_p)}{N \cdot \ell_f}$, which yields substantial savings when $K \ll N$ and $\ell_p \ll \ell_f$.

### Theoretical Analysis

The effectiveness of PoLR is guaranteed by two complementary properties:

- **Correctness Alignment**: As long as the mutual information $I(Z; Y) > 0$ (cluster assignment $Z$ weakly correlates with correctness $Y$), restricting to the dominant cluster does not systematically reduce accuracy. Empirically, NMI $\leq 0.18$ is sufficient.
- **Structural Skew**: Efficiency gains depend on the dominance of the largest cluster, $\kappa = |C^*|/N$. Proposition 1 provides a formal lower bound on efficiency: $\eta \geq 1 - (K/M) \cdot \kappa^{-1}$.

Key insight: accuracy preservation relies on correctness alignment, while efficiency improvement relies on structural skew — the two are independent, so high efficiency is achievable even under low NMI.

### Loss & Training
PoLR is a purely inference-time method requiring **no training or fine-tuning**. All operations are performed within the inference pipeline. Clustering overhead is negligible ($k_t = 2$–$17$ ms). The default prefix length is $L_p = 256$.

## Key Experimental Results

### Main Results

| Dataset | Model | N | SC Acc | PoLR Δ | Token Efficiency η | Clustering Overhead $k_t$ |
|--------|------|---|--------|--------|-------------|-------------|
| GSM8K | QWQ32B | 51 | 90.8% | −0.3% | 47.6% | 11.2ms |
| Math500 | QWQ32B | 51 | 91.8% | +0.2% | 51.8% | 11.2ms |
| Math500 | DSQ7B | 31 | 89.6% | +0.1% | 48.5% | 5.1ms |
| AIME25 | DSQ7B | 31 | 35.3% | +0.0% | 48.4% | 3.9ms |
| AIME25 | Phi-4-15B | 31 | 32.0% | **+4.0%** | 54.8% | 5.9ms |
| GPQA-D | QWQ32B | 51 | 68.7% | **+1.5%** | 53.8% | 17.4ms |
| GPQA-D | MiMo-7B | 51 | 65.7% | −0.5% | 51.4% | 9.0ms |

PoLR matches or exceeds SC accuracy in the vast majority of settings, with typical token efficiency of 40%–60% and negligible clustering overhead in the millisecond range.

### Ablation Study (Complementarity with Adaptive Methods, GPQA-Diamond)

| Method | DSQ7B N=31 Acc | PExp | QWQ32B N=31 Acc | PExp |
|------|---------------|------|-----------------|------|
| SC | 55.25 | 31.00 | 68.19 | 31.00 |
| AC | 55.20 | 13.54 | 67.93 | 13.00 |
| **PoLR+AC** | **55.56** | **10.53** | **68.33** | **8.72** |
| ESC | 54.85 | 14.74 | 67.73 | 14.89 |
| **PoLR+ESC** | **54.85** | **10.71** | **67.73** | **9.10** |

Used as a pre-filtering step, PoLR is fully complementary to AC and ESC, further reducing the average number of expanded paths by approximately 31% and achieving roughly 75% total compute savings relative to SC.

### Key Findings
- **Prefixes encode early consensus**: On Math500 with $L_p = 32$, 64% of chains share identical prefixes (EPM = 125/500), with accuracy identical to full SC (89.8%).
- **Efficiency is driven by structural skew**: NMI (cluster–correctness mutual information) is only $\leq 0.18$, yet strong structural skew (large $\kappa$) still yields 50%–58% efficiency.
- **Consistent across models and scales**: Results hold for DSQ1.5B, DSQ7B, QWQ32B, MiMo-7B, Phi-4-15B, and Qwen2.5-Math-7B, from 1.5B to 32B parameters.
- **Occasional accuracy improvements**: PoLR improves accuracy on AIME25 by +2.7% for DSQ7B and +4.0% for Phi-4-15B, attributed to filtering out noisy reasoning paths.
- **Robust to clustering method**: Agglomerative clustering, K-Means, and DBSCAN all perform comparably; TF-IDF matches neural encoders in accuracy at orders-of-magnitude lower cost.

## Highlights & Insights
- Prefix consistency is an underutilized inference-time signal — a large proportion of reasoning chains converge on the same solution strategy within the first few steps.
- The theoretical analysis cleanly separates "safety" (correctness alignment; low NMI suffices) from "efficiency" (structural skew is the critical factor).
- The fully training-free, plug-and-play design makes PoLR directly applicable as a preprocessing step for any SC variant.
- Clustering overhead is effectively negligible (millisecond scale), making PoLR a genuine "free lunch."

## Limitations & Future Work
- Occasional large accuracy drops are observed on AIME25/QWQ32B (−10%), occurring in high-difficulty settings with very small datasets (30 problems).
- Prefix length $L_p$ is a hyperparameter; while $L_p = 256$ generalizes well empirically, the optimal value may be task-dependent.
- Theoretical guarantees are approximate, grounded in empirical correlations between mutual information and skew rather than strict PAC bounds.
- The method may be less effective on tasks requiring high creativity or exploratory reasoning (e.g., open-ended generation), where prefixes are inherently diverse.
- Evaluation is currently limited to mathematical, STEM, and commonsense reasoning benchmarks, without coverage of broader NLP tasks.

## Related Work & Insights
- PoLR is complementary rather than competing with Adaptive Consistency and Early-Stop SC — PoLR prunes *before* generation, while the latter truncate *after* generation.
- Ji et al. (2025) exploit prefix consistency at training time; PoLR is the first to do so at inference time without fine-tuning.
- Insight: The prefix consistency phenomenon suggests that LLMs "know" the solution direction early in the reasoning process, with subsequent generation being largely confirmatory — this may represent a fundamental bottleneck for reasoning token efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ The first inference-time method exploiting prefix consistency; the concept is concise and elegant, though the core mechanism (cluster-and-prune) is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five benchmarks × six models × three values of $N$ × ten repetitions, with complementarity experiments against AC/ESC and robustness analyses.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; theoretical analysis and experimental results are well aligned; figures and tables are intuitively designed.
- Value: ⭐⭐⭐⭐ A practically significant improvement to SC decoding with substantial token savings, directly deployable in production environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Nudging the Boundaries of LLM Reasoning](nudging_the_boundaries_of_llm_reasoning.md)
- [\[ACL 2026\] Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment](../../ACL2026/llm_reasoning/which_reasoning_trajectories_teach_students_to_reason_better_a_simple_metric_of_.md)
- [\[ICLR 2026\] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)
- [\[ICLR 2026\] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)
- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
