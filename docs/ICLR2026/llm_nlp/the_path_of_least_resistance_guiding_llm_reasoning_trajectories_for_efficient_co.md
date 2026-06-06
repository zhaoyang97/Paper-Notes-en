---
title: >-
  [Paper Note] The Path of Least Resistance: Guiding LLM Reasoning Trajectories for Efficient Consistency
description: >-
  [ICLR 2026][LLM/NLP][self-consistency] This paper proposes PoLR (Path of Least Resistance), the first inference-time method that exploits reasoning prefix consistency. By clustering short prefixes and expanding only the…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "self-consistency"
  - "inference efficiency"
  - "prefix clustering"
  - "reasoning"
  - "token reduction"
date: 2026-05-08
content_hash: fc732d8690de0779
---

# The Path of Least Resistance: Guiding LLM Reasoning Trajectories for Efficient Consistency

**Conference**: ICLR 2026
**arXiv**: [2601.21494](https://arxiv.org/abs/2601.21494)  
**Code**: None  
**Area**: LLM/NLP
**Keywords**: self-consistency, inference efficiency, prefix clustering, reasoning, token reduction

## TL;DR

This paper proposes PoLR (Path of Least Resistance), the first inference-time method that exploits reasoning prefix consistency. By clustering short prefixes and expanding only the dominant cluster, PoLR serves as an efficient alternative to Self-Consistency, reducing token usage by up to 60% and latency by up to 50%.

## Background & Motivation

Self-Consistency (SC) decoding substantially improves LLM reasoning accuracy by sampling multiple reasoning trajectories and selecting the final answer via majority voting. However, it incurs significant computational overhead, as every reasoning trajectory must be fully unrolled. Existing improvements such as Adaptive Consistency (AC) and Early-Stopping SC (ESC) mitigate this by halting generation once answer-level agreement is reached, but they share a fundamental limitation: **answer-level consistency is only observable after complete reasoning trajectories have been generated**, precluding the use of rich structural information available in early reasoning stages.

The central observation motivating PoLR is that **the prefix of a reasoning trajectory (its first few steps) already encodes strong signals about the final answer**—a phenomenon termed "prefix consistency." Trajectories sharing the same prefix achieve nearly identical accuracy to full SC, implying that the substantial token overhead spent on additional trajectories rarely contributes to the final answer.

## Method

### Overall Architecture

PoLR modifies the standard SC pipeline by introducing a prefix-based selection step:

1. **Prefix Sampling**: Given input question $x$ and model $\mathcal{M}$, generate $N$ short reasoning prefixes $p_i = \text{Prefix}(\mathcal{M}(x, t_i), L_p)$ (implemented via `max_new_tokens = L_p`).
2. **Embedding & Clustering**: Each prefix is encoded as a sparse TF-IDF bag-of-words vector and then clustered via hierarchical clustering (cosine similarity) into $\mathcal{C} = \{C_1, \dots, C_m\}$. The dominant cluster is selected as $C^* = \arg\max_{C_j}|C_j|$.
3. **Expansion**: Only $K$ prefixes from the dominant cluster $C^*$ are expanded into full reasoning trajectories $r_k = \mathcal{M}(x | p_k)$.
4. **Voting**: $\hat{a} = \arg\max_y \sum_{k=1}^K \mathbf{1}[a_k = y]$

### Key Designs

**Token Efficiency Formula**:

$$\eta = 1 - \frac{T_{\text{PoLR}}}{T_{\text{SC}}} = 1 - \frac{N \cdot \ell_p + K \cdot (\ell_f - \ell_p)}{N \cdot \ell_f}$$

where $\ell_p$ denotes average prefix length and $\ell_f$ denotes full reasoning length.

**Justification of Design Choices**:
- **TF-IDF over neural encoders**: Lightweight, model-agnostic, and CPU-friendly; neural encoders introduce clustering overhead far exceeding the marginal accuracy benefit over TF-IDF.
- **Hierarchical clustering**: Well-suited for small $N$ (11–51); requires no pre-specified number of clusters and produces interpretable groupings.
- **$L_p = 256$**: Empirically achieves a favorable balance between accuracy and token efficiency.

### Theoretical Analysis

The correctness and efficiency of PoLR are guaranteed by two complementary properties:

**Correctness Alignment**: Let $Y \in \{0,1\}$ denote the correctness of the final reasoning trajectory and $Z$ denote the cluster assignment of the prefix. The key condition is $I(Z;Y) > 0$, i.e., the clustering at least weakly predicts correctness. When $H(Y|Z)$ is small, cluster identity reliably predicts correctness.

**Structural Skew and Efficiency**: Efficiency is not driven by correctness alignment but by the structural skew of the prefix cluster distribution. Define the skew ratio $\kappa = |C^*|/N$.

**Proposition**: The token efficiency gain of PoLR relative to SC satisfies $\eta \geq 1 - \frac{K}{M} \cdot \kappa^{-1}$, and efficiency increases monotonically with $\kappa$.

Core insight: **Mutual information guarantees safety (no accuracy loss), while skew determines savings**. NMI remains low ($\leq 0.18$), yet efficiency saturates at 50–58%, as prefix clusters exhibit strong structural skew.

### Loss & Training

PoLR is an inference-time method and involves no training or loss function. Its core optimization objective is to minimize token consumption while preserving SC-level accuracy.

## Key Experimental Results

### Main Results

Evaluated on GSM8K, Math500, AIME24/25, and GPQA-Diamond across multiple LLM families:

| Model | Dataset | N | SC Acc | PoLR Δ | η (%) | Overhead kt (ms) |
|------|--------|---|--------|--------|-------|-------------|
| QWQ32B | GSM8K | 51 | 90.8% | -0.3 | 47.6 | 11.2 |
| DSQ7B | Math500 | 31 | 89.6% | +0.1 | 48.5 | 5.1 |
| QWQ32B | GPQA-D | 51 | 68.7% | +1.5 | 53.8 | 11.2 |
| DSQ7B | AIME25 | 31 | 33.7% | +2.7 | - | - |
| Phi-4-15B | AIME25 | 31 | 32.0% | +4.0 | - | - |
| QWQ32B | Math500 | 51 | 91.8% | +0.2 | 51.8 | 11.2 |

**Key Findings**:
- Token efficiency η typically ranges from 40–60%, effectively halving token consumption.
- Clustering overhead kt amounts to only a few milliseconds, and savings translate directly to faster inference.
- Accuracy is maintained or occasionally improved, as PoLR emphasizes the dominant consistent reasoning cluster and filters noisy trajectories.
- The 10-point drop for QWQ32B on AIME25 is an outlier attributable to only 3 out of 30 samples.

### Ablation Study

Preliminary analysis (Math500, GSM8K, DSQ7B, 40 samples) validates prefix consistency:

| Dataset | $L_p$ | Expansion Rate | Accuracy | Exact Prefix Match |
|--------|-------|--------|--------|-------------|
| Math500 | SC | 1.00 | 89.8 | - |
| Math500 | 32 | 0.64 | 89.8 | 125 |
| Math500 | 128 | 0.48 | 89.2 | 5 |
| GSM8K | SC | 1.00 | 79.7 | - |
| GSM8K | 32 | 0.52 | 79.7 | 135 |
| GSM8K | 128 | 0.47 | 79.3 | 30 |

### Key Findings

1. PoLR is robust across different clustering methods, prefix lengths, and cluster selection strategies.
2. PoLR is fully complementary to adaptive inference methods (AC, ESC) and can serve as an upstream filter.
3. Consistent effectiveness across model families and scales (1.5B–32B).
4. Consistent gains are also observed on non-mathematical tasks (StrategyQA).

## Highlights & Insights

1. **"Less is more" paradigm for reasoning efficiency**: Prefix clustering reveals that LLMs encode structural consistency early in reasoning, rendering the majority of subsequent computation redundant.
2. **Elegant unification of theory and practice**: The separation between correctness alignment (mutual information guarantees safety) and structural skew (κ drives efficiency) is analytically clean and practically grounded.
3. **Zero training overhead**: The lightweight combination of TF-IDF and hierarchical clustering makes PoLR a truly plug-and-play replacement.
4. **Complementary by design**: Explicitly positioned as an upstream optimization for SC, PoLR is composable with AC, ESC, and related methods.

## Limitations & Future Work

1. **10-point drop for QWQ32B on AIME25**: Volatility risk exists on challenging benchmarks with very few samples.
2. **Manual specification of prefix length $L_p$**: While 256 works well in most cases, adaptively determining the optimal prefix length remains an open problem.
3. **Dependence on prefix structural skew**: When reasoning paths for a given problem are highly diverse ($\kappa \approx 1/m$), the efficiency gains of PoLR diminish.
4. **Evaluated only on open-source models**: Validation on closed-source models such as GPT-4 is absent.

## Related Work & Insights

- **Self-Consistency** (Wang et al., 2023): The direct baseline for PoLR.
- **Adaptive Consistency** (Aggarwal et al., 2023): Stops generation on demand but still relies on full trajectories.
- **Early-Stopping SC** (Li et al., 2024): Shares similar limitations.
- **Prefix Consistency** (Ji et al., 2025): Exploits prefixes at training time and requires fine-tuning.

The core insight PoLR offers is: **the critical moment for reasoning efficiency optimization lies not at the end (when to stop) but at the beginning (when to branch)**. This insight may inspire further methods that leverage prefix signals during reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ — First inference-time method to exploit prefix consistency as a replacement for SC; conceptually novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 5 benchmarks, 6 models, multiple configurations, and 10 repetitions.
- Writing Quality: ⭐⭐⭐⭐ — Theory and experiments are tightly integrated with a clear overall structure.
- Value: ⭐⭐⭐⭐ — Practically meaningful for efficient inference; a plug-and-play reasoning acceleration solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Reasoning on the Manifold: Bidirectional Consistency for Self-Verification in Diffusion Language Models](../../ICML2026/llm_nlp/reasoning_on_the_manifold_bidirectional_consistency_for_self-verification_in_dif.md)
- [\[ICLR 2026\] KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](kvcomm_enabling_efficient_llm_communication_through_selective_kv_sharing.md)
- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Models](predicting_llm_reasoning_performance_with_small_proxy_models.md)
- [\[ICLR 2026\] From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning](from_assumptions_to_actions_turning_llm_reasoning_into_uncertainty-aware_plannin.md)
- [\[ICML 2026\] Token-Efficient Change Detection in LLM APIs](../../ICML2026/llm_nlp/token-efficient_change_detection_in_llm_apis.md)

</div>

<!-- RELATED:END -->
