---
title: >-
  [Paper Note] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs
description: >-
  [ICML 2025][Interpretability] FlashTrace proposes an efficient multi-token attribution method that reduces the attribution complexity of multi-token targets from $\mathcal{O}(M \cdot N)$ to $\mathcal{O}(N)$ using span-wise aggregation. It also traces importance propagation in reasoning chains via a recursive attribution mechanism, achieving a speedup of over 130x.
tags:
  - "ICML 2025"
  - "Interpretability"
date: 2026-05-08
content_hash: 3da42ffba24e3267
---

# Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs

**Conference**: ICML 2025  
**arXiv**: [2602.01914](https://arxiv.org/abs/2602.01914)  
**Area**: Interpretability  

## TL;DR

FlashTrace proposes an efficient multi-token attribution method that reduces the attribution complexity of multi-token targets from $\mathcal{O}(M \cdot N)$ to $\mathcal{O}(N)$ using span-wise aggregation. It also traces importance propagation in reasoning chains via a recursive attribution mechanism, achieving a speedup of over 130x.

## Background & Motivation

As modern LLMs increasingly rely on extended reasoning chains (such as OpenAI o1, DeepSeek-R1), existing token attribution methods face two key challenges:

**Efficiency Bottleneck**: Attributing a target span of length $M$ requires $\mathcal{O}(M \cdot N)$ operations. For a generation of 5K tokens, Integrated Gradients requires more than 10 hours.

**Fidelity Degradation**: Intermediate reasoning tokens absorb attribution mass, preventing importance from propagating back from the reasoning chain to the original input.

The paper's experiments validate these two issues:
- **Finding 1**: Reasoning tokens absorb most of the attribution mass. As the reasoning chain grows, the proportion of importance assigned to reasoning tokens $\mathbf{T}$ increases from approximately 80% to over 90%.
- **Finding 2**: Reasoning chains degrade the attribution quality on the input. The recovery rate of ground-truth critical input tokens drops from 26% to below 10%.

## Method

### Theoretical Framework

FlashTrace is based on the ALTI/IFR framework, utilizing an L1-norm-based proximity metric:

$$\text{Proximity}(\mathbf{z}, \mathbf{y}) = \max(0, -\|\mathbf{y} - \mathbf{z}\|_1 + \|\mathbf{y}\|_1)$$

Intuitively, it measures how much the magnitude of the target vector $\mathbf{y}$ is reduced after removing the contribution $\mathbf{z}$.

### Span-wise Aggregation

**Core Idea**: Instantly computes attribution for the entire target span rather than token-by-token.

Define the aggregated target: $\mathbf{Y}_S = \sum_{i \in S} \mathbf{y}_i$

Aggregated contribution: $\mathbf{Z}_S = \sum_{i \in S} \mathbf{z}_{j \to i}$

**Key Design**: Factorization is performed by leveraging the linearity of the attention mechanism. The transformation vector $\mathbf{v}_j$ depends only on the source token $j$ and is independent of the target position $i$:

$$\mathbf{F}_{j \to S} = \sum_{i \in S}(\alpha_{i,j}^h \cdot \mathbf{v}_j) = \mathbf{v}_j \cdot \left(\sum_{i \in S} \alpha_{i,j}^h\right)$$

Only one expensive vector transformation $\mathbf{v}_j$ needs to be computed for each source token, reducing the complexity from $\mathcal{O}(M \cdot N)$ to $\mathcal{O}(N)$.

### Recursive Attribution

**First-hop Attribution**: Performs standard attribution on the final output $\mathbf{O}$ to obtain the distribution $\mathbf{w}^{(0)}$.

**Recursive-hop Attribution**: Employs the importance scores from the previous hop as weights for the new target span:

$$\mathbf{Y}^{(1)} = \sum_{j \in \mathbf{T}} w_j^{(0)} \cdot \mathbf{y}_j$$

$$\mathbf{Z}^{(1)} = \sum_{j \in \mathbf{T}} w_j^{(0)} \cdot \mathbf{z}_{k \to j}$$

The efficiency advantage of span-wise aggregation is preserved under the weighted setting: the factorization becomes $\mathbf{v}_k \cdot (\sum_{j \in \mathbf{T}} w_j^{(0)} \alpha_{j,k}^h)$.

### Final Attribution Combination

Through multi-hop attribution, output attribution is propagated back to the original input via the reasoning chain:

$$\mathbf{w}_{\mathbf{I}}^{\text{final}} = \mathbf{w}_{\mathbf{I}}^{(0)} + \sum_{h=1}^{H} \gamma^h \cdot \mathbf{w}_{\mathbf{I}}^{(h)}$$

where $\gamma$ is a decay factor, and $H$ is the number of recursive hops.

## Key Experimental Results

### RULER Benchmark: Long-Context Retrieval

| Metric | Method | mq_q2 | mq_q4 | mv_v2 | mv_v4 |
|---|---|---|---|---|---|
| Recovery Rate ↑ | IFR | 0.471 | 0.328 | 0.575 | 0.452 |
| | AttnLRP | 0.215 | 0.204 | 0.254 | 0.243 |
| | **FlashTrace** | **0.483** | **0.413** | **0.556** | **0.516** |
| RISE ↓ | IFR | 0.075 | 0.115 | 0.069 | 0.073 |
| | **FlashTrace** | **0.068** | **0.113** | **0.069** | **0.070** |

### Reasoning Task: HotpotQA

| Method | Recovery Rate ↑ | RISE ↓ | MAS ↓ |
|---|---|---|---|
| Perturbation | 0.329 | 0.133 | 0.220 |
| CLP | 0.335 | 0.101 | 0.190 |
| IFR | 0.268 | 0.074 | 0.166 |
| AttnLRP | 0.189 | 0.155 | 0.249 |
| **FlashTrace** | **0.384** | **0.033** | **0.128** |

### Efficiency Comparison

FlashTrace achieves a speedup of **over 130x**. For a 10K-token reasoning chain, naive multi-hop methods require hours, whereas FlashTrace completes in seconds.

### Recursive Attribution Analysis

- **Changes in Attribution Distribution from Hop 1 to Hop 2**: Importance shifts from reasoning tokens near the output to earlier reasoning tokens and input context.
- **Even a single recursive hop significantly improves fidelity.**
- **Improvement is consistent across different models and data distributions.**

## Highlights & Insights

- **Elegant Theoretical Derivation**: Leverages the linearity of attention to reduce complexity from $\mathcal{O}(M \cdot N)$ to $\mathcal{O}(N)$.
- **High Practical Utility**: The 130x speedup turns long-horizon reasoning chain attribution from impractical to highly feasible.
- **Generality of Recursive Attribution**: Extends naturally to weighted span settings with zero additional computational overhead.
- **Clear Problem Definition**: Systematically formalizes the multi-token attribution problem for reasoning LLMs.
- **Thorough Experiments**: Validated across various tasks such as long-context retrieval, synthetic reasoning, and multi-step QA.

## Limitations & Future Work

- Proximity-based attribution assumes the effectiveness of the L1 norm in high-dimensional spaces, which may not hold in all scenarios.
- The number of hops for recursive attribution requires manual configuration.
- The aggregation method (summation) of tokens within a span might be overly simplistic.
- Modern gradient-based methods (such as Integrated Gradients) were not systematically compared under the same efficiency budget.
- The applicability to non-autoregressive models (such as encoder-decoder architectures) has not been explored.

## Rating

⭐⭐⭐⭐⭐ (5/5)

This is an exquisite piece of work featuring a clear problem definition, elegant theoretical derivation, a practical technical proposal, and thorough experimental validation. Against the backdrop of the increasing popularity of reasoning LLMs, addressing their interpretability holds crucial timeliness and practical value. The 130x speedup makes attributing long reasoning chains feasible for the first time.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values](deltashap_explaining_prediction_evolutions_in_online_patient_monitoring_with_sha.md)
- [\[ACL 2025\] Enhancing Automated Interpretability with Output-Centric Feature Descriptions](../../ACL2025/interpretability/output_centric_interpretability.md)
- [\[ACL 2025\] A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](../../ACL2025/interpretability/a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)
- [\[ICML 2025\] Foundation Molecular Grammar: Multi-Modal Foundation Models Induce Interpretable Molecular Grammar](foundation_molecular_grammar_multi-modal_foundation_models_induce_interpretable_.md)
- [\[ICLR 2026\] ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training](../../ICLR2026/interpretability/zerotuning_unlocking_the_initial_tokens_power_to_enhance_large_language_models_w.md)

</div>

<!-- RELATED:END -->
