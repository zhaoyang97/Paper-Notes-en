---
title: >-
  [Paper Note] Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference
description: >-
  [ACL 2026][LLM Efficiency][Mixture-of-Experts] This paper abstracts "activated expert counts" in MoE inference into a global budget $B$. It employs dynamic programming for optimal layer-wise Top-K allocation (Alloc-L)…
tags:
  - "ACL 2026"
  - "LLM Efficiency"
  - "Mixture-of-Experts"
  - "expert activation budget"
  - "dynamic programming"
  - "Token-level redistribution"
  - "inference acceleration"
date: 2026-05-08
content_hash: bd193c20b4741042
---

# Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference

**Conference**: ACL 2026  
**arXiv**: [2604.08133](https://arxiv.org/abs/2604.08133)  
**Code**: None  
**Area**: LLM Efficiency / MoE Inference  
**Keywords**: Mixture-of-Experts, expert activation budget, dynamic programming, Token-level redistribution, inference acceleration

## TL;DR
This paper abstracts "activated expert counts" in MoE inference into a global budget $B$. It employs dynamic programming for optimal layer-wise Top-K allocation (Alloc-L), followed by token-level redistribution using global Top-(K·T) selection (Alloc-T). Applied to DeepSeek-V2-Lite, it reduces the activation budget by half while maintaining accuracy, achieving 1.15× prefill speedup and 1.34× decode speedup.

## Background & Motivation

**Background**: Sparse MoE models (e.g., DeepSeek-V2/V3, Qwen-MoE, OLMoE) route each token to Top-K experts. Since inference latency increases nearly linearly with the number of activated experts, reducing K is a natural strategy for acceleration.

**Limitations of Prior Work**: Existing compression routes prioritize "reducing activation" but overlook the impact on model performance. Token-level methods (e.g., Top-P in XMoE, Dynamic-MoE, NAEE, AdapMoE) mostly rely on training-time calibration or fixed thresholds; layer-level methods (e.g., heuristics like LExI) treat all layers uniformly. Experiments show that reducing activations from 6 to 3 per token in DeepSeek-V2-Lite results in a 17% performance drop, falling to nearly 40% when reduced to 2.

**Key Challenge**: Activation quota is a scarce resource that should be distributed neither uniformly across layers nor uniformly across tokens—different layers exhibit significantly different sensitivities to sparsification, and routing distributions vary across different tokens. Prior works typically tackle either the layer or the token dimension, but never both as a unified budget optimization problem.

**Goal**: Find the optimal layer-level allocation $\mathbf{K}^{\ast}=[K_0,\dots,K_{L-1}]$ and token-level activation distribution under a fixed global activation budget $B$ to minimize performance loss.

**Key Insight**: The authors found that layer sensitivity can be accurately quantified using an end-to-end PPL metric (Key technique: when profiling layer $i$, fix **all subsequent layers** to Top-1 and keep **all preceding layers** at $K_{\text{orig}}$ to isolate the compensation effect from deeper layers). Token-level redistribution is then transformed into a global top-selection problem over a candidate set of size $T\times K_{\text{orig}}$.

**Core Idea**: Explicitly model the "activation budget" as a schedulable resource. Use DP for optimal inter-layer allocation + global top-selection for inter-token redistribution—the two are orthogonal and additive.

## Method

### Overall Architecture
Input: Pre-trained MoE model $M$ (with $L$ MoE layers, original Top-K $K_{\text{orig}}$), calibration data $D_{\text{calib}}$, and global activation budget $B$.
Offline Phase (Alloc-L): ① Run "isolated sensitivity profiling" for each layer $i$ to construct a sensitivity matrix $\mathbf{S}\in\mathbb{R}^{L\times K_{\text{orig}}}$; ② Formulate the "inter-layer budget allocation" as a grouped knapsack problem and solve for optimal $\mathbf{K}^{\ast}$ using DP.
Online Phase (Alloc-T): Within each layer $l$, redistribute activations among tokens based on $K_l^{\ast}$—first guarantee $K_{\text{base}}$ experts for each token, then allocate the remaining $(K_l-K_{\text{base}})\cdot T$ slots via global top-K competition among all (token, expert) candidate pairs based on routing scores.

### Key Designs

1.  **Alloc-L: Isolated Sensitivity Profiling + Knapsack DP**:
    - **Function**: Calculates the "relative loss for each Top-K value" per layer without retraining and finds the optimal layer-wise allocation under budget constraints.
    - **Mechanism**: Profiles layers from deep to shallow. When layer $i$ utilizes $k\in\{K_{\text{orig}},\dots,1\}$, all subsequent layers $j>i$ are forced to Top-1 (isolating their compensation) while preceding layers $j<i$ maintain original configurations; record $\mathbf{S}[i,k]=\mathrm{PPL}$. Solve $\arg\min_{\mathbf{K}}\sum_i\mathbf{S}[i,K_i]\ \text{s.t.}\ \sum_i K_i\le B$ using grouped knapsack DP: $\mathrm{DP}[i,b]=\min_{k\le b}(\mathrm{DP}[i-1,b-k]+\mathbf{S}[i,k])$. The complexity $O(L\cdot B\cdot K_{\text{orig}})$ is efficient as $B\le L K_{\text{orig}}$.
    - **Design Motivation**: A direct grid-search over all $\mathbf{K}$ combinations is $O(K_{\text{orig}}^L)$. Without isolating deep-layer compensation, shallow-layer sensitivity is masked (deep layers use more experts to "clean up"). Isolated profiling + DP solves both "measurement accuracy" and "computational efficiency."

2.  **Alloc-T: Base + Global Top-selection Token Redistribution**:
    - **Function**: Redistributes the fixed layer budget of $T\cdot K_l$ activation slots across tokens based on routing score distributions within the same layer—allocating more experts to tokens with flat (high entropy) distributions and fewer to those with sharp (high confidence) distributions.
    - **Mechanism**: Formulated as a 0-1 integer programming problem $\max\sum z_{t,e}w_{t,e}$ with constraints $\sum z_{t,e}\le T\cdot K_l$ and a per-token floor of $K_{\text{base}}$ experts. In practice: ① Secure Top-$K_{\text{base}}$ experts for each token; ② Flatten the remaining $K_{\text{orig}}-K_{\text{base}}$ candidate scores into a pool of size $T\cdot(K_{\text{orig}}-K_{\text{base}})$; ③ Select global top $(K_l-K_{\text{base}})\cdot T$. This involves only two masks and one top-K operation, incurring near-zero overhead.
    - **Design Motivation**: Standard Top-K routing thresholds tokens individually, ignoring the fact that some tokens have high confidence (e.g., score 0.9 for the first expert) while others are uncertain (e.g., scores of 0.25 for the first four). This design treats activations as a schedulable resource, allowing "unconfident" tokens to use more quota. It generalizes standard Top-K when $K_{\text{base}}=K_l$.

3.  **Alloc-MoE: Orthogonal Combination of Layer and Token Levels**:
    - **Function**: Integrates Alloc-L and Alloc-T into a unified framework—Alloc-L determines the average budget $K_l^{\ast}$ per layer, and Alloc-T performs token-level rearrangement under that $K_l^{\ast}$.
    - **Mechanism**: Since they operate on non-overlapping resource dimensions (layer vs. token), they can be combined multiplicatively. Ablations show this combination is optimal or near-optimal across the full budget range of DeepSeek-V2-Lite.
    - **Design Motivation**: Alloc-L alone gains +0.4% on average across 4 budgets, while Alloc-T alone gains +0.93%. Token-level redistribution offers higher gains in aggressive sparsification scenarios, yet the two are complementary, achieving the highest mean of 45.19% vs. 44.19% for Uniform.

### Loss & Training
Requires no retraining: Alloc-L requires one-time offline profiling of $L\cdot K_{\text{orig}}$ PPL instances to obtain $\mathbf{S}$; Alloc-T uses pure inference-time masking and top-K with no additional parameters. Default $K_{\text{base}}=1$ is used (optimal across nearly all model budgets).

## Key Experimental Results

### Main Results
Evaluated on DeepSeek-V2-Lite ($L=26$, $K_{\text{orig}}=6$), Qwen1.5-MoE-A2.7B, and OLMoE-1B-7B across 20 NLU, Reasoning, and Math benchmarks against Uniform, LExI, Dynamic-MoE, and NAEE. The table shows "task group means" for DeepSeek under half budget ($B=78$, average 3 experts per token).

| Task Group | Uniform | Dynamic-MoE | NAEE | LExI | **Alloc-MoE** |
|---|---|---|---|---|---|
| NLU | ~63.0 | ~62.5 | ~62.6 | ~63.4 | **~63.5** |
| Reasoning | ~37.2 | ~37.9 | ~38.0 | ~37.5 | **~38.6** |
| Math | ~31.6 | ~32.3 | ~31.7 | ~31.0 | **~34.4** |

Overall, Alloc-MoE outperformed state-of-the-art in 10 out of 12 budget × task configurations. Performance gains were more pronounced on harder tasks: NLU (+0.05%), Reasoning (+0.70%), Math (+2.15%). In terms of efficiency, it achieved 1.15× prefill and 1.34× decode speedup at half budget—comparable to LExI but with significantly higher accuracy.

### Ablation Study (DeepSeek-V2-Lite Task Group Average)

| Configuration | $B=52$ | $B=78$ | $B=104$ | $B=130$ | Mean |
|---|---|---|---|---|---|
| Uniform | 41.27 | 43.90 | 45.51 | 46.06 | 44.19 |
| +Alloc-L | 41.53 | 44.61 | 45.76 | 46.47 | 44.59 |
| +Alloc-T | 42.83 | 45.42 | 45.93 | 46.30 | 45.12 |
| **+Alloc-L +Alloc-T** | **43.09** | **45.48** | **46.01** | **46.17** | **45.19** |

### Key Findings
- **Budget Tightness**: Alloc-T contributes more (+1.56) when budgets are tighter ($B=52$, 2 experts per layer), suggesting token-level redistribution is more valuable under aggressive sparsification. Contributions equalize as the budget loosens.
- **$K_{\text{base}}$ Ablation**: Sweeping from 0 to 5, $K_{\text{base}}=1$ is near-optimal across all budgets. $K_{\text{base}}\ge 2$ causes significant performance drops under tight budgets because "guaranteeing" more experts reduces redistribution flexibility.
- **Calibration Data**: Sensitivity profiling is robust; using WikiText2, C4, or Pile for Alloc-L results in nearly identical means (44.59 to 44.68).
- **Visualization**: Alloc-L retains more experts in shallow layers and cuts deeper ones, aligned with the intuition that shallow layers extracting general features are more sensitive. Alloc-T shows a strong correlation ($>0.7$) between allocated expert count and routing entropy.

## Highlights & Insights
- **Unified "Activation Budget" Abstraction**: This is the primary highlight—unifying scattered layer/token sparsification decisions into a schedulable resource allows for the immediate use of standard tools like DP and global top-K. This "renaming the problem" approach is transferable to KV cache, attention head, or parallel expert partitioning.
- **Isolated Profiling Trick**: When quantifying layer sensitivity independently, deep layers normally "compensate" for shallow sparsification, making shallow layers appear less critical. Forcing subsequent layers to Top-1 reveals true sensitivity—a reusable profiling paradigm.
- **Zero-Overhead Implementation**: Alloc-T is elegantly implemented as "two masks + one global top-K" without custom kernels, making it easy to integrate into existing MoE routers.
- **Stable Expert Load**: Post-allocation load distribution maintains a Spearman correlation of 0.93–0.99 with the original model and JS divergence $<0.014$, indicating expert specialization is preserved—beneficial for distributed deployment.

## Limitations & Future Work
- Hardware awareness (expert placement, communication cost) was not explicitly modeled; gains in distributed MoE deployments might vary.
- Alloc-L requires $L\cdot K_{\text{orig}}$ offline PPL runs, which may be costly for 100B+ models; proxy small models could be used for sensitivity estimation.
- The framework is entirely post-hoc. Integrating "activation budget awareness" into the training auxiliary loss could further improve robustness.
- Evaluation was limited to three task types and model families, excluding scenarios like long-context or code where activation patterns might differ.

## Related Work & Insights
- **vs XMoE / Dynamic-MoE**: Both utilize token-level dynamic routing, but XMoE requires training-time calibration of Top-P thresholds, and Dynamic-MoE adds entropy regularization to the training objective. Alloc-T is purely post-hoc with zero additional parameters.
- **vs NAEE**: NAEE is limited to skipping the second expert in Top-2 routing and is hard to scale to Top-K>2; Alloc-T is a truly general top-selection.
- **vs LExI**: LExI uses heuristic layer allocation without DP optimality guarantees; Alloc-L provides a global optimal solution under budget constraints.
- **vs MoE Pruning/Quantization**: This work is orthogonal to pruning and quantization and can be combined for further acceleration.

## Rating
- Novelty: ⭐⭐⭐⭐ The "activation budget" abstraction combined with isolated profiling is a fresh combination, though the underlying components (DP/Top-K) are standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive evaluation across 3 models, 4 budgets, and 20 datasets, covering ablations, $K_{\text{base}}$ sweeps, and load analysis. Missing tests on ultra-large models.
- Writing Quality: ⭐⭐⭐⭐ Formulas and algorithms are clear; Figures 3-8 are highly informative. Alloc-L/Alloc-T nomenclature is intuitive.
- Value: ⭐⭐⭐⭐ Pure post-processing and zero training requirements make this highly practical for current MoE frameworks (e.g., vLLM). 1.34× decode speedup at half budget is very attractive for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](../../ICML2026/llm_efficiency/team_temporal-spatial_consistency_guided_expert_activation_for_moe_diffusion_lan.md)
- [\[ACL 2026\] The Hallucination of Specialization: Revealing the "Standing Committee" in Mixture-of-Experts Models](the_illusion_of_specialization_unveiling_the_domain-invariant_34standing_committ.md)
- [\[ICLR 2026\] Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling](../../ICLR2026/llm_efficiency/semantic_parallelism_redefining_efficient_moe_inference_via_model-data_co-schedu.md)
- [\[NeurIPS 2025\] Advancing Expert Specialization for Better MoE](../../NeurIPS2025/llm_efficiency/advancing_expert_specialization_for_better_moe.md)
- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](../../ICML2026/llm_efficiency/probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)

</div>

<!-- RELATED:END -->
