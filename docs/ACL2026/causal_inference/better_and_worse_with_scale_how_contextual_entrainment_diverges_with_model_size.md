---
title: >-
  [Paper Note] Better and Worse with Scale: How Contextual Entrainment Diverges with Model Size
description: >-
  [ACL 2026 Findings][Causal Inference][Contextual entrainment effect] This paper establishes the first scaling laws for the "contextual entrainment effect," discovering that larger models are more resistant to false information in semantic contexts (negative exponent) but more prone to copying irrelevant tokens in non-semantic contexts (positive exponent), revealing opposing scaling behaviors between semantic filtering and mechanical copying functions.
tags:
  - "ACL 2026 Findings"
  - "Causal Inference"
  - "Contextual entrainment effect"
  - "Scaling laws"
  - "Semantic filtering"
  - "Pattern copying"
  - "Robustness"
date: 2026-05-08
content_hash: 6ee768d6ea2c8cee
---

# Better and Worse with Scale: How Contextual Entrainment Diverges with Model Size

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.13275](https://arxiv.org/abs/2604.13275)  
**Code**: None  
**Area**: Causal Reasoning  
**Keywords**: Contextual entrainment effect, Scaling laws, Semantic filtering, Pattern copying, Robustness

## TL;DR
This paper establishes the first scaling laws for the "contextual entrainment effect," discovering that larger models are more resistant to false information in semantic contexts (negative exponent) but more prone to copying irrelevant tokens in non-semantic contexts (positive exponent), revealing opposing scaling behaviors between semantic filtering and mechanical copying functions.

## Background & Motivation

**Background**: LLMs increasingly rely on external context (RAG, user-provided documents), but context information can be noisy, irrelevant, or erroneous. Niu et al. (2025) formalized the "contextual entrainment effect"—the tendency of models to increase the probability of tokens appearing in the context, regardless of their semantic relevance.

**Limitations of Prior Work**: Entrainment effects have been observed at a single model scale, but their relationship with model size remains entirely unknown. Traditional scaling laws describe aggregate loss, masking the evolution of specific behavioral mechanisms.

**Key Challenge**: While intuition suggests that larger, "smarter" models should be more robust, they are also "stronger pattern matchers," which might make them more prone to copying context. Which of these two trends dominates?

**Goal**: Quantify how the contextual entrainment effect changes with model size and establish behavioral scaling laws.

**Key Insight**: Categorize context into four types (counterfactual, related, irrelevant, random) and fit power laws $E(N) = a \cdot N^b$ separately to observe the divergence in the sign of the exponents.

**Core Idea**: Entrainment effects for semantic and non-semantic contexts follow scaling laws in opposite directions—larger models become both "better" and "worse" simultaneously.

## Method

### Overall Architecture

This paper is a pure measurement study aimed at answering whether the contextual entrainment effect strengthens or weakens as models scale. The approach utilizes the LRE (Linear Relational Embedding) factual query dataset, pairing the same query with four types of contexts with varying semantic properties. Logits for each token are extracted across two complete model families, Cerebras-GPT (111M–13B) and Pythia (410M–12B), to calculate the offset with and without context $\Delta_t = \text{logit}(t|\text{ctx}) - \text{logit}(t|\varnothing)$. Behavioral metrics for distractor tokens and gold tokens are analyzed separately, and these metrics are then fitted to a power law against model scale $N$ to observe how the exponent signs diverge by context type.

### Key Designs

**1. Four Context Conditions: Separating Semantic-Driven from Mechanical Copying**

Entrainment may arise from two distinct mechanisms: the model being misled "because it understands the semantics" or "merely copying what it sees." To isolate these, semantic relevance must be controlled. For the same query (e.g., "The capital of Germany is ___", gold=Berlin), four contexts are constructed: Counterfactual ("The capital of Germany is Munich", d=Munich, direct semantic conflict), Related ("The Eiffel Tower is in Paris", d=Paris, semantically related but non-conflicting), Irrelevant ("The water is warm", d=warm, semantically irrelevant), and Random ("Calculator", d=Calculator, purely random token). This spectrum from "strong semantics" to "no semantics" serves as the experimental lever for observing sign divergence.

**2. Power-law Scaling Fitting: Quantifying Behavioral Metrics into Scaling Laws**

To precisely characterize the relationship between entrainment and model size, this paper follows the standard neural scaling form by performing linear regression in log-log space for each behavioral metric: $E(N) = a \cdot N^b$. The exponent $b$, 95% confidence intervals, $R^2$, and p-values are reported, using $R^2 > 0.8$ and $p < 0.01$ as the threshold for "strong evidence." The sign of exponent $b$ is the core conclusion: $b < 0$ indicates the effect decays as models grow, while $b > 0$ indicates it strengthens, compressing the "better or worse" question into a comparable and extrapolatable number.

**3. Baseline Validation and Control: Ruling out Spurious Correlations**

To attribute observed scaling trends to context manipulation, it must be proven that these trends are not inherent to data subsets. The authors verify that without context, gold token logits scale consistently across all four subsets ($b \in [+0.129, +0.134]$, $R^2 > 0.93$), whereas distractor tokens without context show no consistent scaling ($R^2 < 0.25$). This implies that the scaling of distractors is solely introduced by the context, ruling out results driven by data artifacts.

## Key Experimental Results

### Main Results

| Context Type | Exponent $b$ ($\Delta_d$) | 95% CI | $R^2$ | Implication |
|-----------|-------------------|--------|-------|------|
| Counterfactual | -0.330 | [-0.44, -0.22] | 0.926 | Larger models more resistant to false info |
| Related | -0.135 | [-0.16, -0.11] | 0.977 | Larger models more resistant to semantic interference |
| Irrelevant | +0.091 | [+0.05, +0.13] | 0.879 | Larger models more susceptible to irrelevant tokens |
| Random | +0.217 | [+0.14, +0.30] | 0.905 | Larger models more prone to copying random tokens |

### Ablation Study

| Metric | 111M → 13B Change | Description |
|------|----------------|------|
| Counterfactual $\Delta_d$ | 9.69 → 2.30 | 4x decrease, semantic filtering enhanced |
| Random $\Delta_d$ | 0.82 → 1.97 | 2.4x increase, copying mechanism enhanced |
| Related gap ($\Delta_g - \Delta_d$) | 5.71 → 0.55 | 10.3× convergence, semantic distinction improved |
| Random gap | 0.73 → 2.18 | 3.0× divergence, noise sensitivity sharpened |

### Key Findings
- The divergence of exponent signs between semantic and non-semantic contexts is replicated across both Cerebras-GPT and Pythia model families, suggesting this is an inherent property of Transformer scaling.
- A gradient exists rather than a binary split: behaviors align with semantic coherence from counterfactual (strongest negative scaling) to random (strongest positive scaling).
- The convergence-divergence split implies that larger models are more sensitive to context quality—they benefit more from good context but are harmed more by poor context.

## Highlights & Insights
- **The core insight is exceptionally elegant**: The same phenomenon (contextual entrainment) exhibits opposite scaling behaviors based on the semantic nature of the content, moving beyond the simple narrative of "larger is better." For RAG systems, this implies that context quality curation becomes more critical, not less, as models scale.
- **Opposing mechanical explanations** are compelling—pattern matching and semantic filtering are independent functional modules with distinct scaling tracks, where the former resembles induction heads and the latter reflects reasoning capabilities.
- The analysis methodology is transferable to any research question regarding "how behavior changes with model size."

## Limitations & Future Work
- The study is limited to decoder-only Transformers; encoder-only and encoder-decoder architectures may exhibit different entrainment dynamics.
- Scaling is analyzed only at the behavioral level without mechanistic decomposition (e.g., identifying specific attention heads responsible for each behavior).
- The LRE dataset primarily consists of factual queries; entrainment effects may differ in complex reasoning tasks.
- The impact of instruction tuning or RLHF on entrainment scaling remains unexplored.

## Related Work & Insights
- **vs Niu et al. (2025)**: While they found entrainment to be pervasive at a fixed scale, this work extends it to the scaling dimension and discovers the sign divergence.
- **vs Kaplan et al. (2020)**: Traditional scaling laws describe the monotonic decrease of aggregate loss, whereas this work reveals that behavioral metrics can scale in opposite directions.
- **vs Wei et al. (2022)**: While the "emergent abilities" paper focuses on the sudden appearance of capabilities, this work quantifies how existing behaviors change continuously.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First scaling law for contextual entrainment; the sign divergence discovery is novel and counter-intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across two model families with full statistical significance, though missing instruction-tuned models.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative structure is refined and elegant, with the "better and worse" contrast well-integrated throughout.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement](../../ICLR2026/causal_inference/resisting_contextual_interference_in_rag_via_parametric-knowledge_reinforcement.md)
- [\[NeurIPS 2025\] Cyclic Counterfactuals under Shift–Scale Interventions](../../NeurIPS2025/causal_inference/cyclic_counterfactuals_under_shift-scale_interventions.md)
- [\[ICLR 2026\] Frequency-Domain Better than Time-Domain for Causal Structure Recovery in Dynamical Systems on Networks](../../ICLR2026/causal_inference/frequency-domain_better_than_time-domain_for_causal_structure_recovery_in_dynami.md)
- [\[ICML 2026\] Investigating Memory in Model-Free RL with POPGym Arcade](../../ICML2026/causal_inference/investigating_memory_in_model-free_rl_with_popgym_arcade.md)
- [\[ICLR 2026\] Adjusting Prediction Model Through Wasserstein Geodesic for Causal Inference](../../ICLR2026/causal_inference/adjusting_prediction_model_through_wasserstein_geodesic_for_causal_inference.md)

</div>

<!-- RELATED:END -->
