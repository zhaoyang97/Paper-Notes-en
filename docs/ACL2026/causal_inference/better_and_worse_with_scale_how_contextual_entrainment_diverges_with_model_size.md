---
title: >-
  [Paper Note] Better and Worse with Scale: How Contextual Entrainment Diverges with Model Size
description: >-
  [ACL 2026][Causal Inference][Contextual Entrainment] This paper establishes scaling laws for "contextual entrainment" for the first time…
tags:
  - "ACL 2026"
  - "Causal Inference"
  - "Contextual Entrainment"
  - "Scaling Laws"
  - "Semantic Filtering"
  - "Pattern Copying"
  - "Robustness"
date: 2026-05-08
content_hash: bc6630de9cc48fb2
---

# Better and Worse with Scale: How Contextual Entrainment Diverges with Model Size

**Conference**: ACL 2026  
**arXiv**: [2604.13275](https://arxiv.org/abs/2604.13275)  
**Code**: None  
**Area**: Causal Inference  
**Keywords**: Contextual Entrainment, Scaling Laws, Semantic Filtering, Pattern Copying, Robustness

## TL;DR
This paper establishes scaling laws for "contextual entrainment" for the first time, finding that larger models are more resistant to false information in semantic contexts (negative exponent) but more prone to copying irrelevant tokens in non-semantic contexts (positive exponent), revealing the opposing scaling behaviors of semantic filtering and mechanical copying functions.

## Background & Motivation

**Background**: LLMs increasingly rely on external contexts (RAG, user-provided documents), but contextual information can be noisy, irrelevant, or incorrect. Niu et al. (2025) formalized "contextual entrainment"—the tendency for models to boost the probability of tokens appearing in the context, regardless of their semantic relevance.

**Limitations of Prior Work**: Entrainment effects have been observed at single model scales, but their relationship with model size remains unknown. Traditional scaling laws describe aggregate loss, masking the evolution of specific behavioral mechanisms.

**Key Challenge**: Intuitively, larger models are "smarter" and should be more robust, but they are also "stronger pattern matchers" and might be more prone to copying context—which of these trends dominates?

**Goal**: Quantify how contextual entrainment changes with model size and establish behavioral scaling laws.

**Key Insight**: Categorize context into four types (Counterfactual, Related, Irrelevant, Random) and fit power laws $E(N) = a \cdot N^b$ separately to observe the divergence in exponent signs.

**Core Idea**: Entrainment effects in semantic and non-semantic contexts follow scaling laws in opposite directions—larger models are simultaneously "better" and "worse."

## Method

### Overall Architecture
Using the LRE (Linear Relational Embedding) dataset across two model families, Cerebras-GPT (111M-13B) and Pythia (410M-12B), the logit shift $\Delta_t = \text{logit}(t|\text{ctx}) - \text{logit}(t|\varnothing)$ is measured under four context conditions. Distractor and gold tokens are analyzed separately to fit power-law scaling relationships.

### Key Designs

1.  **Systematic Design of Four Context Conditions**:
    - **Function**: Isolate semantic-driven and mechanically-driven entrainment mechanisms.
    - **Mechanism**: For the same query (e.g., "The capital of Germany is ___", gold=Berlin), four contexts are constructed: Related ("The Eiffel Tower is in Paris", d=Paris), Irrelevant ("The water is warm", d=warm), Random ("Calculator", d=Calculator), and Counterfactual ("The capital of Germany is Munich", d=Munich), measuring $\Delta_d$ and $\Delta_g$ under each.
    - **Design Motivation**: Distinguish between "being influenced due to semantic understanding" and "copying because it was seen" by controlling contextual semantic relevance.

2.  **Power Law Scaling Fitting**:
    - **Function**: Quantify the exact relationship between entrainment effects and model size.
    - **Mechanism**: Perform linear regression in log-log space for each metric to fit $E(N) = a \cdot N^b$, reporting the exponent $b$, 95% confidence intervals, $R^2$, and p-values. Strong evidence is defined as $R^2 > 0.8$ and $p < 0.01$.
    - **Design Motivation**: Power laws are the standard form for neural scaling; using them for behavioral metrics enables quantitative prediction.

3.  **Baseline Verification and Control**:
    - **Function**: Exclude dataset spurious correlations and ensure observed trends stem from context manipulation.
    - **Mechanism**: Verify that gold token logits without context scale consistently across all four question subsets ($b \in [+0.129, +0.134]$, $R^2 > 0.93$), while distractor logits without context show no consistent scaling ($R^2 < 0.25$).
    - **Design Motivation**: If different question subsets had different baselines, scaling differences could not be attributed to context conditions.

### Loss & Training
This work is purely analytical and does not involve training.

## Key Experimental Results

### Main Results

| Context Type | Exponent $b$ ($\Delta_d$) | 95% CI | $R^2$ | Implication |
| :--- | :--- | :--- | :--- | :--- |
| Counterfactual | -0.330 | [-0.44, -0.22] | 0.926 | Larger models more resistant to false info |
| Related | -0.135 | [-0.16, -0.11] | 0.977 | Larger models more resistant to semantic interference |
| Irrelevant | +0.091 | [+0.05, +0.13] | 0.879 | Larger models more prone to irrelevant token influence |
| Random | +0.217 | [+0.14, +0.30] | 0.905 | Larger models more prone to copying random tokens |

### Ablation Study

| Metric | 111M → 13B Change | Description |
| :--- | :--- | :--- |
| Counterfactual $\Delta_d$ | 9.69 → 2.30 | 4x decrease, enhanced semantic filtering |
| Random $\Delta_d$ | 0.82 → 1.97 | 2.4x increase, enhanced copying mechanism |
| Related gap ($\Delta_g - \Delta_d$) | 5.71 → 0.55 | 10.3× convergence, improved semantic differentiation |
| Random gap | 0.73 → 2.18 | 3.0× divergence, intensified noise sensitivity |

### Key Findings
- The divergence of exponent signs between semantic and non-semantic contexts is replicated across both Cerebras-GPT and Pythia model families, indicating this is an inherent property of Transformer scaling.
- This represents a gradient rather than a binary split: from counterfactual (strongest negative scaling) to random (strongest positive scaling), aligning with semantic coherence.
- The convergence-divergence split implies that larger models are more sensitive to context quality—they benefit more from good context but are harmed more by poor context.

## Highlights & Insights
- **The core insight is exceptionally elegant**: The same phenomenon (contextual entrainment) exhibits opposite scaling behaviors depending on the semantic nature of the content, moving beyond the simple "bigger is better" narrative. The implication for RAG systems is that context curation becomes more, not less, important as models scale.
- **The opposing explanations for the two mechanisms** are compelling—pattern matching and semantic filtering are independently scaling functional modules, the former resembling induction heads and the latter resembling reasoning capabilities.
- This analytical method can be transferred to any research question regarding "how behavior changes with model size."

## Limitations & Future Work
- Studied only decoder-only Transformers; encoder-only and encoder-decoder architectures may have different entrainment dynamics.
- Scaling is analyzed only at the behavioral level without mechanistic decomposition (e.g., identifying specific attention heads responsible for each behavior).
- The LRE dataset primarily contains factual queries; entrainment effects may differ in more complex reasoning tasks.
- The impact of instruction tuning or RLHF on entrainment scaling was not explored.

## Related Work & Insights
- **vs Niu et al. (2025)**: They found entrainment effects to be pervasive at fixed scales; this paper extends this to the scaling dimension and discovers the sign split.
- **vs Kaplan et al. (2020)**: Traditional scaling laws describe the monotonic decrease of aggregate loss; this paper reveals that behavioral metrics can scale in opposite directions.
- **vs Wei et al. (2022)**: The emergent abilities paper focuses on "which abilities suddenly appear"; this paper quantifies "how existing behaviors change continuously."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First scaling law for contextual entrainment; the sign split discovery is highly novel and counter-intuitive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across two model families with complete statistical significance, though lacks instruction-tuned models.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Concise and elegant narrative structure; the "better and worse" contrast is maintained throughout.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement](../../ICLR2026/causal_inference/resisting_contextual_interference_in_rag_via_parametric-knowledge_reinforcement.md)
- [\[ICML 2026\] Investigating Memory in Model-Free RL with POPGym Arcade](../../ICML2026/causal_inference/investigating_memory_in_model-free_rl_with_popgym_arcade.md)
- [\[NeurIPS 2025\] Cyclic Counterfactuals under Shift–Scale Interventions](../../NeurIPS2025/causal_inference/cyclic_counterfactuals_under_shift-scale_interventions.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](../../AAAI2026/causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)
- [\[ICML 2026\] Density-Guided Robust Counterfactual Explanations on Tabular Data under Model Multiplicity](../../ICML2026/causal_inference/density-guided_robust_counterfactual_explanations_on_tabular_data_under_model_mu.md)

</div>

<!-- RELATED:END -->
