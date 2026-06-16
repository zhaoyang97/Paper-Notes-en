---
title: >-
  [Paper Note] Better and Worse with Scale: How Contextual Entrainment Diverges with Model Size
description: >-
  [ACL 2026][Causal Inference][Paper Note] This paper establishes the first scaling laws for the "contextual entrainment effect," discovering that larger models are more resistant to false information in semantic contexts (negative exponent) but more prone to copying irrelevant tokens in non-semantic contexts (positive exponent), revealing a divergence in the s
tags:
  - ACL 2026
  - Causal Inference
date: 2026-05-08
content_hash: 81aaa0fa8a57c875
---
# Better and Worse with Scale: How Contextual Entrainment Diverges with Model Size

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.13275](https://arxiv.org/abs/2604.13275)  
**Code**: None  
**Area**: Causal Inference  
**Keywords**: Contextual Entrainment Effect, Scaling Laws, Semantic Filtering, Pattern Copying, Robustness

## TL;DR
This paper establishes the first scaling laws for the "contextual entrainment effect," discovering that larger models are more resistant to false information in semantic contexts (negative exponent) but more prone to copying irrelevant tokens in non-semantic contexts (positive exponent), revealing a divergence in the scaling of semantic filtering versus mechanical copying functions.

## Background & Motivation

**Background**: LLMs increasingly rely on external context (e.g., RAG, user-provided documents), but contextual information may be noisy, irrelevant, or incorrect. Niu et al. (2025) formalized "contextual entrainment"—the tendency of models to increase the probability of tokens appearing in the context regardless of their semantic relevance.

**Limitations of Prior Work**: Entrainment has been observed at single model scales, but its relationship with model size remains unknown. Traditional scaling laws describe aggregate loss, which masks the evolution of specific behavioral mechanisms.

**Key Challenge**: Intuitively, larger models should be "smarter" and thus more robust. However, larger models are also "stronger pattern matchers," which might make them more prone to copying context. Which of these two trends dominates?

**Goal**: Quantify how the contextual entrainment effect changes with model size and establish behavioral scaling laws.

**Key Insight**: Categorize context into four types (Counterfactual, Related, Irrelevant, Random) and fit power laws $E(N) = a \cdot N^b$ separately to observe the split in the exponent signs.

**Core Idea**: Entrainment effects for semantic and non-semantic contexts follow scaling laws in opposite directions—larger models simultaneously become "better" and "worse."

## Method

### Overall Architecture

This study is a pure measurement work aimed at answering whether the contextual entrainment effect strengthens or weakens as models scale. The approach involves using the LRE (Linear Relational Embedding) factual query dataset, where each query is paired with four types of context with different semantic properties. Logits for each token are extracted from two complete model families, Cerebras-GPT (111M–13B) and Pythia (410M–12B). The shift in logit with and without context is computed as $\Delta_t = \text{logit}(t|\text{ctx}) - \text{logit}(t|\varnothing)$. Statistics are collected for both distractor and gold tokens. Finally, these behavioral metrics are fitted against model size $N$ using power laws to observe how exponent signs split by context type.

### Key Designs

**1. Four Context Conditions: Decoupling Semantic Drive from Mechanical Copying**

Entrainment may stem from two distinct mechanisms: the model being biased "because it understands the semantics" or "simply because it copies what it sees." To isolate these, semantic relevance must be controlled. For the same query (e.g., "The capital of Germany is ___", gold=Berlin), four contexts are constructed: Counterfactual ("The capital of Germany is Munich", d=Munich, direct semantic conflict), Related ("The Eiffel Tower is in Paris", d=Paris, semantically related but non-conflicting), Irrelevant ("The water is warm", d=warm, semantically unrelated), and Random ("Calculator", d=Calculator, purely random token). Metrics $\Delta_d$ for distractors and $\Delta_g$ for gold tokens are measured under each condition. This spectrum from "strong semantics" to "no semantics" serves as the experimental lever for observing the sign split.

**2. Power Law Fitting: Quantifying Behavior as Scaling Laws**

To precisely characterize the relationship between entrainment and model size, this paper adopts the standard neural scaling form. Linear regression is performed in log-log space for each behavioral metric to fit $E(N) = a \cdot N^b$, reporting the exponent $b$, 95% confidence intervals, $R^2$, and p-values. A threshold of $R^2 > 0.8$ and $p < 0.01$ is used as "strong evidence." The sign of the exponent $b$ itself constitutes the conclusion: $b < 0$ indicates the effect decays as the model grows, while $b > 0$ indicates it strengthens.

**3. Baseline Validation and Control: Excluding Dataset Spurious Correlations**

To attribute observed scaling trends to context manipulation, it must be shown that these trends are not inherent to data subset differences. The paper validates that without context, gold token logits scale consistently across all four subsets ($b \in [+0.129, +0.134]$, $R^2 > 0.93$), whereas distractor tokens without context show no consistent scaling ($R^2 < 0.25$). This ensures that the baselines for all subsets are aligned and that distractor scaling is purely introduced by the context.

## Key Experimental Results

### Main Results

| Context Type | Exponent $b$ ($\Delta_d$) | 95% CI | $R^2$ | Implication |
| :--- | :--- | :--- | :--- | :--- |
| Counterfactual | -0.330 | [-0.44, -0.22] | 0.926 | Larger models resist misinformation better |
| Related | -0.135 | [-0.16, -0.11] | 0.977 | Larger models resist semantic interference better |
| Irrelevant | +0.091 | [+0.05, +0.13] | 0.879 | Larger models are more easily influenced by irrelevant tokens |
| Random | +0.217 | [+0.14, +0.30] | 0.905 | Larger models copy random tokens more easily |

### Ablation Study

| Metric | 111M → 13B Change | Description |
| :--- | :--- | :--- |
| Counterfactual $\Delta_d$ | 9.69 → 2.30 | 4x decrease; enhanced semantic filtering |
| Random $\Delta_d$ | 0.82 → 1.97 | 2.4x increase; enhanced copying mechanism |
| Related gap ($\Delta_g - \Delta_d$) | 5.71 → 0.55 | 10.3× convergence; improved semantic differentiation |
| Random gap | 0.73 → 2.18 | 3.0× divergence; heightened noise sensitivity |

### Key Findings
- The sign split in exponents between semantic and non-semantic contexts is replicated across two independently trained model families (Cerebras-GPT and Pythia), suggesting it is an inherent property of Transformer scaling.
- This is a gradient rather than a binary split: from Counterfactual (strongest negative scaling) to Random (strongest positive scaling), aligning with semantic coherence.
- The convergence-divergence split implies that larger models are more sensitive to context quality—they benefit more from good context but are harmed more by poor context.

## Highlights & Insights
- **The core insight is remarkably elegant**: The same phenomenon (contextual entrainment) exhibits opposite scaling behaviors depending on the semantic nature of the content, moving beyond the simple narrative that "larger models are better." The implication for RAG systems is that as models scale, context curation becomes more important, not less.
- **The dual-mechanism explanation** is compelling: Pattern matching and semantic filtering act as independent functional modules that scale differently, with the former resembling induction heads and the latter reflecting reasoning capabilities.
- The analysis method is transferable to any research question regarding how specific behaviors change with model size.

## Limitations & Future Work
- Focuses only on decoder-only Transformers; encoder-only or encoder-decoder architectures may exhibit different entrainment dynamics.
- Conducts scaling at the behavioral level without mechanistic decomposition (e.g., identifying which specific attention heads are responsible for which behavior).
- The LRE dataset primarily contains factual queries; entrainment effects may differ in complex reasoning tasks.
- Does not explore the impact of instruction tuning or RLHF on entrainment scaling.

## Related Work & Insights
- **vs. Niu et al. (2025)**: They identified the prevalence of entrainment at a fixed scale; this work extends it to the scaling dimension and discovers the sign split.
- **vs. Kaplan et al. (2020)**: Traditional scaling laws describe the monotonic decrease of aggregate loss; this work reveals that behaviors can scale in opposite directions.
- **vs. Wei et al. (2022)**: While the emergent abilities paper focuses on "which abilities suddenly appear," this work quantifies "how existing behaviors change continuously."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First scaling law for contextual entrainment; the sign split discovery is highly novel and counter-intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across two model families with full statistical significance, though lacks instruction-tuned models.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative structure is refined and elegant, with the "better and worse" contrast well-integrated throughout.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ICLR 2026\] Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement](../../ICLR2026/causal_inference/resisting_contextual_interference_in_rag_via_parametric-knowledge_reinforcement.md)
- [\[NeurIPS 2025\] Cyclic Counterfactuals under Shift–Scale Interventions](../../NeurIPS2025/causal_inference/cyclic_counterfactuals_under_shift-scale_interventions.md)
- [\[ICML 2026\] Investigating Memory in Model-Free RL with POPGym Arcade](../../ICML2026/causal_inference/investigating_memory_in_model-free_rl_with_popgym_arcade.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](../../AAAI2026/causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)
- [\[ICML 2026\] Density-Guided Robust Counterfactual Explanations on Tabular Data under Model Multiplicity](../../ICML2026/causal_inference/density-guided_robust_counterfactual_explanations_on_tabular_data_under_model_mu.md)

</div>

<!-- RELATED:END -->
