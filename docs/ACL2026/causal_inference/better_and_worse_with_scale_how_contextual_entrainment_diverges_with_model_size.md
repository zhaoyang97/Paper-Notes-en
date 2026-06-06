---
title: >-
  [Paper Note] Better and Worse with Scale: How Contextual Entrainment Diverges with Model Size
description: >-
  [ACL 2026][Causal Inference][Contextual Entrainment] This paper establishes the first scaling laws for "contextual entrainment," discovering that larger models better resist misinformation in semantic contexts (negative…
tags:
  - "ACL 2026"
  - "Causal Inference"
  - "Contextual Entrainment"
  - "Scaling Laws"
  - "Semantic Filtering"
  - "Pattern Copying"
  - "Robustness"
content_hash: 741b724e343afba9
---

# Better and Worse with Scale: How Contextual Entrainment Diverges with Model Size

**Conference**: ACL 2026
**arXiv**: [2604.13275](https://arxiv.org/abs/2604.13275)  
**Code**: N/A  
**Area**: Causal Inference
**Keywords**: Contextual Entrainment, Scaling Laws, Semantic Filtering, Pattern Copying, Robustness

## TL;DR
This paper establishes the first scaling laws for "contextual entrainment," discovering that larger models better resist misinformation in semantic contexts (negative exponent) but more readily copy irrelevant tokens in non-semantic contexts (positive exponent), revealing opposing scaling behaviors of semantic filtering and mechanical copying functions.

## Method

### Key Designs

1. **Four Context Conditions**: Counterfactual, Related, Irrelevant, Random — systematically separating semantically-driven and mechanically-driven entrainment mechanisms.

2. **Power-Law Scaling Fits**: $E(N) = a \cdot N^b$ in log-log space with $R^2 > 0.8$ and $p < 0.01$ as strong evidence criteria.

## Key Experimental Results

| Context Type | Exponent $b$ (Δ_dstr) | $R^2$ | Implication |
|-------------|----------------------|-------|-------------|
| Counterfactual | -0.330 | 0.926 | Larger models resist misinformation better |
| Related | -0.135 | 0.977 | Larger models resist semantic distractors better |
| Irrelevant | +0.091 | 0.879 | Larger models copy irrelevant tokens more |
| Random | +0.217 | 0.905 | Larger models copy random tokens more |

Sign split replicated across both Cerebras-GPT and Pythia model families.

## Highlights & Insights
- Core insight is elegant: the same phenomenon (entrainment) exhibits opposing scaling based on content semantic nature, transcending the "bigger is better" narrative
- For RAG systems: the larger the model, the more important context quality curation becomes
- The analysis methodology transfers to any "how does behavior change with model size" research question

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement](../../ICLR2026/causal_inference/resisting_contextual_interference_in_rag_via_parametric-knowledge_reinforcement.md)
- [\[ICLR 2026\] Copy-Paste to Mitigate Large Language Model Hallucinations](../../ICLR2026/causal_inference/copy-paste_to_mitigate_large_language_model_hallucinations.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](../../AAAI2026/causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)
- [\[NeurIPS 2025\] Cyclic Counterfactuals under Shift–Scale Interventions](../../NeurIPS2025/causal_inference/cyclic_counterfactuals_under_shift-scale_interventions.md)
- [\[NeurIPS 2025\] Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization](../../NeurIPS2025/causal_inference/bi-level_decision-focused_causal_learning_for_large-scale_marketing_optimization.md)

</div>

<!-- RELATED:END -->
