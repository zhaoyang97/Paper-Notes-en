---
title: >-
  [Paper Note] Token-Importance Guided Direct Preference Optimization (TI-DPO)
description: >-
  [ICLR 2026][LLM Alignment][token-level DPO] TI-DPO is proposed, which precisely quantifies each token's contribution to preference via a hybrid weighting mechanism combining gradient attribution and a Gaussian prior…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "token-level DPO"
  - "gradient attribution"
  - "hybrid weighting"
  - "triplet loss"
  - "fine-grained alignment"
date: 2026-05-08
content_hash: bb8f643a1689282f
---

# Token-Importance Guided Direct Preference Optimization (TI-DPO)

**Conference**: ICLR 2026
**arXiv**: [2505.19653](https://arxiv.org/abs/2505.19653)  
**Code**: [https://github.com/gracefulning/TIDPO](https://github.com/gracefulning/TIDPO)  
**Area**: Alignment RLHF / DPO
**Keywords**: token-level DPO, gradient attribution, hybrid weighting, triplet loss, fine-grained alignment

## TL;DR
TI-DPO is proposed, which precisely quantifies each token's contribution to preference via a hybrid weighting mechanism combining gradient attribution and a Gaussian prior, and incorporates a triplet loss to guide optimization in a continuous semantic space. The method achieves state-of-the-art performance with an average score of 62.3 across 6 benchmarks, while providing interpretable token-level control.

## Background & Motivation

**Background**: DPO optimizes preferences at the sequence level, ignoring the differential importance of individual tokens. Existing token-level methods (TDPO/TIS-DPO) assess importance via probability surrogates, which introduce bias.

**Limitations of Prior Work**:
   - DPO's coarse-grained optimization is sensitive to data noise and suffers from severe distribution shift
   - Probability surrogates in existing token-level methods yield inconsistent outputs
   - The binary "good/bad" contrastive framework cannot finely adjust generation behavior in a continuous semantic space

**Key Challenge**: Simultaneously identifying critical tokens precisely and guiding preference adjustment in a continuous space are both required.

**Core Idea**: Gradient attribution to localize critical tokens + Gaussian prior to correct positional bias + triplet loss for continuous-space guidance

## Method

### Overall Architecture
$\mathcal{L}_{\text{TI-DPO}} = \mathcal{L}_{\text{DPO-w}} + \gamma \mathcal{L}_{\text{triplet}}$

### Key Designs

1. **Hybrid Weighting Mechanism**:

    - Gradient attribution: $I_i = \|\nabla_{e_i}\mathcal{L}_{\text{target}}\|_1$, computing each token embedding's gradient contribution to the final prediction
    - Gaussian prior: $\mathcal{P}(t) = \exp(-\frac{1}{2}(\frac{t-\mu}{\sigma})^2)$, where $\mu=(T-1)/2$, $\sigma=T/4$, correcting the model's U-shaped attention bias (over-attention to leading and trailing tokens)
    - Convex combination: $W = \lambda \cdot \mathcal{I}_{\text{norm}} + (1-\lambda) \cdot \mathcal{P}$
    - Weights are computed independently for $y_w$ and $y_l$

2. **Weighted Token-Level DPO**:

    - $\Delta r_{\text{token}} = \sum_t w_t^w \log\frac{\pi_\theta(y_w^t|\cdot)}{\pi_{\text{ref}}(y_w^t|\cdot)} - \sum_t w_t^l \log\frac{\pi_\theta(y_l^t|\cdot)}{\pi_{\text{ref}}(y_l^t|\cdot)}$
    - The contributions of critical tokens are amplified while noisy tokens are suppressed

3. **Triplet Loss**:

    - An anchor response $y$ is generated from the policy model; in the implicit reward space, $y$ is pulled closer to $y_w$ and pushed away from $y_l$
    - $\mathcal{L}_{\text{triplet}} = \max(0, d(y, y_w) - d(y, y_l) + \alpha)$
    - Provides finer-grained guidance than binary contrastive learning in a continuous semantic space

## Key Experimental Results

### Main Results (Average over 3 Models)

| Method | MMLU | GSM8K | HumanEval | TruthfulQA | IFEval | Avg |
|--------|------|-------|-----------|------------|--------|-----|
| DPO | 65.3 | 69.3 | 61.0 | 56.7 | 70.0 | 57.7 |
| SimPO | 63.5 | 64.7 | 58.2 | 54.2 | 64.7 | 54.5 |
| GRPO | 70.7 | **75.7** | 64.3 | 59.9 | 74.0 | 62.1 |
| **TI-DPO** | 70.0 | 73.0 | **67.0** | **62.0** | **75.7** | **62.3** |

### Ablation Study (Llama-3.2-3B)

| Configuration | General | Math | Code | Reliability |
|---------------|---------|------|------|-------------|
| Full TI-DPO | **65.4** | **80.7** | **33.0** | **86.8** |
| w/o triplet loss | 64.0 | 79.0 | 31.0 | 83.0 |
| Uniform weights | 64.0 | 78.2 | 29.0 | 80.0 |
| w/o Gaussian prior | 64.5 | 79.7 | 31.5 | 82.5 |

### Key Findings
- **TI-DPO matches GRPO**: average 62.3 vs. 62.1, with TI-DPO leading on HumanEval (67 vs. 64.3) and IFEval (75.7 vs. 74)
- **Weight distribution adapts to task**: weights concentrate in [0.2, 0.5] for math tasks (few critical symbols) and shift toward [0.6, 0.8] for safety tasks (requiring comprehensive attention)
- **Noise robustness**: TI-DPO exhibits the least performance degradation as label noise increases
- **Interpretability**: token-level weights can be visualized; e.g., in medical scenarios "medical attention" receives high weight while "painkillers" is down-weighted

## Highlights & Insights
- **Complementary design of gradient attribution and positional prior**: gradient attribution captures semantic importance but suffers from positional bias, while the Gaussian prior corrects this bias — the two are mutually complementary
- **Triplet loss breaks the binary framework**: extends from "good/bad" contrast to continuous-space guidance that aligns with positive samples and distances from negative samples
- **Interpretable token-level control**: beyond performance gains, critical tokens can be visualized — offering direct value for safety auditing

## Limitations & Future Work
- **Computational overhead**: gradient attribution requires additional forward and backward passes
- **Gaussian prior assumption**: assumes important tokens are uniformly distributed within the sequence, which may not hold for certain tasks
- **Future direction**: could integrate Uni-DPO's quality weighting to enable dual-level dynamic re-weighting at both the data level and token level

## Related Work & Insights
- **vs. TDPO/TIS-DPO**: probability surrogates introduce bias; TI-DPO achieves greater accuracy through gradient attribution combined with a Gaussian prior
- **vs. Uni-DPO**: Uni-DPO re-weights at the data level, while TI-DPO re-weights at the token level — the two are orthogonal and can be combined
- **vs. GRPO**: GRPO leverages RL-based exploration, whereas TI-DPO refines supervised signals at the token level — comparable performance achieved through fundamentally different mechanisms

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of hybrid weighting mechanism and triplet loss is elegantly designed
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 benchmarks × 3 models × comprehensive ablation and noise experiments
- Writing Quality: ⭐⭐⭐⭐ Theoretical motivation is clearly articulated
- Value: ⭐⭐⭐⭐ A practical improvement to token-level DPO; interpretability serves as a distinctive advantage

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Is On-Policy Data always the Best Choice for Direct Preference Optimization-based LM Alignment?](is_on-policy_data_always_the_best_choice_for_direct_preference_optimization-base.md)
- [\[ICLR 2026\] SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](safedpo_preference_optimization_safety.md)
- [\[ICLR 2026\] Swap-guided Preference Learning for Personalized RLHF (SPL)](swap-guided_preference_learning_for_personalized_reinforcement_learning_from_hum.md)
- [\[ACL 2026\] S2H-DPO: Hardness-Aware Preference Optimization for Vision-Language Models](../../ACL2026/llm_alignment/s2h-dpo_hardness-aware_preference_optimization_for_vision-language_models.md)
- [\[ICLR 2026\] Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences](displacement-resistant_extensions_of_dpo_with_nonconvex_f-divergences.md)

</div>

<!-- RELATED:END -->
