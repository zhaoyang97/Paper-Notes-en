---
title: >-
  [Paper Note] LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty
description: >-
  [CVPR 2025][LLM Safety][Machine Unlearning] This paper proposes LoTUS, which utilizes logit temperature adjustment and Gumbel-Softmax to smooth predictions of forgotten samples. By dynamically scheduling the temperature, it converges to the target where "forget set accuracy equals unseen set accuracy." This enables efficient unlearning on the large-scale ImageNet-1K benchmark (Avg Gap of 0.0150 on ViT). Furthermore, it introduces RF-JSD, a retraining-free evaluation metric (a…
tags:
  - "CVPR 2025"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Gumbel-Softmax"
  - "Dynamic Temperature"
  - "Large-Scale"
  - "Uncertainty"
date: 2026-05-08
content_hash: d6bcce3d8dd65747
---

# LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty

**Conference**: CVPR 2025  
**arXiv**: [2503.18314](https://arxiv.org/abs/2503.18314)  
**Code**: [https://github.com/cspartalis/LoTUS](https://github.com/cspartalis/LoTUS)  
**Area**: LLM Evaluation  
**Keywords**: Machine Unlearning, Gumbel-Softmax, Dynamic Temperature, Large-Scale, Uncertainty

## TL;DR

This paper proposes LoTUS, which utilizes logit temperature adjustment and Gumbel-Softmax to smooth predictions of forgotten samples. By dynamically scheduling the temperature, it converges to the target where "forget set accuracy equals unseen set accuracy." This enables efficient unlearning on the large-scale ImageNet-1K benchmark (Avg Gap of 0.0150 on ViT). Furthermore, it introduces RF-JSD, a retraining-free evaluation metric (achieving a Pearson correlation of 0.92 with the true JSD).

## Background & Motivation

### Background

**Background**: Machine unlearning requires models to "forget" specific training data. The ideal goal is to approximate retraining from scratch without actually retraining. Prior methods (such as NegGrad, SCRUB, and SalUn) are effective on small scales but infeasible at the ImageNet level.

### Limitations of Prior Work

**Limitations of Prior Work**: (1) Existing methods either fail to converge on large-scale datasets or require prolonged fine-tuning; (2) The gold standard for evaluating unlearning quality is retraining from scratch and calculating JSD, but retraining large models is computationally prohibitive, highlighting a lack of retraining-free evaluation metrics.

### Key Challenge

**Key Challenge**: Unlearning requires precise control: erasing too much degrades performance on the retain set, while erasing too little leaves residual knowledge. This balance is significantly harder to maintain in large-scale settings.

### Key Insight

**Key Insight**: An information-theoretic perspective—separating global information (to be retained) from subset-specific information (to be forgotten). Gumbel-Softmax introduces predictive diversity, while temperature scheduling dynamically controls the intensity of unlearning until the target accuracy is matched.

### Core Idea

**Core Idea**: Gumbel-Softmax + dynamic temperature $\to$ the forget set accuracy converges to the level of the unseen set, achieving information-theory-driven large-scale unlearning.

## Method

### Key Designs

1. **Gumbel-Softmax tempered loss**: $\ell = l \cdot gs(f_{orig}(x), \tau_d) \odot \log s(f_{un}(x)) + (1-l) \cdot gs(f_{orig}(x), \tau \to 0^+) \odot \log s(f_{un}(x))$—using high temperatures to soften labels for forgotten samples (introducing uncertainty) and low temperatures to maintain sharp predictions for retained samples.

2. **Dynamic temperature scheduling**: $\tau_d = \exp(\alpha \cdot (Acc(f_{un}, D_f) - Acc(f_{orig}, D_u)))$—the temperature adaptively adjusts based on the gap between forget and unseen accuracy to achieve automatic convergence.

3. **RF-JSD (Retraining-Free JSD)**: Computes an approximate JSD by randomizing feature subsets, achieving a Pearson correlation of 0.92 with the true JSD, thus enabling unlearning quality assessment without retraining.

### Loss & Training

ViT only requires 3 epochs, and ResNet18 requires 10 epochs. $\alpha=2$.

## Key Experimental Results

| Model/Dataset | LoTUS Avg Gap | LoTUS JSD | Time |
|------------|-------------|-----------|------|
| ViT/TinyImageNet | **0.0150** | 0.03e-4 | 13.41min |
| ViT/CIFAR-100 | **0.0125** | 0.04e-4 | 7.02min |
| ImageNet-1K | Evaluated by RF-JSD | — | — |

Outperforms 8 baseline methods (such as NegGrad+, SCRUB, SalUn, etc.).

### Ablation Study
- Gumbel-Softmax > plain Softmax: Sampling noise introduced by Gumbel breaks model memorization.
- Temperature scheduling is the key to convergence: a fixed temperature fails to balance unlearning and retention.
- RF-JSD vs. JSD PCC = 0.92±0.04: proving that retraining-free evaluation is feasible.

### Key Findings
- Dynamic temperature automatically finds the equilibrium point to achieve "forgetting to the level of unseen data."
- 3 epochs are sufficient (for ViT), demonstrating high efficiency.
- RF-JSD makes large-scale unlearning evaluation feasible.

## Highlights & Insights
- **Information-theory-driven unlearning target**: Setting the target as "forget set accuracy = unseen set accuracy" is both elegant and actionable.
- **Practical value of RF-JSD**: It breaks the limitation of "requiring retraining for evaluation."

## Limitations & Future Work
- Assumes instance-level unlearning (requires modification for class-level unlearning).
- Requires an unseen set that shares a similar distribution with the forget set.
- Limited to classification tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of Gumbel-Softmax + dynamic temperature is simple and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale ImageNet + 8 baselines + RF-JSD.
- Writing Quality: ⭐⭐⭐⭐ Clear information-theoretic motivation.
- Value: ⭐⭐⭐⭐ The first unlearning method scalable to ImageNet.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](../../NeurIPS2025/llm_safety/simu_selective_influence_machine_unlearning.md)
- [\[ACL 2025\] MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models](../../ACL2025/llm_safety/mmunlearner_reformulating_multimodal_machine_unlearning_in_the_era_of_multimodal.md)
- [\[ICCV 2025\] MUNBa: Machine Unlearning via Nash Bargaining](../../ICCV2025/llm_safety/munba_machine_unlearning_via_nash_bargaining.md)
- [\[ICML 2025\] NegMerge: Sign-Consensual Weight Merging for Machine Unlearning](../../ICML2025/llm_safety/negmerge_sign-consensual_weight_merging_for_machine_unlearning.md)
- [\[NeurIPS 2025\] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](../../NeurIPS2025/llm_safety/a_reliable_cryptographic_framework_for_empirical_machine_unl.md)

</div>

<!-- RELATED:END -->
