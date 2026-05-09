---
title: >-
  [Paper Note] Reference-Guided Machine Unlearning
description: >-
  [ICLR 2026][Model Compression][Machine Unlearning] This paper proposes ReGUn (Reference-Guided Unlearning), which leverages an independent held-out dataset as a reference standard for "unseen behavior." Through class-conditional distillation, the model's behavior on forget data is aligned to that on truly unseen data, achieving a superior forgetting–utility trade-off.
tags:
  - ICLR 2026
  - Model Compression
  - Machine Unlearning
  - Reference-Guided
  - Knowledge Distillation
  - Distributional Indistinguishability
  - Privacy Protection
date: 2026-05-08
content_hash: dafc38db92e73774
---

# Reference-Guided Machine Unlearning

**Conference**: ICLR 2026
**arXiv**: [2603.11210](https://arxiv.org/abs/2603.11210)
**Code**: [GitHub](https://github.com/jmirlach/ReGUn)
**Area**: Model Compression / Machine Unlearning
**Keywords**: Machine Unlearning, Reference-Guided, Knowledge Distillation, Distributional Indistinguishability, Privacy Protection

## TL;DR

This paper proposes ReGUn (Reference-Guided Unlearning), which leverages an independent held-out dataset as a reference standard for "unseen behavior." Through class-conditional distillation, the model's behavior on forget data is aligned to that on truly unseen data, achieving a superior forgetting–utility trade-off.

## Background & Motivation

Machine Unlearning aims to remove the influence of specific data from a trained model while preserving general performance — serving as the technical foundation for the "right to be forgotten" under privacy regulations such as GDPR.

**Core Problem**: Existing approximate unlearning methods rely on performance-degradation heuristics (e.g., loss maximization, random labels), which suffer from fundamental flaws:
- **Ill-conditioning**: May produce large or misdirected gradients
- **Generalization damage**: Alters decision boundaries beyond the intended scope
- **Optimization conflict**: Forgetting and stability objectives contradict each other

**Key Insight**: Unlearning should not merely make the model "more wrong"; rather, it should make the model's behavior on forget data indistinguishable from its behavior on truly unseen data.

## Method

### Overall Architecture

ReGUn consists of two core components: **Reference Distribution Construction** (RefDist) and **Unlearning Objective Optimization**.

### 1. Reference Distribution Construction

Given a forget minibatch $B_f = \{(x_i^f, y_i^f)\}_{i=1}^b$, $m$ samples are drawn from the held-out set $\mathcal{D}_h$ via class-histogram matching, and the reference model outputs are aggregated:

$$q(B_f) = \frac{1}{m} \sum_{j=1}^{m} p_\phi(\cdot | \tilde{x}_j)$$

Key designs:
- The reference model uses the initial model $f_{\theta_0}$ (avoiding additional training and reference drift)
- Class-histogram matching controls label prior discrepancy
- All forget samples within the same batch share the same reference distribution

### 2. Unlearning Objective

$$\mathcal{L}(\theta; B_f, B_r) = \lambda_f \frac{1}{|B_f|} \sum_{(x,\cdot) \in B_f} \text{KL}(q(B_f) \| p_\theta(\cdot|x)) + \lambda_r \frac{1}{|B_r|} \sum_{(x,y) \in B_r} \text{CE}(p_\theta(\cdot|x), y)$$

- **Forget term** (KL divergence): Distills predictions on forget samples toward the held-out reference distribution
- **Retain term** (cross-entropy): Anchors updates to preserve performance on retain data
- $\lambda_f, \lambda_r > 0$ control the trade-off between forgetting strength and retain utility

### 3. Data Partitioning

From the original training set $\mathcal{D}_{orig}$:
- 10% is held out as $\mathcal{D}_h$ (used exclusively during the unlearning phase)
- The remainder forms $\mathcal{D}_{train}$, from which the forget set $\mathcal{D}_f$ and validation set $\mathcal{D}_{val}$ are sampled

## Key Experimental Results

### Main Results: ResNet-18 on CIFAR-10 (Forget Ratios 1% / 10% / 50%)

| Method | Forget 1% TestAcc | Forget 1% RMIA_AUC | Forget 10% Gap_Avg | Forget 50% Gap_Avg |
|------|-----------|------------|-------------|-------------|
| Retrain (Oracle) | 94.34 | 49.98 | 0.00 | 0.00 |
| NegGrad | **94.17** | 59.80 | 3.82 | 4.80 |
| Finetune | 90.90 | 54.78 | 2.79 | 2.39 |
| SalUn | 91.63 | **50.09** | 2.48 | 2.00 |
| Amun | 91.84 | 44.17 | **1.46** | — |
| **ReGUn** | 91.98 | 51.35 | 1.49 | **1.55** |

### Ablation Study: Overall Performance Under Different Forget Ratios (GapAvg↓)

| Method | CIFAR-10 1% | CIFAR-10 10% | CIFAR-10 50% | CIFAR-100 |
|------|-------------|--------------|--------------|-----------|
| NegGrad+ | 3.77 | 3.71 | 2.62 | — |
| ℓ1-sparse | 2.73 | 2.49 | 2.09 | — |
| SalUn | 1.64 | 2.48 | 2.00 | — |
| **ReGUn** | **1.49** | **1.49** | **1.55** | — |

**Key Findings**: ReGUn is particularly strong at large forget ratios (50%), achieving the lowest aggregate deviation GapAvg, demonstrating that the reference-guided approach is more stable in large-scale unlearning scenarios.

## Highlights & Insights

1. **Paradigm Shift**: Moves from "making the model more wrong" to "making the model behave as if it had never seen the data," introducing a distributional indistinguishability perspective
2. **Simplicity and Elegance**: Requires only a held-out dataset and KL distillation, with no need for complex repair mechanisms or constrained parameter editing
3. **Class-Conditional Reference**: Achieves instance-level/class-conditional referencing via histogram matching, outperforming global distribution matching
4. **Cross-Architecture Validation**: Performs consistently well on both CNNs (ResNet-18) and Transformers (Swin-T)

## Limitations & Future Work

- Requires an additional held-out dataset (10% of original data), which may be infeasible in data-scarce settings
- The reference model uses the initial model $f_{\theta_0}$, which still retains the influence of forget data (a non-ideal reference)
- Evaluation is limited to random forgetting; settings such as class-wise forgetting remain unexplored
- Membership inference attack evaluation uses offline RMIA, which may underestimate actual privacy risks

## Related Work & Insights

- **Baseline Unlearning Methods**: Finetune, NegGrad, NegGrad+ — simple but limited in effectiveness
- **Constrained Unlearning**: SalUn (saliency-guided), SSD (Fisher information), Amun — introduce restriction mechanisms
- **Reference-Based Methods**: Pseudo-probability replacement, third-party distribution matching — lack instance-level conditional control
- **Exact Unlearning**: SISA and others — computationally expensive but provide exact guarantees

## Rating

| Dimension | Score | Notes |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | Distributional indistinguishability perspective is novel; reference-guided approach is well-motivated |
| Practicality | ⭐⭐⭐⭐ | Method is simple and general, but requires additional held-out data |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Evaluated across multiple architectures, forget ratios, and metrics |
| Writing Quality | ⭐⭐⭐⭐ | Problem formulation is clear; method derivation is rigorous |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](../../ACL2026/model_compression/representation-guided_parameter-efficient_llm_unlearning.md)
- [\[AAAI 2026\] ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs](../../AAAI2026/model_compression/alter_asymmetric_lora_for_token-entropy-guided_unlearning_of.md)
- [\[ICLR 2026\] STAR: Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models](star_similarity-guided_teacher-assisted_refinement_for_super-tiny_function_calli.md)
- [\[NeurIPS 2025\] Distillation Robustifies Unlearning](../../NeurIPS2025/model_compression/distillation_robustifies_unlearning.md)
- [\[ICLR 2026\] KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models](kbvq-moe_klt-guided_svd_with_bias-corrected_vector_quantization_for_moe_large_la.md)

</div>

<!-- RELATED:END -->
