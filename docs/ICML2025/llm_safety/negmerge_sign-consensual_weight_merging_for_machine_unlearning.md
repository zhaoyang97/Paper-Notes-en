---
title: >-
  [Paper Note] NegMerge: Sign-Consensual Weight Merging for Machine Unlearning
description: >-
  [ICML2025][LLM Safety][Machine Unlearning] Proposes NegMerge, which constructs a more effective unlearning vector by merging task vectors from multiple models fine-tuned with different hyperparameters and retaining only sign-consistent weight elements, achieving SOTA unlearning performance in both zero-shot and standard classification scenarios.
tags:
  - "ICML2025"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Task Arithmetic"
  - "Model Merging"
  - "Weight Sign Agreement"
  - "CLIP"
date: 2026-05-08
content_hash: 4601bcf04ccf9071
---

# NegMerge: Sign-Consensual Weight Merging for Machine Unlearning

**Conference**: ICML2025  
**arXiv**: [2410.05583](https://arxiv.org/abs/2410.05583)  
**Code**: [naver-ai/negmerge](https://github.com/naver-ai/negmerge)  
**Area**: Machine Unlearning  
**Keywords**: Machine Unlearning, Task Arithmetic, Model Merging, Weight Sign Agreement, CLIP

## TL;DR

Proposes NegMerge, which constructs a more effective unlearning vector by merging task vectors from multiple models fine-tuned with different hyperparameters and retaining only sign-consistent weight elements, achieving SOTA unlearning performance in both zero-shot and standard classification scenarios.

## Background & Motivation

### Problem Scenario

The "Right to be Forgotten" requires models to erase the influence of specific user data. Since retraining from scratch is prohibitively expensive, machine unlearning aims to selectively remove specific knowledge learned by the model without full retraining.

### Limitations of Prior Work

Task Arithmetic is currently a representative method: it first fine-tunes the pre-trained model on the forget set to obtain a fine-tuned model $\theta_{ft}$, computes the task vector $\tau = \theta_{ft} - \theta_{pre}$, and then subtracts $\tau$ from the pre-trained model to achieve unlearning. However, this method faces two core issues:

**Hyperparameter Sensitivity**: Unlearning performance is highly sensitive to fine-tuning hyperparameters (such as learning rate), with forget set accuracy swinging by up to **15 percentage points**. Heavy validation is required to find a suitable model.

**Unlearning-Retaining Trade-off**: Hyperparameter configurations that yield good unlearning on the forget set often severely degrade retain set performance, and vice versa. Selecting a single model cannot satisfy both needs simultaneously.

### Core Idea

Since a large number of fine-tuned models (typically 10–30) have already been generated during the validation process, why select only one and discard the rest? The core idea of NegMerge is to **utilize all candidate models** by merging task vectors based on sign consensus, synthesizing information from multiple models to construct a superior unlearning vector.

## Method

### Overall Architecture

NegMerge consists of three steps:

**Step 1: Compute diverse task vectors.** Fine-tune the pre-trained model on the forget set using different hyperparameters (learning rate, data augmentation, etc.) to obtain $n$ fine-tuned models $\{\theta_{ft}^{(k)}\}_{k=1}^n$, corresponding to $n$ task vectors:

$$\tau_k = \theta_{ft}^{(k)} - \theta_{pre}, \quad k = 1, \ldots, n$$

**Step 2: Sign consensus filtering.** Analyze the element-wise signs of all task vectors. The core hypothesis is:

- **Sign-consistent elements** $\rightarrow$ Strongly correlated with forget set knowledge, since their directions remain consistent regardless of hyperparameter variations.
- **Sign-inconsistent elements** $\rightarrow$ More likely to be noise introduced by different training configurations, with a weaker relationship to the forget set.

**Step 3: Merging and unlearning.** The final task vector is computed as:

$$\tau_{\text{merged}} = \frac{1}{n} \sum_{k=1}^{n} \left( \tau_k \odot \mathbf{1}_{\text{sign-consistent}} \right)$$

where $\odot$ denotes the Hadamard product (element-wise multiplication), and $\mathbf{1}_{\text{sign-consistent}}$ is a binary mask: it equals 1 when all $\tau_k$ have consistent signs at that position, and 0 otherwise. Finally, unlearning is executed via negation on the pre-trained model using the merged vector:

$$\theta_{\text{new}} = \theta_{pre} - \lambda \cdot \tau_{\text{merged}}$$

### Computational Efficiency Analysis

| Metric | NegMerge Advantages |
|------|-------------|
| Inference Complexity | $O(m)$ vs traditional $O(mn)$, needing to search for $\lambda$ on only a _single_ merged vector |
| Storage Overhead | Dynamically updates masks without needing to store all fine-tuned models |
| Runtime Memory | 90–95% of weights are zeroed out after merging, which is highly sparse and can be accelerated using lookup tables |
| Merging Time | 37 seconds (30 models), situated between Uniform (12s) and TIES (128s) |

## Key Experimental Results

### Experiment 1: Zero-shot Unlearning in CLIP

Unlearning domain-specific knowledge across 8 datasets while preserving ImageNet performance.

| Method | ViT-B/32 Forget↓ | ViT-B/32 Retain | ViT-B/16 Forget↓ | ViT-L/14 Forget↓ |
|------|:-:|:-:|:-:|:-:|
| Pre-trained | 48.13 | 63.33 | 55.49 | 65.19 |
| Task Arithmetic (best) | 23.63 | 60.60 | 20.64 | 19.17 |
| Uniform Merge | 22.50 | 60.55 | 21.51 | 18.10 |
| TIES-Merging | 26.21 | 61.08 | 23.78 | 22.70 |
| MagMax | 25.24 | 60.95 | 24.45 | 21.71 |
| **NegMerge** | **20.76** | 60.36 | **19.24** | **17.32** |

NegMerge achieves the lowest forget set accuracy (i.e., the best unlearning performance) across all backbones, while maintaining retain set accuracy comparable to other methods.

It also leads in the Linear Task Arithmetic scenario: forget set accuracy on ViT-B/32 drops to **8.03%** (compared to 8.88% for the best Task Arithmetic).

### Experiment 2: Standard Classifier Unlearning (CIFAR-10, ResNet-18, 10% Random Unlearning)

| Method | Acc D_r (≃) | Acc D_f (≃) | Acc D_test (≃) | MIA (≃) | Avg. Gap↓ |
|------|:-:|:-:|:-:|:-:|:-:|
| Retrain (Ideal Baseline) | 100.00 | 94.76 | 94.26 | 12.88 | 0.00 |
| SalUn | 99.62 | 97.15 | 93.93 | 14.39 | 1.15 |
| ℓ₁-sparse | 97.74 | 95.81 | 91.59 | 9.84 | 2.26 |
| Task Arithmetic (best) | 98.36 | 94.85 | 91.49 | 10.91 | 1.62 |
| **NegMerge** | — | — | — | — | **Lowest** |

NegMerge has the smallest Avg. Gap (discrepancy from the Retrain baseline), indicating that its unlearned model behavior is closest to the ideal results achieved by retraining from scratch.

### Ablation Study

- **Robustness to Model Count**: As $n$ increases from 10 to 30, NegMerge performance remains consistently stable, unlike the fluctuations observed in single-model selection.
- **Sign Consistency Ratio**: Only 5–10% of the weight elements remain non-zero after merging, demonstrating the high selectivity of sign filtering.
- **Comparison with Other Merging Strategies**: Sign-consensus merging significantly outperforms TIES-Merging (based on magnitude voting) and MagMax (taking maximum values) on unlearning tasks, illustrating that sign consistency is a more intrinsic signal for unlearning scenarios.

## Highlights & Insights

1. **Sparsely Elegant yet Effective Core Idea**: The simple operation of "retaining only sign-consistent elements" drastically outperforms existing methods without introducing additional hyperparameters.
2. **Breaking the Unlearning-Retaining Trade-off**: By fusing complementary information across multiple models, the elusive balance between unlearning and retaining, which is unattainable by a single model, naturally emerges after merging.
3. **Storage Friendly**: There is no need to store all individual model weights, as binary masks can be dynamically maintained during training.
4. **Strong Cross-Task Generality**: Effective across both zero-shot CLIP unlearning and standard classifier unlearning scenarios.
5. **Serendipitous Weight Sparsity**: With over 90% of weights zeroed out after merging, it naturally supports sparse deployment.

## Limitations & Future Work

1. **Limited to Classification Tasks**: Unlearning performance has not yet been validated on generative models (such as diffusion models) or LLMs, requiring broader validation.
2. **Strict Sign Consensus May Be Overly Conservative**: Requiring absolute sign agreement across all models means a single "disagreement" will filter out an element; relaxed consensus (majority voting) might preserve more useful information.
3. **Dependency on Multi-Model Generation**: The method presumes the existence of multiple fine-tuned models under different hyperparameter settings; it cannot be applied if only a single fine-tuning run is available.
4. **Scaling Coefficient $\lambda$ Still Requires Searching**: Although the search space is shrunk from $O(mn)$ to $O(m)$, the choice of $\lambda$ remains an empirical hyperparameter.
5. **Lack of Theoretical Privacy Guarantees**: While MIA is used in experiments to evaluate privacy preservation, no formal mathematical privacy guarantees are provided for the unlearning effect.

## Related Work & Insights

- **Task Arithmetic** (Ilharco et al., 2023): The core foundational method; NegMerge serves as an enhancement to its unlearning pipeline.
- **Model Soups** (Wortsman et al., 2022): Inspired the idea of "utilizing all models produced during validation".
- **TIES-Merging** (Yadav et al., 2023): Employs voting mechanisms to resolve sign conflicts in model merging, but is outperformed by strict sign consensus on unlearning tasks.
- **SalUn** (Fan et al., 2024): A saliency-based unlearning method, but requires access to the retain set.
- **Linear Task Arithmetic** (Ortiz-Jimenez et al., 2023): Linearizes weights in the tangent space; NegMerge is similarly applicable to this variant.

## Rating

- Novelty: ⭐⭐⭐⭐ — Applying sign-consensus merging to machine unlearning is a novel combination with a simple and intuitive design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive ablation across 12 datasets, 4 backbones, and 2 distinct scenarios.
- Writing Quality: ⭐⭐⭐⭐ — Clear, logical flow from motivation through to methods and experiments, with well-designed figures and tables.
- Value: ⭐⭐⭐⭐ — High practical utility with low computational overhead, though currently constrained to visual classification scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging](../../ACL2025/llm_safety/zjuklab_at_semeval-2025_task_4_unlearning_via_model_merging.md)
- [\[ICCV 2025\] MUNBa: Machine Unlearning via Nash Bargaining](../../ICCV2025/llm_safety/munba_machine_unlearning_via_nash_bargaining.md)
- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](../../NeurIPS2025/llm_safety/simu_selective_influence_machine_unlearning.md)
- [\[NeurIPS 2025\] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](../../NeurIPS2025/llm_safety/a_reliable_cryptographic_framework_for_empirical_machine_unl.md)
- [\[CVPR 2025\] LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty](../../CVPR2025/llm_safety/lotus_large-scale_machine_unlearning_with_a_taste_of_uncertainty.md)

</div>

<!-- RELATED:END -->
