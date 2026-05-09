---
title: >-
  [Paper Note] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][Test-time adaptation] DOTA proposes shifting test-time adaptation from a "caching sample instances" paradigm to a "continuously estimating test data distributions" paradigm. By combining online Gaussian discriminant analysis with zero-shot prediction probabilities to estimate per-class distributions, DOTA achieves gradient-free, forgetting-resistant, and efficient test-time adaptation, surpassing all baselines in average accuracy across 10 cross-domain benchmarks.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Test-time adaptation
  - CLIP
  - distribution estimation
  - Gaussian discriminant analysis
  - zero-shot classification
date: 2026-05-08
content_hash: 8aa8244796df1cd2
---

# DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2409.19375](https://arxiv.org/abs/2409.19375)
**Code**: Unavailable (authors indicate release is planned)
**Area**: Multimodal VLM
**Keywords**: Test-time adaptation, CLIP, distribution estimation, Gaussian discriminant analysis, zero-shot classification

## TL;DR
DOTA proposes shifting test-time adaptation from a "caching sample instances" paradigm to a "continuously estimating test data distributions" paradigm. By combining online Gaussian discriminant analysis with zero-shot prediction probabilities to estimate per-class distributions, DOTA achieves gradient-free, forgetting-resistant, and efficient test-time adaptation, surpassing all baselines in average accuracy across 10 cross-domain benchmarks.

## Background & Motivation

**Background** Vision-language foundation models such as CLIP demonstrate strong performance across a wide range of tasks, yet train-test distribution shifts commonly cause performance degradation at deployment. Test-time adaptation (TTA) is an effective low-cost strategy for bridging this gap.

**Limitations of Prior Work** Existing TTA methods fall into two categories: (1) prompt-learning-based approaches (e.g., TPT) require gradient backpropagation and incur high inference costs; (2) cache-based approaches (e.g., TDA, BoostAdapter) store only a limited number of "representative" samples and inevitably discard older samples during cache updates, leading to **catastrophic forgetting**.

**Key Challenge** The fundamental limitation of cache-based methods is that storing discrete samples under a fixed capacity fails to make full use of all test data, and cache replacement causes previously learned distributional information to be lost.

**Goal** To design a TTA method that is gradient-free, capacity-unconstrained, and capable of continuously learning from all test samples.

**Key Insight** Shift from "memorizing instances" to "estimating distributions" — model per-class embeddings as Gaussian distributions and employ online EM parameter estimation weighted by zero-shot prediction probabilities.

**Core Idea** Continuously estimate per-class distributions of test data via online Gaussian discriminant analysis, then compute posterior probabilities through Bayes' theorem to enable adaptation.

## Method

### Overall Architecture
DOTA processes test samples in a streaming fashion. For each new sample, it first obtains prediction probabilities via CLIP zero-shot classification, then uses these probabilities as weights to update per-class Gaussian distribution parameters (mean and covariance), and finally produces an adaptive prediction by combining the zero-shot classifier with the distribution-based classifier.

### Key Designs

1. **Zero-Shot Probability-Weighted Parameter Estimation (Proposition 3.1)**:
    - Function: Estimate per-class Gaussian distribution parameters without labels.
    - Mechanism: The zero-shot prediction probability $P_k^{zs}(y=k|\mathbf{x}_n)$ serves as the posterior weight in the E-step of the EM algorithm; the M-step maximizes the likelihood: $\hat{\boldsymbol{\mu}}_k = \frac{\sum_n P_k^{zs} \mathbf{x}_n}{\sum_n P_k^{zs}}$, with covariance estimated analogously via weighted summation.
    - Design Motivation: Although zero-shot probabilities are imperfect, they provide reasonable soft labels; using them as weights mitigates the influence of erroneous predictions.

2. **Online Distribution Update**:
    - Function: Update distribution parameters incrementally in a streaming, sample-by-sample manner.
    - Mechanism: Maintain an effective sample count $c_k^t$ and distribution parameters for each class, applying incremental updates at each step: $\hat{\boldsymbol{\mu}}_k^t = \frac{c_k^{t-1}\hat{\boldsymbol{\mu}}_k^{t-1} + \sum P_k^{zs}\mathbf{x}_n}{c_k^{t-1} + \sum P_k^{zs}}$. The covariance matrix is averaged across classes to reduce matrix inversion overhead, and shrinkage regularization is applied: $\hat{\Lambda} = [(1-\epsilon)\hat{\Sigma} + \epsilon I]^{-1}$.
    - Design Motivation: The streaming setting requires incremental updates; averaging the covariance across classes reduces $K$ matrix inversions to one, greatly improving efficiency.

3. **Adaptive Fusion Strategy**:
    - Function: Dynamically interpolate between the zero-shot classifier and the test-time classifier.
    - Mechanism: The final probability is $P_k = \text{softmax}(\cos(\mathbf{x}, \mathbf{w}_k)/\tau + \lambda f_k(\mathbf{x}))$, where $\lambda = \min(\rho c, \eta)$ increases as more test samples are observed.
    - Design Motivation: When few test samples are available early on, distribution estimates are unreliable and the zero-shot classifier should dominate; as more samples accumulate, the weight of the test-time classifier is progressively increased.

## Key Experimental Results

### Main Results — Cross-Domain Generalization (Top-1 Accuracy %)

| Method | Aircraft | EuroSAT | Flower | Pets | Avg. (10 datasets) |
|--------|----------|---------|--------|------|--------------------|
| Zero-Shot | 23.22 | 50.42 | 66.99 | 86.92 | 64.59 |
| TDA | 23.91 | 58.00 | 71.42 | 88.63 | 67.53 |
| BoostAdapter | 27.45 | 61.22 | 71.66 | 89.51 | 68.68 |
| **DOTA** | 26.25 | **62.78** | **75.23** | **92.01** | **70.68** |

### Ablation Study

| Configuration | Avg. Accuracy | Notes |
|---------------|--------------|-------|
| Full DOTA | 70.68 | All components |
| w/o fusion (distribution only) | Decreased | Distribution estimates unstable with few early samples |
| Fixed covariance | Decreased | Class-specific covariance aids discrimination |
| Cache method (TDA) | 67.53 | Gap attributable to forgetting |

### Key Findings
- DOTA achieves gains of 8.24% on Flowers102 and 3.09% on Pets, outperforming all cache-based and prompt-based methods.
- Inference is more than 20× faster than TPT, with no gradient computation required.
- Performance improves continuously as more test samples are observed, with no forgetting issue as in cache-based methods.
- The advantage is most pronounced on datasets with large distribution shifts, such as EuroSAT (remote sensing).

## Highlights & Insights
- The paradigm shift from "caching instances" to "estimating distributions" is both elegant and compelling, introducing an information-theoretic perspective into TTA.
- The approach requires no gradient backpropagation, imposes no cache capacity constraint, and offers extremely fast inference, making it highly practical.
- Using CLIP zero-shot probabilities as EM weights is a clean design that circumvents the need for ground-truth labels.

## Limitations & Future Work
- The Gaussian distribution assumption may be overly simplistic for all data types and could degrade in non-Gaussian scenarios.
- When the zero-shot classifier itself is severely inaccurate, distribution estimates derived from its probabilities may be unreliable.
- Validation is limited to classification tasks; extension to other VLM applications such as retrieval and segmentation remains unexplored.

## Related Work & Insights
- **vs. TDA/BoostAdapter**: DOTA replaces instance caching with distribution estimation, fundamentally resolving catastrophic forgetting.
- **vs. TPT/DiffTPT**: DOTA requires no gradient backpropagation and is more than 20× faster, making it suitable for latency-sensitive scenarios.
- **vs. T3A**: T3A adjusts linear classifiers via prototypes; DOTA goes further by estimating the full Gaussian distribution including covariance.
- **vs. HisTPT**: HisTPT leverages historical information but still requires gradient optimization; DOTA is entirely gradient-free.
- **vs. ZERO**: ZERO exploits zero-shot features without online adaptation; DOTA continuously learns from the test stream.

## Additional Notes
- All experiments use the ViT-B/16 CLIP model with a batch size of 1 (single-sample streaming setting).
- Hyperparameters $\omega$, $\sigma^2$, $\rho$, and $\eta$ are selected on a validation set and fixed across all test sets.

## Rating
- Novelty: ⭐⭐⭐⭐ The "cache → distribution" paradigm shift is original, with clear theoretical derivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on 10 cross-domain datasets with thorough ablation.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear, the method is concise, and figures are intuitive.
- Value: ⭐⭐⭐⭐ A highly practical TTA method broadly applicable to VLM deployment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [\[NeurIPS 2025\] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](../../ICCV2025/multimodal_vlm/latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [\[NeurIPS 2025\] TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)
- [\[NeurIPS 2025\] Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models](test-time_spectrum-aware_latent_steering_for_zero-shot_generalization_in_vision-.md)

</div>

<!-- RELATED:END -->
