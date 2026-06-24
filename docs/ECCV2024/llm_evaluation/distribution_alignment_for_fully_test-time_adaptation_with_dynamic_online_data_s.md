---
title: >-
  [Paper Note] Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams
description: >-
  [ECCV 2024][LLM Evaluation][Test-Time Adaptation] A Distribution Alignment (DA) loss is proposed to pull the test-time feature distribution back to the source domain distribution. Combined with a domain shift detection mechanism, this method significantly outperforms existing TTA methods under non-i.i.d. dynamic data streams and continuous domain shift scenarios.
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "Test-Time Adaptation"
  - "Distribution Alignment"
  - "Non-i.i.d."
  - "Domain Shift"
  - "Batch Normalization"
date: 2026-05-08
content_hash: 8203b3654bef2b46
---

# Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams

**Conference**: ECCV 2024  
**arXiv**: [2407.12128](https://arxiv.org/abs/2407.12128)  
**Code**: [Yes](https://github.com/WZq975/DA-TTA)  
**Area**: LLM Evaluation  
**Keywords**: Test-Time Adaptation, Distribution Alignment, Non-i.i.d., Domain Shift, Batch Normalization

## TL;DR

A Distribution Alignment (DA) loss is proposed to pull the test-time feature distribution back to the source domain distribution. Combined with a domain shift detection mechanism, this method significantly outperforms existing TTA methods under non-i.i.d. dynamic data streams and continuous domain shift scenarios.

## Background & Motivation

Test-time adaptation (TTA) aims to adapt models online when facing domain shifts after deployment. Existing methods can be categorized into two major classes:

**Test-Time Batch Normalization (TTBN)**: Normalizes features by replacing training-time global statistics with current test batch statistics.

**Self-training methods**: Optimizes model parameters during test time using entropy minimization or teacher-student frameworks.

While these methods perform well in an **ideal i.i.d. data stream** (where each batch is independently and identically sampled from the target domain), they fail severely in **practical non-i.i.d. data streams**. The reasons are:

- **Non-i.i.d. batches induce label shifts**: In real-world scenarios (such as autonomous driving and robot vision), images are highly temporally correlated, and each batch might exhibit a long-tailed distribution dominated by only a few classes.
- **Failure of TTBN**: Statistics provided by non-i.i.d. batches deviate significantly from the true target domain statistics, leading to incorrect feature normalization.
- **Self-training generates conflicting optimization objectives**: Different long-tailed distributions across different batches lead to gradient conflicts, which may cause model divergence or even collapse.

The authors validated this analysis through key experimental observations: sending source domain data along with i.i.d. and non-i.i.d. target domain data into a TTBN model and recording the average variance of the normalized features. The results showed that: **the feature distribution of the i.i.d. stream was close to that of the source domain (error rate 20%), whereas the feature distribution of the non-i.i.d. stream deviated significantly from the source (error rate 79%)**.

This leads to the core idea: **instead of trying to adapt the model to unpredictable test distributions, the inverse strategy is applied—pulling the test features back to the source domain distribution**. Since the source domain distribution serves as a deterministic reference point, it can eliminate conflicting optimization objectives across different batches.

## Method

### Overall Architecture

The core of DA-TTA is extremely simple: (1) Pre-calculate the mean and variance of the feature distributions at each layer on the source domain data as reference before deployment; (2) Compute the mean and variance of the current batch features during testing; (3) Minimize the discrepancy between the two using the DA loss, pulling the test feature distribution back to the source domain by optimizing the affine parameters of the BN layers. Additionally, a domain shift detection mechanism is designed to handle continuous domain shift scenarios.

### Key Designs

1. **Distribution Alignment Loss**: For intermediate features $\mathbf{X}$ of the model, the test-time mean and variance are calculated channel-wise:

$$m_j = \frac{1}{H \cdot W}\sum_{p=1}^{H \cdot W}\mathbf{X}_{j,p}, \quad d_j^2 = \frac{1}{H \cdot W}\sum_{p=1}^{H \cdot W}(\mathbf{X}_{j,p} - m_j)^2$$

The DA loss measures the discrepancy between the test feature distribution and the pre-calculated source domain statistics:

$$\mathcal{L}_{DA} = \frac{1}{C}\sum_{j=1}^{C}\left(|m_j^T - \bar_m_j^S| + |d_j^{2T} - \bar{d^2}_j^S|\right)$$

Core advantages: (1) All test batches are aligned to **the same source domain distribution**, eliminating conflicting optimization objectives among non-i.i.d. batches; (2) The source domain distribution is compatible with the pre-trained classifier, making the aligned features naturally easier to classify correctly; (3) It is realized solely via linear transformation of the BN layer affine parameters, incurring extremely low computational overhead.

The design motivation stems from the essence of Batch Normalization—the BN affine transformation (scaling + shifting) can linearly control the mean and variance of the feature distribution. Therefore, applying the DA loss to the BN affine layer parameters is the most natural and efficient choice.

2. **Normalization Statistics Initialization**: After deployment, the normalization statistics of the BN layers are updated from the global training statistics to the weighted average of the source domain and the first test batch:

$$\mu_{norm} = \alpha \cdot \mu_{popu} + (1-\alpha) \cdot \mu_{B_1}$$
$$\sigma_{norm}^2 = \alpha \cdot \sigma_{popu}^2 + (1-\alpha) \cdot \sigma_{B_1}^2$$

This provides a more reasonable starting point for subsequent optimization, preventing the model from falling into local optima due to excessively large initial distribution differences.

3. **Domain Shift Detection Mechanism**: In continuous TTA scenarios, the DA loss spikes when a new target domain appears. Domain shift is detected by comparing the average DA loss of a short-term window and a long-term window:

$$\frac{\sum_{i=0}^{p}\mathcal{L}_{DA}^{B_{t-i}}}{p} > \tau \cdot \frac{\sum_{i=0}^{q}\mathcal{L}_{DA}^{B_{t-i}}}{q}$$

Upon detecting a new domain, the affine parameters are reset to their initial states, and the normalization statistics are re-initialized, preventing the adaptation parameters of old domains from negatively affecting the new domain.

### Loss & Training

The final loss combines the DA loss with the entropy minimization (EM) loss weighted by a confidence threshold:

$$\mathcal{L}_{EM} = \sum_m \left[\mathbb{1}(\max_n \hat{y}_n > \theta)\sum_{n=1}^{N} -p(\hat{y}_n)\log p(\hat{y}_n)\right]$$

$$\mathcal{L}_{final} = \mathcal{L}_{DA} + \mathcal{L}_{EM}$$

The DA loss provides a stable optimization direction (aligning all batches to the same target), while the EM loss further improves predictions for highly confident samples under the "protection" of the DA loss. Experiments demonstrate that using DA alone achieves state-of-the-art (SOTA) performance, and incorporating EM yields further improvements.

## Key Experimental Results

### Main Results

**Non-i.i.d. Fully Test-Time Adaptation (Average error rate % across 15 domains)**:

| Method | CIFAR10-C | CIFAR100-C | ImageNet-C | Note |
|------|-----------|------------|------------|------|
| Source (No adaptation) | 43.5 | 46.5 | 82.0 | Baseline |
| TTBN | 75.7 | 55.3 | 95.8 | Severe degradation under non-i.i.d. |
| TENT | 77.0 | 59.1 | 96.7 | Worse than no adaptation |
| SAR | 75.7 | 54.0 | 96.3 | Still poor despite being designed for non-i.i.d. |
| RoTTA | 27.6 | 43.5 | 71.0 | Second best |
| **DA-TTA** | **24.3** | **31.6** | **64.8** | **Outperforms second best by 3.3/7.7/5.9%** |

**Other domain shift datasets (Non-i.i.d. error rate %)**:

| Method | ImageNet-D | ImageNet-R | ImageNet-A |
|------|-----------|------------|------------|
| Source | 70.6 | 63.8 | 90.5 |
| RoTTA | 67.6 | 61.5 | 93.1 |
| **DA-TTA** | **66.5** | **58.5** | **88.6** |

### Ablation Study

**Each component's contribution (Non-i.i.d. error rate %)**:

| Configuration | CIFAR10-C | CIFAR100-C | ImageNet-C | Description |
|------|-----------|------------|------------|------|
| Source | 43.5 | 46.5 | 82.0 | No adaptation |
| w/o low-level DA | 26.6 | 32.6 | 68.1 | High-level layer DA only |
| w/o high-level DA | 24.9 | 33.5 | 67.1 | Low-level layer DA only |
| w/o $\mathcal{L}_{EM}$ | 28.1 | 35.8 | 70.1 | DA alone achieves SOTA |
| **Full DA-TTA** | **24.3** | **31.6** | **64.8** | DA + EM is optimal |

**Key ablation findings**:
- Removing either half of the layers' DA supervision results in relatively minor performance degradation—this is because the feature distributions of adjacent layers in a frozen model are highly correlated, providing mutual constraints between supervised and unsupervised layers.
- TENT (TTBN + EM) performs worse than Source in non-i.i.d. scenarios, but DA + EM significantly improves performance—showing that DA provides a stable optimization direction for EM.

### Key Findings

- **More than half of the existing methods perform worse than no adaptation** (reaching error rates higher than Source) under non-i.i.d. settings, exposing the vulnerability of current TTA methods.
- DA-TTA performs closely on both i.i.d. and non-i.i.d. streams, demonstrating insensitivity to assumptions about data distribution.
- t-SNE visualization confirms that post-DA target domain feature clusters for each category align closely with those of the source domain.
- It is robust to Dirichlet parameters (controlling the degree of non-i.i.d.) and batch size, unlike other methods that collapse as the batch size decreases.

## Highlights & Insights

- **Reverse Thinking**: Instead of adapting the model to unpredictable target distributions, the features are pulled back to the reliable source distribution—solving the conflicting optimization problem simply but fundamentally.
- **Simplicity is Powerful**: Optimizing only the L1 loss of the BN affine parameters achieves SOTA performance, eliminating the need for complex components like teacher networks, memory banks, or pseudo-labels.
- **Thorough Analysis**: The analysis of the failure mechanism of TTBN (derivation of Eq. 2-3 + visualization of feature variance) is highly convincing.

## Limitations & Future Work

- Requires pre-computing feature statistics on source domain data before deployment (a one-time cost), which technically still requires source data access from a strict "source-free" perspective (though methods like EATA and RMT share similar requirements).
- Domain shift detection depends on hyperparameters like window size and threshold $\tau$; extremely rapid domain changes may cause detection lags.
- There is still room for improvement in absolute performance on large-scale datasets such as ImageNet-C (64.8% error rate).
- Adaptation to modern architectures without BN layers (e.g., ViT, LayerNorm networks) has not been explored.

## Related Work & Insights

- Philosophically similar to BUFR (a source-free domain adaptation method), but while BUFR stores marginal distributions for each feature to match them, DA-TTA is much more lightweight by using only the mean and variance.
- The domain shift detection draws inspiration from the tracking approach in EATA, but with a different goal: EATA tracks model outputs to skip redundant samples, whereas DA-TTA tracks feature distributions to detect domain shifts.
- Insight: In highly uncertain scenarios such as TTA, a "conservative" strategy (anchoring to the source domain) can be more robust than an "aggressive" strategy (adapting to the target domain).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The core idea of reverse alignment is simple and novel, and the failure mechanism of TTBN is deeply analyzed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 6 datasets, three settings (i.i.d., non-i.i.d., and continuous domains), multi-dimensional ablation, and robustness analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation derivation, with self-consistent logic from problem analysis to method design.
- **Value**: ⭐⭐⭐⭐ Significant progress made in the practically important scenario of non-i.i.d. TTA, with strong method generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] BATCLIP: Bimodal Online Test-Time Adaptation for CLIP](../../ICCV2025/llm_evaluation/batclip_bimodal_online_test-time_adaptation_for_clip.md)
- [\[ECCV 2024\] Eliminating Warping Shakes for Unsupervised Online Video Stitching](eliminating_warping_shakes_for_unsupervised_online_video_stitching.md)
- [\[ECCV 2024\] Gradient-Regularized Out-of-Distribution Detection](gradient-regularized_out-of-distribution_detection.md)
- [\[ECCV 2024\] Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence](learn_from_the_learnt_source-free_active_domain_adaptation_via_contrastive_sampl.md)
- [\[ICML 2025\] Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time](../../ICML2025/llm_evaluation/bounded_rationality_for_llms_satisficing_alignment_at_inference-time.md)

</div>

<!-- RELATED:END -->
