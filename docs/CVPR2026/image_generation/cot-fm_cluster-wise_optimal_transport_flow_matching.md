---
title: >-
  [Paper Note] COT-FM: Cluster-wise Optimal Transport Flow Matching
description: >-
  [CVPR 2026][Image Generation][Flow Matching] This paper proposes COT-FM, a plug-and-play Flow Matching enhancement framework that clusters target samples…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Flow Matching"
  - "Optimal Transport"
  - "Clustering"
  - "Trajectory Straightening"
  - "Accelerated Sampling"
date: 2026-05-08
content_hash: 03993003e26ef5be
---

# COT-FM: Cluster-wise Optimal Transport Flow Matching

**Conference**: CVPR 2026
**arXiv**: [2603.13395](https://arxiv.org/abs/2603.13395)  
**Code**: [Project Page](https://embodiedai-ntu.github.io/cotfm)  
**Area**: Generative Models / Flow Matching
**Keywords**: Flow Matching, Optimal Transport, Clustering, Trajectory Straightening, Accelerated Sampling

## TL;DR

This paper proposes COT-FM, a plug-and-play Flow Matching enhancement framework that clusters target samples, inverts a pretrained model to recover cluster-wise source distributions, and approximates optimal transport within each cluster. This significantly straightens transport trajectories, simultaneously accelerating sampling and improving generation quality without modifying the model architecture.

## Background & Motivation

Flow Matching (FM) learns a velocity field that maps a simple source distribution to a complex data distribution, generating samples by integrating an ODE along the field at inference time. The core challenge is **trajectory curvature**:

- **Random Coupling**: Although each pair $(x_0, x_1)$ yields a straight path, conflicting velocity directions from different pairs at the same point aggregate into a curved marginal velocity field.
- **Batch Optimal Transport (Batch OT)**: OT is approximated only within small batches, limiting accuracy due to locality constraints.
- **Consequences of Curved Trajectories**: Increased time-discretization error and degraded generation quality under low-step sampling; shortcut methods (e.g., MeanFlow) reduce step count but do not straighten trajectories.

Global OT has cubic complexity in the number of samples, making it infeasible at scale.

## Method

### Overall Architecture

COT-FM alternates between two stages on top of a pretrained FM model:

- **Stage 1**: Cluster target samples → invert ODE to estimate cluster-wise source distributions → compute OT mappings within each cluster.
- **Stage 2**: Fine-tune the FM model using the constructed cluster-wise vector field.

Two alternating rounds suffice for convergence. At inference, only one additional step is required: sample a cluster index $k$, then draw the initial noise from the corresponding source distribution $p_{0,k}$.

### Key Designs

1. **Cluster-wise Source Distribution Identification**: The invertibility of the pretrained FM model is exploited to trace back the source sample for each data point $x_1$ in cluster $\mathcal{C}_k$ by inverting the ODE:

    $\hat{x}_0 := x_1 - \int_0^1 v_\theta(\hat{x}_t, t) \, dt$

   The set of traced-back samples $\hat{X}_{0,k}$ is approximated as a Gaussian distribution $p_{0,k}(x) = \mathcal{N}(x; \boldsymbol{\mu}_{0,k}, \boldsymbol{\Sigma}_{0,k})$. The key insight is that trajectories of the pretrained model are naturally non-crossing, so the inverted source distributions retain inter-cluster separation.

2. **Cluster-wise OT Approximation**: The global OT problem is decomposed into $K$ smaller OT subproblems. For each cluster $\mathcal{C}_k$, samples $X_{0,k} \sim p_{0,k}$ are drawn and the cluster-wise OT mapping $\pi_k = \text{OT}(X_{0,k}, \mathcal{C}_k)$ is computed. This offers two advantages:

    - Reducing the sample count per OT problem improves the accuracy of batch OT approximation.
    - Constraining the source distribution space makes velocity field learning more efficient.

3. **Alternating Optimization**: Stage 1 constructs cluster-wise vector fields → Stage 2 fine-tunes the model using the standard CFM loss:

    $\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, (x_0, x_1) \sim B} \|v_\theta(x_t, t) - (x_1 - x_0)\|_2^2$

   Training batches are sampled proportionally to cluster sizes: $P(k) = \frac{|\mathcal{C}_k|}{\sum_j |\mathcal{C}_j|}$. Empirically, two alternating rounds achieve convergence; a third round leads to slight degradation.

4. **Clustering Strategies**:

    - Supervised (conditional generation): class labels are used directly.
    - Unsupervised (unconditional CIFAR-10): DINO features + K-Means ($K=100$).
    - Non-fixed clustering (robot policy): a learned module predicts the source distribution for new observations.

### Loss & Training

- Standard CFM loss with linear interpolation paths $x_t = (1-t)x_0 + tx_1$.
- No modification to model architecture or input/output mechanisms; only the source-target coupling strategy during training is changed.
- The sole inference-time modification: initial noise is sampled from cluster-wise source distributions rather than the global Gaussian.

## Key Experimental Results

### Main Results

| Dataset | Metric | COT-FM | Prev. SOTA | Gain |
|--------|------|--------|-----------|------|
| 2D Mix-5-Gaussian | Wasserstein ↓ | **0.1995** | 0.5421 (RF) | −63.2% |
| 2D Mix-5-Gaussian | Curvature ↓ | **0.0084** | 0.0104 (OT-CFM) | −19.2% |
| CIFAR-10 (1-step) | FID ↓ | **205.0** | 378.0 (RF) | −45.8% |
| CIFAR-10 (10-step) | FID ↓ | **8.23** | 12.6 (RF) | −34.7% |
| CIFAR-10 (50-step) | FID ↓ | **3.97** | 4.45 (RF) | −10.8% |
| CIFAR-10 (MeanFlow 1-step) | FID ↓ | **2.60** | 2.92 (MeanFlow) | −11.0% |
| ImageNet 256 (SiT-B/2, 10-step) | FID ↓ | **7.52** | 8.25 (RF) | −8.8% |
| LIBERO-Long (1 NFE) | Success Rate ↑ | **94.5%** | 91.5% (2-RF) | +3.0% |

### Ablation Study

| Configuration | FID (50-step) ↓ | Notes |
|------|-----------------|------|
| Rectified Flow (0 iter.) | 4.45 | Baseline |
| COT-FM (1 iter.) | 4.23 | 1 alternating round, −0.22 |
| COT-FM (2 iter.) | **3.97** | Optimal at 2 rounds, −0.48 |
| COT-FM (3 iter.) | 4.17 | Slight degradation / overfitting |
| Uniform cluster sampling | 4.26 | Inferior to proportional sampling |
| Proportional cluster sampling | **3.97** | Optimal with size-proportional sampling |

### Key Findings

- Introducing cluster-wise random coupling alone (without OT) reduces 1-step FID from 378 to 296, demonstrating that clustering itself significantly reduces trajectory crossing.
- COT-FM achieves a 1-step FID of 205 on CIFAR-10 (−45.8% vs. RF), with particularly pronounced gains in low-step regimes.
- On LIBERO robot manipulation tasks, COT-FM with 1 NFE reaches 96.1% (Spatial) and 94.5% (Long) success rates, surpassing FLOWER's 4-NFE results (97.1% and 93.5%).
- Generalization is confirmed by consistent train/test FID gaps (3.97/8.19 vs. 4.45/8.55), with no evidence of overfitting.
- MeanFlow's learned trajectories remain curved, validating that shortcut methods do not straighten the underlying velocity field.

## Highlights & Insights

- **Divide-and-conquer OT is the core insight**: decomposing the intractable global OT into $K$ tractable cluster-wise OT problems strikes a balance between theoretical rigor and computational feasibility.
- Using the invertibility of pretrained FM models to estimate cluster-wise source distributions is an elegant bootstrapping strategy — it requires no additional annotations and naturally inherits the structure already captured by the model.
- Strictly preserving the model architecture and inference pipeline (only the initial sampling is modified) makes COT-FM a genuinely plug-and-play general-purpose enhancement.
- Cross-domain validation (2D point clouds, image generation, robot manipulation) thoroughly demonstrates the generality of the approach.

## Limitations & Future Work

1. Constructing cluster-wise vector fields requires inverting the ODE over the entire training set, incurring computational costs that scale with dataset size.
2. The Gaussian approximation may be inadequate for source distributions with complex geometry, particularly in high-dimensional spaces.
3. Clustering quality directly affects performance — K-Means may not be the optimal choice for high-dimensional feature spaces.
4. Validation is limited to CIFAR-10 and ImageNet 256; experiments have not been extended to higher resolutions or text-conditional generation.
5. The reason why performance degrades at the third alternating round remains unexplored.

## Related Work & Insights

- Compared to k-Rectified Flow (which iteratively refines couplings using self-generated samples), COT-FM avoids the risk of mode collapse.
- Compared to OT-CFM (batch-level OT), COT-FM restricts batch OT to smaller scopes via clustering, substantially improving approximation accuracy.
- Compared to MeanFlow (which learns a mean velocity field), COT-FM fundamentally straightens the velocity field rather than merely skipping steps.
- Implication: the primary avenue for improving Flow Matching lies in coupling strategies rather than model architecture; exploiting data-level structure (clustering) is an underexplored dimension.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of cluster-wise OT and ODE inversion of pretrained models for source distribution estimation is novel and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three-domain validation (2D / image / robotics), multiple baselines, and rich ablations (alternating rounds / generalization / sampling strategies).
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous motivation derivation, clear algorithmic pseudocode, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐⭐ Universally plug-and-play with no architectural or inference-pipeline changes; highly practical with significant gains in low-step regimes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](../../ICCV2025/image_generation/contrastive_flow_matching.md)
- [\[NeurIPS 2025\] Counterfactual Identifiability via Dynamic Optimal Transport](../../NeurIPS2025/image_generation/counterfactual_identifiability_via_dynamic_optimal_transport.md)
- [\[ICCV 2025\] The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation](../../ICCV2025/image_generation/the_curse_of_conditions_analyzing_and_improving_optimal_transport_for_conditiona.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](../../NeurIPS2025/image_generation/on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](vecor_--_velocity_contrastive_regularization_for_flow_matching.md)

</div>

<!-- RELATED:END -->
