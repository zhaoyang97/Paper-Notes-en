---
title: >-
  [Paper Note] TranSUN: A Preemptive Paradigm to Eradicate Retransformation Bias Intrinsically from Regression Models
description: >-
  [NeurIPS 2025][Autonomous Driving][retransformation bias] Addressing the retransformation bias problem in transformed MSE regression models for recommendation systems, we propose TranSUN, a preemptive approach that jointly learns an auxiliary branch to explicitly model the bias, eliminating it from within the model during training. TranSUN offers theoretical unbiasedness guarantees and strong convergence properties, and has been deployed in Taobao homepage "Guess You Like" product and short video recommendation scenarios.
tags:
  - NeurIPS 2025
  - Autonomous Driving
  - retransformation bias
  - regression model
  - recommender system
  - target transformation
  - debiasing
date: 2026-05-08
content_hash: 81cdd36da5f6bb30
---
# TL;DR

To address the retransformation bias problem in transformed MSE regression models for recommender systems, this paper proposes TranSUN, a preemptive debiasing method that introduces an auxiliary branch to explicitly model the bias via joint learning, thereby eliminating bias intrinsically during the training phase. The method offers theoretical unbiasedness guarantees and favorable convergence properties, and has been deployed in the "Guess You Like" item and short-video recommendation scenarios on the Taobao homepage.

## Background & Motivation

1. **Regression models are critical in recommender systems**: Tasks such as dwell time prediction, transaction amount estimation, and customer lifetime value prediction rely heavily on regression models, whose accuracy directly affects recommendation quality and business revenue.
2. **Necessity of target transformation**: Target variables in recommendation scenarios are often highly right-skewed, violating the Gaussian assumption of MSE, which hampers model convergence. Transformations such as $\log(y+1)$ and $\sqrt{y}$ are commonly applied to approximate a Gaussian distribution and improve convergence.
3. **Retransformation bias has been long overlooked**: After transformation, the model learns $\mathbb{E}[T(y)|x]$; applying the inverse $T^{-1}$ to recover the original scale introduces a systematic bias due to Jensen's inequality (e.g., a convex $T^{-1}$ leads to systematic underestimation). This issue has been largely neglected in the recommender systems community.
4. **Limitations of post-hoc correction methods**: Classical debiasing methods such as NTE and Smearing apply corrections externally after training, posing practical challenges in industrial recommender systems, including the need for additional residual statistics and incompatibility with online serving architectures.
5. **Bias harms business metrics**: Taking GMV prediction on Taobao as an example, the $\log$ transformation causes the model to systematically underestimate high-value orders, directly degrading ranking quality and business revenue.
6. **Absence of an intrinsic debiasing paradigm**: All prior methods correct bias externally after training; no approach has been proposed to eliminate bias fundamentally from within the model training process.

## Method

### Overall Architecture: Preemptive Debiasing Paradigm with Joint Bias Learning

- **Function**: On top of a transformed MSE model, an auxiliary branch $z(x;\theta_z)$ is introduced to explicitly learn the bias ratio. Joint optimization during training simultaneously accomplishes the main regression task and the debiasing task.
- **Design Motivation**: Unlike post-hoc methods, embedding debiasing into the training process avoids the engineering complexity of post-hoc correction and fully leverages training data to model the bias.
- **Mechanism**:
  1. Main branch: Standard transformed MSE loss $\mathcal{L}_{\text{MSE}}^T = \mathbb{E}[(f(x;\theta) - T(y))^2]$, learning $\mathbb{E}[T(y)|x]$.
  2. Auxiliary branch: Bias learning loss $\mathcal{L}_{\text{sun}}^T = \mathbb{E}[(z(x;\theta_z) - \text{stop\_grad}[y/(|T^{-1}(f(x;\theta))| + \epsilon)])^2]$, learning the ratio of the ground-truth to the biased prediction.
  3. Total loss: $\mathcal{L}_{\text{TranSUN}}^T = \mathcal{L}_{\text{MSE}}^T + \mathcal{L}_{\text{sun}}^T$.
  4. Inference: $\hat{y}|x = z(x;\theta_z) \cdot (|T^{-1}(f(x;\theta))| + \epsilon)$, recovering an unbiased estimate via ratio correction.
  5. Key detail: `stop_grad` blocks gradients from the bias loss to the main branch, ensuring the two optimization objectives remain independent.

### Key Design 1: Choice of Multiplicative Bias Modeling Scheme

- **Function**: The auxiliary branch is trained to learn $y / T^{-1}(f)$ (ground-truth / biased prediction) rather than $T^{-1}(f) / y$ (biased prediction / ground-truth).
- **Design Motivation**: Not all bias learning schemes guarantee theoretical unbiasedness. Learning $T^{-1}(f)/y$ yields an estimate of $1/\mathbb{E}[1/y]$, which remains biased by Jensen's inequality; learning $y/T^{-1}(f)$ admits a rigorous derivation showing $\hat{y}|x = \mathbb{E}[y|x]$.
- **Mechanism**: By the conditional expectation property of MSE regression, minimizing $\mathcal{L}_{\text{sun}}^T$ yields $z(x;\theta_z) = \mathbb{E}[y/(|T^{-1}(f)| + \epsilon) | x]$; substituting into the inference formula immediately gives $\hat{y}|x = \mathbb{E}[y|x]$. Furthermore, the ratio formulation has intuitively smaller variance, resulting in a smoother loss and more stable training.

### Key Design 2: Generalized TranSUN (GTS) Unified Framework

- **Function**: TranSUN is generalized into a family of regression models, GTS, which supports arbitrary conditional point estimators as the main branch and arbitrary functions $\kappa$ as the linear transformation slope.
- **Design Motivation**: This reveals that the essential mechanism underlying TranSUN's unbiasedness is *conditional linearity* rather than bias learning per se; GTS provides a flexible, plug-and-play framework for adding debiasing capability to arbitrary regression models.
- **Mechanism**:
    - GTS loss = conditional point loss $\mathcal{L}_{\mathcal{H}_q}$ (main branch learning a point estimate of $T(y)|x$) + linear transformation loss (auxiliary branch learning via dynamic slope $\kappa(f)$).
    - Model assumption: $-z(x;\theta_z) + y \cdot \kappa(\mathbb{Q}[T(y)|x]) \sim \mathcal{N}(0,\sigma^2)$, which is essentially a conditionally linear transformed MSE and inherently preserves unbiasedness.
    - Application modes: (a) customize $\mathcal{H}_q$ and $\kappa$ to directly construct novel unbiased regression models; (b) set $T$ as the identity transformation for an existing model to enable plug-and-play debiasing.

## Key Experimental Results

### Experimental Setup
- **Synthetic data**: 8 distributions (right-skewed, left-skewed, symmetric), 3 transformations (linear, logarithmic, square root)
- **Real-world data**: CIKM16 (retail forecasting), DTMart (market mix modeling), Taobao industrial dataset (GMV prediction)
- **Metrics**: SRE (signed relative error), TRE (total ratio error), MRE (mean ratio error), NRMSE, NMAE, XAUC
- **Baselines**: MSE, LogMSE, MAE, WLR, ZILN, MDME, TPM, CREAD, OptDist, NTE, Smearing, SIR

### Main Results

| Method | T(y) | CIKM16 TRE↓ | CIKM16 MRE↓ | DTMart TRE↓ | DTMart MRE↓ |
|------|------|-------------|-------------|-------------|-------------|
| T-MSE | ln(y+1) | 0.3468 | 0.3352 | 0.2894 | 0.1432 |
| **TranSUN** | ln(y+1) | **0.0133** (-96.2%) | **0.0171** (-94.9%) | **0.0803** (-72.3%) | **0.0725** (-49.4%) |
| T-MSE | √y | 0.1386 | 0.1243 | 0.4388 | 0.3421 |
| **TranSUN** | √y | **0.0388** (-72.0%) | **0.0283** (-77.2%) | **0.0907** (-79.3%) | **0.0660** (-80.7%) |

| Comparison | TRE↓ | MRE↓ | NRMSE↓ | NMAE↓ |
|------|------|------|--------|-------|
| LogMSE (no correction) | 0.3451 | 0.3667 | 0.6189 | 0.4528 |
| LogMSE + Smearing | 0.0175 | 0.0477 | 0.5617 | 0.4335 |
| **LogSUN** (intrinsic debiasing) | **0.0123** | 0.0439 | 0.5625 | 0.4333 |

### Key Findings
1. TranSUN maintains $|\text{SRE}| < 0.7\%$ across all transformations and distribution types, whereas transformed MSE exhibits biases exceeding 50% under nonlinear transformations (systematic underestimation under log, systematic overestimation under square root).
2. TranSUN achieves performance comparable to post-hoc correction methods (Smearing, SIR) while requiring no additional residual statistics, enabling direct integration into online serving pipelines.
3. The GTS framework enables plug-and-play debiasing for existing models such as WLR, ZILN, MDME, and OptDist, reducing TRE by over 80% in all cases.
4. TranSUN has been deployed in the core item recommendation and short-video recommendation scenarios of Taobao's "Guess You Like" feature, serving the main traffic with DAU exceeding 300 million.

## Highlights & Insights

- This work is the first to propose a **preemptive paradigm** that eliminates retransformation bias from within the model, overturning the conventional post-hoc correction approach.
- The theoretical unbiasedness guarantee is rigorous and clearly derived, revealing the non-intuitive conclusion that not all explicit bias modeling schemes can guarantee unbiasedness.
- The GTS unified framework is highly general and enables plug-and-play debiasing for arbitrary regression models.
- Large-scale industrial deployment on Taobao (DAU 300M+) provides strong validation of practical utility.

## Limitations & Future Work

- The auxiliary branch increases model parameters and computation, which may affect extremely latency-sensitive scenarios (though the paper claims the overhead is small).
- The theoretical unbiasedness relies on the auxiliary branch perfectly learning the conditional expectation under MSE loss; residual bias may remain when model capacity is insufficient in practice.
- Experiments primarily focus on right-skewed data in e-commerce/retail settings; applicability to other domains (e.g., financial risk, medical prediction) is not extensively validated.
- The use of `stop_grad` decouples the two branches entirely; whether joint optimization could yield additional benefits remains unexplored.

## Related Work & Insights

| Aspect | Ours (TranSUN) | Post-hoc correction (NTE/Smearing) |
|------|------------|-------------------------------|
| Correction timing | Training phase (preemptive) | Applied externally after training (post-hoc) |
| Architectural intrusiveness | Only adds a lightweight auxiliary branch | Does not modify the model but requires extra statistical steps |
| Distributional assumption | No assumption on the form of the transformed distribution | NTE assumes Gaussian distribution |
| Industrial applicability | Directly integrable into online serving | Extra steps increase engineering complexity |

| Aspect | Ours (GTS) | Conditional linear transformation model (CLTM) |
|------|---------|------------------|
| Objective | Unbiased point estimation (value regression) | Interval estimation (transformation model) |
| Slope dependency | $\kappa$ depends on both $x$ and $y$ (via point estimate) | $\beta$ depends on $x$ only |
| Bias guarantee | Theoretically unbiased + experimentally validated | Does not address retransformation bias |

## Rating

- ⭐⭐⭐⭐ Novelty: The preemptive paradigm represents a fundamentally new perspective, revealing the deep connection between the multiplicative scheme and unbiasedness.
- ⭐⭐⭐⭐ Theoretical Depth: Unbiasedness proofs are rigorous; the GTS unified framework provides deep theoretical insights.
- ⭐⭐⭐⭐⭐ Experimental Thoroughness: Three-level validation comprising synthetic, public, and industrial data, including large-scale online deployment.
- ⭐⭐⭐⭐⭐ Value: Deployed in core Taobao scenarios with DAU 300M+, demonstrating outstanding industrial impact.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving](../../CVPR2026/autonomous_driving/drocc_depth-_and_region-guided_3d_occupancy_from_surround-view_cameras_for_auton.md)
- [\[NeurIPS 2025\] URB -- Urban Routing Benchmark for RL-Equipped Connected Autonomous Vehicles](urb_--_urban_routing_benchmark_for_rl-equipped_connected_autonomous_vehicles.md)
- [\[NeurIPS 2025\] Layer-wise Modality Decomposition for Interpretable Multimodal Sensor Fusion](layer-wise_modality_decomposition_for_interpretable_multimodal_sensor_fusion.md)
- [\[NeurIPS 2025\] OpenBox: Annotate Any Bounding Boxes in 3D](openbox_annotate_any_bounding_boxes_in_3d.md)
- [\[NeurIPS 2025\] SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving](sqs_enhancing_sparse_perception_models_via_query-based_splatting_in_autonomous_d.md)

<!-- RELATED:END -->
