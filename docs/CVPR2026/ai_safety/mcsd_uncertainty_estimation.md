---
title: >-
  [Paper Note] Monte Carlo Stochastic Depth for Uncertainty Estimation in Deep Learning
description: >-
  [CVPR 2026][AI Safety][Uncertainty Quantification] This paper formally connects Stochastic Depth (SD) to the Bayesian variational inference framework…
tags:
  - "CVPR 2026"
  - "AI Safety"
  - "Uncertainty Quantification"
  - "Stochastic Depth"
  - "Bayesian Inference"
  - "Object Detection"
  - "Monte Carlo"
date: 2026-05-08
content_hash: b35c1c81d3e83b0e
---

# Monte Carlo Stochastic Depth for Uncertainty Estimation in Deep Learning

**Conference**: CVPR 2026
**arXiv**: [2604.12719](https://arxiv.org/abs/2604.12719)  
**Code**: None  
**Area**: AI Safety / Uncertainty Estimation
**Keywords**: Uncertainty Quantification, Stochastic Depth, Bayesian Inference, Object Detection, Monte Carlo

## TL;DR

This paper formally connects Stochastic Depth (SD) to the Bayesian variational inference framework, proposes Monte Carlo Stochastic Depth (MCSD) as an uncertainty estimation method, and conducts the first systematic benchmark on modern detectors including YOLO and RT-DETR, demonstrating competitive performance against MC Dropout in terms of calibration and uncertainty ranking.

## Background & Motivation

**Background**: Safety-critical systems require reliable uncertainty quantification from DNNs. Monte Carlo Dropout (MCD) reinterprets dropout as approximate Bayesian inference and has become the dominant practical approach. MC DropBlock (MCDB) extends this paradigm to convolutional layers.

**Limitations of Prior Work**: Standard dropout performs poorly in convolutional layers, while Stochastic Depth (SD) is a native regularization technique for residual networks widely adopted in modern architectures such as YOLO and ViT. However, the theoretical foundation for using SD as an inference-time sampling mechanism and systematic empirical validation remain absent.

**Key Challenge**: The formal theoretical connection between SD as a regularizer and Bayesian variational inference has not been established, and its UQ performance on complex multi-task problems such as object detection is unknown.

**Goal**: (1) Establish the theoretical link between MCSD and variational inference; (2) Conduct the first systematic benchmark of MCSD on object detection.

**Key Insight**: The progression from MCD to MCDB reveals a meta-strategy: stochastic regularizers implicitly define approximate posterior distributions. SD is the natural next candidate.

**Core Idea**: Preserve the stochasticity of stochastic depth at inference time, and sample sub-networks of varying depths through $T$ stochastic forward passes, forming an implicit depth ensemble for uncertainty estimation.

## Method

### Overall Architecture

MCSD operates on standard residual networks: at inference time, each residual block independently samples $b_l \sim \text{Bernoulli}(p_l)$, where $b_l=1$ retains the residual path and $b_l=0$ retains only the skip connection. The predictive distribution over $T$ stochastic forward passes provides uncertainty estimates: $p(y_* | x_*, \mathcal{D}) \approx \frac{1}{T} \sum_{t=1}^{T} p(y_* | x_*, W^{(B_t)})$.

### Key Designs

1. **Theoretical Derivation of Stochastic Depth as Variational Inference**:

    - Function: Provides the theoretical foundation for MCSD.
    - Mechanism: Defines the variational distribution $q_\theta(W) \equiv p(B) = \prod_{l=1}^{L} p_l^{b_l}(1-p_l)^{1-b_l}$, i.e., a product of $L$ independent Bernoulli variables. Standard SD training (stochastic forward passes + L2 regularization) is equivalent to optimizing the ELBO: the expected log-likelihood is approximated via MC sampling, and the KL regularization term is approximated by weight decay.
    - Design Motivation: Unlike MCD operating on individual weights or MCDB on weight blocks, MCSD operates on the inclusion/exclusion distribution of entire network stages, producing an implicit ensemble of sub-networks with varying depths.

2. **MCSD Inference Algorithm**:

    - Function: Samples the approximate posterior at inference time by preserving stochasticity.
    - Mechanism: Unlike standard SD inference (deterministic scaling $x_{l+1} = x_l + p_l \cdot \mathcal{F}_l(x_l; W_l)$), MCSD retains stochastic sampling and normalizes features as $A_{res} = A_{res} / p_{keep}$. Each forward pass produces a sub-network with a different depth.
    - Design Motivation: Deterministic inference discards uncertainty information; preserving stochasticity directly leverages the depth distribution learned during training.

3. **Adaptation to Modern Detectors**:

    - Function: Applies MCSD to YOLO, Faster R-CNN, and RT-DETR.
    - Mechanism: MCSD/MCD/MCDB are inserted at the skip connections within residual paths (Bottleneck, Residual Layer, HGBlock) of each detector, and uncertainty estimation quality is compared.
    - Design Motivation: MCSD naturally fits residual architectures without requiring any structural modifications.

### Loss & Training

Standard detection training (classification + regression losses + weight decay). MCSD leverages the existing stochastic depth regularization and requires no additional training or custom losses.

## Key Experimental Results

### Main Results

| Method | Architecture | COCO mAP↑ | ECE↓ | AUARC↑ |
|--------|-------------|-----------|------|--------|
| Deterministic | YOLOv8 | 52.8 | 0.142 | 0.821 |
| MCD | YOLOv8 | 52.5 | 0.128 | 0.835 |
| MCDB | YOLOv8 | 52.3 | 0.135 | 0.829 |
| **MCSD** | YOLOv8 | **52.7** | **0.125** | **0.838** |
| MCD | RT-DETR | 53.1 | 0.118 | 0.842 |
| **MCSD** | RT-DETR | **53.3** | **0.115** | **0.845** |

### Ablation Study

| MC Samples $T$ | mAP | ECE↓ | Inference Time Ratio |
|----------------|-----|------|----------------------|
| 1 (Deterministic) | 52.8 | 0.142 | 1.0× |
| 5 | 52.6 | 0.130 | 4.8× |
| 10 | 52.7 | 0.125 | 9.5× |
| 20 | 52.7 | 0.124 | 19.2× |

### Key Findings

- MCSD slightly outperforms MCD in calibration (ECE) and uncertainty ranking (AUARC) while maintaining competitive mAP.
- The sub-network depth variation induced by MCSD is more "diverse" than the local weight/region dropping of MCD/MCDB.
- Compatible with all architectures incorporating skip connections (both CNN and Transformer).

## Highlights & Insights

- MCSD is an "architecture-native" uncertainty method: SD is already a standard regularizer in modern architectures, and MCSD requires only that stochasticity be preserved at inference time, incurring zero additional training overhead.
- The theoretical derivation unifies MCD, MCDB, and MCSD under the variational inference framework, revealing a spectrum of uncertainty modeling at different granularities (weights → regions → entire layers).

## Limitations & Future Work

- The rigorous computation of the KL divergence term (discrete mixture distribution vs. continuous prior) is mathematically ill-posed; L2 regularization is used as an approximation.
- Evaluation is limited to object detection; segmentation and classification are not addressed.
- The computational overhead of multiple forward passes at inference time remains significant.
- Future work could explore treating network depth as a learnable stochastic variable rather than a fixed probability.

## Related Work & Insights

- **vs. MCD**: MCD operates at the individual weight level and has limited effectiveness for convolutional layers; MCSD operates at the entire residual block level, making it better suited to modern architectures.
- **vs. Deep Ensembles**: Ensemble methods require $N\times$ training and inference costs, whereas MCSD extracts uncertainty from a single model.

## Rating

- Novelty: ⭐⭐⭐⭐ The theoretical derivation fills the formalization gap for MCSD.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic benchmark across three detectors on COCO/COCO-O.
- Writing Quality: ⭐⭐⭐⭐ The theoretical sections are rigorous and clear.
- Value: ⭐⭐⭐⭐ Practically valuable for UQ in safety-critical systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning](../../NeurIPS2025/ai_safety/impact_of_dataset_properties_on_membership_inference_vulnerability_of_deep_trans.md)
- [\[AAAI 2026\] Matrix-Free Two-to-Infinity and One-to-Two Norms Estimation](../../AAAI2026/ai_safety/matrix-free_two-to-infinity_and_one-to-two_norms_estimation.md)
- [\[AAAI 2026\] DeepTracer: Tracing Stolen Model via Deep Coupled Watermarks](../../AAAI2026/ai_safety/deeptracer_tracing_stolen_model_via_deep_coupled_watermarks.md)
- [\[NeurIPS 2025\] Stochastic Regret Guarantees for Online Zeroth- and First-Order Bilevel Optimization](../../NeurIPS2025/ai_safety/stochastic_regret_guarantees_for_online_zeroth-_and_first-order_bilevel_optimiza.md)
- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)

</div>

<!-- RELATED:END -->
