---
title: >-
  [Paper Note] Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection
description: >-
  [CVPR 2025][Object Detection][Anomaly Detection] The DPDL method is proposed to learn multi-Gaussian distribution prototypes and map normal samples to the prototype space via diffusion using the Schrödinger Bridge (while concurrently pushing away anomalous samples). Combined with dispersion feature learning on hyperspherical space to enhance generalization, this method achieves state-of-the-art (SOTA) performance on 9 public anomaly detection datasets (e.g.…
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Anomaly Detection"
  - "Open-Set"
  - "Schrödinger Bridge"
  - "Distribution Prototypes"
  - "Hyperspherical Space"
date: 2026-05-08
content_hash: 5286ffcba3618f55
---

# Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection

**Conference**: CVPR 2025  
**arXiv**: [2502.20981](https://arxiv.org/abs/2502.20981)  
**Code**: None  
**Area**: Others  
**Keywords**: Anomaly Detection, Open-Set, Schrödinger Bridge, Distribution Prototypes, Hyperspherical Space

## TL;DR
The DPDL method is proposed to learn multi-Gaussian distribution prototypes and map normal samples to the prototype space via diffusion using the Schrödinger Bridge (while concurrently pushing away anomalous samples). Combined with dispersion feature learning on hyperspherical space to enhance generalization, this method achieves state-of-the-art (SOTA) performance on 9 public anomaly detection datasets (e.g., outperforming AHL by 5.0% on AITEX and 8.7% on ELPV).

## Background & Motivation

**Background**: Open-set supervised anomaly detection (OSAD) utilizes a small number of known anomaly samples to train models to detect unseen anomaly types during training. Existing methods mainly expand the coverage of anomaly distributions by generating pseudo-anomalies, such as DRA learning decoupled anomaly representations, and AHL simulating heterogeneous anomaly distributions.

**Limitations of Prior Work**: Pseudo-anomaly generation methods suffer from three key issues: (1) they cannot cover all possible anomaly distribution patterns; (2) simulated anomalies inherit the bias of in-distribution data; (3) the inherent diversity of normal samples makes it difficult to delineate the normal/anomalous boundary. These methods essentially attempt to approximate an anomaly distribution that cannot be fully known.

**Key Challenge**: Rather than trying to exhaustively cover all possible anomaly patterns (which is fundamentally impossible), it is more effective to conversely build precise models of the compact distribution boundaries of normal samples, such that any sample deviating from this boundary is classified as an anomaly.

**Goal**: How to learn a compact and discriminative normal distribution boundary amidst high diversity among normal samples, enabling robust generalization to unseen anomalies.

**Key Insight**: Utilizing multi-Gaussian distribution prototypes to cover diverse normal samples, and using the Schrödinger Bridge to learn the optimal transport mapping from normal samples to these prototypes. In the mapped space, normal samples form tight clusters while anomalies are naturally distanced.

**Core Idea**: Directly model the normal distribution rather than indirectly approximating the anomaly distribution—using learnable Gaussian prototypes and Schrödinger Bridge diffusion to map normal samples into a compact space, supplemented by hyperspherical dispersion learning to enhance sample separation.

## Method

### Overall Architecture
An input image is passed through ResNet-18 to extract intermediate features, which are then split into two branches: (1) Distribution Prototype Learning (DPL): maps normal sample features via diffusion to a multi-Gaussian prototype space through the Schrödinger Bridge, while pushing away anomalous samples in a contrastive manner; (2) Dispersion Feature Learning (DFL): utilizes a von Mises-Fisher mixture distribution in hyperspherical space to increase the directional distance between samples. Finally, a multi-instance learning module outputs the anomaly score.

### Key Designs

1. **Distribution Prototype Learning (DPL) + Schrödinger Bridge**:

    - **Function**: Mapping normal samples to a compact prototype space consisting of multiple Gaussian distributions.
    - **Mechanism**: Defining $C$ learnable Gaussian prototypes $\mathcal{P}_{MGP} = \{\mathcal{N}(\mu_c, \sigma_c)\}_{c=1}^C$, and using the Schrödinger Bridge (entropy-regularized optimal transport) to learn the diffusion path from the normal feature distribution $p_0$ to the prototype distribution $p_1$. The SB transport plan $\pi$ is decomposed into a Schrödinger potential function utilizing the Gaussian mixture, allowing for the closed-form solution of the drift function $g(x,t)$. For anomalous samples, a reverse push force is applied to keep them away from the prototypes. The key lies in the joint learning of the prototype parameters $\{\alpha_c, \mu_c, \sigma_c\}$ and the bridge function $\psi_p$.
    - **Design Motivation**: The Schrödinger Bridge is more flexible than directly fitting a Gaussian mixture, allowing it to handle mapping from high-dimensional complex distributions to simple prototype distributions. Furthermore, it possesses an inherent extrapolation capability for unseen normal samples due to the continuity of the diffusion process.

2. **Dispersion Feature Learning (DFL)**:

    - **Function**: Increasing feature dispersion among samples in the hyperspherical space to enhance OOD generalization.
    - **Mechanism**: Mapping intermediate features onto a unit hypersphere and modeling directional features using a von Mises-Fisher (vMF) mixture distribution. The dispersion of the feature space is increased by maximizing the angular distance (minimizing the cosine similarity) between samples. The directional nature of vMF is inherently suited for determining whether samples deviate from the normal direction.
    - **Design Motivation**: Preventing feature collapse (mapping all normal samples to a single point) while maintaining sufficient discriminative power and keeping normal samples compact.

3. **Multi-Instance Learning Anomaly Scoring**:

    - **Function**: Predicting anomaly scores from features.
    - **Mechanism**: Utilizing a multi-instance learning framework, where multiple patch features of an image serve as "instances" inside a "bag", comprehensively evaluating the anomaly degree of each patch and aggregating them into an image-level score.
    - **Design Motivation**: Anomalies are typically local, making MIL inherently suited for handling local anomaly localization.

### Loss & Training
The training loss consists of three components: the optimal transport loss of DPL (negative log-likelihood of normal samples mapped to prototypes + push-away loss for anomalous samples), the directional dispersion loss of DFL, and the binary classification loss of MIL. ResNet-18 is used as the backbone.

## Key Experimental Results

### Main Results (General Setting, 10 anomalous training samples)

| Dataset | DPDL | AHL | DRA | Gain (vs AHL) |
|--------|------|-----|-----|-------------|
| MVTec AD | 97.7% | 97.0% | 95.9% | +0.7% |
| Optical | 98.3% | 97.6% | 96.5% | +0.7% |
| AITEX | 97.5% | 92.5% | 89.3% | +5.0% |
| ELPV | 93.7% | 85.0% | 84.5% | +8.7% |
| Mastcam | 93.4% | 85.5% | 84.8% | +7.9% |
| Hyper-Kvasir | 93.9% | 88.0% | 83.4% | +5.9% |

### Ablation Study

| Configuration | Description |
|------|------|
| DPDL (Full) | Best performance |
| w/o DPL | Suffers the largest performance drop on industrial defect datasets, indicating that DPL is crucial for accurately modeling normal boundaries |
| w/o DFL | Generalization capability to unseen anomalies decreases, demonstrating the necessity of hyperspherical dispersion learning for OOD detection |
| w/o $\mathsf{M}_r$ (push-away loss) | Suffers the largest performance drop, showing that actively pushing away anomalous samples is key |

### Key Findings
- Improvements are limited (<1%) on less challenging datasets (MVTec AD, SDD), where anomaly patterns are simple and existing methods are near saturation.
- Significant improvements (5-8.7%) are achieved on complex datasets such as AITEX, ELPV, and Mastcam, indicating that DPDL holds a noticeable advantage in scenarios with highly diverse normal samples and complex anomaly patterns.
- DPDL maintains a large lead even when trained with only one anomalous sample ($M=1$), showing robustness under extremely sparse anomaly supervision.
- SOTA performance is also achieved under the Hard Setting (where anomaly categories are sampled from a single class only), validating its open-set generalization ability.

## Highlights & Insights
- **Paradigm Shift: From Modeling Anomalies to Modeling Normality**: Instead of attempting to exhaustively define anomaly patterns, the model precisely delineates the distribution boundaries of normal samples. This approach is more practical, as normal patterns are bounded and learnable, whereas anomaly patterns are infinite and non-exhaustive.
- **Innovative Application of Schrödinger Bridge in Anomaly Detection**: As an elegant framework for optimal transport between distributions, SB is creatively utilized to map diverse normal features to a compact prototype space. The closed-form drift function ensures highly efficient training.
- **Dual-Space Learning of Hypersphere + Gaussian Prototypes**: DFL increases separation in the directional space, while DPL tightens the normal boundary in the Euclidean space. The two complement and reinforce each other from different dimensions.

## Limitations & Future Work
- The number of Gaussian prototypes $C$ needs to be predefined, and the optimal $C$ value varies across different datasets, lacking an adaptive determination mechanism.
- The parameter $\epsilon$ for the Schrödinger Bridge is fixed at 0.001, and its sensitivity analysis is insufficient.
- The method is only applied to image-level anomaly detection and has not been extended to pixel-level anomaly segmentation.
- DPDL does not outperform AHL on 2 out of the 9 datasets (BrainMRI, HeadCT). Since these datasets have relatively simplex anomaly patterns, the complex modeling of DPDL does not yield an advantage.

## Related Work & Insights
- **vs AHL**: AHL enhances generalization by simulating heterogeneous anomaly distributions, but still relies on known anomalies to generate pseudo-anomalies; DPDL shifts to modeling the normal distribution, fundamentally bypassing the issue of incomplete anomaly coverage.
- **vs DRA**: DRA learns decoupled holiday representations (seen/pseudo/residual), but its representation capability is limited by the diversity of seen anomalies; DPDL is free from this limitation.
- **vs UAD Methods**: Unsupervised anomaly detection (UAD) methods do not use anomaly information at all. DPDL leverages a small number of anomalous samples to actively push away the decision boundary, offering an optimal compromise between UAD and supervised anomaly detection (SAD).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of Schrödinger Bridge and multi-Gaussian prototypes is highly novel, presenting a thorough paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 datasets, two settings (general + hard), and comprehensive ablation.
- Writing Quality: ⭐⭐⭐ Heavy on mathematical derivations, which may be less accessible to non-expert readers.
- Value: ⭐⭐⭐⭐ Significant practical value for real-world scenarios such as industrial anomaly detection and medical imaging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection](../../ICML2026/object_detection/mixture_prototype_flow_matching_for_open-set_supervised_anomaly_detection.md)
- [\[CVPR 2025\] One-for-More: Continual Diffusion Model for Anomaly Detection](one-for-more_continual_diffusion_model_for_anomaly_detection.md)
- [\[CVPR 2025\] Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection](generalized_diffusion_detector_mining_robust_features_from_diffusion_models_for_.md)
- [\[CVPR 2025\] SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection](simltd_simple_supervised_and_semi-supervised_long-tailed_object_detection.md)
- [\[CVPR 2025\] MulSen-AD: Multi-Sensor Object Anomaly Detection](mulsen_ad_multi_sensor_anomaly_detection.md)

</div>

<!-- RELATED:END -->
