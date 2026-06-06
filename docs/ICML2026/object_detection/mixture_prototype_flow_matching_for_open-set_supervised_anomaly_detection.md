---
title: >-
  [Paper Note] Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection
description: >-
  [ICML 2026][Object Detection][Open-Set Supervised Anomaly Detection] MPFM replaces the traditional "unimodal Gaussian prototype" in OSAD with a learnable **Gaussian Mixture Prototype Space**. It employs flow matching to…
tags:
  - "ICML 2026"
  - "Object Detection"
  - "Open-Set Supervised Anomaly Detection"
  - "Flow Matching"
  - "Gaussian Mixture Prototypes"
  - "Mutual Information Regularization"
  - "OSAD"
date: 2026-05-08
content_hash: 91fa8d485c32e79a
---

# Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection

**Conference**: ICML 2026  
**arXiv**: [2605.02438](https://arxiv.org/abs/2605.02438)  
**Code**: https://github.com/fuyunwang/MPFM-OSAD  
**Area**: Anomaly Detection / Flow Matching / Prototype Learning  
**Keywords**: Open-Set Supervised Anomaly Detection, Flow Matching, Gaussian Mixture Prototypes, Mutual Information Regularization, OSAD

## TL;DR
MPFM replaces the traditional "unimodal Gaussian prototype" in OSAD with a learnable **Gaussian Mixture Prototype Space**. It employs flow matching to directly regress a velocity field in GMM form, supplemented by a mutual information maximization regularizer to prevent prototype collapse. It outperforms all SOTA methods like DRA, AHL, and DPDL on 9 industrial/medical AD datasets under 10/1 anomaly sample settings.

## Background & Motivation

**Background**: Anomaly detection is categorized into three paradigms: unsupervised AD (normal samples only), few-shot AD (extremely few normal samples), and supervised AD (massive anomaly samples). Open-Set Supervised AD (OSAD) is a compromise: training involves a small amount of labeled anomaly images (image-level) and a large amount of normal images, while testing requires identifying both **known and unseen** anomalies. Existing OSAD methods are divided into: (a) data augmentation + outlier exposure (DRA), (b) heterogeneous simulation (AHL), and (c) prototype learning + generative dynamics (DPDL).

**Limitations of Prior Work**: DPDL, the closest SOTA method in the third category, utilizes **simple unimodal Gaussians** as the prototype distribution for normal samples and "pulls" normal features toward these prototypes via diffusion bridges. However, normal samples in industrial scenarios are inherently **multimodal** (e.g., variations in sub-patterns, angles, or lighting within the same category). Forcing a unimodal Gaussian fit treats rare but legitimate normal sub-patterns as anomalies, leading to high false positives and blurred decision boundaries.

**Key Challenge**: The goal is to ensure **compact** density estimation for normal samples (high recall on normal) while allowing the boundary to **extrapolate to unseen anomaly** types (high recall on unknown anomalies). Unimodal Gaussian priors directly undermine the first objective, while discrete multiple Gaussians lack structural correlation, causing a loss of semantic continuity as the model fails to capture transitions between modes.

**Goal**: (1) Replace unimodal Gaussians with a **continuous, structured** multimodal prototype space; (2) Ensure the flow matching velocity field is inherently multimodal rather than regressing a single mean vector; (3) Prevent multiple Gaussian components from collapsing into the same mode.

**Key Insight**: Explicitly model the prototype space as a GMM and learn a continuous transport mapping from the normal feature distribution to this GMM using flow matching. A key observation is that under a standard linear noise schedule, a single-step reverse transition preserves GMM closure (the closure of linear-Gaussian systems), enabling closed-form step-wise sampling without numerical integration.

**Core Idea**: **Use a velocity field in GMM form instead of a single velocity vector for flow matching to propagate multimodal priors from the prior to the transport dynamics, and expand GMM components using mutual information maximization.**

## Method

### Overall Architecture

Input: Training set $\mathcal{Z}_{tr} = \mathcal{Z}_{tr}^{n} \cup \mathcal{Z}_{tr}^{a}$ (Normal $N$ + Anomaly $M$, $N \gg M$, image-level labels only); Output: Anomaly score $S(z)$ for test sample $z \in \mathcal{Z}_{te}$.

Pipeline:
1. **Feature Extraction**: Use a ResNet-18 backbone $f: \mathcal{Z} \to \mathbb{R}^d$ to map images to 1D features.
2. **K-means++ Initialization for GMM**: Run K-means++ on $\mathcal{F}_{tr}^{n}$ to obtain $K$ cluster centers $\mu_k$, mixing weights $\pi_k = |C_k| / N$, and shared variance $s^2 = \frac{1}{dN} \sum_k \sum_{i \in C_k} \| z_0^{n,i} - \mu_k \|_2^2$. This breaks the cyclic dependency between flow network optimization and GMM parameters.
3. **Flow Matching Training**: Under a linear noise schedule $\alpha_t = 1-t, \sigma_t = t$, the velocity is $u = \frac{z_T - z_0}{t}$. Maximize the GMM velocity field likelihood for normal samples and **reversely** minimize it for anomalies (pushing anomalous velocity away from the normal distribution).
4. **MIMR Regularization**: Maximize the mutual information between "prototype assignment $c$" and "flowed features $\psi(z_0^{n,i})$" for normal samples to achieve confident assignment and balanced usage.
5. **Four-module Anomaly Score Prediction**: Global $M_g$ (GMM NLL) + Local $M_a$ (top-O patch scores) + Normal $M_n$ (global pooling) + Residual $M_r$ ($(\psi(z) - \mu_{c^*})/s$ through a classification head). Inference: $S(z) = S_g + S_a + S_r - S_n$.

### Key Designs

1. **Mixture Prototype Flow Learning (MPFL): GMM-based Velocity Field**:
    - **Function**: Replaces the traditional "flow matching $\rightarrow$ unimodal Gaussian" mapping with "flow matching $\rightarrow$ multimodal GMM," where each component corresponds to a normal sub-pattern.
    - **Mechanism**: Instead of using a unimodal conditional Gaussian $q_\theta(u | z_t) = \mathcal{N}(u; \mu_\theta(z_t), s^2 I)$, the velocity field is modeled as a Gaussian Mixture $q_\theta(u | z_t^{n,i}) = \sum_{k=1}^K \pi_k(z_t^{n,i}; \theta) \mathcal{N}(u; \mu_k(z_t^{n,i}; \theta), s^2 I)$, where weights $\pi_k$ and means $\mu_k$ are functions of input features. Training uses NLL: $\mathcal{L}_{NLL} = \mathbb{E} [-\log \sum_k \pi_k \mathcal{N}(u; \mu_k, s^2 I)]$. Under a linear schedule, the reverse transition $q_\theta(z_{t-\Delta t}^{n,i} | z_t^{n,i})$ remains a GMM with coefficients determined by the schedule, eliminating the need for numerical ODE solvers.
    - **Design Motivation**: Unimodal velocity fields force all normal samples toward a single mean, flattening multimodal sub-patterns—the root cause of high false positives in DPDL for industrial AD. Encoding multimodal structures into the velocity field addresses this fundamentally.

2. **Bi-directional Flow Loss**:
    - **Function**: Attracts normal samples to the GMM prototype space while repelling anomalies.
    - **Mechanism**: Normal samples use standard NLL $\mathcal{L}_{flow}^{n} = \mathbb{E}[-\log q_\theta(u | z_t^{n,i})]$, while anomalies use the **negated** $\mathcal{L}_{flow}^{a} = \mathbb{E}[\log q_\theta(u | z_t^{a,i})]$. Maximizing NLL for anomalies ensures their velocity distribution stays away from the normal GMM.
    - **Design Motivation**: Traditional OSAD uses contrastive loss or binary classification in the final feature space; this method applies repulsion at the **velocity field level**, affecting the entire transport trajectory with gradient signals at every step $t$.

3. **Mutual Information Maximization Regularizer (MIMR)**:
    - **Function**: Prevents GMM components from collapsing into the same mode and maintains discriminability.
    - **Mechanism**: Utilizes the GMM posterior $p(c=k | \psi(z_0^{n,i})) = \frac{\pi_k \mathcal{N}(\psi(z_0^{n,i}); \mu_k, s^2 I)}{\sum_j \pi_j \mathcal{N}(\psi(z_0^{n,i}); \mu_j, s^2 I)}$ to maximize mutual information $I(\psi(z_0^{n,i}); c) = H(c) - H(c | \psi(z_0^{n,i}))$. The loss is $\mathcal{L}_{mim} = \mathbb{E}[\sum_k p(c=k|\cdot) \log p(c=k|\cdot)] - \sum_k \pi_k \log \pi_k$. The first term **minimizes** conditional entropy (confident assignment), and the second **maximizes** marginal entropy (balanced usage).
    - **Design Motivation**: Applying entropy constraints directly to the GMM posterior avoids extra discriminative heads or adversarial training.

### Loss & Training

Total Loss:
$\mathcal{L} = \underbrace{\mathcal{L}_{M_a} + \mathcal{L}_{M_n} + \mathcal{L}_{M_r} + \mathcal{L}_{M_g}}_{\text{Anomaly Score Modules}} + \underbrace{\mathcal{L}_{flow}^{n} + \mathcal{L}_{flow}^{a}}_{\text{Flow Matching}} + \lambda \mathcal{L}_{mim}$.

The four anomaly score modules use binary classification loss (BCE) and are trained jointly. $\lambda$ controls MIMR intensity. During inference, $S(z) = S_g(z) + S_a(z) + S_r(z) - S_n(z)$, where $S_n$ is subtracted as it represents the "normality degree."

## Key Experimental Results

### Main Results

Performance (AUC) on 9 AD datasets (MVTec AD, Optical, SDD, etc.) using the general setting of 10 training anomaly samples.

| Dataset | DRA | AHL | DPDL | **MPFM (Ours)** |
|--------|------|------|------|------|
| MVTec AD | 0.959±0.003 | 0.970±0.002 | 0.977±0.002 | **0.982±0.003** |
| Optical | 0.965±0.006 | 0.976±0.004 | 0.983±0.005 | **0.992±0.002** |
| SDD | 0.991±0.005 | — | — | (Best, see paper) |

MPFM ranks first or tied for first across all datasets, with a 0.5% gain over DPDL on MVTec AD and a 0.9% gain on Optical (within the 0.97+ AUC saturation zone).

### Ablation Study

| Configuration | Key Indicator | Description |
|------|---------|------|
| Full MPFM | Best AUC | Complete model. |
| w/o GMM | Significant drop | Validates multimodal velocity field (equivalent to DPDL approach). |
| w/o MIMR | Moderate drop | Validates the anti-collapse regularizer. |
| w/o Reverse flow loss | Large drop | Loss of repulsion at the transport level. |

### Key Findings

- **Multimodal Velocity Field is the Lead Contributor**: Removing the GMM structure (reverting to unimodal) caused the most significant performance drop, confirming that unimodal Gaussians cannot capture sub-pattern diversity.
- **K-means++ Initialization is Critical**: Pre-initializing GMM parameters is essential to break the coupling between the flow network and GMM parameters.
- **MIMR vs. Distance Regularization**: MIMR is more refined than simple mean-distance constraints as it simultaneously manages confident assignment and balanced prototype usage.
- **Shared Variance $s^2$ ensures Stability**: Using a shared variance across components prevents variance explosion or collapse, as discussed in the supplementary material.

## Highlights & Insights

- **Pervasive Multimodality**: Unlike previous methods where only the flow endpoint is a GMM, MPFM ensures the flow process itself is a GMM velocity field, maintaining semantic structure throughout the dynamics.
- **Analytical GMM Closure**: By leveraging the properties of linear-Gaussian systems, the reverse transition is entirely closed-form, allowing inference without numerical ODE solvers.
- **Reverse Flow Matching Loss**: Elegantly implements transport-level repulsion by negating the log-likelihood for anomalies without requiring adversarial training.
- **Subtraction-based Score Combination**: The $S = S_g + S_a + S_r - S_n$ formulation combines perspectives of both "anomalousness" and "non-normality," which is highly effective in practice.

## Limitations & Future Work

- **Hyperparameter $K$**: The number of components $K$ requires manual setting or K-means++ pre-calculation; adaptive selection for different categories is not explored.
- **Backbone Dependency**: Evaluated primarily on ResNet-18; the impact of stronger backbones like CLIP or DINOv2 on prototype structure remains to be seen.
- **MIMR Scope**: MIMR only considers normal samples; additional constraints to ensure anomalies do not occupy any prototypes could be beneficial.
- **Pixel-level Localization**: The study focuses on image-level AUC and lacks evaluation for pixel-level anomaly localization.

## Related Work & Insights

- **vs. DPDL**: DPDL uses independent unimodal Gaussians with simple velocity vectors; MPFM uses a structured continuous GMM velocity field with semantic associations.
- **vs. DRA / AHL**: Those methods focus on augmenting or simulating anomalies (data-centric); MPFM improves normal sample modeling (model-centric).
- **vs. Rectified Flow / SD3**: While the latter are generative, MPFM repurposes flow matching as a discriminative representation framework and density estimator.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of GMM velocity fields, MIMR, and reverse flow loss is a fresh ensemble for OSAD.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive testing on 9 datasets with multiple seeds, though missing pixel-level metrics.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivations and clear motivation via visualization.
- Value: ⭐⭐⭐⭐ Provides a robust baseline for industrial AD and a transferable design philosophy for diffusion-based representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection](../../ICCV2025/object_detection/3dmood_lifting_2d_to_3d_for_monocular_openset_object_detecti.md)
- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](../../AAAI2026/object_detection/vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](../../ICCV2025/object_detection/dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)
- [\[ICLR 2026\] SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](../../ICLR2026/object_detection/spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)
- [\[ICML 2026\] Testing the Test: Score-Direction Instability in Class-Split Anomaly Detection](testing_the_test_score-direction_instability_in_class-split_anomaly_detection.md)

</div>

<!-- RELATED:END -->
