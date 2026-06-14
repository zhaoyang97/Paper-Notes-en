---
title: >-
  [Paper Note] NAPPure: Adversarial Purification for Robust Image Classification under Non-Additive Perturbations
description: >-
  [ICCV 2025][adversarial purification] This paper proposes NAPPure, a framework that jointly optimizes the underlying clean image and perturbation parameters via likelihood maximization…
tags:
  - "ICCV 2025"
  - "adversarial purification"
  - "non-additive perturbations"
  - "diffusion models"
  - "robust classification"
  - "blur attacks"
  - "occlusion attacks"
  - "geometric distortion"
date: 2026-05-08
content_hash: 4f07071cf0f64884
---

# NAPPure: Adversarial Purification for Robust Image Classification under Non-Additive Perturbations

**Conference**: ICCV 2025  
**Code**: None  
**Area**: Other  
**Keywords**: adversarial purification, non-additive perturbations, diffusion models, robust classification, blur attacks, occlusion attacks, geometric distortion

## TL;DR

This paper proposes NAPPure, a framework that jointly optimizes the underlying clean image and perturbation parameters via likelihood maximization, extending adversarial purification beyond additive perturbations to handle blur, occlusion, and geometric distortion. NAPPure achieves an average robust accuracy of 73.93% on GTSRB, compared to only 43.2% for conventional methods.

## Background & Motivation

**Additive vs. Non-Additive Perturbations:**

- Existing adversarial attacks and defenses predominantly focus on **additive perturbations** ($x_{adv} = x + \epsilon$), i.e., directly adding noise to an image.
- However, **non-additive perturbations** are equally prevalent and dangerous in real-world scenarios: blur (e.g., optical films), occlusion (e.g., stickers/patches), and geometric distortion (e.g., warping).
- These non-additive perturbations have been shown to effectively fool classifiers.

**Failure of Existing Adversarial Purification Methods:**

- Methods such as DiffPure and LM inherently assume additive perturbations (i.e., that the perturbed image lies near the clean image in $l_2$ space).
- For non-additive perturbations, the $l_2$ distance between the perturbed and clean images can be large, leading to **semantic drift** during purification—the model may reconstruct the image into content belonging to a different class.
- For example, after a blur attack, DiffPure fails to recover sharp edges and textures.

**Core Insight:** Given knowledge of the perturbation type (e.g., blur/occlusion/distortion), the perturbation can be modeled as a parameterized transformation $x_{adv} = f(x, \epsilon)$, and the underlying clean image and perturbation parameters can be disentangled via likelihood maximization.

## Method

### Overall Architecture

NAPPure purifies adversarial images by jointly optimizing the clean image $x$ and perturbation parameters $\epsilon$, with the optimization objective derived from a Bayesian decomposition:

$$\log p(x, \epsilon | x_{adv}) \propto \log p(x) + \log p(\epsilon) + \log p(x_{adv} | x, \epsilon)$$

### Three Optimization Terms

**1. Image Likelihood Term $\log p(x)$:**  
An EDM diffusion model is used to estimate the data distribution, approximated via the ELBO:

$$-\mathbb{E}_{\sigma, n} [\lambda(\sigma) \| D_\theta(x_\sigma; \sigma) - x \|_2^2]$$

This drives the image toward high-density regions of the data distribution, removing potential perturbations.

**2. Perturbation Prior Term $\log p(\epsilon)$:**  
Represented using an energy-based model, this term constrains the perturbation magnitude:

$$\log p(\epsilon) = -\phi(\epsilon) - \log Z$$

The potential function $\phi(\epsilon)$ attains its minimum at the identity element of the transformation (e.g., $\epsilon_0 = 0$ for additive perturbations, the identity kernel for convolutions).

**3. Image Reconstruction Term $\log p(x_{adv} | x, \epsilon)$:**  
This term constrains the solution to be consistent with the known transformation, preventing semantic drift:

$$\log p(x_{adv} | x, \epsilon) = -\frac{1}{2\sigma^2} \| x_{adv} - f(x, \epsilon) \|_2^2 + C_\sigma$$

### Final Optimization Objective

$$\min_{x, \epsilon} \mathbb{E}_{\sigma, n} \| D_\theta(x_\sigma; \sigma) - x \|_2^2 + \lambda_1 \cdot \phi(\epsilon) + \lambda_2 \cdot \| x_{adv} - f(x, \epsilon) \|_2^2$$

The Adam optimizer is used to alternately update $x$ and $\epsilon$ for $T=500$ iterations.

### Instantiation for Different Perturbation Types

| Perturbation Type | Transform $f$ | Identity $\epsilon_0$ | Potential $\phi$ |
|---|---|---|---|
| Additive | $x + \epsilon$ | $0$ | $\|\epsilon\|_2^2$ |
| Convolution/Blur | $x * \epsilon$ | Identity kernel | $\|\epsilon - \epsilon_0\|_2^2$ |
| Patch/Occlusion | $x \cdot (1-m) + p \cdot m$ | $(x_{adv}, h/2, w/2, 0)$ | $|s|$ |
| Optical Flow/Warp | Optical flow transform | $0$ | $\|\epsilon\|_2^2$ |

### Handling Composite Perturbations (NAPPure-joint)

When multiple perturbation types are present simultaneously, a learnable weight $w \in [0,1]$ is introduced, replacing each base transformation with an interpolated form:

$$\hat{f}(x, \hat{\epsilon}) = w \cdot f(x, \epsilon) + (1-w) \cdot x$$

The composite transformation is constructed as $f = \hat{f}_n \circ \cdots \circ \hat{f}_1$.

### Theoretical Reduction to the Additive Case

When $f(x, \epsilon) = x + \epsilon$ and $p(\epsilon)$ is uniform, NAPPure reduces to $\max_x \log p(x)$, which is equivalent to the standard adversarial purification method LM. This demonstrates that NAPPure is a compatible generalization of conventional approaches.

## Key Experimental Results

### Main Results: Robust Accuracy on GTSRB

| Defense | Convolution | Patch | Optical Flow | Additive | **Average** |
|---|---|---|---|---|---|
| No Defense | 57.42 | 13.67 | 1.56 | 3.12 | 18.95 |
| AT | 61.72 | 19.92 | 19.72 | 47.85 | 37.30 |
| DiffPure | 61.52 | 46.29 | 21.88 | 60.74 | 47.61 |
| LM | 53.32 | 13.67 | 8.79 | 79.07 | 38.71 |
| **NAPPure** | **86.91** | **74.22** | **51.37** | **83.20** | **73.93** |
| NAPPure-joint | 76.17 | 57.23 | 37.37 | 66.40 | 59.29 |

- NAPPure **substantially outperforms** all baselines on every non-additive attack type (average +26.3% vs. DiffPure).
- NAPPure also maintains competitive performance on additive attacks (83.20%).
- NAPPure-joint (without knowledge of the specific attack type) still clearly outperforms all baselines.

### Robust Accuracy on CIFAR-10

| Defense | Convolution | Patch | Optical Flow | Additive | **Average** |
|---|---|---|---|---|---|
| DiffPure | 59.38 | 69.73 | 23.06 | 79.10 | 57.82 |
| LM | 60.16 | 36.13 | 13.09 | 70.12 | 44.88 |
| **NAPPure** | **66.40** | **76.75** | **48.24** | **82.81** | **66.94** |

- The effectiveness of NAPPure is confirmed on CIFAR-10 as well.
- The improvement under optical flow attacks is most pronounced (48.24% vs. 23.06% for DiffPure).

### Composite Attack Experiments (GTSRB)

| Defense | Robust Accuracy |
|---|---|
| No Defense | 12.70 |
| DiffPure | 30.00 |
| LM | 15.82 |
| NAPPure | 37.10 |
| **NAPPure-joint** | **54.49** |

- Under the most challenging setting with all four perturbation types applied simultaneously, NAPPure-joint achieves 54.49%.
- NAPPure-joint outperforms NAPPure under composite attacks, validating the effectiveness of the interpolation technique.

### Key Findings

1. Conventional adversarial purification methods suffer dramatic performance degradation under non-additive perturbations (DiffPure achieves only 21.88% on GTSRB under optical flow attacks).
2. $\lambda_1=0.01, \lambda_2=5$ constitutes the optimal hyperparameter configuration for optical flow attacks.
3. Excessively large $\lambda_1$ over-constrains the perturbation; excessively small $\lambda_1$ introduces new artifacts. Excessively large $\lambda_2$ limits purification flexibility; excessively small $\lambda_2$ causes semantic drift.
4. Improvements on GTSRB are more pronounced than on CIFAR-10, as traffic signs rely heavily on sharp shape boundaries.

## Highlights & Insights

1. **Valuable Problem Formulation**: This is the first systematic study of adversarial purification under non-additive perturbations, addressing an important gap in the literature.
2. **Unified Framework**: The Bayesian decomposition naturally unifies the treatment of different perturbation types while remaining theoretically compatible with conventional additive methods.
3. **Modular Design**: The handling modules for different perturbation types are plug-and-play, offering strong extensibility.
4. **Reduction Theorem**: The proof that NAPPure reduces to LM in the additive case strengthens its theoretical completeness.
5. **Handling Composite Attacks**: NAPPure-joint provides a practical defense when the attack type is unknown.

## Limitations & Future Work

1. **Assumes Known Perturbation Type**: NAPPure requires prior knowledge of the perturbation type, which may not be available in practice.
2. **High Computational Cost**: The 500-step iterative optimization is substantially slower than forward inference, limiting real-time applicability.
3. **Hyperparameter Sensitivity**: $\lambda_1$ and $\lambda_2$ require tuning for different attack types.
4. **Evaluation Limited to $32\times32$ Images**: Whether the approach generalizes to high-resolution images remains unexplored.
5. **Patch Attack Handling**: An auxiliary model must be separately trained to approximate the non-differentiable transformation.

## Related Work & Insights

- **DiffPure / LM**: Standard adversarial purification methods applicable only to additive perturbations.
- **Adversarial Training**: Can naturally extend to non-additive perturbations but generalizes poorly to unseen attack types.
- **Image Restoration**: Tasks such as deblurring and inpainting are related but do not consider adversarial scenarios.
- **Inspiration**: The idea of explicitly modeling perturbation parameters may be applicable to other tasks requiring inverse transformation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — A unified framework extending adversarial purification to non-additive perturbations with clear contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two datasets, four attack types, composite attacks, and thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous mathematical derivations with a good balance between theory and practice.
- **Value**: ⭐⭐⭐⭐ — Addresses the gap in defending against non-additive attacks in practical scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](../../AAAI2026/others/boosting_adversarial_transferability_via_ensemble_non-attention.md)
- [\[ICCV 2025\] Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents](adversarial_data_augmentation_for_single_domain_generalization_via_lyapunov_expo.md)
- [\[ICLR 2026\] Distributionally Robust Classification for Multi-Source Unsupervised Domain Adaptation](../../ICLR2026/others/distributionally_robust_classification_for_multi-source_unsupervised_domain_adap.md)
- [\[ICCV 2025\] Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval](learning_visual_hierarchies_in_hyperbolic_space_for_image_retrieval.md)
- [\[ICCV 2025\] Is Meta-Learning Out? Rethinking Unsupervised Few-Shot Classification with Limited Entropy](is_meta-learning_out_rethinking_unsupervised_few-shot_classification_with_limite.md)

</div>

<!-- RELATED:END -->
