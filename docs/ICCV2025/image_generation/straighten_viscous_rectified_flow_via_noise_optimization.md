---
title: >-
  [Paper Note] Straighten Viscous Rectified Flow via Noise Optimization
description: >-
  [ICCV 2025][Image Generation][Rectified Flow] This paper proposes VRFNO (Viscous Rectified Flow via Noise Optimization), which enhances trajectory distinguishability by introducing a historical velocity term and jointly trains an encoder to optimize noise for constructing optimal couplings, effectively straightening the inference trajectories of Rectified Flow. VRFNO achieves state-of-the-art one-step/few-step generation performance on CIFAR-10 and AFHQ (one-step FID of 4.50…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Rectified Flow"
  - "Noise Optimization"
  - "Few-step Generation"
  - "Trajectory Straightening"
  - "Velocity Field"
date: 2026-05-08
content_hash: 6098fb7daed962ba
---

# Straighten Viscous Rectified Flow via Noise Optimization

**Conference**: ICCV 2025
**arXiv**: [2507.10218](https://arxiv.org/abs/2507.10218)  
**Code**: None  
**Area**: Image Generation
**Keywords**: Rectified Flow, Noise Optimization, Few-step Generation, Trajectory Straightening, Velocity Field

## TL;DR

This paper proposes VRFNO (Viscous Rectified Flow via Noise Optimization), which enhances trajectory distinguishability by introducing a historical velocity term and jointly trains an encoder to optimize noise for constructing optimal couplings, effectively straightening the inference trajectories of Rectified Flow. VRFNO achieves state-of-the-art one-step/few-step generation performance on CIFAR-10 and AFHQ (one-step FID of 4.50, without distillation).

## Background & Motivation

### The Ideal and Reality of Rectified Flow

Rectified Flow (RF) is an ODE-based generative model whose core idea is to construct linear interpolation trajectories between noise and images and learn a constant velocity field. If the model could perfectly learn this constant velocity field, a single step would theoretically suffice to generate images from Gaussian noise. In practice, however, RF fails to learn a truly constant velocity field, leaving inference trajectories curved and requiring multi-step sampling.

### Successes and Limitations of Reflow

RF attributes trajectory curvature to the crossing of reference trajectories and proposes the Reflow operation as a remedy: first generate images with a pretrained model to construct a deterministic coupling, then train a new model on these couplings. Through in-depth analysis, the authors identify two overlooked factors behind Reflow's effectiveness:

**Advantage of deterministic coupling**: Deterministic coupling implies the existence of a learnable deterministic trajectory between noise and images, so training amounts to progressively straightening that trajectory. Randomly matched noise–image pairs lack explicit trajectory relationships, making learning harder.

**Data reuse effect**: Reflow trains by sampling intermediate states at different timesteps along the same trajectory, analogous to multi-timescale optimization in distillation.

Nevertheless, Reflow has clear drawbacks:

- **Distribution bias**: Generated images deviate from real images in distribution, accumulating errors across iterations. Typically only 2–3 rounds are feasible.
- **Storage overhead**: Large amounts of pre-generated data pairs must be stored, consuming substantial computational resources.

### Key Theoretical Insight

The authors formally prove an important result: in high-dimensional spaces, the probability of randomly matched linear interpolation trajectories crossing is $P \sim O(e^{-c(n \times n)})$, which is negligibly small. Therefore, the true cause of trajectory curvature is not "crossing" but **approximate crossing**—the intermediate states of different trajectories are statistically highly similar (especially in early stages dominated by noise), making it difficult for the model to distinguish them and leading to ambiguous predictions.

## Method

### Overall Architecture

VRFNO incorporates two core innovations within a joint encoder–velocity field training framework: (1) a Historical Velocity Term (HVT) as auxiliary input to help the model distinguish similar intermediate states, and (2) an encoder that reparameterizes noise to construct an "optimized coupling" as a substitute for Reflow's deterministic coupling, enabling training directly on real images.

### Key Designs

1. **Historical Velocity Term (HVT)**:

    - Function: The model's predicted velocity at the previous timestep is fed as auxiliary information into the velocity field model at the current timestep.
    - Mechanism: Based on Theorem 2, the velocity difference between any two trajectories exceeds their state difference:
    $\Delta(v_{ref}^{(i)}, v_{ref}^{(j)}) \geq \Delta(X_t^{(i)}, X_t^{(j)})$
      Incorporating velocity information into the model input therefore provides better trajectory discrimination. The resulting Viscous Rectified Flow takes the form:
    $dX_t = v(X_t, t, v_{history}) dt, \quad t \in [\Delta t, 1]$
      During training, $v_{history} = \text{stopgrad}(v_\theta(X_{t-\Delta t}, t-\Delta t, 0))$; during inference, the HVT is set to zero for the first step and uses the previous step's predicted velocity thereafter.
    - Design Motivation: Intermediate states from different trajectories can be very similar (especially early on), yet their directions of motion differ more substantially. Exploiting this more discriminative signal enables more accurate predictions.

2. **Noise Optimization and Optimized Coupling**:

    - Function: An encoder transforms random noise into "optimized noise" such that noise–image pairs satisfy the optimized coupling condition.
    - Mechanism: Encoder $E_\phi$ takes image $X_1$ as input and outputs mean $\mu$ and variance $\sigma^2$; optimized noise is generated via the reparameterization trick:
    $X_0 = \epsilon \cdot \sigma^2 + \mu$
      Optimized coupling is defined: $(X_0, X_1)$ constitutes an optimized coupling when $\|v_\theta(tX_1 + (1-t)X_0) - (X_1 - X_0)\| \leq \varepsilon$.
    - Design Motivation: This avoids reliance on Reflow's pre-generated images (which suffer from distribution bias) by training directly on real dataset images. The encoder identifies a noise subspace better suited to linear trajectory learning for each image, achieving a data reuse effect analogous to Reflow without being limited by data volume.

3. **Stochastic Perturbation in the Encoder**:

    - Function: Random noise $\tau \sim N(0, I)$ is injected into intermediate layers of the encoder.
    - Mechanism: Even when multiple Gaussian noise samples are matched to the same image, the reparameterized means and variances will differ, ensuring generation diversity.
    - Design Motivation: Prevents the encoder from developing memorization effects and ensures that the same image can be matched to different noise samples within the same subspace.

### Loss & Training

The total loss combines a velocity matching loss with KL regularization:

$$L(\theta, \phi) = \underbrace{\mathbb{E}_{t \in p(t)}[d(v_{ref}, v_\theta(X_t, t, v_{history}))]}_{\text{VCL}} + \alpha \underbrace{\frac{1}{2}(\sigma^2 + \mu^2 - 1 - \log(\sigma^2))}_{\text{KLL}}$$

KL regularization constrains the encoder outputs to remain close to a standard Gaussian, preventing overfitting.

Training proceeds in two stages: after convergence with MSE loss in the first stage, LPIPS loss is incorporated for joint training in the second stage until convergence. **No distillation or adversarial training is required.**

At sampling time, an image from the dataset is first fed to the encoder to reparameterize random noise, after which the VRF ODE is solved via the Euler method.

## Key Experimental Results

### Main Results

One-step and few-step generation on CIFAR-10 (without distillation):

| Method | NFE | IS↑ | FID↓ | KID(×10⁻³)↓ |
|--------|-----|-----|------|------------|
| 1-RF | 1 | 1.13 | 379 | 428 |
| 2-RF | 1 | 8.15 | 11.97 | 8.66 |
| CAF | 1 | 8.32 | 4.81 | - |
| TraFlow | 1 | - | 4.50 | - |
| **VRFNO** | **1** | **9.59** | **4.50** | **2.73** |
| 2-RF | 10 | 9.13 | 3.83 | 1.63 |
| CAF | 10 | 9.12 | 3.77 | - |
| **VRFNO** | **10** | **9.51** | **3.36** | **1.31** |

One-step generation on AFHQ at different resolutions:

| Method | Dataset | 64×64 | 128×128 | 256×256 |
|--------|---------|-------|---------|---------|
| 2-RF | AFHQ-CAT | 181.93 | 172.66 | 171.84 |
| **VRFNO** | **AFHQ-CAT** | **28.69** | **27.56** | **27.04** |
| 2-RF | AFHQ-DOG | 200.77 | 192.30 | 189.82 |
| **VRFNO** | **AFHQ-DOG** | **44.64** | **27.21** | **27.37** |

### Ablation Study

Component contributions on CIFAR-10 (FID↓):

| Config | HVT | Noise Opt. | 1-step FID | 5-step FID | 10-step FID |
|--------|-----|-----------|-----------|-----------|------------|
| A | ✗ | ✗ | 379 | 34.81 | 12.70 |
| B | ✓ | ✗ | 332 | 32.50 | 9.34 |
| C | ✗ | ✓ | 4.72 | 4.28 | 4.75 |
| D | ✓ | ✓ | **4.53** | **4.03** | **3.40** |

Trajectory straightness (NFSS↓, lower is straighter):

| Dataset | 2-RF | 3-RF | CAF | VRFNO |
|---------|------|------|-----|-------|
| 2D | 0.067 | 0.053 | 0.058 | **0.054** |
| CIFAR-10 | 0.058 | 0.056 | 0.035 | **0.026** |

### Key Findings

- Noise optimization is the primary source of performance gains: used alone, it reduces one-step FID from 379 to 4.72.
- HVT yields consistent improvements across all step counts, particularly at 10 steps (12.70→9.34 without noise optimization; 4.75→3.40 with noise optimization).
- VRFNO produces the straightest inference trajectories under the NFSS metric (0.026 vs. CAF's 0.035).
- On AFHQ, VRFNO improves over 2-RF by an order of magnitude (e.g., CAT 256×256: 27.04 vs. 171.84).
- The encoder has very few parameters (less than 1/20 of the velocity field model); inference overhead is slightly higher than RF but lower than CAF.

## Highlights & Insights

- The in-depth analysis of why Reflow succeeds is highly valuable: it corrects the intuitive explanation of "trajectory crossing" to "statistical similarity confusion caused by approximate crossing" and provides theoretical support.
- The concept of optimized coupling is more flexible than deterministic coupling: it requires no pretrained model to generate images, incurs no error accumulation, and the encoder's noise subspace focusing effect naturally achieves data reuse.
- The two-stage training strategy (MSE → MSE + LPIPS) is simple and effective, avoiding the complexity of distillation and adversarial training.
- The introduction of the historical velocity term adds modest computational cost but is theoretically well-motivated from an information-theoretic perspective: it provides a prior on the trajectory direction.

## Limitations & Future Work

- Sampling requires drawing one image from the dataset as encoder input, limiting fully unconditional generation scenarios.
- Validation is currently restricted to CIFAR-10 and AFHQ; the method has not yet been extended to large-scale datasets such as ImageNet or high-resolution settings.
- The additional inference time from the encoder (0.305s per one-step generation vs. 0.172s for RF) may become a bottleneck in latency-critical applications.
- The timing for introducing LPIPS in the second stage relies on empirical judgment (switching after convergence), lacking an automated mechanism.

## Related Work & Insights

- VRFNO is most similar to CAF (Constant Acceleration Flow), but CAF requires two velocity field models, resulting in slower inference, whereas VRFNO uses only one velocity field and a lightweight encoder.
- The noise optimization approach differs from prior methods that iteratively optimize noise sample by sample: the encoder performs a linear transformation in a single pass, greatly reducing the number of iterations.
- The joint training framework preserves marginal distribution invariance (Theorem 3), theoretically guaranteeing generation quality.
- The historical velocity term concept can be generalized to other ODE-based generative models.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs](../../ICML2026/image_generation/offline_preference_optimization_for_rectified_flow_with_noise-tracked_pairs.md)
- [\[NeurIPS 2025\] GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer](../../NeurIPS2025/image_generation/guideflow3d_optimization-guided_rectified_flow_for_appearance_transfer.md)
- [\[ICCV 2025\] PINO: Person-Interaction Noise Optimization for Long-Duration and Customizable Motion Generation of Arbitrary-Sized Groups](pino_person-interaction_noise_optimization_for_long-duration_and_customizable_mo.md)
- [\[ICCV 2025\] ReFlex: Text-Guided Editing of Real Images in Rectified Flow via Mid-Step Feature Extraction and Attention Adaptation](reflex_text-guided_editing_of_real_images_in_rectified_flow_via_mid-step_feature.md)
- [\[NeurIPS 2025\] Rectified-CFG++ for Flow Based Models](../../NeurIPS2025/image_generation/rectified-cfg_for_flow_based_models.md)

</div>

<!-- RELATED:END -->
