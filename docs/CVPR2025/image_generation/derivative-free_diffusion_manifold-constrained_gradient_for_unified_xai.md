---
title: >-
  [Paper Note] Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI
description: >-
  [CVPR 2025][Image Generation][Derivative-free gradient estimation] This paper proposes FreeMCG, which utilizes diffusion models to generate particle sets on the data manifold and combines them with the Ensemble Kalman Filter (EnKF) to approximate the projection of model gradients onto the manifold. FreeMCG represents the first framework to unify feature attribution and counterfactual explanations while requiring only black-box model access.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Derivative-free gradient estimation"
  - "diffusion models"
  - "Ensemble Kalman Filter"
  - "counterfactual explanations"
  - "feature attribution"
date: 2026-05-08
content_hash: 852e7ad8a77413da
---

# Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI

**Conference**: CVPR 2025  
**arXiv**: [2411.15265](https://arxiv.org/abs/2411.15265)  
**Code**: None  
**Area**: Explainable AI / XAI  
**Keywords**: Derivative-free gradient estimation, diffusion models, Ensemble Kalman Filter, counterfactual explanations, feature attribution

## TL;DR
This paper proposes FreeMCG, which utilizes diffusion models to generate particle sets on the data manifold and combines them with the Ensemble Kalman Filter (EnKF) to approximate the projection of model gradients onto the manifold. FreeMCG represents the first framework to unify feature attribution and counterfactual explanations while requiring only black-box model access.

## Background & Motivation

**Background**: The most fundamental tool in explainable AI is the model gradient—gradients with respect to inputs are used for feature attribution (determining which pixels are important) and counterfactual explanations (altering the input to flip the prediction). Traditionally, these two tasks have been treated independently in prior literature.

**Limitations of Prior Work**: Gradient-based methods suffer from three main limitations: (1) They require white-box model access (gradients cannot be calculated without knowing the model weights), whereas many models deployed in real-world scenarios are black-boxes; (2) Gradient-based explanations are often unreliable, being closely related to adversarial perturbations and generating updates that deviate from the data manifold; (3) Raw gradients cannot naturally unify feature attribution and counterfactual explanations.

**Key Challenge**: In high-dimensional image space, gradients can point in any direction, whereas meaningful semantic changes occur only along a lower-dimensional data manifold. Consequently, a substantial portion of the gradient information points "off-manifold."

**Goal**: To design a method that requires no model gradient access, yields approximations of manifold-constrained gradients, and can be applied to both feature attribution and counterfactual explanations.

**Key Insight**: Multiplying the gradient by the particle covariance $C_{xx}$ naturally projects it onto the tangent space of the data manifold (Theorem 1). EnKF theory indicates that $C_{xx} \nabla_x f$ can be approximated via $C_{xf}$ (Theorem 2), which completely bypasses gradient computations.

**Core Idea**: Generate particles near the input on the manifold using a diffusion model, and calculate the input-output cross-covariance to approximate the manifold-constrained gradient, thereby achieving gradient-free unified XAI.

## Method

### Overall Architecture
(1) Add noise to input $x$ to obtain $x_t^{(k)}$; (2) Apply Tweedie's formula for denoising to generate manifold particles $x_{0|t}^{(k)}$; (3) Query the black-box classifier to obtain $f(x_{0|t}^{(k)})$; (4) Compute the FreeMCG gradient by multiplying $C_{xf}$ by $(e_c - p)$; (5) Utilize this estimated gradient for feature attribution or counterfactual generation.

### Key Designs

1. **Theoretical Guarantees of Manifold-Constrained Gradients**:

    - Function: Proven assurance that the covariance-preconditioned gradient lies on the data manifold.
    - Mechanism: **Theorem 1** demonstrates that when particles $x^{(k)}$ reside on the manifold, the empirical covariance $C_{xx}$ expands along the tangent space and contracts along the normal directions. Thus, $C_{xx} \nabla_x \log p$ is constrained within the tangent space, avoiding adversarial directions.
    - Design Motivation: To provide rigorous theoretical guarantees that FreeMCG produces semantically meaningful, human-interpretable explanations.

2. **Gradient-Free Approximation via EnKF**:

    - Function: Estimating manifold-constrained gradients without retrieving backpropagation-derived model gradients.
    - Mechanism: **Theorem 2** proves that $C_{xf} = C_{xx} \nabla_x f^T + O(\delta^3)$, where $C_{xf} = \frac{1}{K}\sum_k (x^{(k)} - \bar{x})(f(x^{(k)}) - \bar{f})^T$ requires only the particle states and the corresponding forward outputs of the model. The error term $O(\delta^3)$ becomes negligible when particles are locally clustered.
    - Design Motivation: Black-box compatibility significantly expands the applicability of explanation methods.

3. **Generating Manifold Particles via Diffusion Models**:

    - Function: Efficiently generating particles that are near the original input, reside on the manifold, and can be sampled rapidly.
    - Mechanism: Perturb $x$ with a moderate noise level to obtain $x_t$, then apply the Tweedie formula $D_\theta(x_t) \approx \mathbb{E}[x_0|x_t]$ to project it back to the manifold in a single step, which is substantially faster than full reverse generation.
    - Design Motivation: Since particle quality directly determines the approximation accuracy and explanation fidelity, diffusion models are utilized as the strongest data manifold estimators currently available.

### Loss & Training
FreeMCG is training-free. It utilizes pre-trained diffusion models and classifiers, only requiring forward computations during inference. The key hyperparameters are the noise level $\sigma_t$ and the particle size $K$.

## Key Experimental Results

### Feature Attribution (ImageNet, Deletion/Insertion Metrics)

| Method | Model Access | Performance |
|------|---------|------|
| Vanilla Gradient | White-box | Baseline |
| Integrated Gradients | White-box | Better |
| FreeMCG (Ours) | Black-box | SOTA |

### Counterfactual Explanations (ImageNet, FID/LPIPS/Flip Rate)

| Method | Model Access | Performance |
|------|---------|------|
| DiME | White-box | Baseline |
| ACE | White-box | Moderate |
| FreeMCG (Ours) | Black-box | SOTA |

### Key Findings
- FreeMCG, as a black-box method, outperforms white-box baselines on both tasks, demonstrating that manifold constraints are more crucial than raw gradient access.
- Counterfactual explanations produce semantically meaningful updates rather than pixel-level adversarial perturbations.
- The noise level $\sigma_t$ controls the trade-off between particle dispersion and the quality of the manifold projection.
- Scaling up the particle count $K$ improves gradient approximation fidelity, albeit with diminishing returns.

## Highlights & Insights
- **Unified XAI Framework**: This work is the first to unify feature attribution and counterfactual explanations under a single mathematical formulation. The core insight—that "good gradients should reside on the manifold"—is elegant and intuitive.
- **Theory-Driven Design**: Theorem 1 + Theorem 2 establish a rigorous theoretical pipeline, giving the method a solid mathematical foundation.
- **Black-box Beats White-box**: The bottleneck of white-box gradients is not the lack of information, but rather the inclusion of too much off-manifold noise. Imposing a manifold-constraint projection turns out to be an advantage.

## Limitations & Future Work
- It requires a pre-trained diffusion model, which necessitates additional training when adapting to niche data domains.
- Evaluating $K$ particles requires separate classifier forward passes for each inference step, resulting in high computational overhead.
- The theoretical formulation assumes a locally linear manifold, which might degrade in highly curved areas of the manifold.
- Currently, experiments are restricted to classification tasks.

## Related Work & Insights
- **vs Integrated Gradients**: IG requires white-box access and integrates along a straight line path, whereas FreeMCG operates on black-box settings and projects onto the data manifold.
- **vs DiME/ACE**: Existing counterfactual generation methods necessitate model gradients or rely on adversarially robust classifiers.
- **vs EnKF Diffusion Guidance**: Unlike Zheng et al. who utilize EnKF for image inverse tasks, this work extends the formulation to XAI.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elegant theory, a pioneering unified framework, with the surprising result that black-box beats white-box.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluations on two signature XAI tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Refined theoretical proofs and well-structured, logical motivations.
- Value: ⭐⭐⭐⭐⭐ Highly influential potential for the broader XAI community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Training-Free Constrained Generation with Stable Diffusion Models](../../NeurIPS2025/image_generation/training-free_constrained_generation_with_stable_diffusion_models.md)
- [\[ICML 2025\] Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning](../../ICML2025/image_generation/local_manifold_approximation_and_projection_for_manifold-aware_diffusion_plannin.md)
- [\[CVPR 2025\] Decoupling Training-Free Guided Diffusion by ADMM](decoupling_training-free_guided_diffusion_by_admm.md)
- [\[CVPR 2025\] OmniGen: Unified Image Generation](omnigen_unified_image_generation.md)
- [\[CVPR 2025\] SVFR: A Unified Framework for Generalized Video Face Restoration](svfr_a_unified_framework_for_generalized_video_face_restoration.md)

</div>

<!-- RELATED:END -->
