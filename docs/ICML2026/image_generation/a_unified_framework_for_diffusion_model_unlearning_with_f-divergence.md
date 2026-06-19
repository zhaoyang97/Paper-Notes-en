---
title: >-
  [Paper Note] A Unified Framework for Diffusion Model Unlearning with f-Divergence
description: >-
  [ICML 2026][Image Generation][Diffusion Model] This paper generalizes MSE/KL alignment in diffusion model concept unlearning to arbitrary $f$-divergence, proposes the f-DMU framework, and discovers that closed-form Hellinger loss is more stable and better preserves non-target concepts compared to MSE.
tags:
  - ICML 2026
  - Image Generation
  - Diffusion Model
  - Concept Erasure
  - f-divergence
date: 2026-05-08
content_hash: 55e7c1007a23dd3a
---
# A Unified Framework for Diffusion Model Unlearning with f-Divergence

**Conference**: ICML 2026  
**arXiv**: [2509.21167](https://arxiv.org/abs/2509.21167)  
**Code**: https://github.com/tonellolab/f-DMU  
**Area**: Image Generation / Diffusion Model Unlearning  
**Keywords**: Diffusion Models, Concept Erasure, Model Unlearning, f-divergence, Hellinger Distance  

## TL;DR
This paper generalizes MSE/KL alignment in diffusion model concept unlearning to arbitrary $f$-divergence, proposes the f-DMU framework, and discovers that closed-form Hellinger loss is more stable and better preserves non-target concepts compared to MSE.

## Background & Motivation
**Background**: Text-to-image diffusion models can generate high-quality images but also memorize NSFW content, copyrighted artistic styles, characters, or personal information. Model unlearning aims to directionally delete a concept from a trained model without retraining it from scratch.

**Limitations of Prior Work**: Mainstream fine-tuning unlearning methods typically pull the denoiser output of a target concept toward an anchor concept (e.g., empty concept, parent category, or semantically similar concept). These objectives are usually formulated as MSE, which inherently corresponds to the KL divergence between two Gaussian reverse-process distributions. However, KL/MSE is just one choice of divergence; different tasks may require different trade-offs between erasure strength and fidelity.

**Key Challenge**: Strong erasure easily harms non-target concepts and overall image quality, while mild erasure may retain target features. Existing methods lack a unified perspective to explain why certain losses are more stable or when to use more aggressive ones.

**Goal**: The authors aim to formulate diffusion model concept unlearning as a general $f$-divergence minimization problem, covering previous KL/MSE methods while providing a set of optional closed-form and variational losses.

**Key Insight**: Starting from probability distributions rather than specific network architectures, the paper aligns the reverse process distribution of the original model under an anchor concept with that of the unlearned model under the target concept.

**Core Idea**: Replace the fixed KL divergence with $f$-divergence, allowing the gradient geometry of the divergence to control stability, erasure strength, and prior preservation during the unlearning process.

## Method
The starting point of f-DMU is that many concept erasure methods essentially modify the target generation distribution to resemble the anchor. Denoting the original model as $\Phi$ and the unlearned model as $\hat{\Phi}$, one can compare the divergence between $p_{\Phi}(x_{t-1}|x_t,c)$ and $p_{\hat{\Phi}}(x_{t-1}|x_t,c^*)$, where $c$ is the anchor and $c^*$ is the target to be erased.

### Overall Architecture
The paper formulates the unlearning objective as an expectation over timesteps, samples, and target-anchor pairs: minimizing $D_f$ between reverse-process conditional distributions. When two conditional distributions can be approximated as Gaussians with the same covariance, some $f$-divergences have closed-form solutions, making the loss as computationally efficient as MSE. When no closed-form exists, a variational representation is used to formulate a min-max objective $\min_{\hat{\Phi}} \max_T$, where a discriminator function $T$ estimates the divergence.

Specific instances include KL/MSE, Jeffreys, squared Hellinger, Pearson $\chi^2$, and general $\alpha$-divergence. The paper focuses on comparing closed-form Hellinger, closed-form $\chi^2$, standard MSE/KL, and variational losses.

### Key Designs
**1. Unified Unlearning Objective with $f$-divergence: Integrating various concept erasure losses into a single distribution alignment framework.** Mainstream methods essentially modify the target generation distribution to match an anchor, traditionally using MSE. This paper points out that MSE is merely the KL divergence between two Gaussian reverse-process distributions, thus generalizing the objective to minimize any $f$-divergence $D_f$. A key observation is that changing $f$ does not change the global optimum (which remains distribution alignment) but alters the optimization path and gradient magnitude—effectively adjusting "how aggressive the erasure is" and "how well priors are preserved."

**2. Closed-form Hellinger / $\chi^2$ loss: Changing gradient geometry without additional networks or min-max optimization.** When the two conditional distributions are approximated as Gaussians with matching covariance, some $f$-divergences yield closed-form solutions. The loss remains as cheap as MSE, with the primary difference lying in gradient scaling. The Hellinger loss gradient is approximately $e^{-\text{MSE}}\nabla \text{MSE}$, down-weighting samples with large errors and naturally suppressing outlier updates to reduce image collapse during fine-tuning. Conversely, the $\chi^2$ loss gradient is approximately $e^{\text{MSE}}\nabla \text{MSE}$, amplifying large errors for scenarios requiring stronger erasure.

**3. Variational f-DMU: Extending the framework to arbitrary $f$-divergences without closed-form solutions.** For $f$-divergences without Gaussian closed-form solutions, the paper employs the variational representation $D_f(p\|q)=\sup_T \mathbb{E}_p[T]-\mathbb{E}_q[f^*(T)]$. A discriminator function $T$ is introduced to estimate the divergence, while the unlearned model minimizes this estimate in a $\min_{\hat{\Phi}}\max_T$ min-max setup. The trade-off is the requirement for additional discriminator training and higher risk of distribution noise in small-batch diffusion fine-tuning.

### Loss & Training
Experiments cover Stable Diffusion 1.4, 1.5, 2.1, XL, and extend to SD3 and FLUX in the appendix. Evaluation of concept erasure uses CLIP Score and CLIP Accuracy: lower is better for target concepts, higher is better for retained concepts; KID measures image distribution and quality changes. Closed-form losses require only single-model fine-tuning, while variational losses require an additional discriminator.

## Key Experimental Results

### Main Results
Van Gogh style erasure on SD 2.1 demonstrates the primary effects of closed-form divergences.

| Method | Erased CS↓ | Erased CA↓ | Retained CS↑ | Retained CA↑ | KID↓ |
|--------|------|------|------|------|------|
| ESD | 0.657 | 0.6 | 0.668 | 0.74 | 0.027 |
| CAbl | 0.635 | 0.2 | 0.668 | 0.78 | 0.028 |
| DoCo | 0.737 | 0.9 | 0.691 | 0.86 | 0.033 |
| Hellinger closed-form | 0.624 | 0.2 | 0.672 | 0.78 | 0.027 |
| $\chi^2$ closed-form | 0.628 | 0.1 | 0.672 | 0.76 | 0.028 |
| Hellinger variational | 0.645 | 0.5 | 0.702 | 0.88 | 0.051 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Gradient Magnitude: H-DMU | Gradients strictly smaller than MSE/$\chi^2$ | Exponential decay for large errors, smoother fine-tuning |
| Gradient Magnitude: $\chi^2$-DMU | Gradients significantly larger than MSE | Aggressive erasure, higher risk to retained concepts |
| Sequential Erasure (10 styles) | H2 has lower KID on retained artists | Hellinger favors prior preservation in multi-concept scenarios |
| Nudity erasure: H-DMU | I2P 0.063, MMA Adv. 0.049, MMA S.Adv. 0.042 | Strong performance on both non-adversarial and adversarial prompts |
| Nudity erasure: CAbl | I2P 0.120, MMA Adv. 0.118, MMA S.Adv. 0.141 | Standard MSE methods show weaker robustness than H-DMU |
| Variational losses | Fast erasure but higher KID | Poor divergence estimation in small batches causes distribution shifts |

### Key Findings
- **Hellinger closed-form is the default recommendation**: It typically preserves non-target concepts better without increasing training overhead.
- **$\chi^2$ closed-form serves as a "strong erasure mode"**: Target concepts drop quickly, but surrounding distributions are noticeably altered.
- **Variational f-DMU provides maximum generality**: It allows for more aggressive erasure but requires extra min-max training and poses higher risks to generation quality.
- The theoretical gradient analysis aligns with experimental observations, confirming that divergence selection modifies fine-tuning dynamics rather than just renaming the loss.

## Highlights & Insights
- The most valuable aspect of this paper is reducing empirical unlearning losses to divergence choices, providing an interpretable coordinate system for method selection.
- Hellinger's conservative gradient acts as a per-sample adaptive scaling rather than manual learning rate adjustment; this explains its ability to reduce image collapse while maintaining target erasure.
- Anchor selection and divergence selection are orthogonal knobs: the anchor determines where the target is moved, while the divergence determines how aggressively the transition occurs.

## Limitations & Future Work
- Evaluation relies heavily on CLIP Score, CLIP Accuracy, and KID; these proxy metrics may not fully capture fine-grained human judgment of concept presence.
- Scenarios involving multi-concepts, cross-lingual prompts, combinatorial concepts, and style-object entanglement may still be difficult for a single divergence to handle stably.
- The variational framework suffers from noise in small-batch diffusion fine-tuning, requiring more stable estimation or regularization strategies.
- Future work could combine f-DMU with better anchor selection, closed-form weight editing, or safety filters to create a controllable model governance toolkit.

## Related Work & Insights
- **vs ESD / CAbl**: These use KL/MSE-like alignment. f-DMU shows these are special cases in the $f$-divergence family and provides more stable Hellinger alternatives.
- **vs UCE / RECE / MACE**: These methods rely more on structural editing or closed-form weight updates. f-DMU maintains the fine-tuning framework with broader model architecture applicability.
- **vs DoCo**: DoCo introduces GAN-like variational ideas. f-DMU unifies such min-max objectives from the perspective of variational $f$-divergence, identifying the trade-off of higher quality risks.
- **Insight**: In model unlearning, different tasks should explicitly state whether they favor "fidelity" or "strong erasure"; loss selection should serve this application goal rather than defaulting to MSE.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Strong theoretical and practical connection using $f$-divergence to unify diffusion unlearning goals.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Rich coverage of models, concepts, and attack robustness; human evaluation and actual compliance scenarios could be further strengthened.
- Writing Quality: ⭐⭐⭐⭐☆ Clear mathematical motivation, though the abundance of notation and appendix tables requires a diffusion model background.
- Value: ⭐⭐⭐⭐⭐ Direct methodological value for diffusion model safety, copyright style erasure, and controllable unlearning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[ICCV 2025\] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](../../ICCV2025/image_generation/holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)
- [\[CVPR 2026\] UniPercept: A Unified Diffusion Model for Generalizable Visual Perception](../../CVPR2026/image_generation/unipercept_a_unified_diffusion_model_for_generalizable_visual_perception.md)
- [\[ICML 2026\] Quantifying Error Propagation and Model Collapse in Diffusion Models](quantifying_error_propagation_and_model_collapse_in_diffusion_models.md)
- [\[ICCV 2025\] Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts](../../ICCV2025/image_generation/meta-unlearning_on_diffusion_models_preventing_relearning_unlearned_concepts.md)

</div>

<!-- RELATED:END -->
