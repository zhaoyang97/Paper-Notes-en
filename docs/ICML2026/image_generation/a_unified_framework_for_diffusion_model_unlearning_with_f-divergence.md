---
title: >-
  [Paper Note] A Unified Framework for Diffusion Model Unlearning with f-Divergence
description: >-
  [ICML 2026][Image Generation][Diffusion Model] This paper generalizes MSE/KL alignment in diffusion model concept unlearning to arbitrary $f$-divergence, proposing the f-DMU framework. It identifies that closed-form Hellinger loss is often more stable and better at preserving non-target concepts than MSE.
tags:
  - ICML 2026
  - Image Generation
  - Diffusion Model
  - Concept Erasure
  - f-divergence
date: 2026-05-08
content_hash: 33566b5253b5f98d
---
# A Unified Framework for Diffusion Model Unlearning with f-Divergence

**Conference**: ICML 2026  
**arXiv**: [2509.21167](https://arxiv.org/abs/2509.21167)  
**Code**: https://github.com/tonellolab/f-DMU  
**Area**: Image Generation / Diffusion Model Unlearning  
**Keywords**: Diffusion Models, Concept Erasure, Model Unlearning, $f$-divergence, Hellinger Distance  

## TL;DR
This paper generalizes MSE/KL alignment in diffusion model concept unlearning to arbitrary $f$-divergence, proposing the f-DMU framework. It identifies that closed-form Hellinger loss is often more stable and better at preserving non-target concepts than MSE.

## Background & Motivation
**Background**: Text-to-image diffusion models can generate high-quality images but also memorize NSFW content, copyrighted artistic styles, character images, or personal information. Model unlearning aims to directionally delete a specific concept from a trained model without retraining it from scratch.

**Limitations of Prior Work**: Existing fine-tuning unlearning methods typically pull the denoiser output of a target concept toward an anchor concept, such as a null concept, a parent category, or a semantically similar concept. These objectives are usually formulated as MSE, which essentially corresponds to the KL divergence between two Gaussian reverse-process distributions. The problem is that KL/MSE is only one choice of divergence; different tasks may require different trade-offs between erasure strength and fidelity.

**Key Challenge**: Strong erasure tends to damage non-target concepts and overall image quality, while gentle erasure may retain target features. Existing methods lack a unified perspective to explain "why a certain loss is more stable" and "when to use a more aggressive loss."

**Goal**: The authors aim to formulate diffusion model concept unlearning as a general $f$-divergence minimization problem, covering previous KL/MSE methods while providing a suite of selectable closed-form and variational losses.

**Key Insight**: Starting from probability distributions rather than specific network architectures, the paper aligns the reverse process distribution of the original model under the anchor concept with that of the unlearned model under the target concept.

**Core Idea**: By replacing fixed KL divergence with $f$-divergence, the gradient morphology of the divergence controls stability, erasure intensity, and prior preservation during the unlearning process.

## Method
The starting point of f-DMU is that many concept erasure methods are actually making the generative distribution of the target look like the anchor. If the original model is denoted as $\Phi$ and the unlearned model as $\hat{\Phi}$, one can compare the divergence between $p_{\Phi}(x_{t-1}|x_t,c)$ and $p_{\hat{\Phi}}(x_{t-1}|x_t,c^*)$, where $c$ is the anchor and $c^*$ is the target to be erased.

### Overall Architecture
The paper defines the unlearning objective as the expectation over timesteps, samples, and target-anchor pairs: minimizing $D_f$ between the reverse-process conditional distributions. When both conditional distributions can be approximated as Gaussians with shared covariance, some $f$-divergences have closed-form solutions, making the loss as computationally efficient as MSE. When no closed-form exists, a variational representation is used to formulate the problem as a min-max objective $\min_{\hat{\Phi}} \max_T$, where a discriminator function $T$ estimates the divergence.

Specific instances include KL/MSE, Jeffreys, squared Hellinger, Pearson $\chi^2$, and general $\alpha$-divergence. The paper focuses on comparing closed-form Hellinger, closed-form $\chi^2$, standard MSE/KL, and variational losses.

### Key Designs
**1. $f$-divergence Unified Unlearning Objective: Integrating various concept erasure losses into a single distribution alignment framework.** Mainstream methods are essentially performing the same task—altering the generative distribution of the target concept to resemble that of an anchor concept—and habitually formulate this as MSE. This paper points out that this MSE is merely the KL divergence between two Gaussian reverse-process distributions. Thus, the objective is generalized to minimize an arbitrary $f$-divergence $D_f$, with KL/MSE being a special case. The key observation is that changing $f$ does not alter the global optimum (the optimal solution remains distribution alignment) but changes the optimization path and gradient magnitude—effectively adjusting "how aggressive the erasure is and how well the prior is preserved." Therefore, the real knob to tune in diffusion unlearning is the intensity of distribution shift, rather than being default-locked into MSE geometry.

**2. Closed-form Hellinger / $\chi^2$ loss: Changing gradient morphology without extra networks or min-max optimization.** When the two conditional distributions of the target and anchor concepts are approximated as Gaussians with shared covariance, some $f$-divergences yield closed-form solutions. The loss remains as inexpensive as MSE, with the difference residing entirely in gradient scaling. The gradient of the Hellinger loss is approximately $e^{-\text{MSE}}\nabla \text{MSE}$, which assigns smaller weights to samples with large errors, thereby naturally suppressing outlier updates and reducing sudden image corruption during fine-tuning. Conversely, the $\chi^2$ loss gradient is approximately $e^{\text{MSE}}\nabla \text{MSE}$, amplifying large error samples, which suits scenarios requiring stronger erasure at the cost of some quality loss. This set of closed-form losses forms a spectrum ranging from "high fidelity" to "strong erasure" without increasing training costs.

**3. Variational f-DMU: Extending the framework to arbitrary $f$-divergences without closed-form solutions via variational forms.** Not every $f$-divergence has a closed-form Gaussian solution. In such cases, the variational representation $D_f(p\|q)=\sup_T \mathbb{E}_p[T]-\mathbb{E}_q[f^*(T)]$ is used. A discriminator function $T$ is introduced to estimate the divergence between the two output distributions, and the unlearning model minimizes this estimate, resulting in a $\min_{\hat{\Phi}}\max_T$ min-max objective. The trade-off is that it requires training an additional discriminator, and under the small batches common in diffusion fine-tuning, divergence estimation noise is higher, leading to more aggressive training and higher risks to generative quality—generality is exchanged for stability.

### Loss & Training
Experiments cover Stable Diffusion 1.4, 1.5, 2.1, and XL, with extensions to SD3 and FLUX in the appendix. Evaluation of concept erasure uses CLIP Score and CLIP Accuracy: lower is better for target concepts, higher is better for retained concepts. KID measures changes in image distribution and quality. Closed-form losses require only single-model fine-tuning like MSE, while variational losses require an additional discriminator.

## Key Experimental Results

### Main Results
Van Gogh style erasure on SD 2.1 demonstrates the main effects of closed-form divergence.

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
| Gradient Magnitude: H-DMU | Gradient consistently smaller than MSE and $\chi^2$ | Exponential decay for large error samples leads to smoother fine-tuning |
| Gradient Magnitude: $\chi^2$-DMU | Gradient significantly larger than MSE | More aggressive erasure, but more likely to damage retained concepts |
| Sequential erasure (10 styles) | H2 achieves lower KID on multiple retained artists | Hellinger is more conducive to prior preservation in multi-concept scenarios |
| Nudity erasure: H-DMU | I2P 0.063, MMA Adv. 0.049 | Strong performance against both non-adversarial and adversarial prompts |
| Nudity erasure: CAbl | I2P 0.120, MMA Adv. 0.118 | Standard MSE-based methods are weaker in robustness than H-DMU |
| Variational losses | Faster erasure but higher KID | Coarse divergence estimation in small batches causes larger distribution perturbations |

### Key Findings
- Hellinger closed-form is the default recommendation: it typically preserves non-target concepts better without increasing training overhead.
- $\chi^2$ closed-form functions as a "strong erasure mode": target concepts drop quickly, but surrounding distributions are altered more noticeably.
- Variational f-DMU provides maximum flexibility and aggressive erasure but requires additional min-max training and carries higher generative quality risks.
- Theoretical gradient analysis aligns with experimental results, showing that divergence choice indeed alters fine-tuning dynamics rather than just changing the loss name.

## Highlights & Insights
- The primary value of this paper is reducing many empirical unlearning losses to divergence choices, providing an interpretable coordinate system for method selection.
- Hellinger’s conservative gradient is not a manually tuned learning rate but a per-sample adaptive scaling; this explains why it reduces image corruption while maintaining target erasure.
- Anchor selection and divergence selection are orthogonal knobs: the anchor determines where the target is replaced, while the divergence determines how aggressive the replacement process is.

## Limitations & Future Work
- Evaluation relies heavily on CLIP Score, CLIP Accuracy, and KID; these proxy metrics do not fully capture fine-grained human judgment of "concept persistence."
- Multi-concept, cross-lingual prompts, compositional concepts, and style-object entanglement scenarios may still be difficult for a single divergence to handle stably.
- While general, the variational framework is noisy under the small batches common in diffusion fine-tuning, requiring more stable estimation or regularization strategies.
- Subsequent work could combine f-DMU with better anchor selection, closed-form weight editing, or safety filters to form a controllable model governance toolchain.

## Related Work & Insights
- **vs ESD / CAbl**: These use KL/MSE-style alignment; f-DMU shows these methods are special cases within the $f$-divergence family and provides a more stable Hellinger alternative.
- **vs UCE / RECE / MACE**: These methods rely more on structural editing or closed-form weight updates; f-DMU maintains the fine-tuning framework with broader architecture applicability.
- **vs DoCo**: DoCo introduces GAN-like variational ideas; f-DMU unifies such min-max objectives from the perspective of variational $f$-divergence, noting their aggressiveness and quality risks.
- **Insights**: In model unlearning, different tasks should explicitly declare a preference for "fidelity" vs. "strong erasure"; loss selection should serve this application goal rather than defaulting to MSE.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Strong theoretical and practical connection by unifying diffusion unlearning with $f$-divergence.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Extensive coverage of models, concepts, and attack robustness; human evaluation and real compliance scenarios could be further strengthened.
- Writing Quality: ⭐⭐⭐⭐☆ Mathematical motivation is clear, though heavy notation and appendix tables require background in diffusion models.
- Value: ⭐⭐⭐⭐⭐ Direct methodological value for diffusion model safety, copyright style erasure, and controllable unlearning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[ICCV 2025\] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](../../ICCV2025/image_generation/holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)
- [\[CVPR 2026\] UniPercept: A Unified Diffusion Model for Generalizable Visual Perception](../../CVPR2026/image_generation/unipercept_a_unified_diffusion_model_for_generalizable_visual_perception.md)
- [\[ICML 2026\] Quantifying Error Propagation and Model Collapse in Diffusion Models](quantifying_error_propagation_and_model_collapse_in_diffusion_models.md)
- [\[CVPR 2025\] SVFR: A Unified Framework for Generalized Video Face Restoration](../../CVPR2025/image_generation/svfr_a_unified_framework_for_generalized_video_face_restoration.md)

</div>

<!-- RELATED:END -->
