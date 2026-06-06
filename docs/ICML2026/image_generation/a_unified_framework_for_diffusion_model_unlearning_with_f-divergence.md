---
title: >-
  [Paper Note] A Unified Framework for Diffusion Model Unlearning with f-Divergence
description: >-
  [ICML 2026][Image Generation][Diffusion Models] This paper generalizes MSE/KL alignment in diffusion model concept unlearning to arbitrary $f$-divergence, proposes the f-DMU framework…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Diffusion Models"
  - "Concept Erasure"
  - "Model Unlearning"
  - "$f$-divergence"
  - "Hellinger Distance"
date: 2026-05-08
content_hash: ce500a2c5cf172ca
---

# A Unified Framework for Diffusion Model Unlearning with f-Divergence

**Conference**: ICML 2026  
**arXiv**: [2509.21167](https://arxiv.org/abs/2509.21167)  
**Code**: https://github.com/tonellolab/f-DMU  
**Area**: Image Generation / Diffusion Model Unlearning  
**Keywords**: Diffusion Models, Concept Erasure, Model Unlearning, $f$-divergence, Hellinger Distance  

## TL;DR
This paper generalizes MSE/KL alignment in diffusion model concept unlearning to arbitrary $f$-divergence, proposes the f-DMU framework, and discovers that the closed-form Hellinger loss is often more stable than MSE and better preserves non-target concepts.

## Background & Motivation
**Background**: Text-to-image diffusion models can generate high-quality images but also memorize NSFW content, copyrighted artistic styles, character images, or personal information. Model unlearning aims to directionally delete a specific concept from a trained model without retraining the entire model.

**Limitations of Prior Work**: Mainstream fine-tuning unlearning methods typically pull the denoiser output of a target concept toward an anchor concept (e.g., a null concept, a super-category concept, or a semantically similar concept). These objectives are often formulated as MSE, which essentially corresponds to the KL divergence between two Gaussian reverse-process distributions. The problem is that KL/MSE is just one choice of divergence; different tasks may require different trade-offs between erasure strength and fidelity.

**Key Challenge**: Strong erasure easily harms non-target concepts and overall image quality, while gentle erasure might retain target features. Existing methods lack a unified perspective to explain "why a certain loss is more stable" and "when to use a more aggressive loss."

**Goal**: The authors aim to formulate diffusion model concept unlearning as a general $f$-divergence minimization problem, covering previous KL/MSE methods while providing a set of selectable closed-form and variational losses.

**Key Insight**: The paper starts from probability distributions rather than specific network architectures, aligning the reverse-process distribution of the original model under the anchor concept with the reverse-process distribution of the unlearned model under the target concept.

**Core Idea**: Replace the fixed KL divergence with $f$-divergence, allowing the gradient geometry of the divergence to control stability, erasure strength, and prior preservation during the unlearning process.

## Method
The starting point of f-DMU is that many concept erasure methods are actually making the target generation distribution resemble the anchor. If the original model is denoted as $\Phi$ and the unlearned model as $\hat{\Phi}$, one can compare the divergence between $p_{\Phi}(x_{t-1}|x_t,c)$ and $p_{\hat{\Phi}}(x_{t-1}|x_t,c^*)$, where $c$ is the anchor and $c^*$ is the target to be erased.

### Overall Architecture
The paper formulates the unlearning objective as the expectation over timesteps, samples, and target-anchor pairs: minimizing the $D_f$ between reverse-process conditional distributions. When the two conditional distributions can be approximated as Gaussians with the same covariance, some $f$-divergences have closed-forms, making the loss as computationally efficient as MSE. When no closed-form exists, a variational representation is used to formulate the problem as a $\min_{\hat{\Phi}} \max_T$ min-max objective, where the divergence is estimated by a discriminator function $T$.

Specific instances include KL/MSE, Jeffreys, squared Hellinger, Pearson $\chi^2$, and more general $\alpha$-divergence. The paper focuses on comparing closed-form Hellinger, closed-form $\chi^2$, standard MSE/KL, and variational losses.

### Key Designs
1. **$f$-divergence Unified Unlearning Objective**:
	- Function: Places multiple concept erasure losses into a single probability distribution alignment framework.
	- Mechanism: Minimizes $D_f$ between the original model's anchor distribution and the unlearned model's target distribution. KL is a special case of MSE; the global optimum does not change with $f$, but the optimization path and gradient magnitudes do.
	- Design Motivation: What diffusion unlearning truly needs to control is the strength of the distribution shift rather than being fixed to a single MSE geometry.

2. **Closed-form Hellinger / $\chi^2$ Loss**:
	- Function: Obtains gradient behaviors different from MSE without introducing additional networks or min-max training.
	- Mechanism: The gradient of the Hellinger loss is proportional to $e^{-\text{MSE}}\nabla \text{MSE}$, where large-error samples are weighted less; the $\chi^2$ loss gradient is proportional to $e^{\text{MSE}}\nabla \text{MSE}$, magnifying large-error samples.
	- Design Motivation: Hellinger naturally suppresses outlier updates, reducing image collapse during fine-tuning; $\chi^2$ is suitable for scenarios requiring stronger erasure where some quality loss is acceptable.

3. **Variational f-DMU**:
	- Function: Supports arbitrary $f$-divergences that lack a Gaussian closed-form.
	- Mechanism: Utilizes $D_f(p||q)=\sup_T \mathbb{E}_p[T]-\mathbb{E}_q[f^*(T)]$, where a discriminator $T$ estimates the divergence of the two output distributions, and the unlearned model minimizes this estimate.
	- Design Motivation: Closed-form coverage is limited; the variational form provides universality at the cost of higher estimation noise in small batches and more aggressive training.

### Loss & Training
Experiments cover Stable Diffusion 1.4, 1.5, 2.1, XL, and extend to SD3 and FLUX in the appendix. CLIP Score and CLIP Accuracy are used to evaluate concept erasure: lower is better for the target concept, and higher is better for retained concepts; KID measures changes in image distribution and quality. Closed-form losses require only single-model fine-tuning similar to MSE, while variational losses require training an additional discriminator.

## Key Experimental Results

### Main Results
Van Gogh style erasure on SD 2.1 demonstrates the primary effects of closed-form divergences.

| Method | Target Concept CS↓ | Target Concept CA↓ | Retained Concept CS↑ | Retained Concept CA↑ | KID↓ |
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
| Gradient Magnitude: H-DMU | Gradients are consistently smaller than MSE and $\chi^2$ | Large-error samples are exponentially decayed, smoothing the fine-tuning process |
| Gradient Magnitude: $\chi^2$-DMU | Gradients are significantly larger than MSE | Erasure is more aggressive but more likely to damage retained concepts and generation quality |
| 10 Artistic Styles Sequential Erasure | H2 achieves lower KID on multiple retained artists | Hellinger is more conducive to prior preservation in multi-concept scenarios |
| Nudity erasure: H-DMU | I2P 0.063, MMA Adv. 0.049, MMA S.Adv. 0.042 | Shows strong performance on both non-adversarial and adversarial prompts |
| Nudity erasure: CAbl | I2P 0.120, MMA Adv. 0.118, MMA S.Adv. 0.141 | Standard MSE-based methods are weaker in robustness compared to H-DMU |
| Variational losses | Fast erasure but often higher KID | Coarse divergence estimation in small batches causes larger distribution perturbations |

### Key Findings
- **Closed-form Hellinger is the default recommendation**: It typically preserves non-target concepts better without increasing training overhead.
- **Closed-form $\chi^2$ acts as a "strong erasure mode"**: The target concept drops rapidly, but the surrounding distribution is more significantly altered.
- **Variational f-DMU provides maximum universality and aggressive erasure**: However, it requires additional min-max training and carries higher risks for generation quality.
- The theoretical gradient analysis aligns with experimental observations, indicating that the choice of divergence indeed changes fine-tuning dynamics rather than just being a change in loss naming.

## Highlights & Insights
- The most valuable contribution is reducing many seemingly empirical unlearning losses to a choice of divergence, providing an explainable coordinate system for method selection.
- The conservative gradient of Hellinger is not a manually tuned learning rate but a per-sample adaptive scaling; this explains why it preserves target erasure while reducing mid-training image collapse.
- Anchor selection and divergence selection are two orthogonal knobs: the anchor determines where the target is replaced, while the divergence determines how aggressive the replacement process is.

## Limitations & Future Work
- Evaluation relies mainly on CLIP Score, CLIP Accuracy, and KID; these proxy metrics may not fully capture fine-grained human judgment of "whether a concept still exists."
- Multi-concept, cross-lingual prompts, compositional concepts, and style-object entanglement scenarios may still be difficult for a single divergence to handle stably.
- While universal, the variational framework is noisy under the small batches common in diffusion fine-tuning, requiring more stable estimation or regularization strategies.
- f-DMU can be combined with better anchor selection, closed-form weight editing, or safety filtering to form a controllable model governance toolchain.

## Related Work & Insights
- **vs ESD / CAbl**: These use KL/MSE alignment; f-DMU shows these methods are special cases within the $f$-divergence family and provides a more stable Hellinger alternative.
- **vs UCE / RECE / MACE**: These methods rely more on structural editing or closed-form weight updates; f-DMU maintains the fine-tuning framework with broader model architecture applicability.
- **vs DoCo**: DoCo introduces GAN-like variational ideas; f-DMU unifies such min-max objectives from the perspective of variational $f$-divergence and notes their aggressive nature but higher quality risk.
- **Insight**: In model unlearning, different tasks should explicitly state whether they favor "fidelity" or "strong erasure"; the loss choice should serve this application goal rather than defaulting to MSE.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Strong link between theory and practice in unifying diffusion unlearning objectives with $f$-divergence.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Extensive coverage of models, concepts, and attack robustness; human evaluation and actual compliance scenarios could be further strengthened.
- Writing Quality: ⭐⭐⭐⭐☆ Mathematical motivation is clear, though many symbols and appendix tables require some diffusion model background.
- Value: ⭐⭐⭐⭐⭐ Direct methodological value for diffusion model safety, copyrighted style erasure, and controllable unlearning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Unlearning in Diffusion Models: A Unified Framework Based on KL Divergence and Likelihood Constraints](unlearning_in_diffusion_models_a_unified_framework_with_kl_divergence_and_likeli.md)
- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[ICML 2026\] Quantifying Error Propagation and Model Collapse in Diffusion Models](quantifying_error_propagation_and_model_collapse_in_diffusion_models.md)
- [\[ICLR 2026\] Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](../../ICLR2026/image_generation/continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)
- [\[ICML 2026\] Unified Masked Diffusion Models with Diverse Generation Orders](unifying_masked_diffusion_models_with_various_generation_orders_and_beyond.md)

</div>

<!-- RELATED:END -->
