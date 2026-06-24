---
title: >-
  [Paper Note] When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis
description: >-
  [ICLR 2026][Image Generation][Score Learning] This work reveals the scale separation between geometric and distributional information in score learning under the manifold hypothesis—manifold geometric information intensity is $\Theta(\sigma^{-2})$, which is $O(\sigma^{-2})$ times stronger than distributional information. This proves that the success of diffusion models primarily stems from learning the data manifold rather than the full distribution…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Score Learning"
  - "Manifold Hypothesis"
  - "Geometric Learning"
  - "Distribution Learning"
  - "Rate Separation"
  - "Uniform Sampling"
date: 2026-05-08
content_hash: 6a4dec9153066292
---

# When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis

**Conference**: ICLR 2026  
**arXiv**: [2509.24912](https://arxiv.org/abs/2509.24912)  
**Code**: None  
**Area**: Diffusion Model Theory  
**Keywords**: Score Learning, Manifold Hypothesis, Geometric Learning, Distribution Learning, Rate Separation, Uniform Sampling

## TL;DR
This work reveals the scale separation between geometric and distributional information in score learning under the manifold hypothesis—manifold geometric information intensity is $\Theta(\sigma^{-2})$, which is $O(\sigma^{-2})$ times stronger than distributional information. This proves that the success of diffusion models primarily stems from learning the data manifold rather than the full distribution, and proposes a one-line code modification to generate uniform distributions on the manifold.

## Background & Motivation

**Background**: Score-based methods (diffusion models, Bayesian inverse problems, etc.) are typically interpreted as learning the data distribution in the low-noise limit $\sigma \to 0$. The manifold hypothesis—that data is supported on a low-dimensional manifold in high-dimensional space—is widely adopted.

**Limitations of Prior Work**: (1) Score estimation is extremely difficult in the low-temperature limit, necessitating extensive post-training engineering for stability in practice; (2) existing theoretical analyses mix geometric and distributional information in error bounds, failing to reveal scale separation; (3) even with good score approximations, the recovered distribution might be arbitrary on the manifold.

**Key Challenge**: Distribution learning (complete recovery of $\mu_{\text{data}}$) requires score accuracy of $o(1)$, but practical score errors are often much larger. However, diffusion models still generate realistic samples—suggesting they may not have truly learned the distribution itself.

**Goal**: (1) Why do diffusion models remain effective despite imperfect scores? (2) Can these imperfect scores be leveraged for useful tasks (e.g., uniform sampling)?

**Key Insight**: Perform asymptotic expansion of the score function as $\sigma \to 0$, separating geometric terms (leading order, $\Theta(\sigma^{-2})$) from distributional terms (sub-leading order, $\Theta(1)$).

**Core Idea**: Geometric information in the score function is $\sigma^{-2}$ times stronger than distributional information; thus, geometric learning allows for errors as large as $o(\sigma^{-2})$, which is far more relaxed than the $o(1)$ required for distribution learning.

## Method

### Overall Architecture

Ours does not propose a new model but performs an asymptotic expansion of the Gaussian-smoothed score function as $\sigma \to 0$ under the manifold hypothesis (data supported on a $C^4$ compact boundaryless low-dimensional manifold $\mathcal{M}$). The score is decomposed into two layers of different intensities, proving that "geometric learning is $\sigma^{-2}$ times more relaxed than distribution learning," and providing a one-line code modification for uniform sampling. The theory is built on three quantities: smoothed measures $\mu_\sigma = \text{law}(X + \sigma Z)$ (VE) or $\text{law}(\sqrt{1-\sigma^2}X + \sigma Z)$ (VP), the semi-squared distance function $d_\mathcal{M}(x) = \tfrac{1}{2}\text{dist}^2(x, \mathcal{M})$, and the Riemannian volume measure and metric tensor $g(u)$ on the manifold.

### Key Designs

**1. Asymptotic Expansion of Score: Separating Geometric and Distributional Items by Scale**

A fundamental issue in existing score learning theory is mixing "how far a point is from the manifold" and "how high the density is on the manifold" in the same error bound, obscuring what diffusion models actually learn. This paper performs an expansion of the score as $\sigma \to 0$ (Theorem 3.1), proving it is composed of two layers: the leading order term has intensity $\Theta(\sigma^{-2})$, essentially acting as a projection operator back to the manifold (pure geometry); the sub-leading order term has intensity $\Theta(1)$, encoding the manifold density $p_{\text{data}}$. The vast $\sigma^{-2}$ gap is the "rate separation." It holds because even small offsets from the manifold are amplified by $1/\sigma^2$, making geometric signals dominantly stronger.

**2. Manifold Concentration vs. Distribution Recovery: Different Accuracy Requirements**

Since the score has two layers, the accuracy required for "learning geometry" vs. "learning distribution" differs (Theorem 4.1). Concentrating samples near the manifold (manifold concentration) only requires $o(\sigma^{-2})$ error, as the leading geometric term is $\Theta(\sigma^{-2})$. Precise recovery of $\mu_{\text{data}}$ requires an error of $o(1)$ to prevent the weak $\Theta(1)$ distributional term from being drowned by noise. This $O(\sigma^{-2})$ gap explains why diffusion models work despite imperfect scores: they achieve the accuracy needed to learn manifold geometry ("what looks real") but not necessarily the true distribution ("frequency of real data").

**3. One-Line Code for Uniform Sampling: Offsetting Geometric Terms with Distance Gradients**

Since imperfect scores learn geometry well, they can be used for uniform sampling on the manifold (Theorems 5.1–5.2). This is achieved by replacing the score in Langevin dynamics with $\tilde{s}_\sigma(x) = \nabla \log p_\sigma(x) + \tfrac{1}{\sigma^2} \nabla d_\mathcal{M}(x)$, termed **Tempered Score (TS) Langevin dynamics**:

$$dX_t = \Big[\nabla \log p_\sigma(X_t) + \tfrac{1}{\sigma^2}\nabla d_\mathcal{M}(X_t)\Big] dt + \sqrt{2}\, dW_t.$$

The added term cancels the $\Theta(\sigma^{-2})$ geometric component, leaving only the $\Theta(1)$ distributional term, which is drowned by noise at small $\sigma$. Consequently, the dynamics converge to the uniform distribution on the manifold—the maximum entropy representation. This requires only $o(\sigma^{-2})$ score accuracy and can be implemented at inference without fine-tuning.

**4. Bayesian Inverse Problems: Robustness of Uniform Priors**

Rate separation extends to inverse problems (Theorem 6.1). Using a uniform prior on the manifold for posterior sampling requires only $o(\sigma^{-2})$ score accuracy; conversely, using the data distribution $\mu_{\text{data}}$ as a prior requires $o(1)$. Thus, maximum entropy (uniform) priors are naturally more tolerant of score estimation errors.

## Key Experimental Results

### Synthetic Experiments
- 1D circle in 2D space: Verified geometric recovery and uniform sampling.
- Different score error levels: $o(\sigma^{-2})$ suffices to recover manifold support, but the distribution remains arbitrary.
- After one-line modification: Generates a uniform distribution on the manifold.

### Stable Diffusion 1.5 Experiments
- Validated theoretical predictions on a large-scale pre-trained model.
- Modified sampling algorithm produces samples covering the manifold more uniformly.
- Increased diversity and reduced bias.

### Key Findings
- Experiments verify that the $\Theta(\sigma^{-2})$ rate separation holds in practical models.
- Uniform sampling algorithm works effectively without fine-tuning.
- Standard diffusion models primarily learn manifold geometry rather than exact distributions.

## Highlights & Insights
- **Paradigm Shift**: Proposes a shift from distribution learning to geometric learning, providing practical guidance that perfect distribution recovery is not always necessary.
- **Power of One Line**: Uniform sampling requires only a one-line modification (adding the distance gradient term), making it highly practical.
- **New Explanation for Success**: Diffusion models succeed because they learn "what looks real" (manifold) rather than "the frequency of the real data."
- **Implication for Inverse Problems**: Uniform priors are more robust than data priors against score estimation errors, guiding downstream applications.

## Limitations & Future Work
- Theoretical analysis depends on the manifold hypothesis ($C^4$ compact boundaryless manifold), which real data might not strictly satisfy.
- $\nabla d_\mathcal{M}$ must be estimated in practice, and its accuracy affects sampling quality.
- Initial experiments were conducted on Stable Diffusion 1.5; larger models and more complex tasks remain to be validated.
- The exact prefactor of rate separation is not discussed; the practical gap might be smaller than theoretical predictions.

## Related Work & Insights
- **vs. Stanczuk et al. 2024**: They estimate intrinsic dimensionality but do not address rate separation.
- **vs. Ventura et al. 2024**: Only analyzes linear manifolds (subspaces); Ours applies to general smooth manifolds.
- **vs. De Santi et al. 2025**: They achieve uniform sampling via fine-tuning; Ours achieves it at inference without fine-tuning.
- **vs. Existing Score Learning Theory**: Prior works mix geometric and distributional errors; Ours is the first to separate them.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of rate separation and the proposed geometric learning paradigm are foundational.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic and Stable Diffusion validation provided, though more quantitative evaluation is needed.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical development and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ Profound impact on understanding diffusion models and improving sampling strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generative Model Inversion Through the Lens of the Manifold Hypothesis](../../NeurIPS2025/image_generation/generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)
- [\[ICLR 2026\] Secure Inference for Diffusion Models via Unconditional Scores](secure_inference_for_diffusion_models_via_unconditional_scores.md)
- [\[ICLR 2026\] Learn to Guide Your Diffusion Model](learn_to_guide_your_diffusion_model.md)
- [\[ICLR 2026\] Dragging with Geometry: From Pixels to Geometry-Guided Image Editing](dragging_with_geometry_from_pixels_to_geometry-guided_image_editing.md)
- [\[ICLR 2026\] TempFlow-GRPO: When Timing Matters for GRPO in Flow Models](tempflow-grpo_when_timing_matters_for_grpo_in_flow_models.md)

</div>

<!-- RELATED:END -->
