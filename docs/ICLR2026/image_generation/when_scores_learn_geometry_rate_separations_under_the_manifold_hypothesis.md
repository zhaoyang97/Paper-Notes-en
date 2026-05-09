---
title: >-
  [Paper Note] When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis
description: >-
  [ICLR 2026][Image Generation][Score Learning] Under the manifold hypothesis, this paper reveals a scale separation between geometric and distributional information in score learning — manifold geometry contributes at order $\Theta(\sigma^{-2})$, which dominates distributional information by a factor of $O(\sigma^{-2})$. This establishes that the success of diffusion models stems primarily from learning the data manifold rather than the full distribution, and a one-line code modification suffices to generate the uniform distribution on the manifold.
tags:
  - ICLR 2026
  - Image Generation
  - Score Learning
  - Manifold Hypothesis
  - Geometry Learning
  - Distribution Learning
  - Rate Separation
  - Uniform Sampling
date: 2026-05-08
content_hash: 93d010770d501ed4
---

# When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis

**Conference**: ICLR 2026
**arXiv**: [2509.24912](https://arxiv.org/abs/2509.24912)
**Code**: None
**Area**: Diffusion Model Theory
**Keywords**: Score Learning, Manifold Hypothesis, Geometry Learning, Distribution Learning, Rate Separation, Uniform Sampling

## TL;DR
Under the manifold hypothesis, this paper reveals a scale separation between geometric and distributional information in score learning — manifold geometry contributes at order $\Theta(\sigma^{-2})$, which dominates distributional information by a factor of $O(\sigma^{-2})$. This establishes that the success of diffusion models stems primarily from learning the data manifold rather than the full distribution, and a one-line code modification suffices to generate the uniform distribution on the manifold.

## Background & Motivation

### State of the Field

**Background**: Score-based methods (diffusion models, Bayesian inverse problems, etc.) are conventionally interpreted as learning the data distribution in the low-noise limit $\sigma \to 0$. The manifold hypothesis — that data is supported on a low-dimensional manifold embedded in high-dimensional space — is widely adopted.

**Limitations of Prior Work**: (1) Score estimation in the low-temperature limit is highly unstable, requiring substantial post-training engineering in practice; (2) existing theoretical analyses conflate geometric and distributional information within error bounds, without revealing any scale separation; (3) even with a well-approximated score, the recovered distribution on the manifold may be arbitrary.

**Key Challenge**: Distribution learning (exact recovery of $\mu_{\text{data}}$) requires score accuracy of $o(1)$, whereas practical score errors are typically far larger. Yet diffusion models still generate realistic samples — suggesting they may not truly learn the distribution.

**Goal**: (1) Why do diffusion models remain effective under imperfect scores? (2) Can imperfect scores be leveraged for useful tasks such as uniform sampling?

**Key Insight**: Perform an asymptotic expansion of the score function as $\sigma \to 0$, separating the geometric term (leading order, $\Theta(\sigma^{-2})$) from the distributional term (sub-leading order, $\Theta(1)$).

**Core Idea**: Geometric information in the score dominates distributional information by a factor of $\sigma^{-2}$; consequently, geometry learning tolerates errors as large as $o(\sigma^{-2})$, a far weaker requirement than the $o(1)$ accuracy needed for distribution learning.

## Method

### Core Theory

1. **Asymptotic Expansion of the Score (Central Idea of Theorem 3.1)**:
   - As $\sigma \to 0$, the score function decomposes into:
   - **Leading order** $\Theta(\sigma^{-2})$: a geometric term that essentially acts as a projection operator onto the manifold (pulling points toward the nearest manifold point).
   - **Sub-leading order** $\Theta(1)$: a distributional term encoding the density $p_{\text{data}}$ on the manifold.
   - The two terms are separated by a factor of $\sigma^{-2}$ — this constitutes the "rate separation."

2. **Three Corollaries**:

   **(i) Manifold Concentration vs. Distribution Recovery (Theorem 4.1)**:
   - Manifold concentration (generated samples lie near the manifold): score error $o(\sigma^{-2})$ suffices.
   - Distribution recovery (exact recovery of $\mu_{\text{data}}$): score error must be $o(1)$.
   - The gap is $O(\sigma^{-2})$, which is enormous for small $\sigma$.

   **(ii) Uniform Distribution Generation (Theorems 5.1–5.2)**:
   - Requires only a one-line code modification: replace the score in the sampling algorithm with $\tilde{s}_\sigma(x) = \nabla \log p_\sigma(x) + \frac{1}{\sigma^2} \nabla d_\mathcal{M}(x)$.
   - Required score accuracy: $o(\sigma^{-2})$, the same as for geometry learning.
   - The uniform distribution is the maximum-entropy representation of manifold geometry and requires no fine-tuning at inference time.

   **(iii) Bayesian Inverse Problems (Theorem 6.1)**:
   - When the prior is uniform, posterior sampling requires only $o(\sigma^{-2})$ score accuracy.
   - When the prior is $\mu_{\text{data}}$, stricter $o(1)$ accuracy is required.
   - This implies that the maximum-entropy prior is more robust to score errors.

### Key Mathematical Tools

- **Gaussian Smoothed Measure**: $\mu_\sigma = \text{law}(X + \sigma Z)$ (VE) or $\text{law}(\sqrt{1-\sigma^2}X + \sigma Z)$ (VP).
- **Distance Function**: $d_\mathcal{M}(x) = \frac{1}{2}\text{dist}^2(x, \mathcal{M})$.
- **Riemannian Volume Measure**: $d\mathcal{M}(x)$ and metric tensor $g(u)$.
- **Stationary Distribution of Non-Reversible Dynamics**: When the score is not a gradient field (e.g., in parameterized models), the stationary distribution of Langevin dynamics has no closed-form expression.

### Uniform Sampling Algorithm
Modify the Langevin dynamics as:
$$dX_t = \left[\nabla \log p_\sigma(X_t) + \frac{1}{\sigma^2}\nabla d_\mathcal{M}(X_t)\right] dt + \sqrt{2}\, dW_t$$

The term $\frac{1}{\sigma^2}\nabla d_\mathcal{M}$ precisely cancels the geometric (projection) component of the score, leaving only the distributional term — which is then overwhelmed by noise and effectively ignored, causing the dynamics to converge to the uniform distribution on the manifold.

## Key Experimental Results

### Synthetic Experiments
- A 1D circle embedded in 2D space: validates the feasibility of geometry recovery and uniform sampling.
- Under varying score error levels: $o(\sigma^{-2})$ suffices to recover the manifold support, while the induced distribution remains arbitrary.
- After the one-line modification: the algorithm generates the uniform distribution on the manifold.

### Stable Diffusion 1.5 Experiments
- Theoretical predictions are validated on a large-scale pretrained model.
- The modified sampling algorithm produces samples with more uniform manifold coverage.
- Sample diversity increases and bias decreases.

### Key Findings
- Experiments confirm that the $\Theta(\sigma^{-2})$ rate separation holds in practical models.
- The uniform sampling algorithm operates effectively without any fine-tuning.
- Standard diffusion models primarily learn manifold geometry rather than the precise data distribution.

## Highlights & Insights
- **A Paradigm-Shifting Claim**: The shift from distribution learning to geometry learning is not merely a theoretical finding but a practical guideline — perfect distribution recovery need not be the objective.
- **The Power of One Line**: Uniform sampling requires only a single modification (adding the distance gradient term), making it highly practical.
- **A New Explanation for Diffusion Model Success**: Diffusion models succeed because they learn "what real data looks like" (the manifold), not "the frequency distribution of real data."
- **Implications for Bayesian Inverse Problems**: The uniform (maximum-entropy) prior is more robust to score errors than the data prior, offering important guidance for downstream applications.

## Limitations & Future Work
- The theoretical analysis relies on the manifold hypothesis ($C^4$ compact manifold without boundary), which real data may not strictly satisfy.
- $\nabla d_\mathcal{M}$ must be estimated in practice, and estimation accuracy affects the quality of uniform sampling.
- Preliminary experiments are conducted on Stable Diffusion 1.5; validation on larger models and broader tasks remains to be done.
- The precise constants (prefactors) in the rate separation are not discussed; the practical gap may be smaller than the theoretical prediction.

## Related Work & Insights
- **vs. Stanczuk et al. 2024**: Their work estimates intrinsic dimensionality and does not address rate separation.
- **vs. Ventura et al. 2024**: Analyzes only linear manifolds (subspaces); the present work applies to general smooth manifolds.
- **vs. De Santi et al. 2025**: They achieve uniform sampling via fine-tuning; the present work achieves this at inference time without fine-tuning.
- **vs. Existing Score Learning Theory**: Prior work conflates geometric and distributional errors; this paper is the first to separate the two.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of rate separation and the claim of a geometry learning paradigm are groundbreaking.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both synthetic data and Stable Diffusion, though more quantitative evaluation would strengthen the paper.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical development is clear, illustrations are intuitive, and main contributions are well-highlighted.
- Value: ⭐⭐⭐⭐⭐ Significant implications for both understanding diffusion model principles and improving sampling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generative Model Inversion Through the Lens of the Manifold Hypothesis](../../NeurIPS2025/image_generation/generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)
- [\[ICLR 2026\] The Spacetime of Diffusion Models: An Information Geometry Perspective](the_spacetime_of_diffusion_models_an_information_geometry_perspective.md)
- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[ICLR 2026\] When One Modality Rules Them All: Backdoor Modality Collapse in Multimodal Diffusion Models](when_one_modality_rules_them_all_backdoor_modality_collapse_in_multimodal_diffus.md)
- [\[ICLR 2026\] Concept-TRAK: Understanding how diffusion models learn concepts through concept-level attribution](concept-trak_understanding_how_diffusion_models_learn_concepts_through_concept-l.md)

</div>

<!-- RELATED:END -->
