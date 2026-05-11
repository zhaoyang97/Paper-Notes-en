---
title: >-
  [Paper Note] Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability
description: >-
  [ICLR 2026][Image Generation][memorization detection] This paper demonstrates that norm-based memorization detection metrics are valid only under isotropic log-probability distributions and fail in low-noise anisotropic…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "memorization detection"
  - "anisotropy"
  - "score function"
  - "cosine similarity"
  - "denoising-free detection"
date: 2026-05-08
content_hash: 28bea0132602ffbb
---

# Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability

**Conference**: ICLR 2026
**arXiv**: [2601.20642](https://arxiv.org/abs/2601.20642)
**Code**: [GitHub](https://github.com/rohanasthana/memorization-anisotropy)
**Area**: Diffusion Models / Privacy
**Keywords**: memorization detection, anisotropy, score function, cosine similarity, denoising-free detection

## TL;DR

This paper demonstrates that norm-based memorization detection metrics are valid only under isotropic log-probability distributions and fail in low-noise anisotropic regimes. A denoising-free detection metric is proposed that combines high-noise norms with low-noise angular alignment (cosine similarity), surpassing existing denoising-free methods on SD v1.4/v2.0 while being over 5× faster.

## Background & Motivation

**Background**: Diffusion models inadvertently memorize exact copies of training samples, raising concerns about data privacy, copyright, and evaluation bias. Memorization detection has become an important research direction.

**Limitations of Prior Work**: Mainstream methods (Wen et al., Jeon et al.) detect memorization via the norm of the score difference $\|s_\theta^\Delta(\mathbf{x}_t, t, c)\|$, which essentially measures the overall curvature (Hessian trace) of the log-probability.

**Key Challenge**: Norm-based metrics implicitly assume that the log-probability distribution is isotropic (i.e., Hessian $\propto \mathbf{I}$, with uniform curvature in all directions). However, experiments show that in low-noise regimes the log-probability is in fact anisotropic (eigenvalue variance of the Hessian increases sharply), rendering norms unable to distinguish memorized from non-memorized samples (KL divergence drops from 0.166 under isotropy to 0.022 under anisotropy).

**Goal**: Accurately detect memorization even in anisotropic regimes, without requiring expensive denoising steps.

**Key Insight**: Analysis of the directional relationship between conditional and unconditional scores in low-noise anisotropic regimes reveals that guidance vectors of memorized samples are highly aligned with the unconditional score.

**Core Idea**: The signature of memorization in anisotropic regimes is angular alignment rather than norm spikes. A weighted combination of an "isotropic norm + anisotropic cosine similarity" enables efficient, denoising-free memorization detection.

## Method

### Overall Architecture

Given a pure-noise input $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and prompt $c$, one conditional and one unconditional forward pass are performed at two manually specified timesteps ($t=0$ and $t=T$) to compute the detection metric $\mathcal{M}(\mathbf{x}_T, c)$. Inputs exceeding a threshold are flagged as memorized prompts. No denoising trajectory is required.

### Key Designs

1. **Isotropic Norm Term (High-Noise Regime)**:

    - Function: Computes $\|s_\theta^\Delta(\mathbf{x}_T, t \approx T, c)\|$ at $t \approx T$.
    - Mechanism: At high noise levels, the log-probability is approximately isotropic (near-zero Hessian eigenvalue variance), making the norm an effective memorization indicator, consistent with Wen et al.
    - Design Motivation: Preserves a validated isotropic signal.

2. **Anisotropic Angular Alignment Term (Low-Noise Regime)**:

    - Function: Computes the cosine similarity between the guidance vector and the unconditional score at $t \approx 0$.
    - Mechanism: In memorized samples, the unconditional and conditional modes nearly coincide ($\delta \to 0$), causing $\nabla \log p_t(c|\mathbf{x}_t)$ and $\nabla \log p_t(\mathbf{x}_t)$ to be highly directionally aligned.
    - Theoretical Guarantee: Theorem 1 establishes the lower bound $\cos \geq \frac{1-r}{1+r}$, where $r = \varepsilon + \tau$, with $\varepsilon$ controlling covariance approximation error and $\tau$ controlling mode displacement.

3. **Combined Detection Metric**:
    $$\mathcal{M}(\mathbf{x}_T, c) = \gamma_1 \cdot \text{cos\_sim}(s_\theta^\Delta, s_\theta)|_{t \approx 0} + \gamma_2 \cdot \|s_\theta^\Delta\||_{t \approx T}$$
    - $\gamma_1, \gamma_2$ are determined via small-scale logistic regression (fitted once using only 20 memorized prompts).

### Loss & Training

Inference-time mitigation: The detection metric is used as a loss to optimize the prompt embedding via gradient descent:

$$\mathcal{L}(\mathbf{x}_T, c) = \mathcal{M}(\mathbf{x}_T, c)$$

The resulting corrected embedding $c^\star$ is then used to generate non-memorized content.

## Key Experimental Results

### Detection Performance (SD v1.4 / SD v2.0)

| Method | AUC↑ (n=1) | TPR@1%FPR↑ (n=1) | Time(s)↓ (10 prompts) |
|--------|-----------|------------------|----------------------|
| Ren et al. | 0.846 / 0.848 | 0.116 / 0.000 | 0.05 / 0.07 |
| Wen et al. | 0.976 / 0.948 | 0.896 / 0.739 | 0.40 / 0.80 |
| Jeon et al. | 0.987 / 0.959 | 0.908 / 0.740 | 5.40 / 14.60 |
| **Ours** | **0.994 / 0.953** | **0.935 / 0.791** | **1.10 / 2.20** |

At n=4, the proposed method achieves AUC of 0.999 (SD v1.4) and TPR@1%FPR of 0.984.

### Ablation Study (Component Contributions, SD v1.4 n=1)

| Component | AUC↑ | TPR@1%FPR↑ |
|-----------|------|------------|
| Norm only (isotropic) | 0.976 | 0.896 |
| Cosine only (anisotropic) | 0.923 | 0.424 |
| **Combined (Ours)** | **0.992** | **0.934** |

### Key Findings

- The isotropic norm alone performs well but is insufficiently precise under strict FPR constraints (TPR 0.896); adding the anisotropic cosine term improves this to 0.934.
- Pure cosine similarity performs poorly on SD v2.0 (AUC 0.779), as memorized prompts in v2.0 tend to involve localized memorization (larger $\delta$); however, the combined metric still yields improvement.
- Speed advantage: 5–7× faster than Jeon et al. by avoiding expensive Hessian computations.
- Generalization to Realistic Vision v5.1 is demonstrated (AUC 0.967).
- In mitigation experiments, the proposed method achieves the lowest SSCD similarity (indicating non-memorization) while maintaining CLIP/Aesthetic scores on par with baselines.

## Highlights & Insights

- **Strong Theoretical Contribution**: The failure mechanism of norm-based metrics under anisotropy is rigorously proven, and a theoretical lower bound on angular alignment is established (Theorem 1).
- **Insight on Timestep Mismatch**: Querying at $t=0$ while the input is pure noise $\mathbf{x}_T$ appears contradictory, yet works because memorization information is encoded in the learned log-probability independently of the input sample—this trick entirely eliminates the need for denoising.
- **Two Complementary Predictors**: The isotropic norm and anisotropic angle cover memorization signatures across different noise regimes, improving robustness on borderline cases.

## Limitations & Future Work

- $\gamma_1, \gamma_2$ require a small set of labeled memorized prompts for fitting and are not fully transferable across models (though $\gamma_1=\gamma_2=1$ also yields reasonable performance).
- Detection of localized memorization (partial-region memorization) is limited, as the cosine term is less reliable in this setting.
- Evaluation is restricted to SD v1.4/v2.0/Realistic Vision; newer architectures such as SD3 and Flux are not covered.
- The timestep mismatch phenomenon lacks a rigorous mathematical proof and is supported only by intuitive explanation.

## Related Work & Insights

- **vs. Wen et al.**: The same norm metric is used, but this paper proves its validity is restricted to isotropic regimes; adding the anisotropic term improves TPR@1%FPR by up to 5.2%.
- **vs. Jeon et al.**: Their Hessian-approximated sharpness metric is more accurate but requires 5–14× more computation; the proposed method relies solely on first-order score information.
- **vs. Ross et al.**: Memory is analyzed from a Local Intrinsic Dimensionality geometric perspective but requires a denoising trajectory; the proposed method is entirely denoising-free.
- Insight: Anisotropy analysis is transferable to memorization detection in other generative models (e.g., flow matching).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First introduction of anisotropy analysis to memorization detection; theoretically motivated method design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of detection, mitigation, and ablation; model coverage could be broader.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations, clear experimental setup, and intuitive figures.
- Value: ⭐⭐⭐⭐ Provides a new theoretical perspective and a practical, efficient method for memorization detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AutoDebias: An Automated Framework for Detecting and Mitigating Backdoor Biases in Text-to-Image Models](../../CVPR2026/image_generation/autodebias_automated_framework_for_debiasing_text-to-image_models.md)
- [\[ICLR 2026\] Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models](uni-x_mitigating_modality_conflict_with_a_two-end-separated_architecture_for_uni.md)
- [\[ICLR 2026\] VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model](vfscale_intrinsic_reasoning_through_verifier-free_test-time_scalable_diffusion_m.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[ICLR 2026\] Concept-TRAK: Understanding how diffusion models learn concepts through concept-level attribution](concept-trak_understanding_how_diffusion_models_learn_concepts_through_concept-l.md)

</div>

<!-- RELATED:END -->
