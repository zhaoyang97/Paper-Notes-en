---
title: >-
  [Paper Note] Guiding Noisy Label Conditional Diffusion Models with Score-based Discriminator Correction
description: >-
  [ICCV 2025][Image Generation][Noisy Labels] This paper proposes Score-based Discriminator Correction (SBDC), which trains a lightweight discriminator to correct the generation trajectory of noisy-label conditional diffus…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Noisy Labels"
  - "Discriminator Guidance"
  - "Inference-time Correction"
  - "Conditional Diffusion Models"
  - "Score-based Correction"
date: 2026-05-08
content_hash: f3d23c44dffb4096
---

# Guiding Noisy Label Conditional Diffusion Models with Score-based Discriminator Correction

**Conference**: ICCV 2025
**arXiv**: [2508.19581](https://arxiv.org/abs/2508.19581)
**Code**: N/A
**Area**: Diffusion Models / Image Generation
**Keywords**: Noisy Labels, Discriminator Guidance, Inference-time Correction, Conditional Diffusion Models, Score-based Correction

## TL;DR

This paper proposes Score-based Discriminator Correction (SBDC), which trains a lightweight discriminator to correct the generation trajectory of noisy-label conditional diffusion models at inference time. The discriminator is trained by partitioning the training set into clean and corrupted subsets via noise detection, and the paper finds that applying guidance only during the early-to-middle stages of the sampling process yields optimal results.

## Background & Motivation

Diffusion models rely on large-scale datasets to achieve high-quality image generation; however, such datasets frequently contain mislabeled data. For instance, ImageNet exhibits approximately 6% label noise, and multimodal datasets such as LAION-5B have even higher noise rates. When conditional generative models are trained on noisy data, they learn the corrupted distribution $p(X|\tilde{Y})$ rather than the true distribution $p(X|Y)$, leading to degraded image quality and condition–image misalignment.

Existing solutions face several critical challenges:

**Prohibitive retraining cost**: Transition-matrix–based methods (e.g., TDSM) require multi-stage training, causing errors to propagate across stages; noise detection methods require data cleaning followed by full model retraining, which is impractical for large-scale models.

**Deficiencies of direct approaches**: TDSM degrades severely under high noise rates, frequently generating mislabeled images.

**Inference efficiency**: A solution with low computational overhead that does not require retraining the generative model is needed.

The core motivation of SBDC is: **can the generation trajectory of a noisy model be corrected at inference time via a lightweight auxiliary signal?** Drawing inspiration from Noise Contrastive Estimation (NCE), the paper leverages existing noise detection techniques and discriminator training to achieve this goal.

## Method

### Overall Architecture

The SBDC pipeline consists of two steps:
1. **Training phase**: A noise detection method partitions the dataset into a pseudo-clean set $\mathcal{D}_r$ and a corrupted set $\mathcal{D}_f$, which are used to train a time-dependent discriminator $D_\phi^t$.
2. **Inference phase**: The gradient signal from the discriminator is injected as guidance into the sampling process of the pretrained diffusion model, applied only during the early-to-middle stages of sampling (controlled by a γ-gate).

### Key Designs

1. **Analysis of noisy conditional generation behavior**: The sampling process is decomposed into three phases:

    - **Phase I (Marginalization)**: At large $t$, conditioning information is forgotten; $C(t)$ (confidence) is low and $I(t)$ (instability) is low.
    - **Phase II (Conditioning)**: Multiple modes compete to influence the perturbed identity; $I(t)$ peaks. **This is the phase of most severe class instability.**
    - **Phase III (Refinement)**: The posterior concentrates on a single target; $C(t)$ reflects only the noise rate.

   The quantitative metrics are:

    $C(t) = \mathbb{P}[f(\mathbf{x}_\theta(\mathbf{x}_t, \mathbf{y})) = \mathbf{y}]$

    $I(t) = \mathbb{P}[f(\mathbf{x}_\theta(\mathbf{x}_t, \mathbf{y})) \neq f(\mathbf{x}_\theta(\mathbf{x}_{t-1}, \mathbf{y}))]$

   Core insight: **Once a class transition occurs in Phase II, the error persists through to the final output**; correction should therefore be concentrated in this phase.

2. **Score-based discriminator correction**: Assuming the score network has perfectly learned the noisy distribution, the clean distribution is recovered via:

    $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{y}) = \nabla_{\mathbf{x}_t} \log p_\theta(\mathbf{x}_t|\tilde{\mathbf{y}}) + \nabla_{\mathbf{x}_t} \log \frac{p(\mathbf{x}_t|\mathbf{y})}{p_\theta(\mathbf{x}_t|\tilde{\mathbf{y}})}$

   The second term on the right (the correction term) is approximated by the log-likelihood ratio of the discriminator. The paper establishes a theoretical bound (Theorem 1): when the noise rate is not excessively high, the gradient of the optimal discriminator effectively estimates the true log-likelihood ratio.

3. **γ-gate mechanism**: Based on the phase analysis, discriminator guidance is applied only within a specific interval of the sampling process (the γ-gate), avoiding wasteful computation during Phase I (where conditioning is ineffective) and Phase III (where the trajectory has already converged). Experiments show that this restricted guidance actually improves overall performance.

4. **SiMix data augmentation**: To mitigate discriminator overfitting, the paper proposes Similarity-based Mixup (SiMix), which finds nearest-neighbor samples in feature space for interpolation rather than using random pairing. Specifically, for each sample the nearest neighbor by encoding distance within the batch is found, and linear interpolation is performed with a coefficient sampled from a Beta distribution:

    $\mathbf{z}_i \leftarrow \lambda_i \mathbf{z}_i + (1 - \lambda_i) \mathbf{z}_{\arg\min_{j} \|f_i - f_j\|_2}$

### Loss & Training

The discriminator is trained with a time-weighted binary cross-entropy loss:

$$\mathcal{L}_{adv} = \mathbb{E}_{t, (\mathbf{x}, \mathbf{y}) \sim p_r, \mathbf{x}_t}[-\log D_\theta^t(\mathbf{x}_t, \mathbf{y})] + \mathbb{E}_{t, (\mathbf{x}, \mathbf{y}) \sim p_f, \mathbf{x}_t}[-\log(1 - D_\theta^t(\mathbf{x}_t, \mathbf{y}))]$$

Additionally, Pseudo-clean Shuffle is introduced: a portion of samples from the clean set are randomly drawn, their labels replaced with corrupted labels, and these are designated as negative samples, thereby increasing the discriminator's sensitivity to label–image consistency.

## Key Experimental Results

### Main Results

Generation quality under various noise settings on CIFAR-10 (EDM as the base diffusion model):

| Noise Type | Noise Rate | Method | FID ↓ | IS ↑ | CW-FID ↓ | CW-Den. ↑ |
|------------|------------|--------|-------|------|----------|-----------|
| Symmetric | 20% | EDM | 1.96 | 9.95 | 11.3 | 98.6 |
| Symmetric | 20% | TDSM | 2.36 | 10.04 | 10.9 | 113.1 |
| Symmetric | 20% | **SBDC** | **2.49** | **10.06** | **10.6** | **114.8** |
| Symmetric | 50% | EDM | 2.07 | 9.69 | 38.6 | 66.8 |
| Symmetric | 50% | TDSM | 2.43 | 9.84 | 18.2 | 95.8 |
| Symmetric | 50% | **SBDC** | 2.24 | **9.87** | **15.6** | **98.1** |
| Symmetric | 80% | EDM | 2.15 | 9.67 | 71.7 | 43.0 |
| Symmetric | 80% | TDSM | 2.25 | 9.76 | 59.8 | 52.0 |
| Symmetric | 80% | **SBDC** | **2.30** | **9.71** | **48.2** | **58.0** |

SBDC comprehensively outperforms TDSM on class-conditional metrics (CW-FID, CW-Density, CW-Coverage), with particularly pronounced advantages at high noise rates (80%).

### Ablation Study

| Configuration | Key Effect | Remarks |
|---------------|-----------|---------|
| No γ-gate (full-process guidance) | Performance degradation | Applying guidance in Phases I and III is detrimental |
| No SiMix | Discriminator overfitting | Especially noticeable performance drop on small datasets |
| No Pseudo-clean Shuffle | CW metric degradation | Reduces discriminator sensitivity to label consistency |
| Different γ ranges | Phase II is optimal | Validates the correctness of the three-phase analysis |
| Different guidance weight $w$ | Excessively large $w$ causes distortion | Moderate weight balances correction and generation quality |

### Key Findings

- SBDC's advantage is most pronounced at high noise rates; at 80% noise, CW-FID improves from 59.8 (TDSM) to 48.2.
- TDSM frequently generates mislabeled images under high noise, whereas SBDC effectively corrects such errors.
- Instance noise (asymmetric noise) is more challenging than symmetric noise, yet SBDC maintains its advantage.
- Discriminator training is fast (offering enormous computational savings compared to retraining the diffusion model), and the additional inference overhead is minimal.

## Highlights & Insights

- The paper provides a fine-grained three-phase analysis of conditional diffusion model behavior under noisy labels, identifying Phase II as the optimal intervention window.
- Existing noise detection methods are cleverly leveraged to construct discriminator training data, avoiding additional annotation overhead.
- The γ-gate design simultaneously improves performance and reduces inference overhead, serving as a compelling demonstration of the "less is more" principle.
- SiMix is a novel Mixup variant based on feature similarity rather than random pairing, which benefits discriminator generalization.

## Limitations & Future Work

- Validation is limited to small-scale datasets such as CIFAR-10; experiments on large-scale text-to-image models (e.g., Stable Diffusion) are absent.
- The accuracy of the noise detection step affects final performance and may degrade under extremely high noise rates.
- The theoretical bound relies on Lipschitz conditions and the assumption of an optimal discriminator, which may not be fully satisfied in practice.
- Noise correction in multimodal conditioning scenarios (e.g., joint text–image conditions) remains unexplored.

## Related Work & Insights

- **Distinction from Discriminator Guidance**: DG aims to close the gap between real and synthetic data distributions, whereas SBDC specifically targets the discrepancy between clean and corrupted label distributions.
- TDSM modifies the training objective via a transition matrix, while SBDC operates entirely at inference time, making it more practical.
- Noise detection literature (e.g., SIMIFEAT, SOP) provides reliable pseudo-label sources for discriminator training.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of combining discriminator guidance with noisy-label correction is novel, and the three-phase analysis offers genuine insight.
- **Experimental Thoroughness**: ⭐⭐⭐ — Covers diverse noise settings but is limited to small-scale datasets.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear and experimental organization is sound.
- **Value**: ⭐⭐⭐⭐ — Addresses a practically important problem; the inference-time correction scheme demonstrates strong applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation](../../NeurIPS2025/image_generation/distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)
- [\[ICCV 2025\] UniCombine: Unified Multi-Conditional Combination with Diffusion Transformer](unicombine_unified_multi-conditional_combination_with_diffusion_transformer.md)
- [\[CVPR 2026\] Guiding Diffusion Models with Semantically Degraded Conditions](../../CVPR2026/image_generation/guiding_diffusion_models_with_semantically_degraded_conditions.md)
- [\[ICCV 2025\] ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion](scorehoi_physically_plausible_reconstruction_of_human-object_interaction_via_sco.md)
- [\[ICCV 2025\] HPSv3: Towards Wide-Spectrum Human Preference Score](hpsv3_towards_wide-spectrum_human_preference_score.md)

</div>

<!-- RELATED:END -->
