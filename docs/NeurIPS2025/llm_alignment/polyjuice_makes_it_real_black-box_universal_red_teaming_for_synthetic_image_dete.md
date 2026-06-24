---
title: >-
  [Paper Note] PolyJuice Makes It Real: Black-Box, Universal Red Teaming for Synthetic Image Detectors
description: >-
  [NeurIPS 2025][LLM Alignment][synthetic image detection] This paper proposes PolyJuice, the first black-box, image-agnostic red teaming method for synthetic image detectors (SIDs). By discovering and exploiting a "realism direction" in the latent space of text-to-image (T2I) models, PolyJuice universally steers generated images to fool detectors, achieving an attack success rate of up to 84%.
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "synthetic image detection"
  - "red teaming"
  - "adversarial attack"
  - "text-to-image"
  - "black-box attack"
date: 2026-05-08
content_hash: 994dd76cc4faa7ee
---

# PolyJuice Makes It Real: Black-Box, Universal Red Teaming for Synthetic Image Detectors

**Conference**: NeurIPS 2025
**arXiv**: [2509.15551](https://arxiv.org/abs/2509.15551)  
**Code**: [Project Page](https://sepehrdehdashtian.github.io/Papers/PolyJuice)  
**Area**: LLM Alignment
**Keywords**: synthetic image detection, red teaming, adversarial attack, text-to-image, black-box attack

## TL;DR

This paper proposes PolyJuice, the first black-box, image-agnostic red teaming method for synthetic image detectors (SIDs). By discovering and exploiting a "realism direction" in the latent space of text-to-image (T2I) models, PolyJuice universally steers generated images to fool detectors, achieving an attack success rate of up to 84%.

## Background & Motivation

With rapid advances in T2I generative models (e.g., Stable Diffusion, FLUX), synthetic images are nearly indistinguishable from real ones. Synthetic image detectors (SIDs) serve as a critical line of defense against the risks of generated content, and red teaming can expose blind spots in SIDs to improve detection performance.

However, existing red teaming methods suffer from two fundamental limitations:

**White-box access to SIDs is required**: Existing Unrestricted Adversarial (UA) attacks require access to detector weights or gradients, whereas state-of-the-art detectors (e.g., Reality Defender) are typically closed-source APIs.

**Per-image optimization is required**: Existing methods must individually optimize perturbations or latent directions for each image, incurring high computational costs that scale exponentially with resolution.

These limitations render existing methods infeasible in practical settings.

## Method

### Overall Architecture

The core idea of PolyJuice is that within the latent space of T2I models, there exists an **observable distributional shift** between samples correctly detected as fake (TP) and those misclassified as real (FN). PolyJuice identifies this shift direction and uses it to universally steer all generated images during the generation process.

Workflow:
1. **Offline phase**: Generate a batch of images, obtain black-box SID labels, and discover the guidance direction in latent space.
2. **Online phase**: Apply the guidance direction at each timestep during the generation of new images.

### Key Designs

**Discovering a Universal "Realism Shift" Direction**:

Supervised PCA (SPCA) is used to identify the subspace in the latent space that exhibits the greatest statistical dependence on label variation. Specifically, the following optimization problem is solved:

$$\underset{\mathbf{U}}{\arg\max} \; \text{Tr}\left\{\mathbf{U}^\top \mathbf{Z} \mathbf{H} \mathbf{K}_{\mathbf{YY}} \mathbf{H} \mathbf{Z}^\top \mathbf{U}\right\}, \quad \text{s.t.} \; \mathbf{U}^\top \mathbf{U} = \mathbf{I}$$

where $\mathbf{H} = \mathbf{I}_n - \frac{1}{n} \mathbf{1}_n \mathbf{1}_n^\top$ is the centering matrix and $\mathbf{K}_{\mathbf{YY}}$ is the kernel matrix of labels.

The Hilbert-Schmidt Independence Criterion (HSIC) is employed as the dependence measure. The final guidance direction is a weighted combination of eigenvectors:

$$\boldsymbol{\delta} = \sum_{k=0}^{d-1} \sigma_k \mathbf{U}_k$$

**Time-Varying Guidance** (for diffusion/flow-matching models):

Since the latent space of T2I models is a time-indexed collection $\{\mathcal{Z}_t\}_{t=0}^{T-1}$, a guidance direction is computed separately for each timestep, yielding the direction set $\Delta = \{\boldsymbol{\delta}_0, \boldsymbol{\delta}_1, \ldots, \boldsymbol{\delta}_{T-1}\}$.

At each sampling step, the mapping function is defined as:

$$h_{\boldsymbol{\delta}_t}(\mathbf{z}'_t) = \mathbf{z}'_t + \lambda_t \boldsymbol{\delta}_t, \quad t = 1, \ldots, T-1$$

where $\lambda_t$ controls the guidance strength at each timestep.

**Resolution Transferability**:

A key insight is that the KL-regularized autoencoders used in T2I models focus on perceptual compression, such that **latent spaces at different resolutions maintain similar spatial properties**. Guidance directions computed at low resolution (256×256) can be transferred to high resolution (1024×1024) via interpolation:

$$\boldsymbol{\delta}'_t = \text{Interp}(\boldsymbol{\delta}_t; H', W')$$

### Loss & Training

PolyJuice involves no model training. It only requires:
- Generating 20K TP + 20K FN samples (using COCO training set captions)
- Performing one SPCA decomposition per timestep (closed-form solution, efficient)
- Guidance directions, once computed, can be reused indefinitely

## Key Experimental Results

### Main Results

**Attack Success Rate (%) of PolyJuice vs. Unguided Baseline**:

| T2I Model | Resolution | UFD (Unguided) | UFD (PolyJuice) | RINE (Unguided) | RINE (PolyJuice) |
|-----------|------------|----------------|-----------------|-----------------|------------------|
| SDv3.5 | 256 | 12.8 | **80.6** (+67) | 15.3 | **99.7** (+84) |
| FLUX_dev | 256 | 67.6 | **96.3** (+28) | 52.4 | **81.2** (+28) |
| FLUX_sch | 256 | 61.7 | **83.4** (+21) | 45.4 | **73.8** (+28) |
| SDv3.5 | 512 | 30.5 | **85.0** (+54) | 26.7 | **99.6** (+72) |
| FLUX_dev | 512 | 84.0 | **98.9** (+14) | 77.2 | **96.7** (+19) |
| SDv3.5 | 1024 | 59.3 | **93.3** (+34) | 51.0 | **99.8** (+48) |
| Average | — | 59.6±21.8 | **89.4±6.8** | 53.7±21.1 | **91.7±9.0** |

PolyJuice substantially improves attack success rates across all T2I–SID combinations, increasing the average from ~55% to ~90%.

### Ablation Study

**SID Improvement After Calibration with PolyJuice Attack Samples (FNR)**:

| T2I Model | Resolution | UFD Before | UFD After | RINE Before | RINE After |
|-----------|------------|------------|-----------|-------------|------------|
| SDv3.5 | 256 | 13.4 | **7.5** (-5) | 15.1 | **3.8** (-11) |
| FLUX_dev | 256 | 69.2 | **47.0** (-22) | 52.0 | **21.8** (-30) |
| FLUX_sch | 256 | 64.3 | **43.7** (-20) | 39.6 | **18.4** (-21) |
| SDv3.5 | 512 | 31.1 | **16.1** (-15) | 17.8 | **4.7** (-13) |
| FLUX_dev | 512 | 86.2 | **70.3** (-15) | 69.4 | **41.4** (-28) |

After augmenting datasets with PolyJuice, SID detection performance improves by up to 30%.

**Resolution Transferability** (256→512, evaluated on RINE):

| T2I Model | Unguided | Original 512 Direction | Transferred 256 Direction |
|-----------|----------|------------------------|---------------------------|
| SDv3.5 | 26.7 | 77.6 | **99.6** |
| FLUX_dev | 77.2 | 95.7 | **96.7** |
| FLUX_sch | 62.9 | 79.9 | **84.1** |

Transferred low-resolution directions **even outperform** directions computed directly at high resolution, as SPCA is more stable in lower-dimensional spaces.

**Image Quality Preservation** (FLUX_sch, PolyJuice vs. Unguided):

| Metric | Unguided | PolyJuice |
|--------|----------|-----------|
| FID↓ | 17.65 | **17.23** |
| cFID↓ | 17.81 | **17.41** |
| Precision↑ | 0.495 | 0.485 |
| Recall↑ | 0.485 | **0.498** |
| Density↑ | 0.585 | **0.764** |

PolyJuice guidance has negligible impact on image quality, with FID even improving slightly.

### Key Findings

1. **SDv3.5 is most detectable yet benefits most from guidance**: This indicates that PolyJuice precisely identifies the blind spots of SIDs.
2. **FLUX_dev outperforms FLUX_sch**: More inference steps (50 vs. 4) provide more opportunities for guidance injection.
3. **Spectral fingerprint analysis**: Images generated with PolyJuice have spectra closer to real images, indicating that the guidance direction effectively obscures the frequency-domain fingerprints of T2I models.
4. **CLIP embedding space visualization**: A "realism region" exists in the perceptual space of SIDs; unguided T2I models fail to explore this region, whereas PolyJuice precisely steers generated images into it.

## Highlights & Insights

1. **Black-box + universal = practically deployable**: Only hard labels from the SID are needed to discover the attack direction; no gradients or model weights are required.
2. **Image-agnostic guidance direction**: Computed once and reused indefinitely — directions derived from animal or object images can even be applied to guide the generation of human face images.
3. **Elegant application of SPCA**: The unobservable "realism" attribute is transformed into a statistically measurable direction in latent space.
4. **Dual offensive-defensive utility**: PolyJuice serves not only as an attack tool but also enables improvement of SID performance by up to 30% when the collected attack samples are used for calibration.
5. **Resolution transferability** substantially reduces computational costs by avoiding the generation and processing of high-resolution datasets.

## Limitations & Future Work

1. Sufficient TP/FN samples are needed to discover the guidance direction; for very strong SIDs with very few FN samples, a larger generation budget may be required.
2. The guidance strength $\lambda_t$ requires manual tuning and may vary across different T2I–SID combinations.
3. SPCA identifies linear directions; nonlinear methods may capture more complex deception patterns.
4. Only two SID models (UFD and RINE) are evaluated, leaving generalization to a broader set of detectors unassessed.
5. The interpretability of guidance directions is limited; it remains unclear whether the specific blind spots of SIDs are semantic-level or frequency-domain-level phenomena.

## Related Work & Insights

- Complementary to white-box UA attacks: PolyJuice is the first practical black-box alternative.
- Cross-domain application of SPCA/HSIC: bridging statistical/kernel methods with generative model security.
- Directions for extension: the approach can be generalized to video generation detection, audio deepfake detection, and related domains.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Introduces the first black-box, universal red teaming paradigm for SIDs; the use of SPCA to identify latent space shift directions is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple T2I models and resolutions with image quality, spectral, and transferability analyses; SID coverage is somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, methodology is concise, and visual analysis is rich.
- Value: ⭐⭐⭐⭐⭐ Highly significant for AI security, providing practical tools for both attack and defense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs](gasp_efficient_black-box_generation_of_adversarial_suffixes_for_jailbreaking_llm.md)
- [\[ACL 2025\] Constitutional Classifiers: Defending Against Universal Jailbreaks Across Thousands of Hours of Red Teaming](../../ACL2025/llm_alignment/constitutional_classifiers_defending_against_universal_jailbreaks_across_thousan.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)
- [\[ACL 2025\] M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](../../ACL2025/llm_alignment/m2s_multiturn_to_singleturn_jailbreak_in.md)
- [\[NeurIPS 2025\] Preference Learning with Lie Detectors can Induce Honesty or Evasion](preference_learning_with_lie_detectors_can_induce_honesty_or_evasion.md)

</div>

<!-- RELATED:END -->
