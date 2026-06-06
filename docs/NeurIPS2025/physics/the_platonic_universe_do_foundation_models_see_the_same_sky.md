---
title: >-
  [Paper Note] The Platonic Universe: Do Foundation Models See the Same Sky?
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][Platonic Representation Hypothesis] This paper validates the Platonic Representation Hypothesis (PRH) in an astronomical setting. Using JWST, HSC, Legacy Survey…
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "Platonic Representation Hypothesis"
  - "foundation models"
  - "astronomy"
  - "representation alignment"
  - "cross-modal convergence"
date: 2026-05-08
content_hash: 3fd2fd4e64860029
---

# The Platonic Universe: Do Foundation Models See the Same Sky?

**Conference**: NeurIPS 2025
**arXiv**: [2509.19453](https://arxiv.org/abs/2509.19453)  
**Code**: None (uses public models and the Multimodal Universe dataset)  
**Area**: Physics
**Keywords**: Platonic Representation Hypothesis, foundation models, astronomy, representation alignment, cross-modal convergence

## TL;DR
This paper validates the Platonic Representation Hypothesis (PRH) in an astronomical setting. Using JWST, HSC, Legacy Survey, and DESI spectroscopic data, it measures representation alignment across six foundation models (ViT/ConvNeXt/DINOv2/IJEPA/AstroPT/Specformer) and finds that both intra-modal and cross-modal MKNN scores consistently increase with model scale ($p = 3.31 \times 10^{-5}$), supporting the hypothesis that models of different architectures and modalities converge toward a shared representation.

## Background & Motivation

**Background**: Astronomy is experiencing a "fourth wave" of AI adoption—the influx of foundation models. Multiple groups are independently exploring contrastive, generative, and autoregressive approaches to building astronomical foundation models, yet no consensus exists on the optimal architecture.

**Limitations of Prior Work**: Astronomical observations are fundamentally different projections of the same underlying physics (optical imaging, infrared imaging, spectroscopy), yet models for each modality are typically designed and trained independently, leaving cross-modal knowledge largely underutilized.

**Key Challenge**: It remains unclear whether the astronomical community needs to train dedicated foundation models from scratch, or whether it can leverage the enormous GPU compute already invested in general-purpose vision models.

**Goal**: Quantitatively validate the PRH in an astronomical context—determining whether different neural networks converge to a consistent representation space given sufficient data and compute.

**Key Insight**: Astronomy provides an ideal testbed for the PRH: different observational modalities are mathematical projections of the same physical reality, so if the PRH holds, models should learn similar representations.

**Core Idea**: Even general-purpose vision models pretrained on natural images exhibit cross-modal alignment on astronomical data that increases significantly with model scale, with no clear advantage for astronomy-specific models.

## Method

### Overall Architecture
4 astronomical datasets (HSC/Legacy/JWST images + DESI spectra) × 6 model architectures → embedding extraction → intra-modal and cross-modal MKNN alignment measurement → analysis of the scale–alignment relationship.

### Key Designs

1. **Data modality selection**:

    - HSC: ground-based optical imaging ($z/r/g$ bands), used as the reference baseline
    - DESI Legacy Survey: a distinct ground-based optical survey strategy
    - JWST NIRCam: space-based infrared imaging (F444W/F277W/F090W), representing the most extreme imaging test
    - DESI spectra: 1D spectroscopic data, a modality fundamentally different from imaging
    - The Multimodal Universe (MMU) dataset is used for cross-modal object matching

2. **Model architecture coverage**:

    - Supervised classification: ViT (Base/Large/Huge), ConvNeXtv2 (Nano/Tiny/Base/Large)
    - Self-supervised knowledge distillation: DINOv2 (Small/Base/Large/Giant)
    - Self-supervised prediction: IJEPA
    - Astronomy-specific autoregressive: AstroPTv2 (Small/Base/Large), pretrained on DESI Legacy Survey
    - Astronomical spectral Transformer: Specformer, processing 1D spectra

3. **MKNN alignment metric**:

    - $\text{MKNN}(\mathbf{z}_1, \mathbf{z}_2) = k^{-1} |N_k(\mathbf{z}_1) \cap N_k(\mathbf{z}_2)|$
    - Intra-modal test: alignment between embeddings of different scales within the same modality and architecture
    - Cross-modal test: alignment between embeddings of the same architecture and scale across different modalities
    - PRH prediction: both should increase with model scale

## Key Experimental Results

### Main Results — Intra-modal Alignment (Selected)

| Model Pair | JWST | Legacy | HSC |
|------------|------|--------|-----|
| AstroPTv2 S vs B | 49.7% | 8.1% | 10.3% |
| AstroPTv2 B vs L | 56.2% | 10.0% | 13.5% |
| DINOv2 L vs G | 40.2% | 10.2% | 10.9% |
| ViT L vs H | 32.6% | 4.4% | 5.0% |

### Statistical Tests

| Alignment Type | Fraction Increasing | Binomial Test $p$-value |
|----------------|---------------------|-------------------------|
| Intra-modal | 14/18 (78%) | $p = 1.54 \times 10^{-2}$ |
| Cross-modal | 28/33 (85%) | $p = 3.31 \times 10^{-5}$ |

### Key Findings
- **Significant cross-modal alignment growth**: In 28 out of 33 cross-modal comparisons, MKNN increases with model scale—a result that is highly statistically significant.
- **General models ≈ astronomy-specific models**: AstroPTv2 (astronomy-specific) shows no significantly higher alignment than DINOv2 or ViT (pretrained on natural images).
- **Most extreme cross-modal case also holds**: Models pretrained on natural images also exhibit an increasing alignment trend on Specformer embeddings derived from DESI spectra.
- **JWST yields the highest alignment**: When paired with HSC, MKNN scores for JWST are systematically higher than those for the Legacy Survey.

## Highlights & Insights
- **Scientific justification for transfer learning**: The astronomical community need not train dedicated foundation models from scratch—reusing and fine-tuning models that the ML community has invested GPU-centuries in pretraining substantially reduces computational and carbon costs.
- **Astronomy as a natural testbed for the PRH**: Different observational modalities are mathematical projections of the same physical reality, giving this test stronger physical grounding than evaluations in the natural image domain.
- **Clear practical recommendation**: "focus less on astronomy-specific architectures and more on scale and data diversity."

## Limitations & Future Work
- Some cross-modal matched datasets are small (JWST vs. HSC contains only 1.67K objects), which may limit representativeness.
- MKNN is a single alignment metric; complementary measures such as CKA and mutual information are not explored.
- Additional architecture types such as LLMs and diffusion models are not evaluated.
- Alignment measurement does not imply causal inference—high MKNN does not directly demonstrate that models "understand the same physics."

## Related Work & Insights
- **vs. PRH original paper (Huh et al., ICML 2024)**: The original work is a position paper on general-purpose vision; this paper provides the first quantitative validation on scientific (astronomical multi-modal) data.
- **vs. AstroLLaMA and related astronomical LLMs**: These models pursue domain-specific training; the present results suggest this may be unnecessary.
- **vs. Multimodal Universe (MMU)**: This paper relies on MMU's cross-modal matching infrastructure, further demonstrating the value of the MMU platform.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic validation of the PRH in a scientific domain; a distinctive perspective
- Experimental Thoroughness: ⭐⭐⭐ Broad coverage across 6 architectures × 4 modalities, but limited dataset sizes
- Writing Quality: ⭐⭐⭐⭐⭐ Beautifully written; the analogy from Plato's cave allegory to astronomical observation is compelling
- Value: ⭐⭐⭐⭐ Directly informs strategic directions for the astronomical foundation model community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FAIR Universe HiggsML Uncertainty Dataset and Competition](fair_universe_higgsml_uncertainty_dataset_and_competition.md)
- [\[NeurIPS 2025\] Physics-Guided Machine Learning for Uncertainty Quantification in Turbulence Models](physics-guided_machine_learning_for_uncertainty_quantification_in_turbulence_mod.md)
- [\[AAAI 2026\] Towards a Foundation Model for Partial Differential Equations Across Physics Domains](../../AAAI2026/physics/towards_a_foundation_model_for_partial_differential_equations_across_physics_dom.md)
- [\[NeurIPS 2025\] Transfer Learning Beyond the Standard Model](transfer_learning_beyond_the_standard_model.md)
- [\[NeurIPS 2025\] Vision Transformers for Cosmological Fields: Application to Weak Lensing Mass Maps](vision_transformers_for_cosmological_fields_application_to_weak_lensing_mass_map.md)

</div>

<!-- RELATED:END -->
