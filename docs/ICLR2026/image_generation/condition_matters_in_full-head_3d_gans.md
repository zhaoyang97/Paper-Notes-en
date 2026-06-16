---
title: >-
  [Paper Note] Condition Matters in Full-head 3D GANs
description: >-
  [ICLR2026][3D-aware GAN] This paper identifies that view conditioning in full-head 3D GANs introduces severe directional bias—generation quality is substantially higher at the conditioned viewpoint than at others. To add…
tags:
  - "ICLR2026"
  - "3D-aware GAN"
  - "full-head generation"
  - "semantic conditioning"
  - "view conditioning"
  - "synthetic data"
date: 2026-05-08
content_hash: 98ba8bd36e079aff
---

# Condition Matters in Full-head 3D GANs

**Conference**: ICLR2026
**arXiv**: [2602.07198](https://arxiv.org/abs/2602.07198)  
**Code**: [https://lhyfst.github.io/balancehead/](https://lhyfst.github.io/balancehead/)  
**Area**: Others
**Keywords**: 3D-aware GAN, full-head generation, semantic conditioning, view conditioning, synthetic data

## TL;DR
This paper identifies that view conditioning in full-head 3D GANs introduces severe directional bias—generation quality is substantially higher at the conditioned viewpoint than at others. To address this, the authors propose replacing view conditioning with view-invariant semantic features (frontal CLIP features) and introduce BalanceHead360, a synthetic dataset of 11.2 million 360° full-head images generated via Flux.1 Kontext, achieving for the first time high-fidelity, diverse full-head generation with consistent quality across all viewpoints.

## Background & Motivation

**Background**: 3D-aware GANs (EG3D, PanoHead, SphereHead, HyPlaneHead) adopt tri-plane representations for 3D head generation and inherit EG3D's view conditioning strategy, using camera pose angles as generator conditions.

**Limitations of Prior Work**: (a) **Directional bias**—view conditioning causes the generator to produce substantially higher quality at the conditioned viewpoint than at others, leading to global inconsistency (Fig. 2d–i); (b) inference requires fixing a frontal condition to ensure frontal quality, sacrificing diversity in back-view generation; (c) **data imbalance**—in-the-wild datasets exhibit highly uneven distributions of quality, quantity, and diversity across viewpoints; (d) removing conditioning entirely leads to mode collapse, rendering training infeasible.

**Key Challenge**: Full-head GANs require conditioning to stabilize training (unconditional training collapses), yet view conditioning introduces directional bias. A view-invariant conditioning mechanism is therefore needed.

**Goal**: Design a view-invariant conditioning strategy and construct a viewpoint-balanced dataset so that full-head GANs can generate high-quality outputs across all viewpoints.

**Key Insight**: Use frontal CLIP image features as a shared semantic condition—all viewpoints of the same identity share a single condition—decoupling generative capability from viewing direction.

**Core Idea**: Shift 3D-aware GANs from view conditioning to semantic conditioning by replacing camera pose angles with frontal CLIP features as generator input, thereby eliminating directional bias.

## Method

### Overall Architecture
(1) Construct the BalanceHead360 dataset: use Flux.1 Kontext to expand real frontal images into multi-view synthetic images (11.2 million 360° full-head images), with frontal CLIP features serving as unified condition labels. (2) Train a semantically conditioned 3D-aware GAN (based on the HyPlaneHead architecture), replacing view conditioning with semantic conditioning and incorporating the ViCiCo loss to enhance consistency.

### Key Designs

1. **View-invariant Semantic Conditioning**:

    - Function: Use frontal CLIP image features as a shared condition across all viewpoints.
    - Mechanism: For each subject, generation at all viewpoints is conditioned on the same frontal CLIP feature, so the condition carries no viewpoint information, decoupling generative capacity from viewing direction.
    - Why frontal views: Frontal views contain the most comprehensive semantic information (facial features, hairstyle, clothing) and serve as the optimal identity anchor.
    - Design Motivation: Removing viewpoint information from individual images is infeasible (different viewpoints contain different visual content), but all viewpoints can be anchored to a single reference viewpoint.

2. **BalanceHead360 Dataset Construction**:

    - Function: Construct a dataset of 11.2 million 360° full-head images spanning all viewpoints.
    - Pipeline: ~350k real frontal/lateral images → HyperIQA quality filtering → Flux.1 Kontext frontal image generation → Flux.1 Kontext multi-view expansion via viewpoint-specific prompts → Qwen2.5-VL artifact filtering → VGGHeads pose estimation → ArcFace identity verification.
    - Key Finding: Although 2D generative models do not guarantee strict 3D consistency, 3D-aware GANs naturally filter out inconsistent 2D artifacts through adversarial training and tri-plane representations.
    - Design Motivation: In-the-wild data inevitably suffers from uneven viewpoint distribution; synthetic data enables uniform coverage across all viewpoints.

3. **ViCiCo Loss (View-image and Condition-image Consistency)**:

    - Function: Prevent multi-face artifacts and strengthen consistency between outputs and semantic conditions.
    - Mechanism: Camera labels and/or semantic conditions are randomly shuffled; mismatched pairs are fed into the discriminator as negative samples: $\mathcal{L}_{\text{ViCiCo}} = \log(1 - D((I^+, I, I^m), (r_{\text{cam}}', c_{\text{sem}}')))$
    - Design Motivation: Forces the generator to follow the true semantic distribution rather than stagnating after learning a limited set of modes.

### Loss & Training
Built upon HyPlaneHead (StyleGAN2 backbone + hybrid plane representation). Trained on 8 × H20 GPUs with batch size 32 for 10 days, processing a total of 32 million images.

## Key Experimental Results

### Main Results (FID Evaluation)

| Conditioning | ViCiCo | FID-view ↓ | FID-random ↓ | FID-front ↓ |
|---|---|---|---|---|
| View conditioning | ✗ | 9.67 | 13.82 | 8.42 |
| View + semantic conditioning | ✗ | 8.63 | 46.24 | 5.90 |
| Semantic conditioning | ✗ | - | 4.45 | 4.11 |
| **Semantic conditioning** | **✓** | **-** | **3.67** | **3.51** |

### Ablation Study

| Configuration | Result | Notes |
|---|---|---|
| No conditioning | Training collapse | Early mode collapse |
| Conditioning removed mid-training | Collapse after ~1000k images | Conditioning is necessary |
| Semantic conditioning + ViCiCo | FID-random 3.67 | All-viewpoint consistency, best performance |

### Key Findings
- **Severe directional bias**: Under view conditioning, FID-random (13.82) is substantially higher than FID-view (9.67), indicating significant quality degradation at non-conditioned viewpoints.
- **Semantic conditioning eliminates bias entirely**: FID-random drops from 13.82 to 3.67, achieving consistent generation quality across all viewpoints.
- **Training 3D-aware GANs on 2D synthetic data is effective**: Multi-view images from Flux.1 Kontext lack strict 3D consistency, yet the GAN is naturally robust to such inconsistency.
- **FID-front also improves**: From 8.42 to 3.51, reflecting comprehensive gains from semantic conditioning and larger-scale data.

## Highlights & Insights
- **"Conditioning determines the structure of the generative space"**: Changing the conditioning scheme fundamentally reshapes the 3D space learned by the GAN—an overlooked yet critically important design choice.
- **Synergy between 2D generative models and 3D-aware GANs**: Leveraging the powerful generative capacity of 2D models to produce training data while the 3D GAN naturally filters out 2D inconsistencies—a novel "inconsistency-tolerant" paradigm.
- **Semantic conditioning facilitates continued learning**: Adversarial training is prone to stagnation; semantic conditioning forces the generator to track the data distribution, yielding greater gains at larger scales.
- **11.2 million synthetic images**: Generated using 400 × A10 GPUs over 26 days, breaking the data bottleneck in full-head generation.

## Limitations & Future Work
- **Dependence on Flux.1 Kontext**: Data quality is bounded by the capabilities and biases of the underlying 2D generative model.
- **CLIP features only**: Fine-grained information may be lost; DINOv2 or multimodal encoders could yield better representations.
- **Extremely high training cost**: Reproducing results is challenging.
- **Dynamic expressions not explored**: Extending to talking-head or expression generation is an important future direction.

## Related Work & Insights
- **vs. PanoHead / SphereHead / HyPlaneHead**: All adopt view conditioning and inherit directional bias. This paper provides the first systematic analysis and resolution of this issue.
- **vs. SOAP**: SOAP collects renderings from 24k 3D models, limiting identity diversity. This paper scales to millions of identities using 350k real images.
- **vs. 3DGH**: 3DGH separately models the head and hair to reduce front-back discrepancy; this paper addresses the problem at its root.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The insight is simple yet profound; represents a paradigm-level design shift.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11.2M dataset, three FID metrics, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Directional bias visualizations are highly convincing.
- Value: ⭐⭐⭐⭐⭐ Paradigm-level impact on 3D head generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Scalable GANs with Transformers](../../ICML2026/image_generation/scalable_gans_with_transformers.md)
- [\[ICLR 2026\] Factuality Matters: When Image Generation and Editing Meet Structured Visuals](factuality_matters_when_image_generation_and_editing_meet_structured_visuals.md)
- [\[ICLR 2026\] PI-Light: Physics-Inspired Diffusion for Full-Image Relighting](pi-light_physics-inspired_diffusion_for_full-image_relighting.md)
- [\[ICLR 2026\] Condition Errors Refinement in Autoregressive Image Generation with Diffusion Loss](condition_errors_refinement_in_autoregressive_image_generation_with_diffusion_lo.md)
- [\[ICLR 2026\] Routing Matters in MoE: Scaling Diffusion Transformers with Explicit Routing Guidance](routing_matters_in_moe_scaling_diffusion_transformers_with_explicit_routing_guid.md)

</div>

<!-- RELATED:END -->
