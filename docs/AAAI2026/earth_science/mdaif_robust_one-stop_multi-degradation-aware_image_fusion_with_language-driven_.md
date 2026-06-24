---
title: >-
  [Paper Note] MdaIF: Robust One-Stop Multi-Degradation-Aware Image Fusion with Language-Driven Semantics
description: >-
  [AAAI 2026][Earth Science][Infrared-Visible Fusion] This paper proposes MdaIF, a framework that leverages a vision-language model (VLM) to extract degradation-aware semantic priors for guiding mixture-of-experts (MoE) routing and channel attention modulation, enabling one-stop infrared-visible image fusion across multiple degradation scenarios without requiring degradation-type annotations.
tags:
  - "AAAI 2026"
  - "Earth Science"
  - "Infrared-Visible Fusion"
  - "Degradation-Aware"
  - "Mixture of Experts"
  - "Vision-Language Model"
  - "Channel Attention"
date: 2026-05-08
content_hash: 1e8e7adfb70d62ef
---

# MdaIF: Robust One-Stop Multi-Degradation-Aware Image Fusion with Language-Driven Semantics

**Conference**: AAAI 2026
**arXiv**: [2511.12525](https://arxiv.org/abs/2511.12525)  
**Code**: [https://github.com/doudou845133/MdaIF](https://github.com/doudou845133/MdaIF)  
**Area**: Image Fusion / Adverse Weather Degradation
**Keywords**: Infrared-Visible Fusion, Degradation-Aware, Mixture of Experts, Vision-Language Model, Channel Attention

## TL;DR

This paper proposes MdaIF, a framework that leverages a vision-language model (VLM) to extract degradation-aware semantic priors for guiding mixture-of-experts (MoE) routing and channel attention modulation, enabling one-stop infrared-visible image fusion across multiple degradation scenarios without requiring degradation-type annotations.

## Background & Motivation

**Background**: Infrared-visible image fusion (IVF) aims to integrate infrared thermal radiation information with visible-light texture details. Existing methods have evolved from CNN/GAN-based approaches to Transformers and diffusion models, but most assume high-quality visible images.

**Limitations of Prior Work**:
   - Under adverse weather conditions (haze, rain, snow), visible images suffer severe degradation, making direct fusion ineffective.
   - Cascaded pipelines (restoration followed by fusion) introduce feature misalignment and error accumulation.
   - Existing degradation-aware fusion methods (Text-IF, MMAIF) rely on fixed degradation-type annotations as prompts and employ a single static network for all degradation conditions.

**Key Challenge**: Different degradation types (micron-scale water droplets in haze, millimeter-scale raindrops in rain, ice crystals in snow) correspond to fundamentally distinct atmospheric scattering models, which a fixed network architecture cannot effectively capture. For example, transmission maps effective for dehazing fail in deraining scenarios.

**Goal**: To adaptively handle infrared-degraded visible image fusion across multiple adverse weather conditions without relying on ground-truth degradation type labels.

**Key Insight**: Leveraging VLM scene-understanding capabilities to automatically identify degradation types and extract semantic priors, which then guide MoE-based expert selection for handling different degradations.

**Core Idea**: VLM provides degradation-aware semantic priors → semantic priors guide channel attention modulation via prototype decomposition → modulated features and semantic priors jointly guide MoE routing to select degradation-specific experts for fusion.

## Method

### Overall Architecture

MdaIF consists of four core modules:

1. **Encoder**: Independently encodes infrared and degraded visible images.
2. **Degradation-aware Semantic Prior Extractor (DSPE)**: Extracts semantic priors from degraded visible images using the BLIP-2 VLM.
3. **Degradation-aware Channel Attention Module (DCAM)**: Performs degradation prototype decomposition and channel modulation using semantic priors.
4. **Degradation-aware Mixture of Experts (DMoE)**: Semantic-prior-guided expert routing combined with multi-expert fusion.

### Key Designs

1. **Degradation-aware Semantic Prior Extractor (DSPE)**:

    - Employs pretrained BLIP-2 OPT 2.7B in VQA mode, taking the degraded visible image and an open-ended question prompt as input.
    - Extracts last hidden-layer features $S_{org} \in \mathbb{R}^{S \times C_{org}}$ as raw semantic priors.
    - Compresses dimensionality via MLP + LayerNorm: $S_{embed} = \mathcal{N}_{layer}(\Phi_m^I(S_{org}))$.
    - Re-weights token importance via self-attention to obtain refined semantic priors $S_{prior}$.
    - Semantic priors comprise two components: $S_{weather}$ (weather degradation knowledge) and $S_{scene}$ (scene features).
    - **Key distinction**: Rather than using the VLM solely as a degradation classifier, the method fully exploits its deep semantic scene understanding.

2. **Degradation-aware Channel Attention Module (DCAM)**:

    - Concatenates encoded infrared and visible features along the channel dimension: $F_{in} = \text{Cat}(F_{vi}, F_{ir})$.
    - **Degradation prototype decomposition**: Maps semantic priors through MLP→Sigmoid to produce activation scores $s_K \in \mathbb{R}^K$ for $K$ degradation prototypes.
    - Each degradation prototype $k_i \in \mathbb{R}^C$ encodes channel-wise response intensity; the prototype matrix $W_{proto} \in \mathbb{R}^{K \times C}$ is initialized with orthonormal normalization.
    - Channel weight computation: $w_c = \sigma(\sum_{i=1}^K s_{K_i} \cdot k_i)$.
    - Final modulation: $F_{dcam} = \mathcal{N}_{layer}(F_{in}) \odot \sigma(s_K W_{proto}) + F_{in}$ (residual connection).
    - **Design Motivation**: Different degradation types activate distinct prototype combinations, with each prototype favoring different channel patterns, enabling degradation-adaptive feature enhancement.

3. **Degradation-aware Mixture of Experts (DMoE)**:

    - Multiple expert networks are each specialized for different degradation conditions.
    - Routing strategy: modulated features $F_{dcam}$ interact with semantic priors $S_{prior}$ to establish task-specific routing.
    - $S_{weather}$ enhances degradation texture features in the visible image; $S_{scene}$ enhances target information in both infrared and visible modalities.
    - Avoids expert load imbalance (one expert handling multiple tasks while others remain idle).

### Loss & Training

- Joint optimization of degradation restoration and multimodal fusion (one-stop scheme, not cascaded).
- VLM (BLIP-2) parameters are frozen; only the encoder, DCAM, MoE, and decoder are trained.
- Degradation prototype matrix is initialized with orthonormal normalization and set as a learnable parameter.

## Key Experimental Results

### Main Results

**One-stop methods vs. cascaded methods on MSRS dataset (Strategy I: separate models per degradation / Strategy II: unified model for all degradations)**:

MdaIF outperforms all cascaded combinations under all degradation conditions (Haze/Rain/Snow):

| Method | Haze PSNR↑ | Haze SSIM↑ | Rain PSNR↑ | Rain SSIM↑ | Snow PSNR↑ | Snow SSIM↑ |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| DehazeFormer+SegMiF | 17.051 | 1.046 | — | — | — | — |
| DRSformer+SegMiF | — | — | 17.308 | 0.859 | — | — |
| SnowFormer+SegMiF | — | — | — | — | 16.007 | 0.616 |
| SAGE (strongest cascade) | 17.260 | 1.231 | 17.964 | 0.993 | 17.267 | 0.897 |
| **MdaIF (Ours)** | **18.325** | **1.302** | **18.079** | **1.260** | **17.528** | **1.245** |

MdaIF achieves an average PSNR improvement of approximately 0.6–1.0 dB with substantial SSIM gains.

### Ablation Study

| Degradation Prototype Analysis | Observation |
|-------------------------------|-------------|
| Haze scenario | Prototype 1 has the highest activation (~40%); Prototypes 2/3 are lower |
| Rain scenario | Prototype 2 has the highest activation, notably different from haze |
| Snow scenario | Prototype 3 has the highest activation; latent correlations exist among the three degradations |

- Each prototype learns distinct channel preference patterns (verified via radar chart visualizations), enhancing mixed representational capacity.

### Key Findings

- The one-stop scheme significantly outperforms cascaded approaches, demonstrating that jointly optimizing degradation restoration and fusion effectively avoids error accumulation.
- VLM-extracted semantic priors serve beyond degradation classification, providing deep scene-level understanding that enriches feature interaction.
- The degradation prototype decomposition mechanism leads the model to exhibit differentiated yet interrelated activation patterns across weather conditions.

## Highlights & Insights

- The elimination of degradation-type annotation dependency is the key distinction from Text-IF and MMAIF — replacing manual annotation with VLM understanding improves practical applicability.
- Degradation prototype decomposition transforms the ambiguous notion of "degradation type" into interpretable channel response patterns, offering greater flexibility than direct one-hot routing via degradation labels.
- The MoE design is better suited than a single network for handling heterogeneous degradations, allowing each expert to focus on feature patterns associated with specific scattering models.

## Limitations & Future Work

- Only three weather degradations (haze, rain, snow) are considered; other common degradations such as low-light and overexposure are not addressed.
- BLIP-2 OPT 2.7B is computationally heavy, impacting inference speed and deployment feasibility.
- The number of degradation prototypes $K$ is a hyperparameter; adaptive determination is not discussed.
- Validation is performed only on synthetic degradation datasets; mixed real-world degradations (e.g., simultaneous haze and rain) are not examined.
- The VQA prompt design for VLM may affect prior quality, but the influence of different prompt choices is not analyzed in depth.

## Related Work & Insights

- **Text-IF** (Yi et al. 2024): CLIP-based prompt-guided fusion, but limited to low-light/overexposure conditions and dependent on ground-truth degradation labels.
- **MMAIF** (Cao et al. 2025): Diffusion model + Flan-T5 LLM, also reliant on fixed prompts.
- **SegFormer** (Xie et al. 2021): Backbone for the encoder architecture in this work.
- **Inspiration**: The role of VLMs as "degradation-aware sensors" could be extended to broader low-level vision tasks (a unified model for denoising, super-resolution, and deblurring).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The VLM → degradation prototype → MoE routing pipeline is novel and eliminates reliance on degradation annotations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset evaluation, comparison with multiple cascaded strategies, and degradation prototype visualization.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definition is clear and method motivation is well articulated.
- **Value**: ⭐⭐⭐⭐ One-stop degradation-aware fusion is a practically essential direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ControlFusion: A Controllable Image Fusion Framework with Language-Vision Degradation Prompts](../../NeurIPS2025/earth_science/controlfusion_a_controllable_image_fusion_framework_with_language-vision_degrada.md)
- [\[AAAI 2026\] RENEW: Risk- and Energy-Aware Navigation in Dynamic Waterways](renew_risk-_and_energy-aware_navigation_in_dynamic_waterways.md)
- [\[ICLR 2026\] OmniField: Conditioned Neural Fields for Robust Multimodal Spatiotemporal Learning](../../ICLR2026/earth_science/omnifield_conditioned_neural_fields_for_robust_multimodal_spatiotemporal_learnin.md)
- [\[ECCV 2024\] Semi-supervised Video Desnowing Network via Temporal Decoupling Experts and Distribution-Driven Contrastive Regularization](../../ECCV2024/earth_science/semi-supervised_video_desnowing_network_via_temporal_decoupling_experts_and_dist.md)
- [\[ICLR 2026\] GeoFAR: Geography-Informed Frequency-Aware Super-Resolution for Climate Data](../../ICLR2026/earth_science/geofar_geography-informed_frequency-aware_super-resolution_for_climate_data.md)

</div>

<!-- RELATED:END -->
