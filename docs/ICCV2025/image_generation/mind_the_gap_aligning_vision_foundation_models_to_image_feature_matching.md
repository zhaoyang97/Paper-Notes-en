---
title: >-
  [Paper Note] Mind the Gap: Aligning Vision Foundation Models to Image Feature Matching
description: >-
  [ICCV 2025][Image Generation][Feature Matching] This paper identifies an "alignment gap" in vision foundation models (e.g., DINOv2) for image feature matching: contrastive learning-based models discard instance-level details and lack cross-image interaction mechanisms, causing failures in multi-instance matching scenarios. To address this, the authors propose the IMD framework, which employs diffusion models as feature extractors to preserve instance-level details…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Feature Matching"
  - "Vision Foundation Models"
  - "Diffusion Models"
  - "Cross-Image Interaction"
  - "Multi-Instance Matching"
date: 2026-05-08
content_hash: 31ae7999ac857f2d
---

# Mind the Gap: Aligning Vision Foundation Models to Image Feature Matching

**Conference**: ICCV 2025
**arXiv**: [2507.10318](https://arxiv.org/abs/2507.10318)  
**Code**: N/A  
**Area**: Image Generation
**Keywords**: Feature Matching, Vision Foundation Models, Diffusion Models, Cross-Image Interaction, Multi-Instance Matching

## TL;DR

This paper identifies an "alignment gap" in vision foundation models (e.g., DINOv2) for image feature matching: contrastive learning-based models discard instance-level details and lack cross-image interaction mechanisms, causing failures in multi-instance matching scenarios. To address this, the authors propose the IMD framework, which employs diffusion models as feature extractors to preserve instance-level details, and designs a Cross-Image Interaction Prompt Module (CIPM) for bidirectional information exchange. IMD achieves state-of-the-art performance on standard benchmarks and on the newly introduced multi-instance benchmark IMIM, with a 12% improvement in multi-instance scenarios.

## Background & Motivation

**Background**: Image feature matching is fundamental to 3D reconstruction, visual localization, and related tasks. Leveraging pre-trained vision foundation models (DINOv2, CLIP) to improve matching generalizability has become the dominant paradigm, with methods such as RoMa, OmniGlue, and CasMTR achieving notable advances.

**Limitations of Prior Work**:
- **Alignment Gap**: Foundation models are designed for single-image understanding, whereas matching requires cross-image understanding; contrastive learning objectives (e.g., DINOv2) emphasize global semantic similarity at the expense of instance-level details.
- **Multi-Instance Matching Failure**: When multiple instances of the same category appear in an image (e.g., two cars), global-semantics-based methods cannot disambiguate which instance is the correct correspondence target.
- **Lack of Interaction Mechanism**: Existing methods extract features from each image independently, with inter-image relationships established only in subsequent attention modules, meaning the feature extraction stage lacks cross-image association.

**Key Challenge**: The powerful generalization capability of foundation models conflicts with a fundamental mismatch between their training objectives (single-image understanding / global semantics) and the requirements of matching tasks (cross-image understanding / instance-level details).

**Goal**: (1) What kind of foundation model is better suited for feature matching? (2) How should a cross-image interaction mechanism be designed to convert the single-image understanding capability of foundation models into cross-image understanding?

**Key Insight**: Generative diffusion models' internal representations naturally encode the unique appearance and structural information of each object and instance (as demonstrated by DIFT), and the conditioning mechanism (prompt) of diffusion models provides a natural channel for cross-image interaction.

**Core Idea**: Replace contrastive learning models with diffusion models as the feature extractor to preserve instance-level details, and leverage the prompt mechanism of diffusion models to design a cross-image interaction module (CIPM) that generates personalized prompts to guide feature extraction.

## Method

### Overall Architecture

IMD adopts a standard two-stage matching pipeline (coarse matching → fine matching). Given an image pair $I_A$ and $I_B$:
1. A CLIP image encoder extracts image features, and CIPM generates personalized prompts $P_A$ and $P_B$.
2. The images and prompts are fed into a frozen SD UNet to extract 1/8-resolution coarse features $C_A$ and $C_B$.
3. Self/cross-attention enhances the discriminability of coarse features, and coarse matches $\mathcal{M}_c$ are computed.
4. A dedicated ConvNet encoder extracts 1/2-resolution fine features, which are fused with the transformed coarse features.
5. Local patches are cropped around coarse matches and refined to sub-pixel level matches $\mathcal{M}_f$.

### Key Designs

1. **Diffusion Model as Feature Extractor**:

    - **Function**: A frozen SD 2-1 UNet directly processes clean images (without adding/removing noise) and extracts features from the upsampling blocks.
    - **Mechanism**: The input image $I_0$ is encoded by a VAE into $z_0$; the timestep is set to $t=0$ (no noise added), and the output of upsampling block index=2 is directly extracted as a 640-dimensional feature map.
    - **vs. Contrastive Learning Models**: As shown by DIFT, diffusion model internal feature maps naturally encode the unique visual representations of each object and instance, whereas contrastive learning models compress instance-level details to maximize global semantic similarity.
    - **Design Motivation**: In multi-instance scenarios (e.g., two cars of the same color), global semantics alone cannot distinguish instances; preserving the unique appearance and structural information of each instance is necessary.

2. **Cross-Image Interaction Prompt Module (CIPM)**:

    - **Function**: Generates personalized prompts for each image that incorporate information from the other image, guiding the diffusion model's feature extraction.
    - **Mechanism**: A CLIP image encoder extracts $\mathcal{F}^A$ and $\mathcal{F}^B$; three $1\times1$ convolutions produce Q/K/V; cross-image cross-attention is performed, and an MLP converts the output into prompt embeddings:
    $P_A = \text{MLP} \circ \text{Softmax}(\phi_{Q,I_A} \phi_{K,I_B} / \sqrt{d_k}) \phi_{V,I_B}$
    $P_B = \text{MLP} \circ \text{Softmax}(\phi_{Q,I_B} \phi_{K,I_A} / \sqrt{d_k}) \phi_{V,I_A}$
      Subsequently, $C_A = \text{UNet}(I_A, t, P_A)$ and $C_B = \text{UNet}(I_B, t, P_B)$.
    - **vs. Shared Prompt (SD4Match)**: SD4Match uses a shared prompt constructed by concatenating features from both images, which introduces association at the cost of discriminability. CIPM generates an independent prompt for each image, maintaining discriminability while adding cross-image association.
    - **vs. No Prompt / Empty String**: Ablation experiments demonstrate that cross-image prompts improve IMIM performance by 4.7% over empty string prompts.

3. **Fine Feature Encoder**:

    - **Function**: Produces 1/2-resolution fine features for sub-pixel precise matching.
    - **Mechanism**: Since SD features are only at 1/8 resolution, a ResNet extracts full-resolution features that are fused with upsampled coarse features, without requiring an additional Transformer transformation network.
    - **Design Motivation**: FPN-based approaches are unsuitable for an SD backbone that only provides 1/8 coarse features, necessitating a dedicated fine feature encoder.

### Loss & Training

- Total loss: $\mathcal{L} = \mathcal{L}_c + \alpha \mathcal{L}_{f1} + \beta \mathcal{L}_{f2}$
- $\mathcal{L}_c$: focal loss for coarse matching (following LoFTR)
- $\mathcal{L}_{f1}$: log-likelihood loss for fine matching
- $\mathcal{L}_{f2}$: L2 loss for sub-pixel matching
- $\alpha=1.0$, $\beta=0.25$
- Trained for 30 epochs on the MegaDepth dataset using 8×A100 GPUs
- AdamW optimizer, lr=$4\times10^{-3}$
- UNet is frozen; only CIPM, attention modules, and the fine encoder are trained

## Key Experimental Results

### Main Results

**Multi-Instance Matching + Pose Estimation (Tab. 1)**:

| Method | MegaDepth AUC@5° | AUC@10° | AUC@20° | ScanNet AUC@5° | AUC@10° | IMIM(%) |
|------|----------|---------|---------|----------|---------|---------|
| DINOv2 (zero-shot) | 32.5 | 50.8 | 65.3 | 13.0 | 28.5 | 57.9 |
| DIFT (zero-shot) | 38.4 | 55.9 | 70.5 | 15.7 | 32.0 | 61.2 |
| OmniGlue (sparse) | 47.4 | 65.0 | 77.8 | 31.3 | 50.2 | 77.6 |
| LoFTR (semi-dense) | 52.8 | 69.2 | 81.2 | 16.9 | 33.6 | 68.9 |
| CasMTR | 59.1 | 74.3 | 84.8 | 22.6 | 40.7 | 79.2 |
| PRISM | 60.0 | 74.9 | 85.1 | 23.9 | 41.8 | - |
| **IMD (Ours)** | **61.2** | **76.0** | **85.8** | **29.8** | **48.3** | **88.7** |

AUC@5° on ScanNet improves by 24.6% over PRISM; IMIM multi-instance matching improves by ~12% relative to CasMTR.

**Homography Estimation on HPatches (Tab. 2)**:

| Method | AUC@3px | AUC@5px | AUC@10px |
|------|---------|---------|----------|
| CasMTR | 71.4 | 80.2 | 87.9 |
| PRISM | 71.9 | 80.4 | 88.3 |
| **IMD (Ours)** | **73.9** | **82.0** | **88.9** |

### Ablation Study

**Key Component Ablation (Tab. 4)**:

| Configuration | MegaDepth AUC@5° | AUC@10° | AUC@20° | IMIM(%) |
|------|----------|---------|---------|---------|
| Replace SD with Swin(B) | 57.5 | 73.2 | 83.6 | 74.0 |
| Replace SD with DINOv2(B) | 57.8 | 73.5 | 83.7 | 75.5 |
| Empty string prompt | 58.4 | 73.8 | 84.1 | 84.0 |
| Individual prompt (no interaction) | 59.6 | 74.3 | 84.5 | 85.2 |
| w/o cross-attention (shared prompt) | 60.7 | 75.0 | 85.1 | 87.4 |
| Timestep T=100 | 60.9 | 75.7 | 85.8 | 88.1 |
| **Ours Full (CIPM + T=0)** | **61.2** | **76.0** | **85.8** | **88.7** |

### Key Findings

- **Diffusion Model vs. Contrastive Learning Model**: Replacing SD with DINOv2 drops IMIM from 88.7% to 75.5% (−13.2%), confirming the severe deficiency of contrastive learning models in multi-instance scenarios.
- **Incremental Gains from Interactive Prompts**: IMIM progressively improves from 84.0% to 88.7% as the prompt design advances from empty string → individual prompt → shared prompt → CIPM, demonstrating the dual necessity of cross-image interaction and personalized prompts.
- **Timestep $t=0$ is Optimal**: Adding noise ($t=100$) degrades performance, as matching requires precise low-level details rather than semantic abstraction.
- **Model Capacity is Not the Key Factor**: SD-tiny (75% fewer parameters) still performs favorably on IMIM, indicating that improvements stem from the representational properties of diffusion models rather than model scale.
- **Cross-Dataset Generalization**: Trained solely on MegaDepth (outdoor), the model outperforms the second-best method by 24.6% on ScanNet (indoor), demonstrating strong generalization.

## Highlights & Insights

- **Precise Diagnosis of the "Alignment Gap"** — The paper explicitly identifies two misalignment dimensions between foundation models and matching tasks (representation and interaction) and addresses each in a targeted manner. This "diagnose-then-treat" research methodology is instructive.
- **Leveraging the Diffusion Model's Prompt Mechanism for Cross-Image Interaction** — The conditioning mechanism of diffusion models (cross-attention injection of prompts) was originally designed for text-guided generation; here, features from the other image are cleverly converted into prompts to achieve cross-image information injection. This paradigm of "repurposing existing mechanisms for new functions" is transferable to other tasks requiring multi-view interaction.
- **Introduction of the IMIM Benchmark** — Most existing benchmarks (MegaDepth, ScanNet) are dominated by single-instance scenarios, making it impossible to rigorously evaluate multi-instance discrimination. IMIM fills this evaluation gap.

## Limitations & Future Work

- Using a diffusion model as the backbone incurs higher inference costs than DINOv2 or Swin, which may be limiting for real-time applications.
- CIPM requires cross-attention over each image pair, and the $O(n^2)$ complexity may limit applicability to high-resolution inputs.
- Only the semi-dense matching paradigm is evaluated; extension to dense matching (e.g., RoMa) is not explored.
- The IMIM benchmark contains only 100 image pairs, limiting its statistical reliability.
- The effect of more recent diffusion model backbones (e.g., SD-XL, Flux) is not investigated.

## Related Work & Insights

- **vs. RoMa**: RoMa uses DINOv2 as the backbone for dense matching and performs well on standard benchmarks, but is constrained by the limitations of contrastive learning representations in multi-instance scenarios.
- **vs. SD4Match**: SD4Match is the first to explore diffusion models for semantic matching, but its shared prompt (concatenating features from both images) introduces association at the cost of discriminability; IMD's personalized CIPM simultaneously preserves both association and discriminability.
- **vs. DIFT/SD+DINO**: DIFT demonstrates that diffusion model features can serve as semantic correspondences, but only in a zero-shot setting without task-specific training; IMD builds on DIFT's insights to design a complete matching framework.
- **vs. CroCo/CroCov2**: CroCo learns associative features via cross-view completion pre-training, but requires dedicated pre-training; IMD directly leverages the prompt mechanism of existing diffusion models for interaction without additional pre-training.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The identification of the alignment gap and the CIPM design are novel, though using diffusion models for feature extraction is not an entirely new idea.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation on five benchmarks, detailed ablations, model capacity comparisons, and a new benchmark.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear; the two-challenge organizational structure is logically coherent.
- **Value**: ⭐⭐⭐⭐⭐ Achieves state-of-the-art on multiple benchmarks; the findings on multi-instance matching offer meaningful insights to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities](rethinking_the_embodied_gap_in_vision-and-language_navigation_a_holistic_study_o.md)
- [\[ICLR 2026\] Aligning Visual Foundation Encoders to Tokenizers for Diffusion Models](../../ICLR2026/image_generation/aligning_visual_foundation_encoders_to_tokenizers_for_diffusion_models.md)
- [\[ICCV 2025\] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)
- [\[CVPR 2026\] VFM-VAE: Vision Foundation Models Can Be Good Tokenizers for Latent Diffusion Models](../../CVPR2026/image_generation/vfm-vae_vision_foundation_models_can_be_good_tokenizers_for_latent_diffusion_mod.md)
- [\[CVPR 2026\] Probing and Bridging Geometry–Interaction Cues for Affordance Reasoning in Vision Foundation Models](../../CVPR2026/image_generation/probing_and_bridging_geometry-interaction_cues_for_affordance_reasoning_in_visio.md)

</div>

<!-- RELATED:END -->
