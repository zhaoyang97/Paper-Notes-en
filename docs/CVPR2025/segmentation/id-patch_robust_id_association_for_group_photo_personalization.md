---
title: >-
  [Paper Note] ID-Patch: Robust ID Association for Group Photo Personalization
description: >-
  [CVPR 2025][Segmentation][Identity Association] ID-Patch addresses the identity leakage problem in multi-identity image generation by feeding the same facial features into both an ID patch (for spatial control) and an ID embedding (for identity similarity preservation) simultaneously, comprehensively outperforming baseline models in facial similarity, ID-position association accuracy, and generation efficiency.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Identity Association"
  - "Multi-person Photo Generation"
  - "ID Leakage"
  - "Diffusion Models"
  - "Personalized Generation"
date: 2026-05-08
content_hash: e78490c788acd3f4
---

# ID-Patch: Robust ID Association for Group Photo Personalization

**Conference**: CVPR 2025  
**arXiv**: [2411.13632](https://arxiv.org/abs/2411.13632)  
**Code**: [https://github.com/bytedance/ID-Patch](https://github.com/bytedance/ID-Patch)  
**Area**: Image Segmentation  
**Keywords**: Identity Association, Multi-person Photo Generation, ID Leakage, Diffusion Models, Personalized Generation

## TL;DR
ID-Patch addresses the identity leakage problem in multi-identity image generation by feeding the same facial features into both an ID patch (for spatial control) and an ID embedding (for identity similarity preservation) simultaneously, comprehensively outperforming baseline models in facial similarity, ID-position association accuracy, and generation efficiency.

## Background & Motivation

**Background**: Personalized image generation is highly popular within diffusion-based applications. While single-person personalization (e.g., IP-Adapter, InstantID) is well-established, tasks become significantly more complex when multiple specified individuals need to be generated in a single scene (group photo personalization). Users expect both faithful reproduction of each individual's facial features and precise control over their spatial positions.

**Limitations of Prior Work**: The core dilemma of multi-identity generation is identity leakage—when multiple facial features are injected into the same diffusion model, their distinct attributes interfere with one another. This results in reduced facial similarity, misaligned identity positioning (e.g., identity A's face appearing in identity B's matching position), and severe visual artifacts. Existing methods like OMG rely on external segmentation models for post-processing, demanding long execution times (7 times slower than ID-Patch); InstantFamily is faster but exhibits a high occurrence of identity leakage.

**Key Challenge**: Multi-identity generation requires establishing a robust, one-to-one mapping between "preserving individual identity" and "controlling spatial location." However, conventional schemes either inject identities globally (lacking spatial control) or generate faces separately inside segmented regions and then stitch them together (causing visual discontinuities and high computational overhead).

**Goal**: To design a lightweight and robust identity-position association mechanism that simultaneous maintains high-fidelity facial similarity and precise layout control for each subject without requiring additional segmentation models.

**Key Insight**: The authors observe that two complementary representations can be extracted from the same facial feature source: one specialized for injection into spatial control maps to supervise position, and another integrated into text embeddings to maintain semantic-level facial similarities. Sourcing both representations from the same core facial feature enables an intrinsic identity-position binding.

**Core Idea**: Simultaneously generate an ID patch and an ID embedding from facial features; the patch is placed at a designated location in a condition image to construct spatial alignment, while the embedding is incorporated into the text encodings to enforce high-resemblance identity details.

## Method

### Overall Architecture
ID-Patch is built on the SDXL diffusion model. The input comprises multiple reference facial images and a layout condition image (e.g., an OpenPose skeleton map or a blank canvas). For each reference face, the system extracts identity features using a facial encoder and generates: (1) an ID patch—a small visual-level patch positioned within the target area of the condition image; (2) an ID embedding—a semantic-level face description merged into the text prompts. Finally, the diffusion model is steered by both the ID-patch-embedded condition image and the ID-embedding-integrated text condition to output a cohesive group portrait with accurate identity assignments.

### Key Designs

1. **ID Patch Generation & Placement**:

    - **Function**: Localizes precise spatial position signals for each identity directly on the condition image.
    - **Mechanism**: After extracting features from a face encoder (such as ArcFace), a small MLP network maps them into a fixed-scale ID patch image (e.g., $64 \times 64$ pixels). This patch preserves critical facial characteristics (skin tone, outline, etc.) and is placed block-wise on the target coordinates of the condition image. Guided by ControlNet, these patches enter the diffusion model. Since the patch's placement aligns with the target person’s position in the generated space, the model establishes a robust identity-to-location mapping.
    - **Design Motivation**: Compared to verbal descriptions (e.g., "the first person on the left"), a physical ID patch provides unambiguous identity and position signals directly in pixel space. It is cleaner and more elegant than mask-based region-wise generation, completely bypassing external segmentation steps.

2. **ID Embedding Integration**:

    - **Function**: Guarantees high resemblance of the generated face to the reference photo at the semantic level.
    - **Mechanism**: The output of the identical face encoder is mapped via another MLP into multiple token-level embeddings, which are concatenated into the CLIP text encoder sequence. At the cross-attention layer of the diffusion model, these ID tokens interact with the latent representations, consistently injecting identity nuances during denoising. Each identity's embedding is tagged with a location index to prevent cross-attention space crosstalk when multiple identities are processed simultaneously.
    - **Design Motivation**: Although the ID patch handles coarse spatial layouts, fine-grained textures (e.g., eye shape, lip curvature) are difficult to propagate via a small visual patch. The ID embedding compensates for these fine details within a high-dimensional semantic space, working in perfect synergy with the patch.

3. **Dual-Path Training Strategy**:

    - **Function**: Simultaneously optimizes spatial association accuracy and identity preservation capabilities.
    - **Mechanism**: Training is conducted in two stages. Stage one freezes the diffusion backbone and isolates training to the ID patch generator and the ID embedding mapper, learning foundational layout mappings via multi-person datasets. Stage two selectively unfreezes parts of the diffusion model for fine-tuning under a pose-invariant matching loss to maintain ID consistency across divergent views. Additionally, a pose-free option is supported—allowing generation without skeleton references, utilizing solely the coordinates of the ID patch to prompt layout structure.
    - **Design Motivation**: Jointly training all variables from scratch often results in conflicting gradients between ID patches and ID embeddings. Sequential, two-stage training encourages the network to master spatial associations first (stage one) before refining fine-grained identity compliance (stage two).

### Loss & Training
The loss function consists of the standard diffusion denoising loss, a facial ID consistency loss (relying on ArcFace cosine similarity), and an auxiliary face detection objective ensuring the generated faces remain clean and recognizable. Training is executed on large-scale, multi-person scene datasets.

## Key Experimental Results

### Main Results

| Method | Face Sim↑ | ID-Pos Acc↑ | Gen. Time↓ | FID↓ |
|------|-----------|-------------|-----------|------|
| ID-Patch | **0.72** | **94.3%** | **18s** | **42.1** |
| OMG + InstantID | 0.63 | 87.5% | 126s | 48.7 |
| InstantFamily | 0.58 | 82.1% | 22s | 51.3 |
| IP-Adapter (Multi-call) | 0.55 | 79.4% | 34s | 53.8 |

### Ablation Study

| Configuration | Face Sim | ID-Pos Acc | Description |
|------|----------|------------|------|
| Full ID-Patch | 0.72 | 94.3% | Full model with ID patch + ID embedding |
| w/o ID Patch | 0.68 | 71.2% | Without patch, using only embedding, spatial accuracy drops significantly |
| w/o ID Embedding | 0.54 | 92.8% | Without embedding, using only patch, similarity drops significantly |
| w/ Segmentation Post-processing | 0.70 | 91.6% | Assisted with segmentation model, but execution slows down |
| Pose-free Mode | 0.69 | 89.7% | Without using pose skeleton maps |

### Key Findings
- ID patches and ID embeddings both play indispensable and distinct roles: the patch anchors spatial coordinates (omitting it drops layout accuracy from 94.3% to 71.2%), while the embedding guarantees facial resemblance (removing it drops facial similarity from 0.72 to 0.54).
- ID-Patch runs 7 times faster than the segmentation-dependent OMG framework while achieving higher quality, showcasing the distinct superiority of end-to-end setups.
- The Pose-free mode exhibits highly robust performance despite omission of key skeleton coordinates, significantly widening realistic usage scenarios.
- Over complex scenes with 2 to 5 people, ID-Patch demonstrates considerably lower identity leakage rates compared to existing state-of-the-art benchmarks.

## Highlights & Insights
- The **ID patch concept** is refreshingly straightforward yet powerful—coupling identity and coordinate signals together inside a compact visual patch, avoiding complex segmentation masks or spatial cross-attention masks. This paradigm of "propagating identity through localized pixels" is highly extensible to other conditioned generation tasks.
- The **homogeneous dual-representation** design is incredibly elegant. Extracting both the spatial patch and semantic embedding from a unified encoder output guarantees that they reference the absolute same identity, avoiding alignment issues common in multi-stage networks.
- A 7x reduction in generation time directly translates to substantial commercial appeal, proving that a well-modeled pipeline (patch-based spatial mapping) can boost both generation quality and computational efficiency simultaneously.

## Limitations & Future Work
- In dense group settings exceeding 5 people, overlapping ID patches can occur, decreasing overall alignment accuracy.
- The fixed size constraint of the ID patches restricts performance when handling extreme perspective or scale variations (e.g., massive size scaling difference between up-close foreground and distant background characters).
- Testing was mainly limited to the SDXL baseline; potential adaptation to more modern state-of-the-art architectures (such as SD3 or Flux) demands further verification.
- One promising research direction is extending the ID patch into an "attribute patch" encapsulating more diverse details (cloth style, specific expressions) for fine-grained localized control.

## Related Work & Insights
- **vs OMG**: OMG pursues a "generate regional faces separately, mask, and fuse" strategy, while ID-Patch enables end-to-end synthesis. OMG offers certain flexibilities but is highly serial and slow; ID-Patch optimizes for rapid inference with superior resistance to identity leakage.
- **vs InstantFamily**: InstantFamily feeds global IDs through generalized attention modules, omitting explicit layout control. ID-Patch specifies layout using explicit pixel patches, realizing a 12% improvement in identity-position matching accuracy.
- Developed by ByteDance, this methodology is highly valuable and easily adaptable to their industrial-scale image creation suite.

## Rating
- Novelty: ⭐⭐⭐⭐ The ID patch concept is clear and effective, with a clever dual-stream (patch + embedding) execution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Features comprehensive benchmarks, clear ablation studies, and evaluation of practical pose-free configurations.
- Writing Quality: ⭐⭐⭐⭐ Excellent clarity on task definitions and methodologies, backed by strong qualitative results.
- Value: ⭐⭐⭐⭐ Successfully tackles an immediate practical issue in group-photo personalization, carrying high industrial translation potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GroupMamba: Efficient Group-Based Visual State Space Model](groupmamba_efficient_group-based_visual_state_space_model.md)
- [\[CVPR 2025\] Robust 3D Shape Reconstruction in Zero-Shot from a Single Image in the Wild](robust_3d_shape_reconstruction_in_zero-shot_from_a_single_image_in_the_wild.md)
- [\[CVPR 2025\] Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment](robust_audio-visual_segmentation_via_audio-guided_visual_convergent_alignment.md)
- [\[ICCV 2025\] Refer to Any Segmentation Mask Group With Vision-Language Prompts](../../ICCV2025/segmentation/refer_to_any_segmentation_mask_group_with_vision-language_prompts.md)
- [\[ICCV 2025\] CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation](../../ICCV2025/segmentation/corrclip_reconstructing_patch_correlations_in_clip_for_openv.md)

</div>

<!-- RELATED:END -->
