---
title: >-
  [Paper Note] SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation
description: >-
  [CVPR 2026][Medical Imaging][few-shot segmentation] This paper proposes SD-FSMIS, a framework that adapts pretrained Stable Diffusion for few-shot medical image segmentation (FSMIS). Through a Support-Query Interaction m…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "few-shot segmentation"
  - "stable diffusion"
  - "cross-domain"
  - "foundation model"
date: 2026-05-08
content_hash: 57bed5e2fca4e9c7
---

# SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation

**Conference**: CVPR 2026
**arXiv**: [2604.03134](https://arxiv.org/abs/2604.03134)
**Code**: N/A
**Area**: Medical Image Segmentation
**Keywords**: few-shot segmentation, medical imaging, stable diffusion, cross-domain, foundation model

## TL;DR

This paper proposes SD-FSMIS, a framework that adapts pretrained Stable Diffusion for few-shot medical image segmentation (FSMIS). Through a Support-Query Interaction module and a Visual-to-Text Conditioning Transformer, the framework achieves efficient adaptation, with particularly strong performance in cross-domain scenarios.

## Background & Motivation

Few-shot medical image segmentation (FSMIS) aims to segment novel categories from only a handful of annotated samples, addressing the core challenges of data scarcity and domain shift in medical imaging. Existing methods primarily focus on designing more sophisticated matching networks, such as prototype networks and attention mechanisms; however, these architectures trained from scratch exhibit fragile performance in cross-domain settings.

The authors propose a paradigm shift: rather than designing increasingly complex task-specific architectures, they leverage the rich visual priors embedded in large-scale pretrained generative models such as Stable Diffusion. Diffusion models learn rich texture, shape, and contextual representations from massive datasets (e.g., LAION-5B), offering substantial potential for dense prediction tasks — potential that remains largely unexplored in FSMIS.

**Core Problem**: How to efficiently and directly adapt the general visual priors of Stable Diffusion to the FSMIS task?

## Method

### Overall Architecture

SD-FSMIS repurposes the conditional generation architecture of Stable Diffusion to handle support–query interactions in latent space. Both the support set and query set are first encoded into the latent space via a frozen VAE, and the U-Net generates the query segmentation mask conditioned on text embeddings.

### Key Designs

1. **Support-Query Interaction Module (SQI)**: Adapts SD's U-Net self-attention layers for few-shot learning with minimal modifications. An additional cross-attention layer is inserted after the standard self-attention, allowing query features to attend to support features (as Keys and Values), followed by the original text cross-attention. A Query Enhancement (QE) strategy is also incorporated, leveraging support prototype features and cosine similarity to enrich query representations.

2. **Visual-to-Text Conditioning Transformer (VTCT)**: Converts visual cues from the support set into "text-like" embeddings to condition the diffusion model. A frozen image encoder extracts support image features; masked average pooling yields class prototypes, which are then projected into the text embedding space via a learnable MLP. This enables the model to be guided in the "language" that SD understands.

3. **Single-Step Inference**: During inference, the framework does not require an iterative diffusion process. Instead, it generates the query mask latent representation in a single step, which is decoded back to pixel space via the VAE decoder; the final segmentation mask is obtained by averaging across the three channels.

### Loss & Training

The model is trained using MSE loss between the predicted and ground-truth query mask latent representations. An episode-based meta-learning strategy is adopted under a 1-way 1-shot setting. Pseudo-labels generated via supervoxel clustering serve as training annotations, requiring no explicit manual labeling. VAE weights are frozen, and only the newly introduced parameters are trained.

## Key Experimental Results

### Main Results

| Dataset | Metric (Dice %) | Ours | Prev. SOTA (DIFD) | Gain |
|---------|----------------|------|-------------------|------|
| Abd-MRI Setting 1 | Mean Dice | 83.16 | 84.12 | Competitive |
| Abd-CT Setting 1 | Mean Dice | 83.66 | 80.19 | +3.47 |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| w/o SQI | Dice drops | Support-query interaction is critical for adaptation |
| w/o VTCT (null text) | Dice drops | Visual conditioning is more informative than empty text embeddings |
| w/o QE | Dice drops | Query enhancement provides beneficial prototype matching signals |

### Key Findings

- Achieves competitive results under standard FSMIS settings.
- Substantially outperforms SOTA methods in more challenging cross-domain scenarios, demonstrating strong generalization capability.
- Validates the significant potential of visual priors from large-scale generative models for data-efficient medical segmentation.

## Highlights & Insights

- **Paradigm innovation**: Shifting from task-specific network design to adapting pretrained foundation models represents an important transition in the FSMIS field.
- The framework design is minimalist yet effective, achieving FSMIS adaptation through only minimal modifications to SD.
- The VTCT module's approach of translating visual cues into the "language" that SD understands is elegant and well-motivated.
- The outstanding cross-domain generalization ability confirms the practical value of SD's general visual priors in the medical domain.

## Limitations & Future Work

- Relies on channel replication to adapt SD for grayscale medical images, which may not be an elegant solution.
- Validation is currently limited to abdominal MRI/CT data; broader evaluation across more organs and modalities is needed.
- Single-step inference, while efficient, may forego the refinement benefits of iterative diffusion.

## Related Work & Insights

- Similar in spirit to DiffewS in leveraging diffusion models for few-shot segmentation, but DiffewS targets natural images whereas SD-FSMIS is specifically adapted to the medical domain.
- Offers valuable insights for other dense prediction tasks requiring cross-domain generalization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic exploration of Stable Diffusion for FSMIS.
- **Technical Depth**: ⭐⭐⭐⭐ — SQI and VTCT are well-motivated and soundly designed.
- **Experimental Thoroughness**: ⭐⭐⭐ — Dataset coverage could be broader.
- **Value**: ⭐⭐⭐⭐ — Cross-domain generalization capability has practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)
- [\[CVPR 2026\] MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification](muse_harnessing_precise_and_diverse_semantics_for_few-shot_whole_slide_image_cla.md)
- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)

</div>

<!-- RELATED:END -->
