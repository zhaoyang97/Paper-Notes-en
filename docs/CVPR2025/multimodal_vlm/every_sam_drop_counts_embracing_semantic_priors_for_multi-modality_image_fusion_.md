---
title: >-
  [Paper Note] Every SAM Drop Counts: Embracing Semantic Priors for Multi-Modality Image Fusion and Beyond
description: >-
  [CVPR 2025][Multimodal VLM][Infrared-visible image fusion] Utilizes SAM's semantic priors to enhance infrared-visible image fusion via a persistent attention module, then transfers semantic knowledge to an ultra-lightweight student sub-network of only 0.136M parameters using bi-level optimization knowledge distillation, achieving SAM-free inference in 10.47ms while outperforming all dedicated fusion methods by over 3+ mIoU on segmentation tasks.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Infrared-visible image fusion"
  - "SAM semantic priors"
  - "Knowledge distillation"
  - "Bi-level optimization"
  - "Lightweight model"
date: 2026-05-08
content_hash: 0ce1f07e37d76d44
---

# Every SAM Drop Counts: Embracing Semantic Priors for Multi-Modality Image Fusion and Beyond

**Conference**: CVPR 2025  
**arXiv**: [2503.01210](https://arxiv.org/abs/2503.01210)  
**Code**: [https://github.com/RollingPlain/SAGE_IVIF](https://github.com/RollingPlain/SAGE_IVIF)  
**Area**: Multi-Modal VLM  
**Keywords**: Infrared-visible image fusion, SAM semantic priors, Knowledge distillation, Bi-level optimization, Lightweight model

## TL;DR
Utilizes SAM's semantic priors to enhance infrared-visible image fusion via a persistent attention module, then transfers semantic knowledge to an ultra-lightweight student sub-network of only 0.136M parameters using bi-level optimization knowledge distillation, achieving SAM-free inference in 10.47ms while outperforming all dedicated fusion methods by over 3+ mIoU on segmentation tasks.

## Background & Motivation

**Background**: Infrared-visible image fusion (IVIF) integrates complementary information from both modalities into a single image to serve downstream tasks like object detection and semantic segmentation. Existing fusion methods primarily focus on visual quality (contrast, sharpness), which lacks adaptation for downstream tasks.

**Limitations of Prior Work**: (1) Fusion images optimized solely for visual quality are not necessarily optimal for segmentation tasks—some visually appealing fusion results might actually interfere with the segmenter. (2) Integrating large-scale pre-trained models (e.g., SAM) as semantic priors can improve task performance, but the massive computational overhead of SAM during inference is unacceptable.

**Key Challenge**: Semantic-level guidance is required to improve the downstream task friendliness of fusion images, but the semantic model (SAM, 632M parameters) is too computationally heavy to be used during inference.

**Goal**: Leverage SAM's semantic priors to enhance fusion quality and downstream task performance, while eliminating reliance on SAM during inference via knowledge distillation.

**Key Insight**: SAM is utilized only during training—generating semantic patches to guide the main network (teacher), whose knowledge is then distilled into a lightweight sub-network (student), so that only the student network is used during inference.

**Core Idea**: Employs SAM as a "semantic trainer during training," compressing its knowledge into an ultra-lightweight 0.136M fusion network via bi-level optimization and triplet distillation loss.

## Method

### Overall Architecture
During training: SAM generates semantic patches from infrared/visible images $\rightarrow$ the main network (teacher) fuses source images + semantic patches via the SPA module $\rightarrow$ generates a reference fusion image $I_{ref}$. Simultaneously, the lightweight sub-network (student) generates a fusion image $I_f$ using only the source images. Bi-level optimization alternately updates the teacher and student, with a triplet loss aligning the two models. During inference: only the student network is used.

### Key Designs

1. **Semantic Persistent Attention Module (SPA)**:

    - **Function**: Fuses SAM's semantic patches with the complete information of the source images.
    - **Mechanism**: SPA contains a Persistent Repository (PR) that stores the latent representation $Z$ of the source image features and key-value pairs $(K_{src}, V_{src})$. The encoded semantic patches from SAM act as queries to perform cross-attention over the PR's key-value pairs, complementing the incomplete semantic patches into fused features containing full-scene context.
    - **Design Motivation**: SAM's semantic patches only cover parts of the scene (a single segmentation region), while PR provides endpoints/anchors for the entire scene, preventing local semantic information from losing global context.

2. **Triplet Distillation Loss**:

    - **Function**: Aligns teacher and student outputs from three dimensions.
    - **Mechanism**: $\mathcal{L}_{fea}$ (feature alignment): cosine similarity between correspond-scale features of the teacher and student; $\mathcal{L}_{context}$ (context): Sobel gradient L1 + MSE to maintain structural and illumination consistency; $\mathcal{L}_{cs}$ (contrastive semantic): constructs positive and negative sample pairs using SAM encoder features and binary masks to pull close the semantic representations of the fusion map and reference map.
    - **Design Motivation**: Ablation shows that removing $\mathcal{L}_{fea}$ drops FMB mIoU by 9.6 percentage points (making it the most critical); all three losses address different aspects and are mutually indispensable.

3. **Bi-Level Optimization Distillation (DARTS-like)**:

    - **Function**: Alternately updates teacher and student to achieve mutual adaptation.
    - **Mechanism**: Instead of training the teacher first and then distilling to the student (offline), a DARTS-like alternating scheme is adopted—the teacher network has an additional segmentation loss $\mathcal{L}_{seg}$, while the student network follows the teacher via triplet loss. The bi-directional gradient allow the teacher to adjust based on student feedback.
    - **Design Motivation**: Offline distillation yields an FMB mIoU of only 50.4, whereas bi-level optimization reaches 61.2 (+10.8), proving mutual adaptation is crucial.

### Loss & Training
Teacher: $\mathcal{L}_{seg}$ (CE segmentation loss of X-Decoder). Distillation: $\mathcal{L} = \mathcal{L}_{fea} + \mathcal{L}_{context} + \mathcal{L}_{cs}$. DARTS-like alternating optimization.

## Key Experimental Results

### Main Results

| Method | FMB mIoU | MFNet mIoU | Inference Time | Parameters |
|------|---------|----------|---------|--------|
| DDFM | 58.2 | - | 280K ms | 552.7M |
| SegMiF | 57.3 | 52.4 | - | - |
| EMMA | 55.8 | - | 25.73ms | 1.516M |
| **SAGE** | **61.2** | **54.0** | **10.47ms** | **0.136M** |

### Ablation Study

| Configuration | FMB mIoU | Description |
|------|---------|------|
| Without SAM | 56.5 | -4.7 |
| Without $\mathcal{L}_{fea}$ | 51.6 | -9.6, most critical |
| Without $\mathcal{L}_{cs}$ | 54.3 | -6.9 |
| Offline distillation | 50.4 | -10.8, bi-level optimization is crucial |
| **Full SAGE** | **61.2** | Coordination of all components |

### Key Findings
- **Value of SAM Semantic Priors brings +4.7 mIoU**, proving the importance of high-level semantic guidance for fusion quality and downstream tasks.
- **Extreme Efficiency**: 0.136M parameters and 10.47ms inference time, 26,700$\times$ faster and 4,000$\times$ smaller than DDFM, and 2.5$\times$ faster and 11$\times$ smaller than EMMA.
- **Bi-level Optimization >> Offline Distillation** (+10.8 mIoU); alternating training enables co-evolution of teacher and student.

## Highlights & Insights
- **The "Large model for training, small model for inference" paradigm** is highly practical—efficiently compressing SAM's semantic knowledge into a 0.136M network. This strategy can be generalized to any scenario where large model capabilities are required but inference budget is constrained.
- **The design of the Persistent Repository (PR)** solves the problem of local semantic patches lacking global context, effectively serving as a cross-attention-based memory augmentation mechanism.

## Limitations & Future Work
- The quality of semantic patches generated by SAM directly affects distillation performance; low-quality segmentation might introduce noise.
- Only verified on infrared-visible image fusion; generalization to other multi-modality fusion tasks (e.g., CT-MRI, multispectral) remains unconfirmed.
- The segmentation loss relies on X-Decoder; the performance when utilizing other open-vocabulary segmentation models remains unknown.

## Related Work & Insights
- **vs SegMiF**: Both optimize fusion and segmentation concurrently, but SegMiF integrates the segmenter directly into the inference pipeline, increasing overhead. SAGE incorporates segmentation guidance during training via distillation and avoids it entirely during inference.
- **vs EMMA**: EMMA uses a larger model (1.516M) but achieves worse performance (55.8 vs 61.2 mIoU), indicating code/model size is not the key; the manner in which semantic priors are introduced is more critical.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of SAM priors and bi-level distillation is novel and practical, with a well-designed SPA module.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on multiple datasets, featuring detailed component-wise ablation (SAM, SPA, distillation losses, and distillation schemes).
- Writing Quality: ⭐⭐⭐⭐ The methodology workflow is clear, and the efficiency data is highly convincing.
- Value: ⭐⭐⭐⭐ Holds direct engineering value for multi-modal fusion, with generalizable distillation concepts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ASAP: Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding](asap_advancing_semantic_alignment_promotes_multi-modal_manipulation_de.md)
- [\[CVPR 2025\] ASAP: Advancing Semantic Alignment for Multi-Modal Manipulation Detection](asap_advancing_semantic_alignment_promotes_multi-modal_manipulation_detecting_an.md)
- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](../../AAAI2026/multimodal_vlm/harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [\[CVPR 2025\] Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices](multi-layer_visual_feature_fusion_in_multimodal_llms_methods_analysis_and_best_p.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](../../CVPR2026/multimodal_vlm/multi-modal_image_fusion_via_intervention-stable_feature_learning.md)

</div>

<!-- RELATED:END -->
