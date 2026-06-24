---
title: >-
  [Paper Note] FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning
description: >-
  [CVPR 2025][Human Understanding][Face Security] FSFM proposes the first self-supervised pre-training framework specialized for face security tasks. By employing the CRFR-P facial masking strategy coupled with a dual-task collaborative learning of MIM/ID, it learns "3C" representations of real faces (intra-region consistency, inter-region coherence, and local-to-global correspondence). It surpasses task-specific SOTA methods across three major tasks: Deepfake Detection…
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Face Security"
  - "Self-Supervised Pre-training"
  - "Masked Image Modeling"
  - "Instance Discrimination"
  - "Foundation Model"
date: 2026-05-08
content_hash: e940d784bc196c7e
---

# FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning

**Conference**: CVPR 2025  
**arXiv**: [2412.12032](https://arxiv.org/abs/2412.12032)  
**Code**: [https://fsfm-3c.github.io](https://fsfm-3c.github.io)  
**Area**: Human Understanding  
**Keywords**: Face Security, Self-Supervised Pre-training, Masked Image Modeling, Instance Discrimination, Foundation Model

## TL;DR
FSFM proposes the first self-supervised pre-training framework specialized for face security tasks. By employing the CRFR-P facial masking strategy coupled with a dual-task collaborative learning of MIM/ID, it learns "3C" representations of real faces (intra-region consistency, inter-region coherence, and local-to-global correspondence). It surpasses task-specific SOTA methods across three major tasks: Deepfake Detection, Face Anti-Spoofing, and Diffusion Forgery Detection.

## Background & Motivation

**Background**: The field of face security faces three threat tasks: Deepfake Detection (DfD), Face Anti-Spoofing (FAS), and Diffusion Forgery Detection (DiFF). The vast majority of existing methods adopt fully supervised learning, utilizing various backbones (Xception, EfficientNet, ViT) initialized with ImageNet pre-trained weights, with independent specialized methods for each task.

**Limitations of Prior Work**: (1) Fully supervised learning requires extensive annotations or generative augmentation, which is costly and poorly scalable; (2) ImageNet pre-training provides natural image representations that lack the face-specific "authenticity" representation, limiting the models' generalization ability on face security tasks; (3) DfD focuses on digital manipulation artifacts while FAS focuses on physical spoofing cues; the two are treated as incompatible, independent tasks, lacking a unified foundational representation.

**Key Challenge**: Existing facial self-supervised methods (such as FaRL, MARLIN, etc.) primarily learn salient facial features required for facial analysis tasks (expression recognition, attribute analysis). However, forged and spoofed faces are highly similar to authentic faces in these salient features—they neglect the "authenticity" representation required to distinguish genuine faces from fake ones.

**Goal**: How to learn a robust and transferable foundational representation from a large volume of unlabeled real face images that can simultaneously enhance cross-dataset deepfake detection, cross-domain anti-spoofing, and unseen diffusion forgery detection?

**Key Insight**: Leverage the complementary advantages of two self-supervised paradigms—MIM (masked image modeling) which excels at local pixel-level context awareness, and ID (instance discrimination) which excels at global semantic alignment—and design a face-structure-aware masking strategy to inject facial priors.

**Core Idea**: Use the CRFR-P facial masking strategy to drive MIM in capturing intra- and inter-region relationships, while coupling ID self-distillation to establish local-to-global correspondence. These three learning objectives (3C) jointly pre-train a general face security foundation model.

## Method

### Overall Architecture
The input consists of unlabeled real face images. The CRFR-P masking strategy generates two masks (a complete mask M and a facial region mask $M_{fr}$). The MIM network (online encoder + online decoder) reconstructs the masked regions from the visible patches, while the ID network (online branch + target branch updated via EMA) aligns the masked local features with the global features of the complete image. After pre-training, the online encoder (a vanilla ViT) serves as the general foundation model, which can be applied to downstream face security tasks with simple fine-tuning.

### Key Designs

1. **CRFR-P Facial Masking Strategy (Covering Random Facial Region + Proportional)**:

    - **Function**: Generates facial masks that simultaneously promote intra-region consistency and inter-region coherence.
    - **Mechanism**: First, a face parser is used to segment the face into 8 semantic regions (eyes, nose, mouth, eyebrows, facial boundary, hair, skin, background). Then: (1) one randomly selected facial region is completely covered (e.g., the entire nose), forcing the model to infer the masked region from other regions, thereby learning inter-region coherence; (2) the remaining regions are uniformly masked according to a proportion, ensuring each region retains some visible patches, which allows the model to learn the internal texture consistency of each region.
    - **Design Motivation**: Random masking tends to completely cover small yet information-dense areas (like eyes) while potentially allowing "shortcuts" through remaining patches in the same region. CRFR-P forces the complete coverage of a region to eliminate shortcuts, while proportional sampling ensures all regions participate in learning.

2. **Dual-Task MIM Reconstruction Objective**:

    - **Function**: Supervises mask reconstruction on two levels: global pixel reconstruction + specific facial region reconstruction.
    - **Mechanism**: The online encoder only processes visible patches, while the online decoder completes all patches. The loss is $\mathcal{L}_{rec} = \mathcal{L}_{rec}^m + \lambda_{fr}\mathcal{L}_{rec}^{fr}$, where $\mathcal{L}_{rec}^m$ is the MSE reconstruction loss for all masked patches, and $\mathcal{L}_{rec}^{fr}$ is the MSE loss targeted specifically at the fully covered facial region.
    - **Design Motivation**: The additional facial-region loss $\mathcal{L}_{rec}^{fr}$ explicitly reinforces the learning of inter-region coherence—the completely covered region can only be inferred from information in other regions, preventing the shortcut of copying from remaining patches in the same region.

3. **ID Self-Distillation Network (Local-to-Global Correspondence)**:

    - **Function**: Establishes alignment between masked local features and the global semantics of the complete image.
    - **Mechanism**: The online branch receives patches processed by the CRFR-P mask, while the target branch receives complete patches (unmasked), with both branches sharing the encoder architecture. Symmetric representation decoders $D_o^r$ and $D_t^r$ map the encoder outputs into a decoupled space, and then minimize the negative cosine similarity $\mathcal{L}_{sim}$ via projection and prediction heads. The target branch parameters are updated via EMA, employing a stop-gradient to backpropagate gradients only through the online branch.
    - **Design Motivation**: Selecting complete patches rather than another masked view as the target branch input ensures semantic integrity. Introducing representation decoders decouples the feature spaces for pixel reconstruction and semantic alignment, keeping low-level MIM tasks from interfering with high-level ID semantic learning.

### Loss & Training
- Total loss: $\mathcal{L} = \mathcal{L}_{rec}^m + \lambda_{fr}\mathcal{L}_{rec}^{fr} + \lambda_{cl}\mathcal{L}_{sim}$
- Pre-train a vanilla ViT-B on VGGFace2 (3.1 million real face images).
- After pre-training, only the online encoder $E_o$ is kept and simply fine-tuned on various downstream task datasets.

## Key Experimental Results

### Main Results

**Cross-dataset Deepfake Detection (Trained on FF++, cross-domain testing Video-level AUC %)**:

| Method | Pre-training | CDFV2 | DFDC | DFDCP | WDF | Average |
|------|--------|-------|------|-------|-----|------|
| **FSFM ViT-B** | SSL(VF2) | **91.44** | **83.47** | **89.71** | **86.96** | **87.90** |
| ViT-B | Sup(IN) | 86.24 | 74.48 | 82.11 | 81.20 | 81.01 |
| MAE ViT-B | SSL(IN) | 79.51 | 75.93 | 87.10 | 80.96 | 80.88 |
| DINO ViT-B | SSL(IN) | 80.47 | 76.90 | 84.64 | 82.06 | 81.02 |
| SBIs (CVPR'22) | Init(IN) | 93.18 | 72.42 | 86.15 | - | - |

FSFM's average AUC is 6.9 percentage points higher than ImageNet supervised pre-training, ~7 percentage points higher than visual self-supervised methods like MAE/DINO, and exceeds the specialized SOTA method SBIs by about 11 percentage points on DFDC.

### Ablation Study

| Configuration | Average AUC | Note |
|------|---------|------|
| CRFR-P + MIM + ID (Full) | Highest | Full 3C components |
| Random mask + MIM + ID | Decreased | Lacks facial structure prior |
| CRFR-P + MIM only | Decreased | Lacks global semantic alignment |
| CRFR-P + ID only | Decreased | Lacks pixel-level context awareness |
| Fasking-I + MIM + ID | Decreased | Masking strategy inferior to CRFR-P |
| FRP + MIM + ID | Moderate | Has consistency but lacks coherence |
| CRFR-R + MIM + ID | Moderate | Has coherence but random part may block small regions |

### Key Findings
- The CRFR-P masking strategy is the largest contributor—it outperforms both random masking and Fasking-I because it simultaneously ensures intra-region consistency and inter-region coherence.
- MIM and ID are indeed complementary—using either alone is inferior to combining them, as MIM provides local detail awareness and ID provides global semantics.
- Larger pre-training data scales yield better performance—increasing unlabeled real face data further improves performance (a "free lunch" effect).
- A single vanilla ViT can unify and serve all three major face security tasks without requiring task-specific architectures.

## Highlights & Insights
- **CRFR-P Mask Design**: It cleverly injects facial priors into the general MIM framework using facial semantic partitioning. The combination of "covering one region + proportionally covering others" both prevents shortcuts and ensures coverage, presenting a simple yet highly effective design.
- **3C Collaborative Design**: The two-level reconstruction loss (global + regional) of MIM and the self-distillation of ID construct three levels of learning objectives (pixel $\rightarrow$ region $\rightarrow$ instance). This progressive approach covers the full spectrum of representation from low-level to high-level.
- **Generality Validation**: Using a single pre-trained model with simple fine-tuning surpasses task-specific SOTA methods across cross-dataset DfD, cross-domain FAS, and unseen diffusion forgery, providing strong evidence of generalization.

## Limitations & Future Work
- CRFR-P relies on off-the-shelf face parsers; if parsing fails (e.g., extreme poses, occlusions), the masking strategy may degrade to random masking.
- Only ViT-B (86M) was utilized; the potential of larger scales like ViT-L/H or larger pre-training datasets remains to be fully explored.
- Pre-training only uses images and does not exploit the temporal information of videos, which limits the potential improvement for video-level forgery detection.
- The 10 evaluation datasets are primarily facial, and cross-modal scenarios (e.g., voiceprint + face) have not been covered.

## Related Work & Insights
- **vs MAE**: MAE uses random masking to pre-train on ImageNet, lacking facial structure priors. FSFM injects facial semantics via CRFR-P, significantly outperforming MAE on facial tasks.
- **vs DINO**: DINO performs only instance discrimination without pixel-level reconstruction. FSFM proves that the combination of MIM+ID is more effective than either paradigm alone for face security tasks.
- **vs FaRL**: FaRL relies on contrastive learning of image-text pairs, which requires additional textual data. FSFM only needs unlabeled face images, requiring less data and focusing more on "authenticity" representations.
- **vs MCF**: MCF also combines MIM+ID but is pre-trained on LFW and lacks an effective facial masking strategy. FSFM outperforms MCF across all metrics.

## Rating
- Novelty: ⭐⭐⭐⭐ The CRFR-P masking strategy is novel and effective, though the broad framework combining MIM+ID has prior counterparts.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 datasets, 3 major tasks, comparisons with multiple pre-training baselines, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and good unification of the 3C concepts.
- Value: ⭐⭐⭐⭐⭐ A milestone study in the direction of face security foundation models, with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Bi-Level Optimization for Self-Supervised AI-Generated Face Detection](../../ICCV2025/human_understanding/bi-level_optimization_for_self-supervised_ai-generated_face_detection.md)
- [\[ECCV 2024\] VideoClusterNet: Self-Supervised and Adaptive Face Clustering for Videos](../../ECCV2024/human_understanding/videoclusternet_self-supervised_and_adaptive_face_clustering_for_videos.md)
- [\[CVPR 2026\] Action Motifs: Self-Supervised Hierarchical Representation of Human Body Movements](../../CVPR2026/human_understanding/action_motifs_self-supervised_hierarchical_representation_of_human_body_movement.md)
- [\[CVPR 2025\] ControlFace: Harnessing Facial Parametric Control for Face Rigging](controlface_harnessing_facial_parametric_control_for_face_rigging.md)
- [\[ECCV 2024\] Pose-Aware Self-Supervised Learning with Viewpoint Trajectory Regularization](../../ECCV2024/human_understanding/pose-aware_self-supervised_learning_with_viewpoint_trajectory_regularization.md)

</div>

<!-- RELATED:END -->
