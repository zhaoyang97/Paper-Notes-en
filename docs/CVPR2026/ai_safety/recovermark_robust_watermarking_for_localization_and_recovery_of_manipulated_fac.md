---
title: >-
  [Paper Note] RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces
description: >-
  [CVPR 2026][AI Safety][Face tampering detection] This paper proposes RecoverMark, a robust watermarking framework that embeds facial content itself as a watermark into the background region, simultaneously achieving tampering localization, original content recovery, and copyright verification while remaining effective under watermark removal attacks.
tags:
  - CVPR 2026
  - AI Safety
  - Face tampering detection
  - robust watermarking
  - tampering localization
  - content recovery
  - copyright verification
date: 2026-05-08
content_hash: 3750fbcbb8de3bcf
---

# RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces

**Conference**: CVPR 2026
**arXiv**: [2602.20618](https://arxiv.org/abs/2602.20618)
**Code**: To be released (authors state it will be made public upon acceptance)
**Area**: AI Security
**Keywords**: Face tampering detection, robust watermarking, tampering localization, content recovery, copyright verification

## TL;DR

This paper proposes RecoverMark, a robust watermarking framework that embeds facial content itself as a watermark into the background region, simultaneously achieving tampering localization, original content recovery, and copyright verification while remaining effective under watermark removal attacks.

## Background & Motivation

**Background**: AIGC technologies such as Stable Diffusion and various GAN variants have made face manipulation extremely accessible, posing serious threats to the authenticity of visual content and intellectual property.

**Limitations of Prior Work**: Existing active defense methods (e.g., EditGuard, OmniGuard) adopt a dual-watermarking strategy combining fragile and robust watermarks. Fragile watermarks handle tampering detection and localization, while robust watermarks handle copyright authentication. However, these methods assume attackers are unaware of the watermark's existence and fail completely under watermark removal attacks (e.g., low-pass filtering, regeneration attacks).

**Key Challenge**: In dual-watermarking frameworks, the two types of watermarks interfere with each other and the limited embedding capacity further degrades the effectiveness of the fragile watermark; moreover, existing methods overlook the critical need for recovering content in tampered regions.

**Goal**: To design a unified robust watermarking framework capable of simultaneously performing tampering localization, content recovery, and copyright verification even under adversarial watermark removal attacks.

**Key Insight**: The paper exploits a key real-world constraint — attackers must preserve the semantic consistency of the background to avoid visual detection. Accordingly, facial content is embedded as a watermark into the surrounding background, such that if the background remains intact, the watermark can be extracted.

**Core Idea**: The protected facial content is robustly embedded into the background region as a watermark, and a two-stage progressive training strategy is employed to simultaneously achieve tampering localization, content recovery, and copyright verification.

## Method

### Overall Architecture

RecoverMark proceeds as follows: a segmentation tool (MTCNN/YOLOSeg/SAM2) first separates the original image into a salient region (face) $I_{sal}$ and background $I_{bg}$; a watermark encoder Enc compresses the face into a latent representation; a hiding network HNet embeds this representation into the background to produce a container image $I_{cntr}$; an extraction network ENet and decoder Dec then recover the facial information $I'_{sal}$ from the container image.

### Key Designs

1. **Watermark Encoder/Decoder (Enc/Dec)**: Based on the CEILNet architecture, it compresses facial content into a latent representation suitable for embedding. The mechanism leverages the image's own content as the watermark to enhance extraction robustness. The design motivation is to avoid using watermarks independent of the image, thereby natively supporting content recovery.

2. **Two-Stage Progressive Training**:

    - **Stage 1 (Initial Training)**: Trains four networks — Enc, HNet, ENet, and Dec — optimizing three losses: a fidelity loss (container image should approximate the original background), a watermark loss (extracted face should approximate the original face), and a clean loss (extraction from a background without an embedded watermark should yield an all-white image, preventing false detections).
    - **Stage 2 (Robustness Enhancement Training)**: Freezes Enc and Dec, and introduces a perturbation layer between HNet and ENet to simulate three attack categories: salient-region processing (adding noise to the face region to prevent extraction from relying on facial information), global processing (JPEG compression, Gaussian noise, low-pass filtering), and advanced attacks (regeneration attacks). A progressive strategy is adopted: the hardest regeneration attack is trained first, followed by gradual introduction of other perturbations.

3. **Recovery, Localization, and Copyright Verification**: After extracting the hidden facial information $I'_{sal}$ from a suspicious image: (1) a difference comparison between $I'_{sal}$ and the face region of the suspicious image generates a tampering localization mask; (2) the normalized cross-correlation (NCC > 0.95) between $I'_{sal}$ and the original face is used for copyright verification.

### Loss & Training

The total loss is: $\mathcal{L}_{sum} = \alpha_1 \mathcal{L}_{fidelity} + \alpha_2 \mathcal{L}_{wm} + \alpha_3 \mathcal{L}_{clean}$, where all three weights are set to 1.

Progressive training strategy: regeneration attacks occupy the first half of total training epochs, with the remaining perturbations equally distributed over the second half. Experiments show that introducing the regeneration attack first is critical — delayed introduction leads to failure against the strongest attacks.

## Key Experimental Results

### Main Results

Localization performance (F1/AUC) on the ID dataset (CelebA) under Structpix2pix manipulation:

| Method | Regen. F1 | Regen. AUC | Noise F1 | Noise AUC | JPEG F1 | JPEG AUC | LPF F1 | LPF AUC | Lattice F1 | Lattice AUC |
|------|-----------|-------------|--------|---------|--------|---------|--------|---------|-----------|------------|
| MVSS-Net | 0.041 | 0.723 | 0.062 | 0.711 | 0.157 | 0.776 | 0.184 | 0.755 | 0.034 | 0.719 |
| EditGuard | 0.090 | 0.610 | 0.528 | 0.932 | 0.552 | 0.954 | 0.090 | 0.658 | 0.438 | 0.930 |
| OmniGuard | 0.105 | 0.655 | 0.127 | 0.659 | 0.315 | 0.890 | 0.146 | 0.743 | 0.113 | 0.689 |
| **RecoverMark** | **0.855** | **0.993** | **0.876** | **0.992** | **0.867** | **0.993** | **0.830** | **0.989** | **0.842** | **0.991** |

### Ablation Study / Recovery Performance

Face recovery quality comparison on the ID dataset (PSNR/MS-SSIM):

| Method | Regen. PSNR | Regen. MS-SSIM | Noise PSNR | Noise MS-SSIM | JPEG PSNR | JPEG MS-SSIM | Lattice PSNR | Lattice MS-SSIM |
|------|-------------|-----------------|---------|-------------|---------|-------------|-------------|----------------|
| Imuge+ | 7.252 | 0.339 | 10.432 | 0.563 | 10.778 | 0.629 | 9.089 | 0.424 |
| **RecoverMark** | **22.154** | **0.607** | **23.276** | **0.657** | **23.314** | **0.680** | **23.230** | **0.655** |

### Key Findings

- RecoverMark substantially outperforms all baseline methods across all attack types, achieving F1 > 0.7 and AUC > 0.98.
- Strong performance is maintained on the unseen Lattice attack (F1 = 0.842), demonstrating strong generalization.
- Recovery PSNR exceeds Imuge+ by approximately 12–15 dB, indicating significantly improved content recovery quality.
- Copyright verification success rate reaches 99.9%.
- Embedding fidelity remains high when the face occupies ≤ 60% of the image, but degrades noticeably beyond this threshold.

## Highlights & Insights

- **Unified Framework**: This is the first work to integrate tampering localization, content recovery, and copyright verification into a single robust watermarking framework, eliminating the capacity competition inherent in dual-watermarking approaches.
- **Exploiting Real-World Constraints**: The paper cleverly exploits the constraint that attackers must preserve background consistency, enabling robust extraction by embedding facial content into the background.
- **Progressive Training Strategy**: Analogous to learning from hard to easy, the framework first trains on the hardest regeneration attack before progressively incorporating other perturbations.
- **Generalization to Unseen Attacks**: Effectiveness on the Lattice attack, which is absent from training, confirms the generalizability of the robustness.

## Limitations & Future Work

- Embedding quality degrades when the face occupies more than 60% of the image, indicating limited capacity.
- Validation is currently conducted only at 256×256 resolution; performance in high-resolution settings remains unknown.
- The accuracy of the segmentation tool directly affects embedding and extraction quality.
- The framework has only been validated for face manipulation scenarios and has not yet been extended to other types of image tampering.

## Related Work & Insights

- Passive detection methods (MVSS-Net, HiFi-Net) rely on tampering artifacts, which can be easily eliminated by post-processing.
- Active defense methods (EditGuard, OmniGuard) rely on fragile watermarks and are vulnerable to removal attacks.
- Self-recovering watermarking methods (Imuge/Imuge+) represent early attempts at DNN-based joint localization and recovery, but are also based on fragile embedding.
- The core contribution of RecoverMark is a paradigm shift from fragile to robust watermarking while preserving sensitivity to tampering detection.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of embedding the face itself as a watermark into the background is novel, and the unified three-task framework is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across ID/OOD datasets, multiple seen/unseen attacks, multiple manipulation types, capacity analysis, and both qualitative and quantitative results.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, grounded in a practical scenario (courtroom evidence), with internally consistent logic.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to facial content protection; the unified framework reduces deployment complexity in practice.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering](clustermark_towards_robust_watermarking_for_autoregressive_image_generators_with.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[CVPR 2026\] AdvMark: Decoupling Defense Strategies for Robust Image Watermarking](decoupling_defense_strategies_for_robust_image_watermarking.md)
- [\[CVPR 2026\] TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking](tiacam_text-anchored_invariant_feature_learning_with_auto-augmentation_for_camer.md)
- [\[CVPR 2026\] Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection](tutor-student_reinforcement_learning_a_dynamic_curriculum_for_robust_deepfake_de.md)

<!-- RELATED:END -->
