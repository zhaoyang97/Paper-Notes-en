---
title: >-
  [Paper Note] Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection
description: >-
  [CVPR 2025][AI Safety][Face forgery detection] This paper proposes Forensics Adapter, a lightweight adapter network with only 5.7M parameters that learns blending boundary features of face forged images in parallel with a frozen CLIP. Highly generalizable cross-dataset face forgery detection is achieved via a triple objective: masked boundary prediction, patch-level contrastive learning, and sample-level contrastive learning, achieving an AUC of 0.914 on CDF-v1.
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Face forgery detection"
  - "CLIP adaptation"
  - "blending boundary"
  - "contrastive learning"
  - "cross-dataset generalization"
date: 2026-05-08
content_hash: 8de8e1387e1a5c2d
---

# Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection

**Conference**: CVPR 2025  
**arXiv**: [2411.19715](https://arxiv.org/abs/2411.19715)  
**Code**: [https://github.com/OUC-VAS/ForensicsAdapter](https://github.com/OUC-VAS/ForensicsAdapter)  
**Area**: AI Security  
**Keywords**: Face forgery detection, CLIP adaptation, blending boundary, contrastive learning, cross-dataset generalization

## TL;DR

This paper proposes Forensics Adapter, a lightweight adapter network with only 5.7M parameters that learns blending boundary features of face forged images in parallel with a frozen CLIP. Highly generalizable cross-dataset face forgery detection is achieved via a triple objective: masked boundary prediction, patch-level contrastive learning, and sample-level contrastive learning, achieving an AUC of 0.914 on CDF-v1.

## Background & Motivation

**Background**: The core challenge of Deepfake detection is cross-domain generalization—detectors trained on one forgery method exhibit sharp performance degradation when facing unseen forgery methods. Recently, researchers have attempted to leverage general representations from vision foundation models like CLIP to improve generalization.

**Limitations of Prior Work**: Directly fine-tuning CLIP for forgery detection destroys its general representations, leading to reduced cross-domain generalization. Conversely, fully freezing CLIP fails to capture forgery-specific detailed features (such as blending boundaries, i.e., the stitching seams between forged and real regions).

**Key Challenge**: The need to preserve CLIP's general representations (for generalization) while learning forgery-specific discriminative features (for detection accuracy) conflicts when parameters are shared.

**Key Insight**: Utilizing an independent, lightweight adapter network specifically to learn blending boundary features, while making minimal modifications to CLIP (1×1 convolutions and attention bias), allows the two streams to work complementarily.

**Core Idea**: Independent adapter learning blending boundaries + minimal CLIP modifications + triple contrastive objectives = highly generalizable cross-domain forgery detection.

## Method

### Key Designs

1. **Dual-Stream Architecture (Adapter + CLIP)**:

    - **Function**: Separates general representations from task-specific representations.
    - **Mechanism**: The adapter stream utilizes learnable query tokens to interact with CLIP vision tokens, learning blending boundary maps via cross-attention. The CLIP stream undergoes minimal modifications with trainable 1×1 convolutions and attention biases added to the tokens. Features from both streams are fused for classification.
    - **Design Motivation**: The adapter parameters are independent (5.7M), avoiding disruption to CLIP's pretrained representations.

2. **Blending Boundary Prediction ($\mathcal{L}_1$)**:

    - **Function**: Forces the adapter to explicitly learn boundary features of forged regions.
    - **Mechanism**: The forgery mask is Gaussian-blurred to generate a blending boundary map, and the adapter's output is used to predict this map via MSE loss. This is computed only on forged samples.
    - **Design Motivation**: Blending boundaries serve as a common forgery artifact across different methods—regardless of the forgery generation method, splicing boundaries always exist.

3. **Patch-Level and Sample-Level Contrastive Learning ($\mathcal{L}_2 + \mathcal{L}_3$)**:

    - **Function**: Pulls identical patch/sample features together while pushing dissimilar ones apart.
    - **Mechanism**: $\mathcal{L}_2$ performs patch-level contrastive learning (each patch of the 14×14 feature map determines positive/negative pairs based on the blending boundary mask), and $\mathcal{L}_3$ performs sample-level contrastive learning.
    - **Design Motivation**: Patch-level contrastive learning learns local boundary discrimination, while sample-level contrastive learning learns global real/forgery differentiation.

### Loss & Training

$\mathcal{L} = 10\mathcal{L}_0 + 200\mathcal{L}_1 + 20\mathcal{L}_2 + 10\mathcal{L}_3 + 1.5\mathcal{L}_4$. Here, $\mathcal{L}_0$ is the binary classification BCE loss, and $\mathcal{L}_4$ is the vision-language alignment loss for ForAda++. The model is trained solely on FaceForensics++.

## Key Experimental Results

### Main Results

Cross-dataset frame-level AUC:

| Dataset | ForAda | Prev. SOTA | Gain |
|--------|--------|--------|------|
| CDF-v1 | **0.914** | 0.867 | +4.7% |
| CDF-v2 | **0.900** | 0.869 | +3.1% |
| DFDC | **0.843** | 0.758 | +8.5% |
| DFD | **0.933** | 0.915 | +1.8% |

### Ablation Study

| Removed Loss | CDF-v2 AUC |
|-----------|-----------|
| Full Model | **0.914** |
| -$\mathcal{L}_1$ (boundary) | 0.904 (-1.0%) |
| -$\mathcal{L}_2$ (patch contrastive) | 0.818 (-9.6%) |
| -$\mathcal{L}_3$ (sample contrastive) | 0.904 (-1.0%) |

### Key Findings
- **Patch-level contrastive learning is of vital importance**: Removing it leads to an AUC drop of 9.6%, indicating that local boundary discrimination is key to cross-domain generalization.
- **DFDC achieves the highest gain**: +8.5%, indicating that blending boundary features have better generalization capabilities on low-quality data.
- **Text enhancement in ForAda++ is limited**: An average improvement of only +1.3% suggests that vision features have already captured most of the necessary information.

## Highlights & Insights
- **Blending boundaries serve as universal forgery clues**—forgeries from different methods consistently exhibit splicing boundaries, which possesses better generalization ability than learning specific forgery patterns (e.g., frequency artifacts).
- **A lightweight adapter with 5.7M parameters outperforms full fine-tuning**—demonstrating that keeping CLIP frozen and appending lightweight plug-in modules is a superior way to leverage foundation models.

## Limitations & Future Work
- Training requires manipulation masks (which are unavailable in real-world Deepfake datasets).
- Text prompts are limited to only 5 facial regions, potentially missing other forged areas.
- The performance of 0.803 on WildDeepfake still leaves room for improvement.
- The text encoder is frozen, without joint adaptation of the vision-language space.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of a dual-stream adapter and blending boundaries is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation studies conducted across 6 cross-domain datasets at both frame and video levels.
- Writing Quality: ⭐⭐⭐⭐ Clear experimental design.
- Value: ⭐⭐⭐⭐ Provides an efficient and practical solution for CLIP-based forgery detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Towards General Visual-Linguistic Face Forgery Detection](towards_general_visual-linguistic_face_forgery_detection.md)
- [\[CVPR 2025\] Stacking Brick by Brick: Aligned Feature Isolation for Incremental Face Forgery Detection](stacking_brick_by_brick_aligned_feature_isolation_for_incremental_face_forgery_d.md)
- [\[CVPR 2026\] A Sanity Check for Multi-In-Domain Face Forgery Detection in the Real World](../../CVPR2026/ai_safety/a_sanity_check_for_multi-in-domain_face_forgery_detection_in_the_real_world.md)
- [\[CVPR 2026\] DiffusionFF: A Diffusion-based Framework for Joint Face Forgery Detection and Fine-Grained Artifact Localization](../../CVPR2026/ai_safety/diffusionff_a_diffusion-based_framework_for_joint_face_forgery_detection_and_fin.md)
- [\[CVPR 2026\] Beyond \[CLS\] Token: Query-Driven Token-Level Forgery Purification for Generalizable Deepfake Detection](../../CVPR2026/ai_safety/beyond_cls_token_query-driven_token-level_forgery_purification_for_generalizable.md)

</div>

<!-- RELATED:END -->
