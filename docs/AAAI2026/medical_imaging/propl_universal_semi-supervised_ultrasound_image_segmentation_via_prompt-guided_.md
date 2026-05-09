---
title: >-
  [Paper Note] ProPL: Universal Semi-Supervised Ultrasound Image Segmentation via Prompt-Guided Pseudo-Labeling
description: >-
  [AAAI 2026][Medical Imaging][universal segmentation] ProPL is proposed as a framework that, for the first time, achieves universal semi-supervised ultrasound image segmentation via a shared visual encoder, prompt-guided dual decoders, and uncertainty-driven pseudo-label calibration. With only 1/16 labeled data across 5 organs and 8 tasks, it surpasses fully supervised methods by 5.18% mDice.
tags:
  - AAAI 2026
  - Medical Imaging
  - universal segmentation
  - semi-supervised learning
  - pseudo-labeling
  - prompt guidance
  - ultrasound imaging
date: 2026-05-08
content_hash: 0acafa795bfe3f95
---

# ProPL: Universal Semi-Supervised Ultrasound Image Segmentation via Prompt-Guided Pseudo-Labeling

**Conference**: AAAI 2026
**arXiv**: [2511.15057](https://arxiv.org/abs/2511.15057)
**Code**: [https://github.com/WUTCM-Lab/ProPL](https://github.com/WUTCM-Lab/ProPL)
**Area**: Medical Imaging / Ultrasound Segmentation
**Keywords**: universal segmentation, semi-supervised learning, pseudo-labeling, prompt guidance, ultrasound imaging

## TL;DR

ProPL is proposed as a framework that, for the first time, achieves universal semi-supervised ultrasound image segmentation via a shared visual encoder, prompt-guided dual decoders, and uncertainty-driven pseudo-label calibration. With only 1/16 labeled data across 5 organs and 8 tasks, it surpasses fully supervised methods by 5.18% mDice.

## Background & Motivation

**State of the Field**: Ultrasound image segmentation is critical for computer-aided diagnosis, yet existing methods are typically designed for specific organs or tasks, exhibiting poor generalizability.

**Limitations of Prior Work**:
- Fully supervised methods require large amounts of annotated data, which is particularly costly for ultrasound images due to speckle noise, acoustic shadowing, and tissue artifacts that blur boundaries.
- Semi-supervised methods reduce annotation requirements but remain confined to single tasks.
- Universal segmentation frameworks (e.g., DoDNet, UniSeg) support only fully supervised settings and are thus constrained by annotation availability.

**Core Problem**: How to construct a universal ultrasound segmentation framework capable of handling multiple organs and tasks simultaneously while requiring only minimal annotations?

**Starting Point**: Combining prompt learning for task adaptation with dual-decoder mutual learning to generate reliable pseudo-labels.

## Method

### Overall Architecture

Input ultrasound images → shared ConvNeXt-Tiny encoder → dual decoders (standard decoder $\mathcal{G}_{sd}$ + prompt decoder $\mathcal{G}_{pd}$) → mutual pseudo-label supervision between decoders. Task prompts are encoded via BERT and injected into the prompt decoder.

### Key Designs

1. **Prompting-upon-Decoding (PuD)**:

    - **Function**: Injects task-specific textual prompts into the decoding process.
    - **Mechanism**: Task descriptions are encoded via BERT to obtain $\bm{t}$, aligned in dimensionality through 1D convolution and linear projection, and then injected into decoding features via multi-head cross-attention: $\bm{h}_k = \bm{z}_k' + \alpha \cdot \text{MHCA}(Q=\bm{z}_k', K=\bm{\tau}, V=\bm{\tau})$
    - **Design Motivation**: Unlike one-hot encodings or learnable prompts, textual prompts carry richer semantics and generalize to unseen tasks; the learnable scalar $\alpha$ controls the influence of the prompt.

2. **Uncertainty-driven Pseudo-Label Calibration (UPLC)**:

    - **Function**: Filters and calibrates pseudo-labels based on prediction uncertainty.
    - **Mechanism**: Each decoder independently generates predictions; regions of disagreement are treated as high-uncertainty, and only high-confidence regions are used for mutual pseudo-label supervision.
    - **Design Motivation**: Raw pseudo-labels introduce noise; the divergence between the two decoders serves as an uncertainty signal for filtering unreliable predictions.

3. **Universal Ultrasound Dataset**:

    - 6,400 images spanning 5 organs (breast, fetal, cardiac, ovary, thyroid) across 8 segmentation tasks.
    - Labeled data partitions: 1/16, 1/8, and 1/4 settings.

## Key Experimental Results

### Main Results (1/16 Labeled Data)

| Method Type | Method | mDice% | mIoU% |
|-------------|--------|--------|-------|
| Single-task supervised | U-Net | 75.17 | 64.76 |
| Single-task semi-supervised | UniMatch | 79.38 | 69.66 |
| Universal supervised | DoDNet | 62.99 | 50.04 |
| Universal supervised | CLIP-UM | 63.70 | 51.27 |
| **Universal semi-supervised** | **ProPL** | **80.35** | **70.63** |

### Different Labeling Ratios

| Labeling Ratio | ProPL mDice | Gain over Runner-up |
|----------------|-------------|---------------------|
| 1/16 | 80.35% | +0.97% |
| 1/8 | 82.56% | +2.2% |
| 1/4 | 83.70% | +1.32% |

### Ablation Study (1/16)

| Configuration | mDice | mIoU |
|---------------|-------|------|
| w/o prompt (PuD) | 60.76 | 52.57 |
| w/o UPLC | 77.85 | 67.23 |
| Full ProPL | **80.35** | **70.63** |

### Key Findings
- Removing task prompts causes a **19.59%** drop in mDice (80.35→60.76), underscoring the critical role of prompts in universal models.
- UPLC contributes a 2.5% mDice improvement, confirming the effectiveness of uncertainty-based pseudo-label filtering.
- ProPL requires only 712 MB of GPU memory and dominates the performance–efficiency Pareto frontier over all baselines.
- Universal supervised methods (DoDNet, CLIP-UM) perform poorly on ultrasound data (~63% mDice), indicating that universal models benefit substantially from semi-supervised augmentation.

## Highlights & Insights
- **First formulation of "universal semi-supervised ultrasound segmentation"**: integrates multi-organ multi-task generality with the practical constraint of limited annotations.
- **Textual prompts vs. one-hot/learnable prompts**: textual prompts add only 18 s/epoch overhead but provide substantially richer semantics; their removal causes model collapse.
- **Dual-decoder mutual learning**: high-confidence predictions from one decoder serve as pseudo-labels for the other, with inter-decoder disagreement used to estimate uncertainty.

## Limitations & Future Work
- The dataset covers only 2D ultrasound images and does not extend to 3D volumetric ultrasound.
- Prompt templates require manual design; automated prompt generation may yield further improvements.
- The threshold in UPLC requires manual tuning; adaptive threshold strategies warrant further exploration.
- Cross-modality generalization (e.g., whether the framework transfers to CT/MRI) remains unvalidated.

## Related Work & Insights
- **vs. UniMatch (semi-supervised)**: UniMatch is the single-task semi-supervised state of the art; ProPL surpasses it by 0.97% in the universal setting.
- **vs. DoDNet/UniSeg (universal supervised)**: These methods perform poorly on ultrasound data; ProPL compensates for annotation scarcity through semi-supervised learning.
- **vs. SAM-based methods**: SAM-based approaches require additional interactive prompts (points, scribbles), whereas ProPL requires only textual task descriptions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First definition of universal semi-supervised ultrasound segmentation with a well-motivated framework design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Eight tasks, multiple labeling ratios, detailed ablations, and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear and contributions are well-articulated.
- **Value**: ⭐⭐⭐⭐ Both the dataset and the framework offer practical contributions to clinical ultrasound segmentation.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation](bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)
- [\[CVPR 2026\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](../../CVPR2026/medical_imaging/a_semisupervised_framework_for_breast_ultrasound_s.md)
- [\[AAAI 2026\] DeNAS-ViT: Data Efficient NAS-Optimized Vision Transformer for Ultrasound Image Segmentation](denas-vit_data_efficient_nas-optimized_vision_transformer_for_ultrasound_image_s.md)
- [\[AAAI 2026\] DualFete: Revisiting Teacher-Student Interactions from a Feedback Perspective for Semi-supervised Medical Image Segmentation](dualfete_revisiting_teacher-student_interactions_from_a_feedback_perspective_for.md)
- [\[CVPR 2026\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](../../CVPR2026/medical_imaging/semitooth_a_generalizable_semisupervised_framework.md)

<!-- RELATED:END -->
