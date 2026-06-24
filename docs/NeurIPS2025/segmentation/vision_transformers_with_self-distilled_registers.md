---
title: >-
  [Paper Note] Vision Transformers with Self-Distilled Registers
description: >-
  [NeurIPS 2025 Spotlight][Segmentation][Vision Transformer] This paper proposes PH-Reg (Post Hoc Registers), an efficient self-distillation approach that retrofits register tokens into existing pretrained ViTs without labeled data or full retraining. By combining test-time augmentation-based teacher feature denoising with student self-distillation, PH-Reg effectively eliminates artifact tokens in ViT dense features, improving performance on segmentation and depth estimation.
tags:
  - "NeurIPS 2025 Spotlight"
  - "Segmentation"
  - "Vision Transformer"
  - "Register Token"
  - "Self-Distillation"
  - "Feature Denoising"
  - "Open-Vocabulary Segmentation"
date: 2026-05-08
content_hash: eb4a32fe2df5a4d9
---

# Vision Transformers with Self-Distilled Registers

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.21501](https://arxiv.org/abs/2505.21501)  
**Code**: [GitHub](https://github.com/yinjiechen/PH-Reg)  
**Area**: Image Segmentation
**Keywords**: Vision Transformer, Register Token, Self-Distillation, Feature Denoising, Open-Vocabulary Segmentation

## TL;DR

This paper proposes PH-Reg (Post Hoc Registers), an efficient self-distillation approach that retrofits register tokens into existing pretrained ViTs without labeled data or full retraining. By combining test-time augmentation-based teacher feature denoising with student self-distillation, PH-Reg effectively eliminates artifact tokens in ViT dense features, improving performance on segmentation and depth estimation.

## Background & Motivation

Vision Transformers have become the dominant architecture for visual modeling, demonstrating remarkable scalability across classification, detection, and segmentation tasks. However, recent studies have identified **artifact tokens** in ViT dense features—abnormal activations that are inconsistent with local image semantics, degrading fine-grained spatial localization and negatively affecting tasks requiring high spatial precision, such as semantic segmentation and part correspondence.

The existing remedy is to incorporate **register tokens**—randomly initialized learnable embeddings that participate in self-attention but are discarded from the output. Register tokens effectively "absorb" artifact patterns, yielding cleaner dense features. However, this approach requires **training from scratch**, which is computationally prohibitive for large-scale pretrained models such as CLIP and DINOv2.

The core motivation is: **Can register tokens be added post hoc to existing large-scale pretrained ViTs?** This requires addressing two challenges:
1. How to obtain clean training targets without labeled data?
2. How to effectively suppress artifacts by fine-tuning only a minimal number of parameters?

## Method

### Overall Architecture

PH-Reg is a self-distillation framework in which both the teacher and student networks are initialized from the same pretrained weights. The teacher network remains frozen with its original architecture and generates denoised dense features via test-time augmentation (TTA) as distillation targets. The student network introduces only register tokens and a small set of unfrozen parameters, learning to produce clean dense representations through distillation.

### Key Designs

1. **Efficient Denoising of Teacher Representations**: The key observation is that **artifact tokens do not shift statically with image content**—if the image is translated by some amount, the artifacts do not shift accordingly. Exploiting this property, $n$ random augmentations (horizontal/vertical shifts and flips) are applied to the input image, each with a shift magnitude that is an integer multiple of the patch size $k$. Teacher features $F_i = f_{\text{teacher}}(\mathcal{I}_i)$ are extracted for each augmented image, inverse-transformed back to the original coordinate frame, and averaged across positions with a weighted accumulation. The resulting denoised features $Q/K$ (accumulated features / counts) are computed without any gradient computation. This is equivalent to the MSE-optimal solution and is **approximately two orders of magnitude faster than the neural-field-based DVT method** (<200ms).

2. **Design of the Student Network**: $m$ register tokens are added to the original ViT, so that $m + 1 + \frac{H}{k} \times \frac{W}{k}$ tokens participate in self-attention. Ablation studies determine the optimal unfreezing strategy: in addition to the register tokens, the positional embeddings, convolutional patch embedding layer, and the final attention layer are unfrozen. Key findings include: (1) even a single register token substantially improves feature quality (the 99th-percentile cosine similarity of the 1-register configuration exceeds the 50th-percentile of the raw model); (2) 16 registers offer the best cost-performance trade-off; and (3) positional embeddings are not the sole source of artifacts (in contrast to the assumption in prior work DVT).

3. **Learning and Optimization**: A multi-objective distillation loss combining cosine similarity and MSE is used to ensure alignment in both direction and magnitude: $\text{Loss}_{\text{total}} = 1 - \text{cossim}(\text{target}, \text{predicted}) + \text{MSE}(\text{target}, \text{predicted})$. The entire training pipeline requires only a set of unlabeled images.

### Loss & Training

Distillation is performed using unlabeled images from COCO Captions. The teacher network is frozen; 10 augmented views are used to generate targets. The unfrozen components of the student network include the register tokens, positional embeddings, convolutional patch embedding, and the final attention layer. The default configuration uses 16 register tokens. No segmentation or depth annotations are required during training.

## Key Experimental Results

### Main Results

**Open-Vocabulary Semantic Segmentation (mIoU%, OpenAI CLIP ViT-B/16)**:

| Method | VOC21 | PC60 | Object | VOC20 | PC59 | Stuff | City | ADE | Avg. |
|--------|-------|------|--------|-------|------|-------|------|-----|------|
| MaskCLIP | 49.27 | 25.46 | 26.94 | 66.56 | 28.62 | 18.80 | 28.33 | 13.70 | 32.21 |
| SCLIP | 59.62 | 31.74 | 33.52 | 81.53 | 34.46 | 22.65 | 32.34 | 16.45 | 40.08 |
| NACLIP | 58.88 | 32.20 | 33.15 | 79.70 | 35.16 | 23.30 | 35.48 | 17.42 | 39.41 |
| NACLIP+DVT | 60.25 | 32.73 | 32.89 | 80.26 | 35.91 | 23.41 | 36.31 | 17.54 | 39.91 |
| **PH-Reg** | **63.01** | **34.52** | **35.27** | 83.05 | **37.88** | **24.66** | **37.17** | **19.22** | **41.85** |

PH-Reg achieves the best results on 7 out of 8 benchmarks, with an average gain of 1.94%.

**Linear Probing for Segmentation and Depth Estimation**:

| Method | VOC21 mIoU | ADE mIoU | NYUd RMSE↓ | NYUd δ₁↑ |
|--------|-----------|---------|-----------|----------|
| CLIP | 73.88 | 35.78 | 0.6843 | 64.93 |
| CLIP+DVT | 74.74 | 36.39 | 0.6800 | 65.07 |
| **PH-Reg (CLIP)** | **75.32** | **38.07** | **0.6746** | **68.17** |
| DINOv2 | 84.13 | 47.82 | 0.4566 | 82.92 |
| DINOv2+DVT | 85.43 | 48.86 | 0.4329 | 85.23 |
| **PH-Reg (DINOv2)** | 84.85 | 48.66 | **0.4306** | **86.35** |

### Ablation Study

| Configuration | VOC21 | 8-Bench Avg. | Notes |
|--------------|-------|-------------|-------|
| Vanilla MaskCLIP | 49.27 | 32.21 | Baseline |
| Denoising only (10× aug) | 51.41 | 34.55 | TTA denoising only, +2.34 |
| Distill, no reg, no denoise | 61.16 | 40.68 | Distillation w/o register or denoising |
| Distill, with reg, no denoise | 61.27 | 40.66 | With register, no denoising |
| Distill, no reg, with denoise | 62.48 | 41.48 | With denoising, no register |
| **Full Pipeline** | **63.01** | **41.85** | Register + denoising |

Roughly half of the total improvement is attributable to register tokens and the other half to the teacher denoising process.

### Key Findings

- Artifact tokens are not always high-norm—in some models, artifact tokens exhibit lower norms than normal tokens, challenging prior assumptions.
- Positional embeddings cannot fully account for artifact formation (contrary to DVT's hypothesis), though unfreezing them still has a positive effect.
- DVT's static artifact assumption does not hold for CLIP-type models, limiting DVT's effectiveness on these backbones.
- The shifting ratio affects denoising quality; the optimal range is 10–15%.
- PH-Reg generalizes across multiple ViT backbones: OpenAI CLIP, OpenCLIP, DFN-CLIP, and DINOv2.

## Highlights & Insights

- The core idea of **"average to denoise"** is remarkably simple and elegant: TTA-based averaging is mathematically equivalent to the MSE-optimal solution, yet requires no gradient computation whatsoever.
- The self-distillation design enables training without any labeled data, substantially lowering the practical barrier to adoption.
- New findings on the nature of artifact tokens (non-static, not always high-norm) advance the community's understanding of ViT internal mechanisms.
- The number of additional parameters introduced by register tokens is minimal (on the order of thousands), and the inference overhead is negligible.

## Limitations & Future Work

- Validation is currently limited to ViT-B/16; the effectiveness on larger models (ViT-L, ViT-G) remains unexplored.
- The denoising process requires multiple forward passes (10 by default), which, while far faster than DVT, still incurs additional inference cost.
- The interaction between distillation and denoising warrants further analysis—why do the two components contribute roughly equal improvements?
- ClearCLIP slightly outperforms PH-Reg on VOC20, suggesting that q-q attention carries unique localization value.

## Related Work & Insights

- Compared to DVT (neural-field-based denoising), PH-Reg is more efficient, more general, and relies on fewer assumptions.
- The post hoc register token paradigm is potentially extensible to other self-attention architectures beyond vision.
- The TTA denoising idea is orthogonal to other dense prediction methods and could be combined with them.
- Attention-modification methods such as SCLIP and NACLIP are orthogonal to PH-Reg, leaving room for potential synergistic combinations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The post hoc addition of register tokens is both practical and creative; TTA-based denoising is elegantly concise.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Eight segmentation benchmarks, multi-backbone validation, depth estimation, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, fair experimental setup, and identical backbone models used across all baselines.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a low-cost solution for improving dense features in a wide range of existing pretrained ViTs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity](../../ICCV2025/segmentation/legrad_an_explainability_method_for_vision_transformers_via_feature_formation_se.md)
- [\[CVPR 2025\] DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers](../../CVPR2025/segmentation/da-vpt_semantic-guided_visual_prompt_tuning_for_vision_transformers.md)
- [\[ICLR 2026\] Revisiting \[CLS\] and Patch Token Interaction in Vision Transformers](../../ICLR2026/segmentation/revisiting_cls_and_patch_token_interaction_in_vision_transformers.md)
- [\[CVPR 2025\] COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training](../../CVPR2025/segmentation/cosmos_cross-modality_self-distillation_for_vision_language_pre-training.md)
- [\[ECCV 2024\] SiLC: Improving Vision Language Pretraining with Self-Distillation](../../ECCV2024/segmentation/silc_improving_vision_language_pretraining_with_self-distillation.md)

</div>

<!-- RELATED:END -->
