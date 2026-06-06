---
title: >-
  [Paper Note] ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection
description: >-
  [ICCV 2025][Image Generation][Forgery image detection] This paper proposes ForgeLens, a feature-guided framework built upon a frozen CLIP-ViT backbone. Through a lightweight Weight-Shared Guidance Module (WSGM) and a For…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Forgery image detection"
  - "CLIP-ViT"
  - "data efficiency"
  - "feature guidance"
  - "generalization"
date: 2026-05-08
content_hash: 385debd9b33c1eb2
---

# ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection

**Conference**: ICCV 2025
**arXiv**: [2408.13697](https://arxiv.org/abs/2408.13697)  
**Code**: [GitHub](https://github.com/Yingjian-Chen/ForgeLens)  
**Area**: Image Generation
**Keywords**: Forgery image detection, CLIP-ViT, data efficiency, feature guidance, generalization

## TL;DR

This paper proposes ForgeLens, a feature-guided framework built upon a frozen CLIP-ViT backbone. Through a lightweight Weight-Shared Guidance Module (WSGM) and a Forgery-Aware Feature Integrator (FAFormer), it steers the frozen pretrained network to focus on forgery-relevant features, achieving state-of-the-art generalization performance with only 1% of training data.

## Background & Motivation

The rapid advancement of GANs and diffusion models has enabled the generation of highly realistic forged images, posing serious threats to social security. An ideal forgery image detector should satisfy two objectives: (1) **high generalizability**—detecting images produced by unseen generative techniques; and (2) **data efficiency**—achieving optimal performance with minimal training data.

Existing methods exhibit clear limitations:

**Specialized detection methods** (e.g., FreqNet, F3Net): perform well on training distributions but generalize poorly to unseen generative models, as different generators produce unique artifact signatures and full training leads to overfitting.

**Frozen network methods** (e.g., UniFD): leverage pretrained models such as CLIP to extract general image features with a linear classifier, yielding good generalizability but limited accuracy, since general-purpose features contain substantial forgery-irrelevant information.

The authors illustrate this dilemma via t-SNE visualizations:
- ResNet50 (fully trained): real/fake clusters are well-separated on seen data (ProGAN) but collapse on unseen data.
- Frozen CLIP-ViT: features cluster by image category (cars, cats, etc.) rather than by authenticity, indicating insufficient discriminative power for forgery detection.

**Core Motivation**: Can one preserve frozen pretrained weights (ensuring generalizability) while *guiding* the network to attend to forgery-specific features (improving accuracy), with as few trainable parameters as possible (ensuring data efficiency)?

## Method

### Overall Architecture

ForgeLens adopts an image encoder-only architecture consisting of three steps: (1) WSGM guides the frozen CLIP-ViT to focus on forgery features; (2) FAFormer integrates forgery-relevant information across stages; (3) a linear classifier performs final prediction. Training follows a two-stage strategy: WSGM is trained first, followed by FAFormer.

### Key Designs

1. **Weight-Shared Guidance Module (WSGM)**:

    - **Function**: Inserted between the frozen ViT's MHSA and MLP layers as a lightweight trainable module, steering general features toward forgery-specific representations.
    - **Mechanism**: For $n$ ViT blocks, $m$ WSGMs are shared across blocks, with each WSGM responsible for $n/m$ blocks. WSGM adopts a bottleneck structure:
    $\text{WSGM}(z) = W_{com} \cdot \text{ReLU}(W_{mid} \cdot \text{ReLU}(W_{exp} \cdot z))$
      where $W_{exp} \in \mathbb{R}^{\hat{d} \times d}$ (expansion), $W_{mid} \in \mathbb{R}^{\hat{d} \times \hat{d}}$ (intermediate transformation), $W_{com} \in \mathbb{R}^{d \times \hat{d}}$ (compression), and bottleneck dimension $\hat{d} \ll d$.
      A residual connection is applied: $z_l' = \text{WSGM}_k(z_l) + z_l$
    - **Design Motivation**: (1) Placement between MHSA and MLP allows refinement of global features before MLP processing, progressively enhancing forgery focus in the frozen encoder; (2) weight sharing substantially reduces parameter count; (3) unlike Adapter/LoRA fine-tuning, WSGM acts as a **guidance mechanism** rather than a parameter adjustment.

2. **Forgery-Aware Feature Integrator (FAFormer)**:

    - **Function**: Integrates CLS tokens from each ViT stage, fusing shallow fine-grained texture features with deep semantic features.
    - **Mechanism**: CLS tokens from each stage are concatenated with a newly introduced Focus CLS token and fed into a standard ViT block:
    $c_0 = [CLS_{focus}; CLS_1; CLS_2; \ldots; CLS_N]$
      After multiple self-attention layers, the Focus CLS token aggregates forgery-relevant information from all stages and serves as the final classification representation.
    - **Design Motivation**: Shallow features capture subtle textures and artifacts (e.g., facial contours, hair), while deep features encode semantic and structural information. Transformer self-attention naturally integrates multi-stage features without additional embedding transformations, as CLS tokens inherently satisfy Transformer input requirements.

3. **Two-Stage Training Strategy**:

    - **Function**: Ensures training stability and effective forgery feature guidance.
    - **Mechanism**: Stage one freezes CLIP-ViT and trains only WSGM; stage two freezes all preceding network components and trains only FAFormer.
    - **Design Motivation**: Sequential training avoids instability that may arise from simultaneously optimizing multiple lightweight modules.

### Loss & Training

- Binary Cross-Entropy loss for real/fake classification.
- Adam optimizer with $\beta=(0.9, 0.999)$.
- Input resolution 224×224; only horizontal flip augmentation is applied during training.
- Training set uses only 4 categories of ProGAN-generated images (car, cat, chair, horse).

## Key Experimental Results

### Main Results

| Method Category | Method | Avg. Acc (%) | Avg. AP (%) | Notes |
|----------------|--------|-------------|------------|-------|
| Specialized | FreqNet | 85.09 | 91.95 | Strong on seen GANs, weak on diffusion models |
| Preprocessing | NPR | 87.56 | 92.76 | Exploits upsampling artifacts |
| Frozen network | UniFD | 81.38 | 90.14 | Baseline frozen CLIP-ViT |
| Frozen network | FatFormer | 90.86 | 98.16 | Uses text + image encoders |
| Frozen network | C2P-CLIP | 93.79 | 98.66 | Prev. SOTA (requires two encoders) |
| **Ours** | **ForgeLens** | **94.99** | **98.83** | Image encoder only, simpler architecture |

ForgeLens achieves 94.99% average accuracy across 19 generative model test sets, surpassing the baseline UniFD by 13.61% and the previous SOTA C2P-CLIP by 1.2%, while employing a simpler architecture (image encoder only).

### Ablation Study

| Configuration | Avg. Acc (%) | Avg. AP (%) | Notes |
|--------------|-------------|------------|-------|
| Baseline (no WSGM, no FAFormer) | 81.38 | 90.14 | UniFD baseline |
| + FAFormer only | 87.89 | 92.26 | Multi-stage feature fusion +6.51 |
| + WSGM only | 94.52 | 98.12 | Forgery feature guidance +13.14 |
| **+ WSGM + FAFormer** | **94.99** | **98.83** | Full ForgeLens |

Comparison of WSGM against fine-tuning methods:

| Method | Avg. Acc (%) | Avg. AP (%) |
|--------|-------------|------------|
| No adaptation | 81.38 | 90.14 |
| Adapter | 85.49 (+4.11) | 94.29 |
| LoRA | 86.96 (+5.58) | 96.41 |
| **WSGM** | **94.52 (+12.86)** | **98.12** |

WSGM substantially outperforms both Adapter and LoRA, demonstrating that guidance mechanisms are superior to parameter fine-tuning approaches.

### Key Findings

1. **Remarkable data efficiency**: Using only 1% of training data (~72 images), ForgeLens achieves performance comparable to full-data training, which is critical for rapidly responding to emerging forgery techniques.
2. **WSGM is the primary contributor**: It accounts for a 13.14% accuracy gain, far exceeding FAFormer's contribution of 6.51%.
3. **CAM visualizations** confirm that ForgeLens successfully focuses on forgery-relevant regions (e.g., facial contours, hair), whereas the original CLIP-ViT distributes attention broadly across the image.
4. The model maintains robustness under perturbations including Gaussian noise, Gaussian blur, JPEG compression, and random cropping.

## Highlights & Insights

- **Philosophical distinction between "guidance" and "fine-tuning"**: WSGM does not modify network parameters (as in LoRA's low-rank updates) but redirects information flow within the frozen network—a design philosophy that simultaneously preserves generalizability and improves accuracy.
- **Impressive data efficiency**: The model is operational with 1% of training data, vastly outperforming specialized methods that require large training sets.
- **Architectural simplicity**: Using only an image encoder (without a text encoder), ForgeLens is simpler yet more effective than dual-encoder methods such as FatFormer and C2P-CLIP.

## Limitations & Future Work

1. The training set is limited to ProGAN (a GAN-based method); adaptability to emerging generative paradigms (e.g., video generation, 3D generation) remains to be validated.
2. The bottleneck dimension $\hat{d}$ and number of sharing groups $m$ in WSGM require ablation studies to determine.
3. Only binary classification (real/fake) is evaluated; finer-grained source attribution tasks are not explored.
4. Misclassifications persist on certain diffusion model subsets (e.g., Guided with 200 steps + CFG).

## Related Work & Insights

- UniFD established the paradigm of "frozen CLIP + linear classifier"; ForgeLens builds upon this foundation and substantially improves accuracy through a guidance mechanism.
- The comparison with PEFT methods such as LoRA and Adapter reveals a fundamental distinction between "guidance" and "fine-tuning."
- The multi-stage CLS token aggregation idea in FAFormer is transferable to other tasks requiring multi-level feature fusion.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The guidance mechanism of WSGM is novel, and the data efficiency results are particularly noteworthy.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluation spans 19 generative models, with comprehensive ablations, comparisons against multiple fine-tuning methods, and robustness assessments.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated with intuitive and compelling visualizations.
- **Value**: ⭐⭐⭐⭐⭐ — Highly significant for practical AI-generated image detection; the ability to deploy with 1% training data offers substantial application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis](deepshield_fortifying_deepfake_video_detection_with_local_and_global_forgery_ana.md)
- [\[ICCV 2025\] Semantic Discrepancy-aware Detector for Image Forgery Identification](semantic_discrepancy-aware_detector_for_image_forgery_identification.md)
- [\[ICCV 2025\] M2SFormer: Multi-Spectral and Multi-Scale Attention with Edge-Aware Difficulty Guidance for Image Forgery Localization](m2sformer_multi-spectral_and_multi-scale_attention_with_edge-aware_difficulty_gu.md)
- [\[ICCV 2025\] Domain Generalizable Portrait Style Transfer](domain_generalizable_portrait_style_transfer.md)
- [\[AAAI 2026\] Creating Blank Canvas Against AI-Enabled Image Forgery](../../AAAI2026/image_generation/creating_blank_canvas_against_ai-enabled_image_forgery.md)

</div>

<!-- RELATED:END -->
