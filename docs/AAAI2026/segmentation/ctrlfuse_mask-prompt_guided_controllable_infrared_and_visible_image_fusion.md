---
title: >-
  [Paper Note] CtrlFuse: Mask-Prompt Guided Controllable Infrared and Visible Image Fusion
description: >-
  [AAAI2026][Segmentation][infrared-visible image fusion] This paper proposes CtrlFuse, which achieves interactive controllable infrared-visible image fusion by fine-tuning SAM with mask prompt guidance…
tags:
  - "AAAI2026"
  - "Segmentation"
  - "infrared-visible image fusion"
  - "controllable fusion"
  - "mask prompt"
  - "SAM"
  - "semantic segmentation"
date: 2026-05-08
content_hash: db02d6577bd67775
---

# CtrlFuse: Mask-Prompt Guided Controllable Infrared and Visible Image Fusion

**Conference**: AAAI2026
**arXiv**: [2601.08619](https://arxiv.org/abs/2601.08619)  
**Code**: [Sevryy/CtrlFuse](https://github.com/Sevryy/CtrlFuse)  
**Area**: Image Segmentation
**Keywords**: infrared-visible image fusion, controllable fusion, mask prompt, SAM, semantic segmentation

## TL;DR

This paper proposes CtrlFuse, which achieves interactive controllable infrared-visible image fusion by fine-tuning SAM with mask prompt guidance, simultaneously improving fusion quality and downstream segmentation/detection performance.

## Background & Motivation

Infrared-visible image fusion aims to combine complementary information from two modalities to provide all-weather perception for intelligent autonomous systems. Visible images offer rich color and high resolution but degrade under low-light conditions; infrared images compensate for poor illumination but lack texture information.

Existing methods suffer from two fundamental limitations:

1. **Pixel-level fusion methods** focus solely on pixel consistency between the fused and source images, ignoring the adaptability of fused images to downstream perception tasks.
2. **Task-driven fusion methods** implicitly learn fixed semantic categories via cascaded detection/segmentation models, and cannot dynamically control attention to specific targets according to varying application requirements.

For instance, although existing methods learn target semantics during training, they still underperform in practical vehicle segmentation scenarios. This indicates the need for a semantically controllable multimodal fusion architecture capable of dynamic, demand-driven fusion.

## Core Problem

How to construct an interactive and controllable multimodal image fusion framework that enables users to dynamically specify semantic targets of interest via mask prompts, while achieving mutual reinforcement between fusion quality and downstream task performance?

## Method

### Overall Architecture

CtrlFuse consists of four core components:

- **Multimodal backbone encoder-decoder**: Extracts infrared features $F_{ir}$ and visible features $F_{vis}$ separately, concatenates them, and generates a reference image $I_{ref}$ through the decoder.
- **Reference Prompt Encoder (RPE)**: Dynamically encodes task-relevant semantic prompts under mask guidance.
- **Prompt-Semantic Fusion Module (PSFM)**: Explicitly injects semantic prompts into the fusion features.
- **Frozen SAM**: Provides strong semantic-aware foundational capabilities.

### Reference Prompt Encoder

Taking the infrared branch as an example:

1. The mask prompt is applied to $F_{ir}$ via Hadamard product followed by average pooling to obtain the target feature $F_t$.
2. $F_{ir}$ and $F_{ref}$ are each concatenated with $F_t$ and convolved to generate support feature $F_{supp}$ and query feature $F_{qry}$.
3. Learnable queries $Q \in \mathbb{R}^{N \times C}$ ($N=40$) extract category-relevant information from $F_{supp}$ via cross-attention to obtain $Q'$.
4. $Q'$ further interacts with $F_{qry}$ through cross-attention to generate reference prompt $P'$, which is passed through the frozen SAM Prompt Encoder to produce the final prompt embedding $P$.

$$Q' = \text{SelfAttn}_1(\text{CrossAttn}_1(Q, F_{supp}))$$
$$P' = \text{SelfAttn}_2(\text{CrossAttn}_2(Q', F_{qry}))$$

### Prompt-Semantic Fusion Module

1. The encoded feature $F$ is downsampled and flattened into sequence $F_{seq}$.
2. $F_{seq}$ and prompt embedding $P$ are fused via cross-attention.
3. The spatial dimensions are restored and upsampled, then element-wise multiplied with SAM segmentation mask $M$ to obtain the category-enhanced feature $F^p$.

$$F^p = M \cdot \text{Up}(\text{View}(\text{CrossAttn}(F_{seq}, P)))$$

The final fusion feature is obtained by element-wise addition of the preliminary fusion feature $F_{ref}$ with the infrared and visible prompt features $F_{ir}^p$ and $F_{vis}^p$, which is then fed into the decoder to generate the final fused image.

### Loss & Training

End-to-end training jointly optimizes the fusion loss $\mathcal{L}_{fusion}$ and segmentation loss $\mathcal{L}_{seg}$.

## Key Experimental Results

### Fusion Quality (Three Datasets)

| Dataset | Best Metrics | Specific Performance |
|--------|---------|---------|
| FMB | PSNR/Q_abf/N_abf best | PSNR=63.292, Q_abf=0.719 |
| DroneVehicle | MSE/PSNR/SCD best | PSNR=60.317, SCD=1.552 |
| MSRS | PSNR/N_abf best | PSNR=64.75, N_abf=0.018 |

### Semantic Segmentation (MSRS)

- mIoU=0.7963, best among 8 methods
- Best performance on Car, Curve, Guardrail, and Color Tone categories

### Object Detection (DroneVehicle)

- Overall AP@\[0.5:0.95\] = 0.525, best overall
- Car class AP=0.651, bus class AP=0.521, both best

### Ablation Study (MSRS)

| Configuration | SSIM | SCD | Conclusion |
|------|------|-----|------|
| w/o Prompt | 0.933 | 1.635 | Prompt is critical for structural preservation |
| w/o Seg | 0.939 | 1.636 | Segmentation branch improves fusion quality |
| w/o Vis | 0.915 | 1.681 | Visible branch is indispensable |
| w/o Ir | 0.938 | 1.622 | Infrared branch is indispensable |
| Exchange SQ | 0.924 | 1.659 | Original support/query design is superior |
| **Full Model** | **0.969** | **1.726** | All components contribute synergistically |

## Highlights & Insights

1. **Interactive controllable fusion**: For the first time, mask prompts are introduced in infrared-visible fusion to enable interactive dynamic fusion, allowing users to specify targets of interest.
2. **Fusion-segmentation co-enhancement**: Joint optimization enables fusion quality and segmentation performance to mutually reinforce each other; the fine-tuned SAM branch even surpasses the original SAM model on segmentation.
3. **Robustness to prompt quality**: Even with incomplete or low-quality masks (annotating only partial targets), the fusion results can still effectively highlight target regions.
4. **Versatile prompt sources**: Grounded-SAM can be directly used to generate mask prompts from text, enabling controllable fusion on new datasets without annotated data.

## Limitations & Future Work

1. **Dependency on mask prompt input**: Additional masks are required as guidance, increasing usage complexity; the quality of automated prompt generation pipelines (e.g., text-to-mask) directly affects the final outcome.
2. **Bottleneck of frozen SAM**: Both the SAM image encoder and mask decoder are frozen, limiting adaptation to the infrared modality; lightweight adapter fine-tuning could be considered.
3. **Misclassification in categorization**: The paper is categorized under 3d_vision, whereas it belongs to image fusion / multimodal perception.
4. **Grayscale-only fusion output**: The final fused image is single-channel $I_{\mathcal{F}} \in \mathbb{R}^{1 \times H \times W}$, discarding color information from the visible modality.
5. **Computational overhead not discussed**: Inference speed and GPU memory consumption when using SAM as an auxiliary network are not sufficiently addressed.

## Related Work & Insights

| Method | Characteristics | Limitations |
|------|------|------|
| SeAFusion | Segmentation-driven, joint optimization | Fixed semantic categories, non-controllable |
| PSFusion | High-level visual task driven | Implicit semantic learning, no interaction |
| SDCFusion | Segmentation-driven + depth decomposition | Still constrained by predefined categories |
| LDFusion | CLIP text-guided | Coarse text semantics, difficult fine-grained control |
| **CtrlFuse** | **Mask prompt + SAM fine-tuning** | **Explicit semantic injection, interactive control** |

The fundamental distinction between CtrlFuse and existing task-driven methods lies in transitioning from "fixed-category implicit semantic learning" to "mask-prompt-guided explicit controllable semantic injection," leveraging SAM's strong zero-shot generalization to achieve dynamic fusion over arbitrary semantic targets.

### Inspirations

1. **Transfer of the prompt tuning paradigm**: Introducing the prompt tuning idea from NLP/vision foundation models into low-level image fusion tasks suggests that the "foundation model + prompt" paradigm is generalizable to other low-level vision tasks (denoising, super-resolution, etc.).
2. **Task synergy optimization**: The mutual reinforcement between fusion and segmentation indicates that positive transfer effects among tasks in multi-task joint training warrant further investigation.
3. **Controllability as an evaluation dimension**: Beyond traditional pixel-level metrics, controllability should become an important evaluation dimension for multimodal fusion methods.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to achieve mask-prompt-based interactive controllable fusion in infrared-visible image fusion
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three datasets, three task categories (fusion/segmentation/detection), complete ablation study
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, detailed method description, rich figures and tables
- Value: ⭐⭐⭐⭐ — Introduces a controllable paradigm for multimodal fusion with high practical value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HCLFuse: Revisiting Generative Infrared and Visible Image Fusion Based on Human Cognitive Laws](../../NeurIPS2025/segmentation/revisiting_generative_infrared_and_visible_image_fusion_based_on_human_cognitive.md)
- [\[AAAI 2026\] Text-guided Controllable Diffusion for Realistic Camouflage Images Generation](text-guided_controllable_diffusion_for_realistic_camouflage_images_generation.md)
- [\[AAAI 2026\] Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models](symmetrical_flow_matching_unified_image_generation_segmentation_and_classificati.md)
- [\[AAAI 2026\] Breaking the Stealth-Potency Trade-off in Clean-Image Backdoors with Generative Trigger Optimization](breaking_the_stealth-potency_trade-off_in_clean-image_backdoors_with_generative_.md)
- [\[AAAI 2026\] JoDiffusion: Jointly Diffusing Image with Pixel-Level Annotations for Semantic Segmentation Promotion](jodiffusion_jointly_diffusing_image_with_pixel-level_annotations_for_semantic_se.md)

</div>

<!-- RELATED:END -->
