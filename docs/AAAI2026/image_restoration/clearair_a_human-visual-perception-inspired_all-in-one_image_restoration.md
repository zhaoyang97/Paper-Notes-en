---
title: >-
  [Paper Note] ClearAIR: A Human-Visual-Perception-Inspired All-in-One Image Restoration
description: >-
  [AAAI 2026][Image Restoration][All-in-One Image Restoration] Inspired by human visual perception (HVP), this paper proposes ClearAIR, a coarse-to-fine unified image restoration framework that progressively recovers image…
tags:
  - "AAAI 2026"
  - "Image Restoration"
  - "All-in-One Image Restoration"
  - "Human Visual Perception"
  - "MLLM-based IQA"
  - "Semantic Guidance"
  - "Self-supervised Learning"
date: 2026-05-08
content_hash: acc611a7d5926eb2
---

# ClearAIR: A Human-Visual-Perception-Inspired All-in-One Image Restoration

**Conference**: AAAI 2026
**arXiv**: [2601.02763](https://arxiv.org/abs/2601.02763)
**Code**: Not released
**Area**: Image Restoration
**Keywords**: All-in-One Image Restoration, Human Visual Perception, MLLM-based IQA, Semantic Guidance, Self-supervised Learning

## TL;DR

Inspired by human visual perception (HVP), this paper proposes ClearAIR, a coarse-to-fine unified image restoration framework that progressively recovers image quality through four stages — MLLM-based quality assessment → semantic region perception → degradation type identification → internal clue reuse — achieving state-of-the-art performance across multiple degradation tasks.

## Background & Motivation

**Poor generalization of task-specific models**: Early image restoration methods design dedicated networks for individual degradations (denoising, dehazing, deraining, etc.), making cross-task generalization infeasible and incurring high deployment costs.

**General models still require multiple instances**: Although general restoration models such as NAFNet and Restormer can handle multiple degradation types, each degradation still requires a separate model instance, leading to complex inference pipelines.

**Existing AiOIR methods overlook spatially non-uniform degradation**: All-in-One methods such as AirNet and PromptIR apply uniform processing strategies over the entire image, ignoring spatial variations in degradation distribution and severity.

**Texture complexity affects restoration difficulty**: Even when degradation is uniformly distributed, flat regions and complex texture regions present significantly different restoration challenges, and a uniform strategy leads to over-smoothing or artifacts.

**Lack of hierarchical perceptual mechanisms**: Human vision processes scenes globally before locally, whereas existing methods lack a progressive perceptual pipeline from global structure to local detail.

**Insufficient fine detail recovery**: Existing methods remain limited in recovering fine-grained textures and fail to exploit the intrinsic structural information within images.

## Method

### Overall Architecture

ClearAIR emulates the hierarchical processing pipeline of human visual perception and consists of four core components:

1. **MLLM-based IQA** (global quality assessment) → 2. **SGU** (region-level semantic perception) → 3. **Task Identifier** (degradation type recognition) → 4. **ICRM** (internal clue reuse)

The restoration backbone is Restormer, with Prompt Transformer Block counts of [3, 5, 6, 8] across four levels and channel dimensions of [48, 96, 192, 384].

### Key Design 1: MLLM-based Overall Assessment

- Employs DeQA (an MLLM-based image quality assessment model) as the global quality perceiver.
- A visual encoder encodes the degraded image into visual tokens, which are compressed by a vision abstractor and fused with text tokens before being fed into the MLLM.
- The hidden state $\mathcal{Q}$ from the layer preceding the "quality level" token is extracted as the quality representation.
- A Quality Guidance Module (QGM) injects this representation into the restoration backbone via affine transformation: $\mathbf{X}_{out} = \mathbf{X}_{in} \odot \text{Linear}(\mathbf{F}_q) + \text{Linear}(\mathbf{F}_q)$

### Key Design 2: Semantic Guidance Unit (SGU) + Task Identifier

**Region-level perception (SGU)**:
- Pretrained SAM2 is applied to the degraded image to generate $N_m$ binary semantic masks.
- Mask Average Pooling (MAP) computes the mean features within each mask region and broadcasts them back to the corresponding positions, yielding a semantic prior $\mathbf{F}_{sem}$.
- A mask dropout strategy is introduced during training, randomly removing a subset of masks and merging them into the background to improve robustness.
- Semantic features interact with the backbone via Semantic Cross-Attention (SCA).

**Degradation recognition (Task Identifier)**:
- DA-CLIP generates a content embedding $\mathbf{F}_c \in \mathbb{R}^{1 \times 512}$ and a degradation embedding $\mathbf{F}_d \in \mathbb{R}^{1 \times 512}$.
- The degradation embedding is passed through an MLP + Softmax to produce weighted combinations of a learnable prompt set $\mathcal{P}$, yielding the degradation prompt $\mathbf{F}_p$.
- A Degradation-Aware Module (DAM) performs content-aware enhancement via cross-attention using $\mathbf{F}_c$, while $\mathbf{F}_p$ generates a degradation mask for spatial feature modulation.

### Key Design 3: Internal Clue Reuse Mechanism (ICRM)

- Weak and strong augmentations are sequentially applied to the restored output $\mathbf{I}_r$.
- The L2 distance between the weakly and strongly augmented outputs serves as an internal consistency loss: $\mathcal{L}_{inter} = \gamma \cdot \|\mathbf{I}_r^w - \mathbf{I}_r^s\|_2^2$
- This self-supervised approach mines intrinsic structural information within the image to enhance detail recovery.
- No additional annotations are required; the mechanism exploits the internal statistics of the image itself.

### Loss & Training

- Total loss: $\mathcal{L}_{total} = \mathcal{L}_1 + \alpha \cdot \mathcal{L}_{inter}$, where $\alpha = 0.25$ and $\gamma = 0.05$.
- Optimizer: AdamW ($\beta_1=0.9$, $\beta_2=0.999$), learning rate $2 \times 10^{-4}$, batch size 4.
- Training for 300K iterations; inputs randomly cropped to 256×256 with random horizontal/vertical flipping.
- Hardware: NVIDIA RTX 4090.

## Key Experimental Results

### Three Degradations (Denoising + Dehazing + Deraining)

| Method | Params | SOTS (PSNR/SSIM) | Rain100L | BSD68 σ=15 | BSD68 σ=25 | BSD68 σ=50 | Avg |
|---|---|---|---|---|---|---|---|
| PromptIR | 36M | 30.58/.974 | 36.37/.972 | 33.98/.933 | 31.31/.888 | 28.06/.799 | 32.06/.913 |
| AdaIR | 29M | 31.06/.980 | 38.64/.983 | 34.12/.934 | 31.45/.892 | 28.19/.802 | 32.69/.918 |
| VLU-Net | 35M | 30.71/.980 | 38.93/.984 | 34.13/.935 | 31.48/.892 | 28.23/.804 | 32.70/.919 |
| **ClearAIR** | **31M** | **31.08/.981** | 38.61/.984 | **34.18/.935** | 31.50/.891 | **28.31/.804** | **32.74/.919** |

### Five Degradations (+ Deblurring + Low-light Enhancement)

| Method | SOTS | Rain100L | BSD68 σ=25 | GoPro | LOLv1 | Avg |
|---|---|---|---|---|---|---|
| Perceive-IR | 28.19/.964 | 37.25/.977 | 31.44/.887 | 29.46/.886 | 22.81/.833 | 29.84/.909 |
| AdaIR | 30.53/.978 | 38.02/.981 | 31.35/.888 | 28.12/.858 | 23.00/.845 | 30.20/.910 |
| **ClearAIR** | 30.12/.978 | 38.20/.982 | **31.53/.888** | **29.67/.887** | 22.83/.846 | **30.45/.916** |

### All-Weather (Snow + Rain-Haze + Raindrops)

| Method | Snow100K-S | Snow100K-L | Outdoor-Rain | RainDrop | Avg |
|---|---|---|---|---|---|
| Histoformer | 37.41/.966 | 32.16/.926 | 32.08/.939 | 33.06/.944 | 33.68/.945 |
| **ClearAIR** | **37.79/.967** | **32.53/.932** | **32.45/.941** | 32.82/.942 | **33.90/.946** |

### Composited Degradations (CDD-11)

ClearAIR achieves 29.34 dB / 0.886 SSIM, surpassing OneRestore (28.72 dB) by 0.62 dB.

### Ablation Study

- Perception order: "How-Where-What" (ours) achieves the best performance at 38.21 dB, outperforming "What-How-Where" (38.04) and "Where-What-How" (37.89).
- Component contributions: Removing any one of IQA, SGU, TI, or ICRM leads to performance degradation; all four components together achieve the best result of 38.21/0.986.

## Highlights & Insights

1. **Novel HVP-inspired progressive design**: The four-stage pipeline adapts the human "global-before-local" visual perception principle to AiOIR, yielding a logically coherent framework.
2. **MLLM-driven quality assessment**: This work is among the first to incorporate MLLM-based IQA as a global prior for image restoration, leveraging cross-modal understanding to enrich degradation representations.
3. **Region-level adaptive processing**: The combination of SGU and Task Identifier enables differentiated handling of spatially non-uniform degradation, addressing a fundamental limitation of uniform processing strategies.
4. **Annotation-free self-supervised detail recovery via ICRM**: The mechanism cleverly exploits intrinsic image statistics, using augmentation consistency constraints to enhance texture recovery without additional labels.
5. **Comprehensive SOTA across four AiOIR settings**: With a moderate parameter count of 31M, ClearAIR consistently outperforms recent methods including AdaIR and VLU-Net.

## Limitations & Future Work

1. **Inference efficiency concerns**: The framework incorporates three large pretrained models — DeQA, SAM2, and DA-CLIP — which likely incur substantial inference overhead and latency; the paper does not report inference speed.
2. **Heavy reliance on pretrained models**: The framework's performance is strongly dependent on the quality of DeQA, SAM2, and DA-CLIP, and their behavior under severe degradation is not sufficiently discussed.
3. **Limited gain from ICRM**: In the ablation study, removing ICRM results in only a 0.18 dB drop (38.03→38.21), suggesting that the design complexity is not fully justified by the performance gain.
4. **Moderate performance on low-light enhancement**: In the five-degradation setting, ClearAIR achieves 22.83 dB on LOLv1, falling short of AdaIR's 23.00 dB, indicating limited benefit from global quality guidance for low-light scenarios.
5. **Lack of large-scale real-world evaluation**: Validation is primarily conducted on synthetic datasets; generalization to real-world degradation scenarios remains to be further verified.

## Related Work & Insights

- **AirNet** (CVPR'22): Contrastive learning encodes degradation representations; ClearAIR extends this with region-level perception.
- **PromptIR** (NeurIPS'23): Visual prompts guide multi-degradation processing; ClearAIR generates degradation prompts explicitly via DA-CLIP rather than relying purely on learnable parameters.
- **Perceive-IR** (TIP'25): Jointly perceives degradation type and severity; ClearAIR further introduces MLLM-based global assessment and semantic region segmentation.
- **AdaIR** (ICLR'25): Adaptive image restoration; ClearAIR surpasses it by 0.25 dB in average PSNR on the five-degradation setting.
- **VLU-Net** (CVPR'25): Vision-language unified restoration; ClearAIR outperforms it by 0.04 dB on three degradations with fewer parameters.
- **OneRestore** (ECCV'24): Composite degradation restoration; ClearAIR surpasses it by 0.62 dB on CDD-11.

## Rating

- Novelty: ⭐⭐⭐⭐ — The HVP-inspired four-stage framework is conceptually novel, and incorporating MLLM-based IQA into restoration is an innovative contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four AiOIR settings, ablation studies, and qualitative comparisons provide broad coverage.
- Writing Quality: ⭐⭐⭐⭐ — The paper is well-structured, the HVP analogy is consistently maintained throughout, and mathematical expressions are presented rigorously.
- Value: ⭐⭐⭐⭐ — The work advances the research direction of region-level adaptive perception in the AiOIR field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](../../ICCV2025/image_restoration/eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)
- [\[CVPR 2026\] Bridging the Perception Gap in Image Super-Resolution Evaluation](../../CVPR2026/image_restoration/bridging_the_perception_gap_in_image_super-resolution_evaluation.md)
- [\[AAAI 2026\] Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration](clear_nights_ahead_towards_multi-weather_nighttime_image_res.md)
- [\[ACL 2026\] Diffusion-CAM: Faithful Visual Explanations for dMLLMs](../../ACL2026/image_restoration/diffusion-cam_faithful_visual_explanations_for_dmllms.md)
- [\[AAAI 2026\] ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement](iclr_inter-chrominance_and_luminance_interaction_for_natural_color_restoration_i.md)

</div>

<!-- RELATED:END -->
