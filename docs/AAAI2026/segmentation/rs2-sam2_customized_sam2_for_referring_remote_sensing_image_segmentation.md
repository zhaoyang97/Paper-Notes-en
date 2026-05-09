---
title: >-
  [Paper Note] RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation
description: >-
  [AAAI 2026][Segmentation][SAM2] This paper proposes RS2-SAM2, a framework that injects textual information into the SAM2 image encoding process via a Bidirectional Hierarchical Fusion Module (BHFM) and designs a Mask Prompt Generator (MPG) to supply SAM2 with dense prompts, achieving state-of-the-art performance on referring remote sensing image segmentation.
tags:
  - AAAI 2026
  - Segmentation
  - SAM2
  - remote sensing imagery
  - referring segmentation
  - multimodal fusion
  - dense prompts
date: 2026-05-08
content_hash: 3a33460ff26def47
---

# RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation

**Conference**: AAAI 2026
**arXiv**: [2503.07266](https://arxiv.org/abs/2503.07266)
**Code**: [https://github.com/rongfu-dsb/RS2-SAM2](https://github.com/rongfu-dsb/RS2-SAM2)
**Area**: Segmentation
**Keywords**: SAM2, remote sensing imagery, referring segmentation, multimodal fusion, dense prompts

## TL;DR

This paper proposes RS2-SAM2, a framework that injects textual information into the SAM2 image encoding process via a Bidirectional Hierarchical Fusion Module (BHFM) and designs a Mask Prompt Generator (MPG) to supply SAM2 with dense prompts, achieving state-of-the-art performance on referring remote sensing image segmentation.

## Background & Motivation

Referring Remote Sensing Image Segmentation (RRSIS) aims to segment target objects from aerial imagery based on natural language descriptions. Compared to referring segmentation on natural images, remote sensing scenes present unique challenges: **diverse object spatial scales, complex scene contexts, and ambiguous object boundaries**, combined with low foreground–background contrast, making targets difficult to distinguish.

SAM2 performs well on natural image segmentation, yet directly applying it to RRSIS faces **two major bottlenecks**:

**Insufficient visual–language alignment**: SAM2's performance degrades in remote sensing scenarios due to low target discriminability. Existing adaptation methods such as SAM2-Adapter perform only single-modal adjustments, lacking hierarchical visual–language interaction and thus failing to understand textual information at a fine-grained level.

**Absence of text-guided prompt generation**: SAM2 lacks the ability to integrate text prompts. Existing work such as EVF-SAM generates sparse prompts via joint encoding and MLP, but sparse prompts are insufficient for subtle or inconspicuous targets in remote sensing scenes and cannot provide pixel-level precise guidance.

The core ideas of this work are: (1) performing both remote sensing feature adaptation and text alignment during SAM2 image encoding; and (2) generating dense pseudo-mask prompts to replace sparse prompts, supplying pixel-level positional information.

## Method

### Overall Architecture

RS2-SAM2 consists of four components: a **Union Encoder**, a **Bidirectional Hierarchical Fusion Module (BHFM)**, a **Mask Prompt Generator (MPG)**, and **SAM2**. Given a remote sensing image and a text description, the Union Encoder produces aligned visual/textual embeddings and a multimodal CLS token; BHFM is embedded within the SAM2 encoder to fuse text and visual information layer by layer; MPG leverages visual embeddings and the CLS token to generate a pseudo-mask as a dense prompt for SAM2; and the SAM2 decoder ultimately outputs high-precision segmentation masks.

### Key Designs

#### 1. **Union Encoder**

- BEiT-3 is adopted as the Union Encoder to jointly encode visual and textual inputs.
- Images are divided into non-overlapping patches and projected into $P_v \in \mathbb{R}^{N_p \times D}$; text is tokenized via XLM-Roberta.
- The multimodal representation $U_0 = [V_0; T_0] \in \mathbb{R}^{(N_p+N_t+1) \times D}$ is concatenated and passed through multimodal fusion, then decomposed into visual CLS token $V_{cls}$, visual embeddings $V$, and text embeddings $T$.
- **Design Motivation**: Joint encoding achieves early semantic-space alignment between vision and language, providing a strong foundation for subsequent modules.

#### 2. **Bidirectional Hierarchical Fusion Module (BHFM)**

This is the core innovation of the paper, embedded at every layer of the SAM2 image encoder:

- **Dimensionality reduction and cross-attention**: SAM2 image features $F_i$ are projected to a lower dimension via a linear layer, and text features $T_i$ are projected to the matching dimension; **bidirectional cross-attention** is then applied:

$$F_i'' = \text{MHCA}(F_i', T_i') + F_i'$$
$$T_i'' = \text{MHCA}(T_i', F_i') + T_i'$$

- **Weighted text preservation**: To maintain text integrity, text features are blended with a weight coefficient $\alpha_t = 0.2$: $T_{i+1} = (1-\alpha_t)T_i + \alpha_t \cdot \text{Linear}(T_i'')$
- **Visual feature enhancement**: After skip connections, visual features are processed through both an MLP branch and a linear branch, then blended with weight $\alpha_i = 0.5$.
- **Post-encoding high-level guidance**: After encoding, the original text features $T$ further guide visual features $F$ via cross-attention, producing text-guided hierarchical features $F_{en}$.

**Design Motivation**: Layer-wise injection of text information makes SAM2 more sensitive to referred targets; bidirectional interaction allows visual and text features to mutually reinforce each other; hierarchical interaction from global to local facilitates fine-grained understanding.

#### 3. **Mask Prompt Generator (MPG)**

- The multimodal CLS token $V_{cls}$ serves as query, while visual embeddings $V$ serve as key-value for cross-attention computation.
- The interaction result is element-wise multiplied with $V_{cls}$ to further align multimodal tokens with visual information.
- Visual embeddings are reshaped into a 2D feature map; $V_{cls}$ is projected via a linear layer, broadcast to the same spatial size, and element-wise multiplied.
- An MLP generates the pseudo-mask $M_p \in \mathbb{R}^{H_u/p \times W_u/p}$, which is upsampled and fed to SAM2 as a dense prompt.

**Design Motivation**: The visual embeddings produced by joint encoding are already well-aligned with the text embeddings in semantic space. Exploiting this property together with the CLS token enables the generation of high-quality prior masks that provide pixel-level guidance.

### Loss & Training

$$\mathcal{L} = \lambda_{ce}\mathcal{L}_{ce} + \lambda_{dice}\mathcal{L}_{dice} + \lambda_{tbl}\mathcal{L}_{tbl}$$

- $\mathcal{L}_{ce}$: cross-entropy loss ($\lambda_{ce}=1$)
- $\mathcal{L}_{dice}$: DICE loss ($\lambda_{dice}=0.1$)
- $\mathcal{L}_{tbl}$: **Text-guided Boundary Loss** ($\lambda_{tbl}=0.2$), a novel loss proposed in this work:
    - Computes absolute differences between horizontally and vertically adjacent pixels as boundary gradients.
    - Text embeddings are abstracted into a sentence embedding and projected to a scalar via a linear layer, serving as a text-guided boundary weight.
    - MSE is used to measure boundary similarity between the predicted mask and the GT mask under the text-guided weighting.

Training setup: 8 × RTX 4090; 60 epochs on RefSegRS and 40 epochs on RRSIS-D; AdamW optimizer; batch size 1. SAM2 uses SAM2-Hiera-Large pretrained weights; the Union Encoder uses BEiT-3-Large.

## Key Experimental Results

### Main Results

**RefSegRS Dataset (Test)**:

| Method | Pr@0.5 | Pr@0.7 | Pr@0.9 | oIoU | mIoU |
|--------|--------|--------|--------|------|------|
| LAVT | 51.84 | 17.34 | 2.09 | 71.86 | 47.40 |
| RMSIN | 79.20 | 42.98 | 3.25 | 75.72 | 62.58 |
| FIANet | 84.09 | 61.86 | 7.10 | 78.32 | 68.67 |
| **RS2-SAM2** | **84.31** | **70.89** | **21.19** | **80.87** | **73.90** |

**RRSIS-D Dataset (Test)**:

| Method | Pr@0.5 | Pr@0.7 | Pr@0.9 | oIoU | mIoU |
|--------|--------|--------|--------|------|------|
| RMSIN | 74.26 | 55.93 | 24.53 | 77.79 | 64.20 |
| FIANet | 74.46 | 56.31 | 24.13 | 76.91 | 64.01 |
| **RS2-SAM2** | **77.56** | **61.76** | **29.73** | **78.99** | **66.72** |

### Ablation Study

| Configuration | Pr@0.5 | mIoU | oIoU | Note |
|---------------|--------|------|------|------|
| Baseline (SAM2 + Union Encoder) | 35.17 | 36.64 | 55.51 | Baseline |
| + $\mathcal{L}_{tbl}$ | 39.79 | 38.63 | 57.36 | Boundary loss is effective |
| + $\mathcal{L}_{tbl}$ + MPG | 71.00 | 60.20 | 70.89 | Large contribution from mask prompts (+21.57% mIoU) |
| + $\mathcal{L}_{tbl}$ + BHFM | 81.89 | 68.71 | 78.36 | Fusion module contributes most (+30.08% mIoU) |
| + All (RS2-SAM2) | **84.31** | **73.90** | **80.87** | All three components are complementary |

**BHFM Structure Ablation**:

| Configuration | mIoU | Note |
|---------------|------|------|
| Linear (no text interaction) | 68.19 | Adaptation alone is insufficient |
| Uni (unidirectional enhancement) | 70.10 | Lacks feedback |
| **Bi (bidirectional enhancement)** | **73.90** | Optimal |

### Key Findings

- BHFM is the highest-contributing component; used alone it raises mIoU from 38.63% to 68.71%.
- Dense prompts from MPG (+21.57% mIoU) substantially outperform the sparse prompt approach of EVF-SAM.
- Bidirectional interaction outperforms unidirectional interaction (73.90% vs. 70.10%), validating the necessity of mutual visual–text enhancement.
- Both intra-encoding (BL) and post-encoding (BC) BHFM components are indispensable; removing either causes significant degradation.
- The text-guided boundary loss yields particularly notable gains at high-precision thresholds such as Pr@0.9.

## Highlights & Insights

1. The **hierarchical bidirectional fusion design** is distinctive: textual information is injected at every layer of the SAM2 encoder, enabling progressive alignment from global to local rather than a single late fusion.
2. The strategy of **replacing sparse prompts with dense prompts** is especially effective in remote sensing scenarios, where targets are frequently ambiguous and sparse point/box prompts cannot adequately guide segmentation.
3. The **text-guided boundary loss** cleverly leverages textual semantics to weight boundary constraints, directly addressing the low foreground–background contrast characteristic of remote sensing targets.

## Limitations & Future Work

- Two encoders (BEiT-3 and SAM2 Hiera) are required, with input resolutions of 224 and 1024 respectively, resulting in considerable computational overhead.
- Validation is limited to RRSIS datasets; broader remote sensing tasks such as change detection and instance segmentation remain unexplored.
- The Union Encoder is fixed as BEiT-3; alternative multimodal encoders have not been investigated.
- SAM2's memory mechanism is not utilized, precluding video-level remote sensing segmentation.

## Related Work & Insights

- The dense prompt generation approach could be generalized to other segmentation tasks requiring fine-grained guidance, such as medical image segmentation.
- The BHFM design can serve as a general multimodal feature adaptation scheme.
- The text-guided boundary loss concept is applicable to other segmentation scenarios involving ambiguous boundaries.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combined design of hierarchical bidirectional fusion and dense prompt generation is original, though individual components are not entirely novel in isolation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Two datasets, comprehensive ablation studies, evaluation across multiple threshold metrics)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with rich figures and tables)
- Value: ⭐⭐⭐⭐ (Provides an effective adaptation of SAM2 for referring remote sensing image segmentation with significant empirical improvements)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval](../../ICLR2026/segmentation/efficient-sam2_accelerating_sam2_with_object-aware_visual_encoding_and_memory_re.md)
- [\[AAAI 2026\] S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing](s5_scalable_semi-supervised_semantic_segmentation_in_remote_sensing.md)
- [\[NeurIPS 2025\] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](../../NeurIPS2025/segmentation/sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)
- [\[ICCV 2025\] Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild](../../ICCV2025/segmentation/correspondence_as_video_test-time_adaption_on_sam2_for_reference_segmentation_in.md)
- [\[ICCV 2025\] Dynamic Dictionary Learning for Remote Sensing Image Segmentation](../../ICCV2025/segmentation/dynamic_dictionary_learning_for_remote_sensing_image_segmentation.md)

</div>

<!-- RELATED:END -->
