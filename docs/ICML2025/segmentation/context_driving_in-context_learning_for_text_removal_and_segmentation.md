---
title: >-
  [Paper Note] ConText: Driving In-context Learning for Text Removal and Segmentation
description: >-
  [ICML 2025][Segmentation][Visual In-context Learning] This work applies the visual in-context learning (V-ICL) paradigm to OCR tasks for the first time. It proposes three key designs: task-chaining prompting, context-aware aggregation (CAA), and self-prompting (SP) strategies. ConText significantly outperforms existing general V-ICL models and task-specific models in text removal and segmentation tasks, achieving improvements of +4.50 PSNR and +3.34% fgIoU, respectively.
tags:
  - "ICML 2025"
  - "Segmentation"
  - "Visual In-context Learning"
  - "Text Segmentation"
  - "Text Removal"
  - "Task-chaining Reasoning"
  - "MAE"
date: 2026-05-08
content_hash: ffef857f8efa65ff
---

# ConText: Driving In-context Learning for Text Removal and Segmentation

**Conference**: ICML 2025  
**arXiv**: [2506.03799](https://arxiv.org/abs/2506.03799)  
**Code**: [https://github.com/Ferenas/ConText](https://github.com/Ferenas/ConText)  
**Area**: Segmentation  
**Keywords**: Visual In-context Learning, Text Segmentation, Text Removal, Task-chaining Reasoning, MAE

## TL;DR

This work applies the visual in-context learning (V-ICL) paradigm to OCR tasks for the first time. It proposes three key designs: task-chaining prompting, context-aware aggregation (CAA), and self-prompting (SP) strategies. ConText significantly outperforms existing general V-ICL models and task-specific models in text removal and segmentation tasks, achieving improvements of +4.50 PSNR and +3.34% fgIoU, respectively.

## Background & Motivation

Visual In-context Learning (V-ICL) is borrowed from the in-context learning (ICL) paradigm in NLP, where the core idea is to guide the model to make predictions on new queries using a few input-output examples (demonstrations) as context. Existing V-ICL methods are mainly based on Masked Autoencoders (MAE), slicing two pairs of image-labels into a grid input and performing masked reconstruction after masking the query label region.

However, existing methods face three core problems:

**Single-task centric**: Models can only perform inference for a single task at a time, failing to leverage the intrinsic correlation between tasks to enhance the ICL capability.

**Context-free fusion**: The feature fusion of input-output is only linearly superimposed within a single pair, lacking context-aware information interaction across demonstrations.

**Assumed context homogeneity**: Traditional methods require demonstrations and queries to have the exact same category of objects (e.g., airplane to airplane). However, the high degree of heterogeneity in fonts, styles, and languages in text recognition makes it very difficult to search for "same-category" demonstrations.

The core motivation of this paper is: **Can multiple related visual tasks be chained into an integrated prompt, similar to Chain-of-Thought in LLMs, to enhance the inference capability of V-ICL through complementary logic between tasks?**

## Method

### Overall Architecture

ConText is based on the MAE architecture of ViT-L + decoder (with SegGPT as the baseline), enhancing the baseline through three specifically designed modules:

1. **Task Chaining**: Extends the original `image-label` pair to an `image-removal-segmentation` triplet.
2. **Context-Aware Aggregation (CAA)**: Inject patterns of the prompt into the query representation using cross-attention.
3. **Self-Prompting (SP)**: Employs identical samples as both demonstration and query with a certain probability to maintain the in-context learnability.

### Key Designs

#### 1. Task Chaining

Core observation: There is an implicit task-level logical correlation between text segmentation and text removal—the segmentation mask should theoretically correspond to the visual difference between the original image and the text-removed image.

- Extends the original prompt from `[Image, Label]` to `[Image, Removal, Segmentation]`.
- New composite input: $\mathbf{F} = [\mathbf{F}_I, \mathbf{F}_O, \mathbf{F}_Y] \in \mathbb{R}^{3 \times 2h \times 3w}$.
- During training, mask reconstruction is simultaneously performed on both removal and segmentation labels, maintaining spatial consistency between their masks: $\mathbf{M}_O = \mathbf{M}_Y$.
- Reconstructs labels for both tasks separately using a weight-sharing decoder.
- During inference, the removal and segmentation labels of the query are masked to generate all task outputs in an end-to-end manner.

Pilot experiments show: introducing the removal label to assist segmentation on Painter (Rem-based Seg.) improves TotalText fgIoU from 60.60% to 63.22% (+2.62%); conversely, introducing segmentation to assist removal (Seg-based Rem.) improves SCUT-Ens PSNR from 36.15 to 37.02 (+0.87), validating the effectiveness of the task chain.

#### 2. Context-Aware Aggregation (CAA)

Inspired by mechanistic studies of ICL (where the label position progressively extracts demonstration information in shallow layers, and the final label absorbs all information), a two-step fusion is designed:

**Step 1: Context-free fusion** (similar to the baseline's linear fusion)

$$\tilde{\mathbf{F}}_1 = [\mathbf{I}_i + \tilde{\mathbf{O}}_i + \alpha_y \tilde{\mathbf{Y}}_i, \quad \mathbf{I}_j + \alpha_o \tilde{\mathbf{O}}_j + \tilde{\mathbf{Y}}_j]$$

where $\alpha_o$ and $\alpha_y$ are learnable weights controlling the contributions of removal and segmentation features. This step focuses on the inter-task fusion within the demonstration.

**Step 2: Cross-demonstration context fusion (CAA)**

$$\tilde{\mathbf{F}}_2 = [\phi(\tilde{\mathbf{F}}_{O_i}, \tilde{\mathbf{F}}_j), \phi(\tilde{\mathbf{F}}_{Y_i}, \tilde{\mathbf{F}}_j), \phi(\tilde{\mathbf{F}}_{O_j}, \tilde{\mathbf{F}}_i), \phi(\tilde{\mathbf{F}}_{Y_j}, \tilde{\mathbf{F}}_i)]$$

where $\phi(\text{query}, \text{key/value})$ represents a shared cross-attention mapping. The final label representation is $\tilde{\mathbf{F}}_1 + \tilde{\mathbf{F}}_2$, where each label feature explicitly extracts information from the other demonstration, achieving a more comprehensive in-context understanding.

#### 3. Self-Prompting

To address the visual heterogeneity in text recognition (massive variations in font, style, and language, making it difficult to find "homogeneous" demonstrations), the model is trained with a probability of $p=0.2$ using the identical input-output pair as both the demonstration and the query ($\tilde{\mathbf{F}}_i = \tilde{\mathbf{F}}_j$).

This forces the model to maintain task-specific inference capabilities while generalizing in-context learning, preventing degradation into a specialist that requires no demonstration. However, excessively high self-prompting probability (e.g., 0.6) degrades performance due to reduced demonstration diversity.

### Loss & Training

- **Reconstruction Loss**: smooth-L1 loss is applied separately to text removal and segmentation labels for pixel reconstruction, with the removal loss weight set to 0.3.
- **Pixel-level Supervision**: Introduces an additional cross-entropy-based pixel-level supervision $\mathcal{L}_{pix}$ (weight: 1.0) via a lightweight decoder (two convolutional layers) used only during training to enhance fine-grained character-level recognition.
- **Mask Ratio**: 85% (validated as optimal by ablation study).
- **Optimizer**: AdamW, learning rate 0.0001, weight decay 0.1, cosine schedule.
- **Training Scale**: 16×A100 (80GB), batch size 64 (2 per GPU + 2 gradient accumulation steps), 150 epochs.
- **Two Training Modes**:
    - ConText: Trained only on HierText (for ablation).
    - ConText$_V$: HierText + TextSeg + TotalText + SCUT-EnsText (for comparison with specialists).

## Key Experimental Results

### Main Results

**Comparison with general V-ICL models** (trained only on HierText):

| Method | Text Removal PSNR (Avg) | Text Removal FID (Avg) | Text Segmentation fgIoU (Avg) |
|------|---------------------|--------------------|-----------------------|
| MAE-VQGAN | 28.30 | 41.70 | 6.97% |
| Painter Rem.+Seg. | 32.34 | 24.73 | 67.16% |
| SegGPT Rem.+Seg. | 33.05 | 24.34 | 68.34% |
| **ConText** | **38.36** | **11.04** | **76.77%** |

**Comparison with text segmentation specialists** (ConText$_V$):

| Method | HierText fgIoU | TotalText fgIoU | *FST fgIoU | TextSeg fgIoU |
|------|----------------|-----------------|-----------|-------------|
| Hi-SAM | 77.76 | 84.59 | - | 88.77 |
| EAFormer | - | 82.73 | 72.63 | 88.06 |
| UPOCR | - | - | - | 88.76 |
| **ConText$_V$** | **81.21** | **85.19** | **75.90** | **89.74** |

**Comparison with text removal specialists** (SCUT-EnsText):

| Method | PSNR↑ | MSSIM↑ | MSE↓ | FID↓ |
|------|-------|--------|------|------|
| ViTEraser | 36.87 | 97.51% | 0.05 | 10.15 |
| UPOCR | 37.14 | 97.62% | 0.04 | 10.47 |
| **ConText$_V$** | **40.83** | **98.76%** | **0.03** | 11.63 |

### Ablation Study

**Effectiveness of each module** (TotalText segmentation / SCUT-Ens removal, RS = Random demonstration):

| Configuration | Seg. fgIoU (RS / GT Gap) | Rem. PSNR (RS / GT Gap) | Description |
|------|--------------------------|-------------------------|------|
| Baseline (SegGPT) | 68.53 / +1.57 | 34.42 / +0.17 | Multi-task fine-tuning SegGPT |
| + Linear Fusion | 72.14 / +1.08 | 35.75 / +0.41 | Context-free fusion, +3.61% |
| + CAA | 79.14 / +0.65 | 38.59 / +0.37 | Cross-attention fusion, +10.61% |
| + CAA + SP-0.2 | 78.02 / +3.98 | 37.67 / +1.42 | Best trade-off, significantly restores ICL capability |
| + CAA + SP-0.6 | 77.14 / +5.83 | 36.12 / +2.13 | Stronger ICL capability but degraded task performance |

**Computational Cost**:

| Configuration | Training Time / epoch | Inference Time | FLOPs Increment |
|------|--------------|---------|----------|
| Baseline | 3.8 min | 0.09 sec | 666.76G |
| + SP | 4.2 min | 0.09 sec | 0% |
| + CAA | 4.6 min | 0.12 sec | +2% |
| + CAA + SP | 4.8 min | 0.12 sec | +2% |

### Key Findings

1. **Task-chaining prompting is effective**: Pilot experiments demonstrate that leveraging underlying task logic (segmentation ↔ removal) significantly improves ICL inference, with Rem-based Seg. gaining +2.62% fgIoU.
2. **CAA is the primary performance driver**: Moving from baseline to +CAA, segmentation fgIoU increases by +10.61% and removal PSNR improves by +4.17, although it leads to the model collapsing into a specialist (the gap between GT and RS is very small).
3. **SP restores in-context learnability**: SP-0.2 widens the RS-GT performance gap from +0.65 to +3.98 (in segmentation), proving that the model can learn information from the demonstration.
4. **Training-free prompting capability**: On the unseen PromptText dataset, ConText demonstrates the ability to comprehend hand-marked visual prompts (circles, boxes, scribbles), significantly outperforming all specialists and general V-ICL models.
5. **Multi-demonstration and dual-inference**: 5-shot inference brings a +0.62% fgIoU and +0.78 PSNR improvement, while dual-inference yields +0.25% fgIoU and +0.44 PSNR.

## Highlights & Insights

1. **Representing Chain-of-Thought in vision**: Transfers the Chain-of-Thought reasoning from NLP to vision tasks. Providing rich intermediate representations via the image→removal→segmentation task chain fundamentally enhances the V-ICL paradigm.
2. **Quantifying in-context learnability**: Proposes using the performance gap between RS and GT demonstrations to measure the in-context learnability of the model. ConText achieves an in-context gap of +5.39% fgIoU on text segmentation, far exceeding other baselines.
3. **Lightweight yet critical designs**: CAA adds only 2% FLOPs, and SP introduces zero overhead during inference, but their combination delivers a qualitative leap.
4. **First V-ICL framework for OCR**: Opens up a new direction for V-ICL application in fine-grained text recognition tasks.

## Limitations & Future Work

1. **Dependence on segmentation labels**: For datasets lacking human-annotated removal labels, the framework relies on ViTEraser to generate pseudo-labels, which introduces noise.
2. **Synthetic-to-real domain gap**: Underperforms some specialists on the SCUT-Syn synthetic dataset, exposing domain transfer problems.
3. **Limited gains from multi-demonstration**: 5-shot only outperforms 1-shot by 0.62%, indicating that searching for "identical-category" demonstrations for heterogeneous texts remains a challenge to be explored.
4. **Sensitivity of the self-prompting probability**: There is a clear trade-off between task performance and ICL capability in SP-0.2 and SP-0.6. Automating the selection of the optimal value remains unsolved.
5. **Deficiency in FID metrics**: On SCUT-EnsText, ConText$_V$'s FID (11.63) is slightly higher than ViTEraser's (10.15), leaving room to improve the perceptual consistency of generation quality.

## Related Work & Insights

- **V-ICL Baselines**: MAE-VQGAN, Painter, and SegGPT comprise the mainstream paradigm of composited-prompting V-ICL.
- **Task-chaining Inspiration**: Inspired by Chain-of-Thought (Wei et al., 2022) in NLP and Chain-of-Thought reasoning without prompting (Wang & Zhou, 2024).
- **ICL Mechanism**: The hypothesis of label positions serving as informational anchors (Wang et al., 2023a; Yu & Ananiadou, 2024) directly inspired the design of CAA.
- **Text Segmentation SOTA**: Hi-SAM for hierarchical text segmentation based on SAM, and EAFormer with edge-aware Transformer.
- **Text Removal SOTA**: ViTEraser's one-stage ViT solution, and UPOCR's unified pixel-level OCR interface.
- **Future Inspiration**: The task-chaining concept can be extended to other visual task pairs with latent logical correlations (e.g., detection ↔ segmentation, depth ↔ normal estimation).

## Rating

| Dimension | Rating | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | Applies V-ICL to OCR for the first time; task-chaining prompting is a novel visual CoT. |
| Technical Depth | ⭐⭐⭐⭐ | CAA and SP designs are theoretically grounded, with comprehensive ablation. |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | 6 datasets, multi-dimensional metrics, rich ablation and analysis. |
| Writing Quality | ⭐⭐⭐⭐ | Clear narrative, progressively building enhancements from baseline characteristics. |
| Practical Value | ⭐⭐⭐⭐ | Open-source code, end-to-end multi-tasking, with only +2% extra FLOPs. |
| Overall Rating | ⭐⭐⭐⭐ | A solid piece of work that successfully extends V-ICL to the OCR field and achieves significant SOTA results. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CAVIS: Context-Aware Video Instance Segmentation](../../ICCV2025/segmentation/cavis_context-aware_video_instance_segmentation.md)
- [\[CVPR 2026\] INSID3: Training-Free In-Context Segmentation with DINOv3](../../CVPR2026/segmentation/insid3_training-free_in-context_segmentation_with_dinov3.md)
- [\[CVPR 2025\] The Power of Context: How Multimodality Improves Image Super-Resolution](../../CVPR2025/segmentation/the_power_of_context_how_multimodality_improves_image_super-resolution.md)
- [\[ICCV 2025\] SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation](../../ICCV2025/segmentation/score_scene_context_matters_in_openvocabulary_remote_sensing.md)
- [\[CVPR 2026\] Annotation-Efficient Coreset Selection for Context-dependent Segmentation](../../CVPR2026/segmentation/annotation-efficient_coreset_selection_for_context-dependent_segmentation.md)

</div>

<!-- RELATED:END -->
