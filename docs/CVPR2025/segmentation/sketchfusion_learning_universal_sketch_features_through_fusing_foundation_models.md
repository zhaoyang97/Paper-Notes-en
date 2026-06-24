---
title: >-
  [Paper Note] SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models
description: >-
  [CVPR 2025][Segmentation][Sketch Feature Representation] Introduces SketchFusion, which dynamically injects CLIP visual features into the denoising process of Stable Diffusion to compensate for SD's high-frequency bias and sketch feature deficiencies. Combined with adaptive multi-scale feature aggregation, it establishes the first universal sketch feature representation in the foundation model era, achieving state-of-the-art (SOTA) performance across four tasks: retrieval…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Sketch Feature Representation"
  - "Foundation Model Fusion"
  - "Stable Diffusion"
  - "CLIP"
  - "Frequency Domain Analysis"
date: 2026-05-08
content_hash: 544fcba4b0c8b45f
---

# SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models

**Conference**: CVPR 2025  
**arXiv**: [2503.14129](https://arxiv.org/abs/2503.14129)  
**Code**: None (Project page to be released)  
**Area**: Segmentation/Sketch Understanding  
**Keywords**: Sketch Feature Representation, Foundation Model Fusion, Stable Diffusion, CLIP, Frequency Domain Analysis

## TL;DR

Introduces SketchFusion, which dynamically injects CLIP visual features into the denoising process of Stable Diffusion to compensate for SD's high-frequency bias and sketch feature deficiencies. Combined with adaptive multi-scale feature aggregation, it establishes the first universal sketch feature representation in the foundation model era, achieving state-of-the-art (SOTA) performance across four tasks: retrieval, recognition, segmentation, and correspondence learning.

## Background & Motivation

Due to its abstract, sparse, and cross-modal nature, sketches require feature representations that are fundamentally different from those of natural images. Although foundation models (such as SD, CLIP, and DINO) perform exceptionally well across various visual tasks, their efficacy in sketch understanding remains under-explored.

The authors reveal two fundamental limitations of SD as a sketch feature extractor through systematic pilot experiments: (1) features extracted by SD from abstract, sparse sketches are far inferior to those from photos, as SD's pre-training is primarily based on natural images; (2) frequency domain analysis indicates that SD's UNet exhibits an inherent high-frequency bias—systematically enhancing high-frequency components (edge details) while suppressing low-frequency components (global semantic structure), which is particularly unfavorable for dense prediction tasks (like segmentation) that require capturing global semantics.

**Key Insight**: SD features possess strong spatial perception but inaccurate semantics, whereas CLIP features are semantically precise but spatially sparse. The two are highly complementary—CLIP provides precisely the low-frequency semantic components that SD lacks.

## Method

### Overall Architecture

SketchFusion keeps the SD and CLIP models frozen while training only three lightweight components: (1) a 1D convolutional layer to inject CLIP visual features into various layers of the SD UNet; (2) a ResNet aggregation network to unify multi-scale features; (3) branch weights to automatically select the optimal feature combination. Different downstream tasks train these components using task-specific losses.

### Key Design 1: CLIP Feature Injection

**Function**: Injects CLIP semantic information into various layers of the SD denoising process to compensate for SD's deficiencies in sketch feature extraction.

**Mechanism**: Patch features $f_\mathbf{v} \in \mathbb{R}^{h/p \times w/p \times d}$ are extracted from the penultimate layer of the CLIP vision encoder, dimensionally adjusted via a learnable 1D convolution $\mathcal{C}(\cdot)$, and added to the intermediate features of each upsampling layer in the SD UNet: $\hat{f}_\mathbf{u}^n = f_\mathbf{u}^n + \mathcal{C}(f_\mathbf{v})$. Dynamic injection is performed simultaneously across all timesteps and layers.

**Design Motivation**: CLIP visual and textual embeddings are inherently aligned, with CLIP visual features providing richer semantic information than text prompts. Multi-layer injection allows SD to leverage semantic guidance from CLIP at all stages of denoising. The 1D convolution only handles dimension adaptation, keeping computational overhead extremely low. PCA analysis confirms that the injected features contain both the high-frequency spatial details of SD and the low-frequency semantic components of CLIP.

### Key Design 2: Dynamic Feature Aggregation

**Function**: Automatically selects the optimal combination of features from different layers of the SD UNet, eliminating the need for manual layer selection.

**Mechanism**: CLIP-enhanced features $\{\hat{f}_\mathbf{u}^n\}_{n=1}^3$ are extracted from the first three upsampling layers of the UNet, unified to the same resolution of $60 \times 60 \times d$ through three ResNet blocks, and then weighted and summed using learnable weights $\{\alpha_n\}$ to obtain the final feature map.

**Design Motivation**: Different layers capture features of varying semantic granularities—shallow layers are detailed (suitable for correspondence learning), while deep layers are abstract (suitable for recognition). Manually selecting the optimal layer requires extensive tuning for different tasks. Automatic weighting allows the model to adaptively determine the contribution of each layer.

### Key Design 3: Unified Multi-Task Adaptation

**Function**: Adapts a single feature extraction framework to four types of tasks: retrieval, recognition, segmentation, and correspondence.

**Mechanism**: Globally pooled features + triplet loss are used for retrieval/recognition; dense features + pixel-wise loss are used for segmentation and correspondence learning. All tasks share the same SD+CLIP feature extractor, with only the injection layer, aggregation network, and branch weights being trained.

**Design Motivation**: Existing methods design specialized architectures for each task category, whereas this work demonstrates the feasibility of a unified feature representation.

### Loss & Training

**Task-Specific**: Retrieval and recognition employ triplet loss; segmentation uses cross-entropy; correspondence learning relies on pixel-wise matching loss. Across all tasks, only the lightweight components are trained while SD and CLIP remain frozen.

## Key Experimental Results

### Category-Level Zero-Shot Sketch-Based Image Retrieval (ZS-SBIR)

| Method | Sketchy mAP@200 | TU-Berlin mAP@all | Quick,Draw! mAP@all |
|------|-----------|-----------|-----------|
| B-CLIP | 0.250 | 0.228 | 0.080 |
| B-SD | 0.558 | 0.510 | 0.179 |
| SD-PL (SOTA) | 0.746 | 0.680 | 0.231 |
| **SketchFusion** | **0.761** (+1.5%) | **0.695** (+1.5%) | **0.242** (+1.1%) |

### Sketch Segmentation

| Method | SketchSeg-150K mIoU |
|------|-----|
| B-SD | 35.72 |
| SD-PL | 47.89 |
| **SketchFusion** | **77.31** (+29.42%) |

### Sketch-Photo Correspondence Learning

| Method | PCK@0.1 |
|------|---------|
| B-SD | 33.12 |
| **SketchFusion** | **54.34** (+21.22%) |

### Key Findings

- The improvement on the segmentation task is the most striking (+29.42%), validating the crucial role of low-frequency semantic compensation for dense prediction tasks.
- Direct fine-tuning of SD+CLIP (B-Finetuning) instead leads to severe degradation (mAP 0.120 vs 0.761), proving the validity of the frozen-backbone + lightweight-injection-layer strategy.
- Simple concatenation of SD and CLIP features (B-SD+CLIP) brings improvements but is far inferior to the injection strategy (0.588 vs 0.761), indicating that injection during the denoising process is much more effective than post-processing fusion.
- Frequency domain analysis clearly illustrates the high-frequency bias of SD and the low-frequency complementarity of CLIP, providing a theoretical foundation for feature fusion.

## Highlights & Insights

1. **Frequency Domain Perspective**: This is the first work to analyze the limitations of SD on sketches from a frequency domain perspective, identifying the high-frequency bias issue and addressing it using CLIP's low-frequency semantics.
2. **Universality**: A single framework and feature representation span across four diverse tasks (retrieval, recognition, segmentation, and correspondence), achieving SOTA performance in all of them.
3. **Efficiency**: By keeping both foundation models frozen and training only the lightweight 1D convolution and aggregation network, it avoids catastrophic forgetting and high fine-tuning costs.

## Limitations & Future Work

- Inference requires running both SD and CLIP large models simultaneously, resulting in relatively high memory and computational overhead.
- Using empty prompts instead of class-specific prompts may limit the utilization of textual semantics.
- Validated only in the sketch domain; universality to other sparse visual inputs (e.g., line drawings, medical images) remains to be explored.
- Based on SD v2.1; newer versions (e.g., SDXL, SD3) might exhibit different frequency domain characteristics.

## Related Work & Insights

- **SD-PL**: The previous SOTA sketch feature method, which uses a single SD model with manual layer selection. This work comprehensively surpasses it through CLIP fusion and adaptive aggregation.
- **Vision Fusion**: Hybrid model methods such as SD+DINO inspired the complementary fusion strategy in this paper.
- **Frequency Domain Analysis**: Analytical tools from classical CV literature are introduced into foundation model analysis, revealing the inherent bias of UNet.

## Rating

⭐⭐⭐⭐ — Deep pilot experimental analysis (discovery of frequency bias), elegant method design (injection instead of fine-tuning), and SOTA results across all four tasks. The +29.42% improvement in segmentation is highly impressive. The analysis of SD feature limitations is also valuable to the broader community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Assessing and Learning Alignment of Unimodal Vision and Language Models (SAIL)](assessing_and_learning_alignment_of_unimodal_vision_and_language_model.md)
- [\[CVPR 2025\] Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video](uni4d_unifying_visual_foundation_models_for_4d_modeling_from_a_single_video.md)
- [\[CVPR 2025\] Universal Domain Adaptation for Semantic Segmentation](universal_domain_adaptation_for_semantic_segmentation.md)
- [\[ECCV 2024\] GiT: Towards Generalist Vision Transformer through Universal Language Interface](../../ECCV2024/segmentation/git_towards_generalist_vision_transformer_through_universal_language_interface.md)
- [\[CVPR 2025\] The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation](the_devil_is_in_low-level_features_for_cross-domain_few-shot_segmentation.md)

</div>

<!-- RELATED:END -->
