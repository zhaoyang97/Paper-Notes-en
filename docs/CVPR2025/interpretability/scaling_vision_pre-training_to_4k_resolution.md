---
title: >-
  [Paper Note] Scaling Vision Pre-Training to 4K Resolution
description: >-
  [CVPR 2025][Interpretability][High-Resolution Pre-Training] This paper proposes PS3 (Pre-training with Scale-Selective Scaling), which scales CLIP-style vision pre-training to 4K resolution with near-constant computational overhead by replacing global image contrastive learning with localized region and local caption contrastive learning. Combined with top-down/bottom-up patch selection mechanisms, the VILA-HD multimodal large language model is constructed…
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "High-Resolution Pre-Training"
  - "Vision Encoder"
  - "Adaptive Patch Selection"
  - "Contrastive Learning"
  - "Multimodal Large Language Models"
date: 2026-05-08
content_hash: 9bea0e56c8a1fdd6
---

# Scaling Vision Pre-Training to 4K Resolution

**Conference**: CVPR 2025  
**arXiv**: [2503.19903](https://arxiv.org/abs/2503.19903)  
**Code**: [https://nvlabs.github.io/PS3](https://nvlabs.github.io/PS3)  
**Area**: Interpretability  
**Keywords**: High-Resolution Pre-Training, Vision Encoder, Adaptive Patch Selection, Contrastive Learning, Multimodal Large Language Models

## TL;DR

This paper proposes PS3 (Pre-training with Scale-Selective Scaling), which scales CLIP-style vision pre-training to 4K resolution with near-constant computational overhead by replacing global image contrastive learning with localized region and local caption contrastive learning. Combined with top-down/bottom-up patch selection mechanisms, the VILA-HD multimodal large language model is constructed, significantly outperforming GPT-4o and Qwen2.5-VL on high-resolution perception tasks.

## Background & Motivation

Modern vision models (CLIP, SigLIP) are pre-trained only at low resolutions (e.g., $378 \times 378$) and cannot perceive fine-grained visual details, whereas daily tasks (such as recognizing road signs while driving) demand high-resolution perception. The fundamental bottleneck is **computational cost**: the computation of ViT grows quadratically with the number of pixels (the fourth power of resolution). Existing methods (AnyRes, S²) attempt to run pre-trained low-resolution models at higher resolutions in a training-free manner, but this prevents the model from learning high-quality high-resolution representations from large-scale pre-training data. **Key Challenge**: The information demand of high-resolution pre-training versus the explosive growth of computational costs. The **Key Insight** of this work comes from the top-down selection mechanism of human vision: **there is no need to digest the entire high-resolution image; instead, one only needs to process local regions at high resolution and align them with local descriptions.** This decouples the region size from the full-image resolution, keeping the cost nearly constant.

## Method

### Overall Architecture

PS3 consists of three stages: Stage 1 encodes the global low-resolution features using a ViT ($378 \times 378 \rightarrow 27 \times 27$ tokens); Stage 2 selects salient or text-relevant local regions based on the low-resolution features and lightweight high-resolution auxiliary features; Stage 3 uses the same ViT to process multi-scale high-resolution patches of the selected regions, injecting global context via the KV cache of Stage 1. Pre-training utilizes 75M high-resolution images and 282M local caption-bounding box pairs, and is jointly optimized via localized contrastive loss and bounding box supervision.

### Key Designs

1. **Localized Contrastive Loss**:
    - **Function**: Learning language-aligned fine-grained visual representations at 4K resolution with near-constant cost.
    - **Mechanism**: Instead of contrasting the full image with a global caption, the model extracts high-resolution features from local regions and contrasts them with detailed local captions. During pre-training, at most 2560 high-resolution patches are selected. Global low-resolution contrastive learning is mixed to preserve global feature quality.
    - **Design Motivation**: To recognize the text on a stop sign, the model only needs to process the local region near the sign and align it with the text "stop sign" instead of the whole 4K image. This reduces the pre-training computation by 79 times compared to SigLIP's global contrast.

2. **Top-down / Bottom-up Patch Selection**:
    - **Function**: Selecting local regions that require high-resolution processing based on text prompts (top-down) or image saliency (bottom-up).
    - **Mechanism**: Cosine similarity is computed between low-resolution visual features and text embeddings (or a learnable vector) to obtain selection scores. Additionally, a lightweight ConvNeXt (3 layers) is used to extract auxiliary high-resolution selection scores at 1512 resolution, which are then fused with the former. During training, bounding box ground truths are used for segmentation supervision of the selection scores (cross-entropy + DICE loss).
    - **Design Motivation**: Low-resolution features cannot localize fine-grained details, which is compensated for by the auxiliary high-resolution encoder. Top-down selection enables the MLLM to selectively process relevant regions based on the user's question, while bottom-up selection identifies naturally salient regions in the image.

3. **High-Resolution Multi-Scale Feature Extraction + KV Cache**:
    - **Function**: Processing multi-scale patches of selected regions while injecting global context.
    - **Mechanism**: High-resolution images are resized to three predefined scales (756/1512/3780), and top-k patches are selected from each scale proportionally to the resolution. Scale-aware positional embeddings are added to allow the ViT to distinguish tokens of different scales at the same spatial location. The KV cache from Stage 1 is reused in self-attention to provide global context.
    - **Design Motivation**: Multi-scale processing ensures visual information across different granularities is captured. The KV cache prevents the localized high-resolution encoding from losing global semantic context, analogous to context reuse in LLMs.

### Loss & Training

- **Contrastive Loss**: Sigmoid contrastive loss from SigLIP is used, with both the ViT and text encoder initialized from SigLIP-SO400M.
- **Selection Score Supervision**: Position-wise cross-entropy + DICE loss, with the ground truth generated from bounding boxes.
- **Using GT Selection Scores during Training (Teacher Forcing)**: Avoids selecting irrelevant regions due to inaccurate model predictions.
- **Pooling Only Tokens within the Box**: Prevents aligning irrelevant features outside the box with the text.
- **Avoiding Intra-image Contrast**: Ensures each image appears only once in the same batch to prevent mismatching of different regions within the same image.
- **VILA-HD Micro-tuning Data**: 500k patch selection fine-tuning data + 225k synthetic high-resolution VQA data (pasting low-resolution images onto 4K backgrounds).

## Key Experimental Results

### Main Results (Average of 7 High-Resolution Sensitive Benchmarks)

| Vision Encoder | Max Resolution | HR Token Count | TextVQA | DocVQA | OCRBench | 7-avg |
|---|---|---|---|---|---|---|
| SigLIP | 378 | 0 | 62.3 | 51.9 | 387 | 49.9 |
| AnyRes | 1512 | 3136 | 67.4 | 67.9 | 468 | 56.3 |
| S² | 1512 | 2916 | 66.1 | 78.3 | 526 | 60.8 |
| **PS3** | 1512 | 3645 | **69.3** | **79.4** | **534** | **63.2** |
| **PS3** | 3780 | 3840 | **69.8** | **79.1** | **543** | **63.9** |

PS3 outperforms S² by 2.4% and AnyRes by 6.9%. Furthermore, PS3 can scale to 4K whereas AnyRes and S² cannot.

### 4KPro Benchmark Comparison with SOTA

| Model | Selection Ratio | Latency | Acc |
|---|---|---|---|
| GPT-4o | - | - | 59.7 |
| Qwen2-VL-7B | - | 3.61s | 71.0 |
| Qwen2.5-VL-7B | - | 2.98s | 68.3 |
| **VILA-HD-4K** | 18% | 1.22s | 71.0 |
| **VILA-HD-4K** | 35% | 1.78s | **75.8** |

VILA-HD-4K outperforms GPT-4o by 16.1% and Qwen2.5-VL by 7.5%, while being 1.67x faster.

### Key Findings

- **Constant-Cost Scaling**: Scaling from 756 to 4K is possible by selecting a fixed number of patches, where 1512 $\rightarrow$ 4K still yields a $+3.1\%$ improvement.
- **Test-Time Scaling**: Training with a $20\%$ selection ratio and testing with a $44\%$ selection ratio provides an additional $1.2\%$ improvement.
- **Existing Benchmarks Do Not Require 4K**: Analysis reveals that the minimum recognizable resolution (MRR) of DocVQA and others is only about 1K. 4KPro is the first benchmark that genuinely requires 4K resolution.
- In the comparison of vision encoders, PS3 outperforms SOTA encoders such as SigLIP2 and Perception Encoder.

## Highlights & Insights

- **A New Dimension of Scaling Laws**: Revealing the scaling laws of resolution $\rightarrow$ performance in vision pre-training, as well as the feasibility of constant-cost and test-time scaling.
- **Top-Down Selection Mechanism**: Inspired by human vision, enabling the MLLM to "glance at the global image first, then focus on relevant regions" like humans do.
- **4KPro Benchmark**: Exposing the "pseudo-high-resolution" issue in existing benchmarks — where image resolutions are high but the questions do not require high resolution.

## Limitations & Future Work

- PS3 pre-training relies on 75M high-resolution images and 282M local captions, where data collection depends on an MLLM captioner (Qwen2-VL) rather than being completely self-curated.
- The "pasting images onto 4K backgrounds" strategy for high-resolution fine-tuning data is a simplistic workaround that may introduce distribution shifts.
- 4KPro comprises only 4 scene categories, which limits the evaluation coverage.
- The KV cache in Stage 3 increases the VRAM demands of the ViT, potentially limiting its application on larger models.

## Related Work & Insights

- **vs AnyRes**: AnyRes splits the image into tiles and feeds them into the original ViT but lacks high-resolution pre-training; PS3 outperforms it by $6.9\%$.
- **vs S²**: S² performs training-free multi-scale feature fusion; PS3, backed by pre-training, outperforms it by $2.4\%$.
- **vs SigLIP2**: PS3 overall outperforms SigLIP2 across 23 benchmarks, showing that high-resolution pre-training is more crucial than better low-resolution pre-training.
- **vs Perception Encoder**: Meta's PE also pre-trains vision encoders, but it does not support 4K and underperforms compared to PS3.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Extends CLIP-style pre-training to 4K for the first time, with an extremely elegant localized contrastive + patch-selection design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation including 4 types of scaling analysis, SOTA comparisons, a new benchmark, encoder comparisons, and ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Features clear logic, exquisite figures/tables, and highly convincing scaling analysis.
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm for high-resolution visual perception, and 4KPro fills the gap in benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dataset Distillation for Pre-Trained Self-Supervised Vision Models](../../NeurIPS2025/interpretability/dataset_distillation_for_pre-trained_self-supervised_vision_models.md)
- [\[ICLR 2026\] Evolution of Concepts in Language Model Pre-Training](../../ICLR2026/interpretability/evolution_of_concepts_in_language_model_pre-training.md)
- [\[ECCV 2024\] POA: Pre-training Once for Models of All Sizes](../../ECCV2024/interpretability/poa_pre-training_once_for_models_of_all_sizes.md)
- [\[ICLR 2026\] Learning to See Before Seeing: Demystifying LLM Visual Priors from Language Pre-training](../../ICLR2026/interpretability/learning_to_see_before_seeing_demystifying_llm_visual_priors_from_language_pre-t.md)
- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](../../NeurIPS2025/interpretability/towards_scaling_laws_for_symbolic_regression.md)

</div>

<!-- RELATED:END -->
