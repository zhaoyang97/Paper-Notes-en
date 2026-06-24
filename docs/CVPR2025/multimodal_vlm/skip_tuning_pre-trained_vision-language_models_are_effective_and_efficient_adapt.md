---
title: >-
  [Paper Note] Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves
description: >-
  [CVPR 2025][Multimodal VLM][VLM fine-tuning] This paper reveals that prompt tuning with frozen VLM parameters neither facilitates knowledge transfer nor significantly improves efficiency (only reducing memory by 6% and time by 16%). It proposes Skip Tuning, which shortens the gradient propagation flow of full fine-tuning through Layer-wise Skipping (LSkip) and Class-wise Skipping (CSkip), achieving 15× speedup and 6.4× memory efficiency while delivering superior accuracy.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "VLM fine-tuning"
  - "Skip connection"
  - "efficient adaptation"
  - "prompt tuning alternative"
  - "CLIP"
date: 2026-05-08
content_hash: c31ebe4563f2b7d5
---

# Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves

**Conference**: CVPR 2025  
**arXiv**: [2412.11509](https://arxiv.org/abs/2412.11509)  
**Code**: [https://github.com/Koorye/SkipTuning](https://github.com/Koorye/SkipTuning)  
**Area**: Multimodal VLM  
**Keywords**: VLM fine-tuning, Skip connection, efficient adaptation, prompt tuning alternative, CLIP

## TL;DR
This paper reveals that prompt tuning with frozen VLM parameters neither facilitates knowledge transfer nor significantly improves efficiency (only reducing memory by 6% and time by 16%). It proposes Skip Tuning, which shortens the gradient propagation flow of full fine-tuning through Layer-wise Skipping (LSkip) and Class-wise Skipping (CSkip), achieving 15× speedup and 6.4× memory efficiency while delivering superior accuracy.

## Background & Motivation

**Background**: Prompt tuning (such as CoOp, MaPLe, PromptSRC, etc.) is considered an effective and efficient paradigm for adapting pre-trained VLMs like CLIP to downstream tasks. It learns a small number of context vectors while keeping the VLM parameters frozen.

**Limitations of Prior Work**: Compared to the full fine-tuning (FT) baseline, although prompt tuning reduces the parameter count to 1/51200, memory usage only decreases by 6.3% and training time by only 15.8%—because the frozen parameters still require forward propagation, which consumes substantial memory. More importantly, the classification accuracy of FT is higher than that of CoOp by 3.49% (on base classes) and 4.49% (on new classes), indicating that freezing parameters actually limits knowledge transfer.

**Key Challenge**: The "parameter efficiency" (fewer trainable parameters) pursued by PT is less critical in practical deployment than actual "memory/time efficiency," yet PT's high parameter efficiency does not translate into high memory/time efficiency.

**Goal**: To directly optimize the memory and time efficiency of the FT baseline without introducing additional prompts or adapters, thereby outperforming PT methods in both efficiency and performance.

**Key Insight**: An analysis of Feature-Gradient Propagation Flows (FGPF) reveals that most shallow layers contribute very little to task-specific knowledge (Feature Sensitivity is close to zero), and most class tokens have very small gradients for specific training images (low Gradient Dependence). Skipping these parts can significantly reduce computational cost.

**Core Idea**: Caching shallow features to skip forward/backward passes (LSkip) and filtering out irrelevant class tokens to reduce width (CSkip) are combined to make FT significantly faster and more resource-efficient than PT.

## Method

### Overall Architecture
Before training starts, a single forward pass is performed through the first $\omega$ layers of CLIP to cache the intermediate features. During training, forward and backward propagation start only from layer $\omega+1$ onwards (LSkip). For the text encoder, only the top-$r \times M$ most relevant class tokens and exponentially sampled extra classes are preserved for each training sample (CSkip). Finally, the model is trained with the standard ITM loss.

### Key Designs

1. **Layer-wise Skipping (LSkip)**:

    - **Function**: Shortens the length of FGPF.
    - **Mechanism**: Measures the contribution of each layer to FT using Feature Sensitivity (FS)—calculated as the Euclidean distance of each layer's output before and after FT. It is found that the FS of shallow layers is nearly zero, whereas the FS of deep layers is significant. Consequently, only the last $N-\omega$ layers with high FS are fine-tuned, while the features of the first $\omega$ layers are cached.
    - **Design Motivation**: The first 9 layers of ViT-B/16 exhibit near-zero FS, with only the last 3 layers showing significant FS. Caching the first 9 layers saves 75% of the layer computations.

2. **Class-wise Skipping (CSkip)**:

    - **Function**: Reduces the width of FGPF.
    - **Mechanism**: Measures the contribution of each class token to each training image using Gradient Dependence. Most classes are found to have extremely small gradients. Only the top-$r \times M$ most relevant classes are kept ($r=0.05$), and a small number of the remaining classes are kept via exponential decay sampling to maintain generalization.
    - **Design Motivation**: Among 1000 classes, usually only about 50 are meaningful for a given training image. Removing the remaining 950 classes not only reduces the computation but also filters out noise from irrelevant gradients.

### Loss & Training
Standard ITM loss is used for full-parameter fine-tuning (only on the last $N-\omega$ layers). No additional prompt or adapter parameters are introduced.

## Key Experimental Results

### Main Results

| Method | Base-New H | Time Efficiency | Memory Efficiency |
|------|-----------|---------|---------|
| CoOp | ~72 | 1× | 1× |
| PromptSRC | ~74 | 1× | 1× |
| **Skip Tuning** | **Best H** | **15× Faster** | **6.4× Saved** |
| LoRA | ~73 | 1× | 1× |
| **Skip Tuning vs LoRA** | **+3.59% H** | **3.8× Faster** | **3.9× Saved** |

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| FT baseline | Good accuracy but slow | Full computation |
| +LSkip (ω=9) | No drop in accuracy, 3× speedup | Shallow layers frozen + cached |
| +CSkip (r=0.05) | Slight accuracy gain, 2× speedup | Irrelevant classes removed |
| LSkip+CSkip | **Optimal efficiency + accuracy** | Dual speedup |

### Key Findings
- **PT's efficiency advantage is overestimated**: CoOp only saves 6.3% memory compared to FT, because the large number of frozen parameters still require forward propagation.
- **Shallow layers contribute almost nothing to task knowledge**: Feature Sensitivity of the first 9/12 layers is close to zero.
- **CSkip actually improves generalization**: Removing irrelevant class tokens reduces gradient noise, which has a positive impact on new class performance.

## Highlights & Insights
- **Challenges the common perception that "PT is more efficient than FT"**—when properly optimized, FT can outperform PT in both efficiency and effectiveness.
- **Surpasses prompt tuning and adapter methods without any extra parameters/modules**, achieving extreme simplicity.
- **The FGPF analysis framework** can be generalized to other scenarios requiring efficient fine-tuning.

## Limitations & Future Work
- The selection of $\omega$ depends on Feature Sensitivity analysis; different backbones may require re-determination.
- The exponential sampling strategy for CSkip is heuristic; adaptive sampling can be explored.
- Only validated on CLIP; the effectiveness on other VLMs such as BLIP/SigLIP remains unknown.

## Related Work & Insights
- **vs CoOp/MaPLe/PromptSRC**: These methods introduce extra prompt parameters but gain limited efficiency. Skip Tuning introduces zero extra parameters and is faster.
- **vs LoRA/Adapter**: These methods introduce trainable low-rank matrices/modules. Skip Tuning is simpler and more efficient (3.8× faster, 3.9× memory saved).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Exceptional empirical analysis challenging PT and an elegant, minimalist solution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive outperformance across four benchmarks: base-to-new, cross-dataset, domain, and few-shot.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous and persuasive logic, from challenging existing perceptions to presenting the solution.
- Value: ⭐⭐⭐⭐⭐ Paradigm-shifting significance for the efficient adaptation of VLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[CVPR 2025\] Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?](can_large_vision-language_models_correct_semantic_grounding_errors_by_themselves.md)
- [\[ICCV 2025\] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning](../../ICCV2025/multimodal_vlm/from_holistic_to_localized_local_enhanced_adapters_for_efficient_visual_instruct.md)
- [\[CVPR 2025\] FastVLM: Efficient Vision Encoding for Vision Language Models](fastvlm_efficient_vision_encoding_for_vision_language_models.md)
- [\[ECCV 2024\] Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models](../../ECCV2024/multimodal_vlm/omniview-tuning_boosting_viewpoint_invariance_of_vision-language_pre-training_mo.md)

</div>

<!-- RELATED:END -->
