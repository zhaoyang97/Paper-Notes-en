---
title: >-
  [Paper Note] Embracing Collaboration Over Competition: Condensing Multiple Prompts for Visual In-Context Learning
description: >-
  [CVPR 2025][Model Compression][Visual ICL] Proposes Condenser to condense multiple Visual ICL prompt candidates into a single prompt via Patch-wise Cross-Attention, enabling multi-prompt collaboration instead of competitive selection. It achieves $46.63$ mIoU with $16$ input prompts (vs. $44.14$ for a single prompt) on segmentation, detection, and colorization, while being $15\times$ faster in inference than evaluating prompts one-by-one.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Visual ICL"
  - "Prompt Condensation"
  - "Cross-Attention Fusion"
  - "MAE-VQGAN"
  - "Multi-Prompt Collaboration"
date: 2026-05-08
content_hash: c675b27ed4c9c5db
---

# Embracing Collaboration Over Competition: Condensing Multiple Prompts for Visual In-Context Learning

**Conference**: CVPR 2025  
**arXiv**: [2504.21263](https://arxiv.org/abs/2504.21263)  
**Code**: [https://github.com/gimpong/CVPR25-Condenser](https://github.com/gimpong/CVPR25-Condenser)  
**Area**: Model Compression  
**Keywords**: Visual ICL, Prompt Condensation, Cross-Attention Fusion, MAE-VQGAN, Multi-Prompt Collaboration

## TL;DR
Proposes Condenser to condense multiple Visual ICL prompt candidates into a single prompt via Patch-wise Cross-Attention, enabling multi-prompt collaboration instead of competitive selection. It achieves $46.63$ mIoU with $16$ input prompts (vs. $44.14$ for a single prompt) on segmentation, detection, and colorization, while being $15\times$ faster in inference than evaluating prompts one-by-one.

## Background & Motivation

**Background**: Visual ICL enables models to learn new tasks at inference time by providing "input image-label" prompt pairs. Existing methods select the "best single prompt" from a candidate prompt pool, but different prompts may contain complementary information.

**Limitations of Prior Work**: (1) Single-prompt selection may miss useful information present in other prompts. (2) Evaluating $K$ prompts one-by-one for selection incurs high computational overhead ($K\times$ inference time). (3) Simply averaging multiple prompts loses spatial local information.

**Key Challenge**: Utilizing more prompts provides more information, but the current paradigm only allows selecting one. Merging multiple prompts requires aggregating heterogeneous information while maintaining spatial consistency.

**Goal**: Design a lightweight module to condense $K$ prompts into a single high-quality prompt, making multi-prompt collaboration possible.

**Key Insight**: Patch-wise cross-attention — each patch of the query image only performs attention aggregation with patches at the same spatial location across the $K$ prompts. This locality maintains spatial consistency while aggregating complementary information from multiple prompts.

**Core Idea**: Use Patch-wise Cross-Attention to condense $K$ candidate prompts into one, achieving "multi-prompt collaboration" instead of "single-prompt competition".

## Method

### Overall Architecture
$K$ candidate prompts (image-label pairs) $\rightarrow$ self-attention within each prompt to identify informative patches $\rightarrow$ Patch-wise Cross-Attention (PCA) to aggregate each position of the query across the same positions of the $K$ prompts $\rightarrow$ condensed single prompt (image + label) $\rightarrow$ fed into the frozen MAE-VQGAN backbone for inference.

### Key Designs

1. **Patch-wise Cross-Attention (PCA)**:

    - Function: Aggregate information across prompts while maintaining spatial consistency.
    - Mechanism: The patch at position $(i,j)$ of the query image only performs attention aggregation with the patches at position $(i,j)$ across the $K$ prompts. Compared with global cross-attention ($44.87$ mIoU), PCA ($46.63$ mIoU) performs better because local consistency is crucial for visual tasks.
    - Design Motivation: Global cross-attention would mix information from different spatial locations, destroying the spatial structure of the prompt.

2. **Separate Image/Label Condensation**:

    - Function: Prompt images and labels are condensed separately, sharing the same key source.
    - Mechanism: Two PCA modules are used — one to condense the prompt image $\mathbf{F}_{c*}^I$ and another to condense the prompt label $\mathbf{F}_{c*}^L$. Both share keys derived from the image space, ensuring that label condensation is also content-guided by the image.
    - Design Motivation: Labels and images must maintain their correspondence; sharing keys naturally guarantees this alignment.

3. **End-to-End Training (Token Prediction Loss)**:

    - Function: Optimize condensation quality via end-to-end gradients from the backbone.
    - Mechanism: $\mathcal{L}_{TP}$ (Token Prediction) ensures that the output of the frozen backbone using the condensed prompt closely matches the ground truth. $\mathcal{L}_{PA}$ (Pre-Alignment) uses cosine similarity to pre-align condensed features with target features. Ablations show that removing $\mathcal{L}_{TP}$ causes mIoU to collapse from $46.63$ to $8.66$.
    - Design Motivation: Without end-to-end feedback, the condensation module cannot learn meaningful fusion at all.

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{TP} + \lambda \cdot \mathcal{L}_{PA}$$ The MAE-VQGAN backbone is completely frozen, training only the Condenser module.

## Key Experimental Results

### Main Results

| Method | K | Seg. mIoU | Det. mIoU | Inference Time |
|------|---|----------|----------|---------|
| Single Prompt | 1 | 44.14 | 43.22 | 66ms |
| Prompt-SelF (Selection) | 16 | - | - | 989ms |
| **Condenser** | **16** | **46.63** | **44.64** | **66ms** |

### Ablation Study

| Fusion Method | Seg. mIoU |
|---------|----------|
| Average Pooling | 17.23 |
| Global Cross-Attention | 44.87 |
| **Patch-wise Cross-Attention** | **46.63** |
| Without $\mathcal{L}_{TP}$ | 8.66 |

### Key Findings
- **Multi-prompt collaboration >> Single-prompt selection**: $46.63$ mIoU at $K=16$ vs. $44.14$ mIoU at $K=1$ ($+2.49$ mIoU).
- **Nearly zero inference overhead**: The inference time after condensation is identical to a single prompt (66ms), which is $15\times$ faster than evaluating prompts one-by-one.
- **Performance scales with $K$**: Performance increases monotonically from $K=1 \rightarrow 2 \rightarrow 4 \rightarrow 8 \rightarrow 16 \rightarrow 32$.
- **Simple average is completely ineffective** ($17.23$ mIoU): Condensation is not merely simple feature averaging.

## Highlights & Insights
- **Paradigm shift of "collaboration over competition"**: Shifting from "selecting the single best" to "fusing all into one", which is simple yet highly effective.
- **Spatial consistency of the patch-wise design** is a crucial design choice — spatial information must be preserved in visual tasks.

## Limitations & Future Work
- The Condenser module itself requires training and may need to be retrained for new tasks.
- As $K$ increases, the inputs to Condenser grow, but inference time remains constant — indicating diminishing marginal returns for excessively large $K$.
- Only validated on the MAE-VQGAN backend.

## Related Work & Insights
- **vs. Prompt-SelF**: Evaluates prompt candidates one-by-one to select the best one, which is $15\times$ slower and ultimately uses only a single prompt. Condenser utilizes all $K$ prompts and is much faster.
- **vs. InMeMo**: Reached $43.14$ mIoU. Condenser outperforms this with $K=1$ ($44.14$ mIoU) and reaches $46.63$ mIoU with $K=16$.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of multi-prompt condensation is novel, and the PCA design is reasonable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three tasks evaluated (segmentation, detection, colorization), comprehensive ablations on $K$ values and fusion methods.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative comparing collaboration vs. competition.
- Value: ⭐⭐⭐⭐ Makes a significant contribution to the field of Visual ICL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition](../../ACL2025/model_compression/teamlora_boosting_low-rank_adaptation_with_expert_collaboration_and_competition.md)
- [\[CVPR 2025\] ECVC: Exploiting Non-Local Correlations in Multiple Frames for Contextual Video Compression](ecvc_exploiting_non-local_correlations_in_multiple_frames_for_contextual_video_c.md)
- [\[CVPR 2026\] Balanced Dataset Distillation via Modeling Multiple Visual Pattern Distribution](../../CVPR2026/model_compression/balanced_dataset_distillation_via_modeling_multiple_visual_pattern_distribution.md)
- [\[CVPR 2025\] Mamba-Adaptor: State Space Model Adaptor for Visual Recognition](mamba-adaptor_state_space_model_adaptor_for_visual_recognition.md)
- [\[CVPR 2025\] MobileMamba: Lightweight Multi-Receptive Visual Mamba Network](mobilemamba_lightweight_multi-receptive_visual_mamba_network.md)

</div>

<!-- RELATED:END -->
