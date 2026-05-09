---
title: >-
  [Paper Note] Attention Retention for Continual Learning with Vision Transformers
description: >-
  [AAAI 2026][LLM Safety][Continual Learning] This paper proposes ARCL-ViT, a framework that prevents attention drift in Vision Transformers during continual learning via a two-step strategy of attention mask generation and gradient masking. It achieves state-of-the-art results on ImageNet-R and CIFAR-100, demonstrating that preserving attention patterns is key to mitigating catastrophic forgetting.
tags:
  - AAAI 2026
  - LLM Safety
  - Continual Learning
  - Vision Transformer
  - Attention Retention
  - Catastrophic Forgetting
  - Gradient Masking
date: 2026-05-08
content_hash: 8645a9f99f80083d
---

# Attention Retention for Continual Learning with Vision Transformers

**Conference**: AAAI 2026
**arXiv**: [2602.05454](https://arxiv.org/abs/2602.05454)
**Code**: None
**Area**: LLM Safety
**Keywords**: Continual Learning, Vision Transformer, Attention Retention, Catastrophic Forgetting, Gradient Masking

## TL;DR
This paper proposes ARCL-ViT, a framework that prevents attention drift in Vision Transformers during continual learning via a two-step strategy of attention mask generation and gradient masking. It achieves state-of-the-art results on ImageNet-R and CIFAR-100, demonstrating that preserving attention patterns is key to mitigating catastrophic forgetting.

## Background & Motivation
**Background**: Continual learning requires models to retain performance on previous tasks while learning new ones. ViTs are increasingly adopted in CL settings.

**Limitations of Prior Work**: (a) Catastrophic forgetting in ViTs manifests as **attention drift**; (b) regularization methods (e.g., EWC) offer limited effectiveness for ViTs; (c) expansion methods (e.g., DualPrompt) introduce substantial additional parameters.

**Key Challenge**: Updating parameters to learn new tasks may disrupt the attention allocation established for discriminative features of old tasks.

**Goal**: Directly prevent the destruction of attention patterns corresponding to previously learned tasks in ViTs.

**Key Insight**: Inspired by the selective attention mechanism of the human V1 visual cortex — maintaining sustained focus on important features.

**Core Idea**: Generate attention masks from previous tasks and zero out gradients of Q/K/V weights in the corresponding regions during new task training, directly preventing attention drift.

## Method

### Overall Architecture
Input: a sequence of tasks arriving sequentially. Output: a ViT capable of handling all learned tasks. Two steps: (1) layer-wise rollout to extract attention maps → adaptive thresholding → binary mask; (2) mask-based zeroing of Q/K/V weight gradients.

### Key Designs

1. **Attention Mask Generation**:

    - Function: Extract attention regions to be protected from the previous task.
    - Mechanism: Layer-wise rollout extracts $\mathbf{U}_{t-1}$; instance-adaptive thresholding generates $\bar{\mathbf{M}}_{t-1}$.
    - Design Motivation: Identify attention regions critical to discriminative features of old tasks.

2. **Gradient Masking**:

    - Function: Protect old attention patterns during new task training.
    - Mechanism: $\nabla \mathbf{W}'_{\theta,t} = \nabla \mathbf{W}_{\theta,t} \odot (1 - \bar{\mathbf{M}}_{t-1})$, with Adam-compatible scaling $\Delta\mathbf{W}'_{\theta,t} = (\nabla\mathbf{W}'_{\theta,t} / \nabla\mathbf{W}_{\theta,t}) \odot \Delta\mathbf{W}_{\theta,t}$.
    - Design Motivation: Directly block modifications to critical regions of old tasks at the gradient level, with compatibility for Adam optimizer.

3. **Instance-Adaptive Thresholding**:

    - Function: Generate sample-specific thresholds for binarization.
    - Design Motivation: Attention distributions vary considerably across different tasks and samples.

### Loss & Training
Standard cross-entropy loss is used. The gradient mask is applied after backpropagation, requiring no modification to the loss function.

## Key Experimental Results

### Main Results

| Method | 10S-ImageNet-R | 20S-ImageNet-R | 10S-CIFAR-100 |
|--------|---------------|---------------|--------------|
| CODA-Prompt | 75.45% | - | 86–89% |
| OS-Prompt++ | - | 73.77% | - |
| **ARCL-ViT** | **SOTA** | **SOTA** | **~87%** |

### Ablation Study

| Configuration | Performance | Note |
|--------------|-------------|------|
| Full model | Best | Attention mask + gradient mask |
| w/o gradient mask | Severe degradation | Equivalent to Seq-FT |
| w/o adaptive threshold | Slight drop | Global threshold lacks flexibility |
| Different pre-training schemes | Robust | Insensitive to pre-training choice |

### Key Findings
- Attention drift is the primary cause of catastrophic forgetting in ViTs, clearly demonstrated through visualization.
- Gradient masking outperforms regularization and expansion methods.
- The approach is robust on long task sequences (20S) and across different pre-training schemes.

## Highlights & Insights
- **Precise Problem Formulation**: Attributing catastrophic forgetting to attention drift is well-motivated and supported by compelling visual evidence.
- **Biologically Inspired Elegant Solution**: Simple gradient masking achieves attention retention in a concise yet effective manner.
- **Adam Compatibility Design**: The gradient/update scaling trick is a critical engineering contribution for practical applicability.

## Limitations & Future Work
- Storing attention masks for previous tasks incurs memory overhead that grows linearly with the number of tasks.
- Binary masks may be overly coarse; soft masks could offer greater flexibility.
- Validation is limited to ViTs; applicability to CNNs remains unexplored.
- Task boundary information is required during training.

## Related Work & Insights
- **vs. EWC**: EWC applies regularization in parameter space, whereas ARCL-ViT directly protects the attention space, yielding better effectiveness for ViTs.
- **vs. DualPrompt/CODA-Prompt**: Expansion methods increase model size; ARCL-ViT introduces no additional parameters.
- **vs. PackNet**: A conceptually similar idea transferred from CNN pruning to attention protection in ViTs.

## Rating
- Novelty: ⭐⭐⭐⭐ The attention drift perspective is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets, settings, and visualizations.
- Writing Quality: ⭐⭐⭐⭐ Clear and intuitive presentation.
- Value: ⭐⭐⭐⭐ Practical contribution to continual learning with ViTs.

## Additional Remarks
- The methodology and experimental design offer useful reference for related research areas.
- Future work should validate generalizability and scalability across more diverse scenarios and larger scales.
- Potential research value exists in combining this approach with recent methods (e.g., RL/MCTS or multimodal frameworks).
- Deployment feasibility and computational efficiency should be assessed against practical application requirements.
- The choice of datasets and evaluation metrics may affect the generalizability of conclusions; cross-validation on additional benchmarks is recommended.

## Additional Remarks
- The methodology and experimental design offer useful reference for related research areas.
- Future work should validate generalizability and scalability across more diverse scenarios and larger scales.
- Potential research value exists in combining this approach with recent related work (e.g., intersections with RL/MCTS or multimodal methods).

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds](catformer_when_continual_learning_meets_spiking_transformers_with_dynamic_thresh.md)
- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](../../CVPR2026/llm_safety/elastic_weight_consolidation_done_right_for_continual_learning.md)
- [\[NeurIPS 2025\] Finding Structure in Continual Learning](../../NeurIPS2025/llm_safety/finding_structure_in_continual_learning.md)
- [\[AAAI 2026\] PANDA: Patch and Distribution-Aware Augmentation for Long-Tailed Exemplar-Free Continual Learning](panda_--_patch_and_distribution-aware_augmentation_for_long-tailed_exemplar-free.md)
- [\[AAAI 2026\] Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting](learning_from_the_undesirable_robust_adaptation_of_language_models_without_forge.md)

<!-- RELATED:END -->
