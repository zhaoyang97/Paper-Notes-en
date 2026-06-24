---
title: >-
  [Paper Note] Masking Meets Supervision: A Strong Learning Alliance
description: >-
  [CVPR 2025][Model Compression][Supervised Learning] This work proposes Masked Sub-branch (MaskSub)—a generic framework that introduces high-ratio (50%) mask augmentation into supervised learning. By utilizing a self-distillation structure with a main branch (unmasked) and a sub-branch (masked), it addresses the training instability caused by strong mask augmentation. It consistently improves performance across various scenarios, including DeiT-III, MAE fine-tuning…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Supervised Learning"
  - "Mask Augmentation"
  - "Self-Distillation"
  - "ViT Training"
  - "Regularization"
date: 2026-05-08
content_hash: 7adc8561c21df114
---

# Masking Meets Supervision: A Strong Learning Alliance

**Conference**: CVPR 2025  
**arXiv**: [2306.11339](https://arxiv.org/abs/2306.11339)  
**Code**: [https://github.com/naver-ai/augsub](https://github.com/naver-ai/augsub)  
**Area**: Model Compression/Training Optimization  
**Keywords**: Supervised Learning, Mask Augmentation, Self-Distillation, ViT Training, Regularization

## TL;DR

This work proposes Masked Sub-branch (MaskSub)—a generic framework that introduces high-ratio (50%) mask augmentation into supervised learning. By utilizing a self-distillation structure with a main branch (unmasked) and a sub-branch (masked), it addresses the training instability caused by strong mask augmentation. It consistently improves performance across various scenarios, including DeiT-III, MAE fine-tuning, CLIP fine-tuning, BERT training, as well as ResNet/Swin architectures.

## Background & Motivation

Masked Image Modeling (such as MAE) achieves powerful self-supervised pre-training through high-ratio random masking (>50%), but supervised learning has been unable to effectively leverage such strong mask augmentation:

- **Failure of high mask ratios in supervised learning**: When the mask ratio exceeds 50%, direct training with cross-entropy causes severe degradation—the standard training loss fails to converge, and validation accuracy drops.
- **Regularization intensity exceeding the limit**: High-ratio masking acts as an extremely strong regularization. Once it exceeds the optimal regularization window, it exerts a detrimental effect on loss convergence.
- **Room for improvement in supervised learning**: Existing SOTA training recipes (DeiT-III, RSB) already feature meticulously tuned combinations of regularizations, making it highly challenging to add extra strong regularization on top of them.
- **Narrowing gap between MIM and supervised learning**: As demonstrated by DeiT-III, modern supervised training recipes have caught up with the performance of MAE pre-training. However, both paradigms have distinct advantages, and integrating the regularization capability of masking into supervised learning remains a key challenge.

Core Motivation: To design an architecture that isolates strong mask augmentation from the main training pipeline, providing a relaxed learning target for the sub-branch via self-distillation (using the main branch output as a soft label) to avoid disrupting the convergence of the main loss.

## Method

### Overall Architecture

MaskSub adds a sub-branch on top of standard supervised training. The main branch $f_\theta(\mathbf{x}|r_{mask}=0)$ remains unmasked and is trained using standard cross-entropy. The sub-branch $f_\theta(\mathbf{x}|r_{mask}=r)$ undergoes high-ratio masking (50% by default) and is trained using the softmax output of the main branch as a distillation target. The two branches **share all parameters**, and the total loss is the average of the main branch's CE loss and the sub-branch's distillation loss.

### Key Designs

1. **Self-distillation sub-branch structure**:
    - Function: Decouple the training signal of strong mask augmentation from the main loss to avoid mutual interference.
    - Mechanism: The main branch is trained normally (with stop-gradient applied when serving as a target), while the sub-branch uses the softmax output of the main branch $\sigma(f_\theta(\mathbf{x}|r_{mask}=0))$ as a soft label instead of the one-hot ground-truth. The total loss = $\frac{1}{2}[\text{CE}(o_1, \text{label}) + \text{CE}(o_2, \text{softmax}(o_1.\text{detach}()))]$. The sub-branch uses MAE-style masking—directly removing masked tokens to reduce computational cost (50% mask ≈ 1.5× standard training cost).
    - Design Motivation: (1) Directly training the sub-branch with hard labels leads to instability due to the high difficulty introduced by masking; (2) The soft labels from the main branch are relaxed and adaptive—when the main branch performs well, the sub-branch target is close to the ground truth (hard task), and when the main branch performs poorly, the target is blurry (easy task), achieving automatic difficulty control.

2. **Automatic difficulty control mechanism**:
    - Function: Adaptively adjust the learning difficulty of the sub-branch as training progresses.
    - Mechanism: Validated by analyzing gradient magnitudes: in the early stages of training, the sub-branch gradients are small (targets are relaxed, and learning is easy); as the performance of the main branch improves, the sub-branch gradients gradually increase (targets approach ground truth, and learning becomes harder). This aligns with findings in knowledge distillation literature—weak teachers yield simple targets, while strong teachers yield challenging targets.
    - Design Motivation: This can be viewed as sample-wise mask augmentation—applying challenging mask training only to samples that the main branch has already successfully classified, while placing no additional pressure on failed samples.

3. **Extension to other Drop regularizations**:
    - Function: The MaskSub framework is not limited to mask augmentation and can be extended to all drop-based regularizations.
    - Mechanism: Three variants are proposed—(1) **MaskSub**: the sub-branch performs random patch masking, yielding the largest performance gain; (2) **DropSub**: the sub-branch performs strong dropout (element-wise random dropping), reviving the otherwise abandoned dropout in ViT training; (3) **PathSub**: the sub-branch uses a higher drop-path probability. All variants only require replacing $r_{mask}$ with the corresponding drop probability.
    - Design Motivation: Drop-based regularizations (dropout, drop-path) also degrade supervised learning when they are overly strong. The MaskSub framework provides a unified solution.

### Loss & Training

- Main branch loss: $\mathcal{L}_{main} = \text{CE}(\sigma(f_\theta(\mathbf{x}|r=0)), \mathbf{y})$ (or BCE version)
- Sub-branch loss: $\mathcal{L}_{sub} = \text{CE}(\sigma(f_\theta(\mathbf{x}|r)), \sigma(f_\theta(\mathbf{x}|r=0)).\text{detach()})$
- Total loss: $\mathcal{L} = \frac{1}{2}(\mathcal{L}_{main} + \mathcal{L}_{sub})$
- No extra hyperparameters—the mask ratio is consistently set to 50% and kept unchanged across all experiments.

## Key Experimental Results

### Main Results (ImageNet-1k from Scratch, DeiT-III Recipe)

| Network | DeiT-III 400ep | + MaskSub | DeiT-III 800ep | + MaskSub |
|------|---------------|-----------|---------------|-----------|
| ViT-S/16 | 80.4 | **81.1** (+0.7) | 81.4 | **81.7** (+0.3) |
| ViT-B/16 | 83.5 | **84.1** (+0.6) | 83.8 | **84.2** (+0.4) |
| ViT-L/16 | 84.5 | **85.2** (+0.7) | 84.9 | **85.3** (+0.4) |
| ViT-H/14 | 85.1 | **85.7** (+0.6) | 85.2 | **85.7** (+0.5) |

### Fine-tuning Experiments

| Pre-training Method | Network | Baseline | + MaskSub |
|-----------|------|----------|-----------|
| MAE | ViT-B/16 | 83.6 | **83.9** (+0.3) |
| MAE | ViT-H/14 | 86.9 | **87.2** (+0.3) |
| BEiTv2 | ViT-B/16 | 85.5 | **85.6** (+0.1) |
| CLIP | ViT-B/16 | 84.8 | **85.2** (+0.4) |
| CLIP | ViT-L/14 | 87.5 | **87.8** (+0.3) |

### Hierarchical Architectures (ResNet + Swin Transformer)

| Network | Baseline | + MaskSub |
|------|----------|-----------|
| ResNet-50 (RSB) | 79.7 | **80.0** (+0.3) |
| ResNet-101 (RSB) | 81.4 | **82.1** (+0.7) |
| ResNet-152 (RSB) | 81.8 | **82.8** (+1.0) |
| Swin-T | 81.3 | **81.4** (+0.1) |
| Swin-S | 83.0 | **83.4** (+0.4) |
| Swin-B | 83.5 | **83.9** (+0.4) |

### Comparison with SOTA Pre-training Methods

| Method | ViT-B Epochs | Top-1 | Computational Cost |
|------|-------------|-------|---------|
| MAE | 1600 | 83.6 | - |
| DeiT-III | 800 | 83.8 | ×1.0 |
| CoSub | 800 | 84.2 | ×2.0 |
| **MaskSub** | **400** | **84.1** | **×0.75** |
| **MaskSub** | **800** | **84.2** | **×1.5** |

### Key Findings

- **ViT-H + MaskSub 400ep (85.7) > ViT-H DeiT-III 800ep (85.2)**: Outperforms the original recipe with half the training budget.
- MaskSub 400ep reaches or exceeds the fine-tuning accuracy of MAE 1600ep pre-training—demonstrating extreme efficiency for supervised learning with MaskSub.
- Achieves a 1.0% gain (81.8→82.8) on ResNet-152, showing a significant improvement even on top of the already highly optimized RSB recipe.
- Training analysis confirms that MaskSub simultaneously accelerates the convergence of both the standard loss and the mask loss—it is not a simple regularization trade-off.
- The pattern of sub-branch gradients growing from small to large validates the hypothesis of automatic difficulty control.
- Compared with CoSub: achieves comparable performance while requiring only 75% of the computational cost (MaskSub 400ep vs CoSub 800ep).
- No extra data augmentation, no extra parameters, and no extra optimizer steps—achieving ultimate simplicity.

## Highlights & Insights

- **Extremely Simple Design**: The core implementation requires only ~10 lines of PyTorch code (see Algorithm 1), requires no hyperparameter tuning (a consistent 50% mask), and introduces no additional model parameters.
- **Elegant Application of Decoupling**: Isolates the "strong regularization" from the main training flow into a sub-branch, connected via self-distillation. This both protects the convergence of the main loss and fully harvests the regularization benefits of masking.
- **Deep Insight into Automatic Difficulty Control**: MaskSub essentially achieves sample-wise and epoch-wise adaptive regularization strength—relaxed in the early phase and tightened in the late phase, consistent with the spirit of curriculum learning.
- **Cross-Architecture Universality**: Applicable from ViT to ResNet and Swin, from supervised pre-training to MAE/BEiT/CLIP fine-tuning, and even to BERT. This broad applicability demonstrates the generality of the framework.

## Limitations & Future Work

- For hierarchical architectures (ResNet/Swin), MAE-style token removal is inapplicable, requiring filling with zero values or mask tokens, which doubles the computational cost.
- Although a uniform 50% mask ratio is simple, it may not be optimal—the optimal ratio might vary across different tasks, models, and training stages.
- Only classification performance has been demonstrated; the impact on dense prediction tasks like detection and segmentation remains unvalidated.
- Though theoretical differences from methods like CoSub (co-training vs. self-distillation) are discussed, deeper analytical investigations are lacking.
- The stop-gradient design for sub-branch gradients prevents the regularization effect of masking from backpropagating to affect the main branch—whether more optimal gradient designs exist deserves further exploration.

## Related Work & Insights

- Essential difference from MAE: MAE uses masking for self-supervised reconstruction, whereas MaskSub uses masking as regularization for supervised learning—their targets are entirely different despite sharing the masking operation.
- Difference from CoSub: CoSub uses drop-path to build a sub-branch for co-training (mutual learning between two branches), while MaskSub uses masking to construct a sub-branch for one-way self-distillation (the main branch teaches the sub-branch). MaskSub is more general and computationally efficient.
- Difference from SupMAE: SupMAE incorporates supervised loss into MAE training (MAE recipe + supervision), while MaskSub introduces masking into supervised training (supervised recipe + masking)—representing diametrically opposite directions of integration.
- Insight: The fundamental reason why strong regularization "fails" in supervised learning is not that the regularization itself is harmful, but because directly applying it to the main loss disrupts convergence. Decoupling branches can unlock the otherwise unused potential of regularization.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing mask augmentation into the sub-branch self-distillation for supervised learning is simple and effective, though it bears similarities to existing self-distillation or dual-branch frameworks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering ViT/ResNet/Swin, pre-training/fine-tuning/BERT, and multiple recipes.
- Writing Quality: ⭐⭐⭐⭐⭐ Thorough analysis (loss curves, gradient analysis, difficulty control), simple and intuitive pseudo-code, and excellent manuscript structure.
- Value: ⭐⭐⭐⭐⭐ Provides a generic, parameter-free, plug-and-play improvement solution for supervised training with broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Disentangling Latent Shifts of In-Context Learning with Weak Supervision](../../NeurIPS2025/model_compression/disentangling_latent_shifts_of_in-context_learning_with_weak_supervision.md)
- [\[ICML 2025\] Weak-to-Strong Jailbreaking on Large Language Models](../../ICML2025/model_compression/weak-to-strong_jailbreaking_on_large_language_models.md)
- [\[NeurIPS 2025\] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](../../NeurIPS2025/model_compression/synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)
- [\[CVPR 2025\] Incremental Object Keypoint Learning (KAMP)](incremental_object_keypoint_learning.md)
- [\[ICCV 2025\] ACAM-KD: Adaptive and Cooperative Attention Masking for Knowledge Distillation](../../ICCV2025/model_compression/acam-kd_adaptive_and_cooperative_attention_masking_for_knowledge_distillation.md)

</div>

<!-- RELATED:END -->
