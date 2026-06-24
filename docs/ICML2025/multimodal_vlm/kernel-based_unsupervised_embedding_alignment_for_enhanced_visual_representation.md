---
title: >-
  [Paper Note] Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models
description: >-
  [ICML2025][Multimodal VLM][CLIP] Proposed Kernel-based Unsupervised Embedding Alignment (KUEA), a method that aligns the visual representations of CLIP and DINOv2 in a kernel space. By fine-tuning solely on image data, it enhances CLIP's fine-grained perception while maintaining compatibility with the text encoder, thereby boosting the performance of downstream MLLMs.
tags:
  - "ICML2025"
  - "Multimodal VLM"
  - "CLIP"
  - "DINOv2"
  - "Kernel Alignment"
  - "Visual Representation Enhancement"
  - "Zero-shot Classification"
  - "MLLMs"
date: 2026-05-08
content_hash: 332a7ecfee74f94a
---

# Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models

**Conference**: ICML2025  
**arXiv**: [2506.02557](https://arxiv.org/abs/2506.02557)  
**Code**: [peterant330/KUEA](https://github.com/peterant330/KUEA)  
**Area**: Visual Representation / Multimodal VLMs  
**Keywords**: CLIP, DINOv2, Kernel Alignment, Visual Representation Enhancement, Zero-shot Classification, MLLMs

## TL;DR

Proposed Kernel-based Unsupervised Embedding Alignment (KUEA), a method that aligns the visual representations of CLIP and DINOv2 in a kernel space. By fine-tuning solely on image data, it enhances CLIP's fine-grained perception while maintaining compatibility with the text encoder, thereby boosting the performance of downstream MLLMs.

## Background & Motivation

CLIP achieves strong zero-shot capabilities through global image-text contrastive learning, but its vision encoder exhibits significant deficiencies in fine-grained perception (e.g., color, spatial relationships, and counting). This limitation propagates to downstream MLLMs (such as LLaVA and OpenFlamingo) that rely on CLIP as their vision encoder.

Self-supervised vision models like DINOv2 excel at capturing pixel-level details compared to CLIP, but their feature spaces are incompatible with text encoders. Existing approaches face a dilemma:

- **Multi-encoder Fusion** (e.g., Eagle, Cambrian): High computational overhead.
- **Self-supervised Fine-tuning of CLIP** (e.g., MaskCLIP): Disrupts image-text alignment, discarding zero-shot capabilities.
- **Re-training CLIP**: Prohibitively expensive, and the newly learned embeddings are incompatible with existing downstream models.

The core insight of this paper is that samples with similar semantics but distinct visual appearances are highly clustered in CLIP's feature space, whereas DINOv2 can distinguish them. If CLIP can learn the **relative, pairwise relationships** from DINOv2 while maintaining the global structure of its original feature space, it can enjoy the best of both worlds.

## Method

### Core Idea: Kernel Space Alignment

Instead of aligning the feature vectors of the two models directly (as their dimensions and spatial structures differ drastically), the method aligns their **kernel matrices**—specifically, the pairwise relative similarity structure between samples.

### Kernel Function Definition

A normalized polynomial kernel is employed:

$$k_{\text{poly}(\gamma,c,d)}(\mathbf{x}, \mathbf{y}) = (\gamma \mathbf{x}^T \mathbf{y} + c)^d$$

Normalization operation:

$$\tilde{k}(\mathbf{x}, \mathbf{y}) = \frac{k(\mathbf{x}, \mathbf{y})}{\sqrt{k(\mathbf{x}, \mathbf{x}) \cdot k(\mathbf{y}, \mathbf{y})}}$$

By default, a degree-3 polynomial kernel is used. On the DINOv2 side, the kernel parameters are fixed ($\gamma = 1/d_{emb}$, $c=1$), while on the CLIP side, the kernel parameters are set to be trainable.

### Alignment Objective

For each randomly sampled pair of images $(I_i, I_j)$, the difference between the two kernel function values is minimized:

$$\min_\theta \; \mathbb{E}_{I_i, I_j \sim \mathcal{D}} \left[ \left( k_1(f_\theta(I_i), f_\theta(I_j)) - k_2(g(I_i), g(I_j)) \right)^2 \right]$$

where $f_\theta$ represents the trainable CLIP vision encoder, and $g$ represents the frozen DINOv2 model.

**Key Theoretical Guarantee** (Proposition 3.1): Based on Hoeffding's inequality, it is proven that an unbiased estimator of the gradient can be obtained with finite sample pairs. This supports stochastic gradient optimization and enables scalability to large-scale data.

### Regularization for Preserving Image-Text Alignment

To prevent the fine-tuned features from shifting too far from the original CLIP embeddings, an L2 regularization term is introduced:

$$\mathcal{L} = w \cdot \mathcal{L}_{\text{kernel}} + \mathbb{E}_{I_i} \left[ \| f_\theta(I_i) - f_{\theta_0}(I_i) \|_2^2 \right]$$

**Theoretical Guarantee** (Proposition 3.2): If $\|f_\theta(I) - f_{\theta_0}(I)\|_2 \leq \lambda$, the upper bound of the drift in image-text cosine similarity is $2\lambda / \max\{\|f_\theta(I)\|_2, \|f_{\theta_0}(I)\|_2\}$. This indicates that the regularization directly constrains the shift in image-text alignment.

### Training Configuration

- Training Data: ImageNet-1K training set (1.28 million images, **images only, no text**)
- Hardware: 2× RTX 4090, alignment takes approximately 30 hours for ViT-L-14
- The text encoder is fully frozen; only the vision encoder is fine-tuned

## Key Experimental Results

### Zero-Shot Classification (Average Accuracy across 12 Datasets)

| Model | Original CLIP | Projection | Feature | DIVA | **Kernel (Ours)** |
|------|-----------|------------|---------|------|-------------------|
| ViT-B-16 | 61.22 | 54.17 | 61.41 | — | **62.04** (+0.82) |
| ViT-L-14 | 65.26 | 54.07 | 65.73 | 65.45 | **66.54** (+1.28) |
| ViT-L-14-336 | 66.10 | 54.53 | 66.52 | 65.30 | **67.13** (+1.03) |

The projection-based approach leads to severe degradation in zero-shot performance (by -7% to -12%), and direct feature alignment yields negligible improvements. In contrast, only kernel space alignment effectively enhances performance while maintaining compatibility.

### Image-Text Retrieval (Flickr30K / COCO)

After alignment, retrieval performance increases instead of dropping. For ViT-L-14-336 on Flickr30K, Image→Text R@1 increases from 83.0 to 84.6, and Text→Image R@1 improves from 64.78 to 67.08, demonstrating that the image-text alignment is effectively preserved.

### Fine-Grained Perception Tasks (Linear Probe)

| Task | Original ViT-L-14 | ViT-L-14 +align | Gain |
|------|-------------|----------------|------|
| SVHN | 65.20 | **69.39** | +4.19 |
| GTSRB | 72.94 | **74.51** | +1.57 |
| CLEVR Distance | 22.97 | **30.82** | +7.85 |
| CLEVR Counts | 41.25 | **49.67** | +8.42 |

Improvements are particularly significant on counting and spatial reasoning tasks (+5.51% on average), validating that kernel alignment effectively transfers the fine-grained visual perception capabilities of DINOv2.

### Downstream MLLM Performance Improvement

**LLaVA-1.5-7B** (ViT-L-14-336):
- Simply replacing the vision encoder (without tuning the LLM): Average score improves from 60.06 to 60.88 (+0.82)
- Replacing the vision encoder + LoRA fine-tuning the LLM: Average score improves from 60.18 to **61.70** (+1.52), with an average improvement of 3.05% on localization tasks

**OpenFlamingo-3B** (4-shot):
- Average score improves from 45.54 to **46.16** (+0.62), without fine-tuning the LLM

## Highlights & Insights

1. **Kernel Space Alignment instead of Feature Space Alignment**: Bypasses the issue of vast structural differences between the feature spaces of the two models. By aligning only the relative pairwise relations between samples, it preserves the global structure of the original feature space.
2. **Lightweight Fine-tuning using Only Image Data**: Requires no image-text pairs and no retraining of the text encoder. The training process can be completed on ImageNet-1K using just 2× RTX 4090 GPUs.
3. **Plug-and-Play and Compatible with Downstream Models**: The aligned vision encoder can directly replace the original CLIP in LLaVA/OpenFlamingo without requiring a re-run of the vision-language alignment stage.
4. **Solid Theoretical Guarantees**: Proposition 3.1 proves the feasibility of stochastic optimization, and Proposition 3.2 provides an explicit upper bound for the drift in image-text alignment.
5. **Strong Generalization**: Beyond the CLIP+DINOv2 combination, its effectiveness is validated across other model pairings such as SigLIP, DFN, and MetaCLIP.

## Limitations & Future Work

1. **Constrained Data Scale**: Fine-tuning is only conducted on ImageNet-1K (1.28M images). Scaling up to larger datasets (e.g., DataComp) could yield further improvements.
2. **Evaluation Limited to Smaller Models**: MLLM evaluation is restricted to LLaVA-7B and OpenFlamingo-3B, without scaling up to 70B-grade models.
3. **Narrow Choice of Target Models**: The work primarily focuses on CLIP↔DINOv2. The alignment efficacy on other vision foundation models (such as BEiT and MAE) is not fully explored.
4. **Modest Performance Gains**: The average improvement in zero-shot classification is around 1%, which may not be significant enough for certain application scenarios.
5. **Token-level Alignment Unexplored**: The current method only aligns [CLS] token-level embeddings. Patch-token-level alignment could potentially enhance dense prediction performance.

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective of aligning in the kernel space is highly novel, bypassing the difficulties of direct feature-level alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers multiple dimensions including zero-shot classification, retrieval, fine-grained perception, and MLLMs, complemented by a complete ablation study.
- Writing Quality: ⭐⭐⭐⭐ — The theoretical derivations are clear, and the motivations are compellingly articulated.
- Value: ⭐⭐⭐⭐ — This lightweight, plug-and-play solution offers strong utility and provides a new paradigm for visual enhancement in MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](../../ICCV2025/multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[ACL 2025\] OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference](../../ACL2025/multimodal_vlm/omnialign-v_towards_enhanced_alignment_of_mllms_with_human_preference.md)
- [\[ICCV 2025\] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models](../../ICCV2025/multimodal_vlm/multi-cache_enhanced_prototype_learning_for_test-time_generalization_of_vision-l.md)
- [\[CVPR 2026\] TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment](../../CVPR2026/multimodal_vlm/tipsv2_patch_text_alignment.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)

</div>

<!-- RELATED:END -->
