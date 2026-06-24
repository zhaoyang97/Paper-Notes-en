---
title: >-
  [Paper Note] Post-pre-training for Modality Alignment in Vision-Language Foundation Models
description: >-
  [CVPR 2025][Multimodal VLM][CLIP] CLIP-Refine is proposed, a "post-pre-training" approach positioned between pre-training and fine-tuning. By utilizing two key techniques—Random Feature Alignment (RaFA) and Hybrid Contrastive Distillation (HyCD)—it narrows CLIP's modality gap and enhances zero-shot performance with only 1 epoch of training on a small dataset.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "CLIP"
  - "Modality Alignment"
  - "Post-pre-training"
  - "Knowledge Distillation"
  - "Feature Space"
date: 2026-05-08
content_hash: 746bb679dc9120cb
---

# Post-pre-training for Modality Alignment in Vision-Language Foundation Models

**Conference**: CVPR 2025  
**arXiv**: [2504.12717](https://arxiv.org/abs/2504.12717)  
**Code**: [https://github.com/yshinya6/clip-refine](https://github.com/yshinya6/clip-refine)  
**Area**: Multimodal VLM  
**Keywords**: CLIP, Modality Alignment, Post-pre-training, Knowledge Distillation, Feature Space

## TL;DR

CLIP-Refine is proposed, a "post-pre-training" approach positioned between pre-training and fine-tuning. By utilizing two key techniques—Random Feature Alignment (RaFA) and Hybrid Contrastive Distillation (HyCD)—it narrows CLIP's modality gap and enhances zero-shot performance with only 1 epoch of training on a small dataset.

## Background & Motivation

CLIP pre-trained image-text encoders suffer from a "modality gap" where image and text features cluster in distinct regions of the feature space, limiting performance on downstream tasks such as cross-modal retrieval and classification.

Existing solutions have two main limitations:
- **Pre-training methods**: Require training from scratch, relying on millions of datasets and a large number of GPUs, which is computationally expensive.
- **Fine-tuning methods**: Although computationally efficient, they degrade zero-shot transfer performance as they focus on specific target tasks.

The authors propose a new training stage called "post-pre-training." The goal is to improve the modality alignment and zero-shot generalization capabilities of off-the-shelf pre-trained CLIP using lightweight computational resources and small datasets (e.g., COCO Captions + a single GPU).

The key challenge is that directly minimizing the image-text feature distance ($\mathcal{L}_{\text{align}}$) collapses the uniformity of the feature space, leading to degraded generalization performance, while contrastive learning suffers from catastrophic forgetting due to small batch sizes.

## Method

### Overall Architecture

CLIP-Refine consists of two components: RaFA (Random Feature Alignment) and HyCD (Hybrid Contrastive Distillation). It performs post-pre-training by jointly optimizing the objective function $\min_{\theta_V, \theta_T} \mathcal{L}_{\text{RaFA}} + \mathcal{L}_{\text{HyCD}}$. The entire process requires only 1 epoch and can be completed on a single A100 GPU.

### Key Designs

1. **Random Feature Alignment (RaFA)**:
    - Function: Indirectly narrows the modality gap while preserving feature space uniformity.
    - Mechanism: Instead of directly minimizing the distance between image and text features, features from both modalities are matched to a shared prior distribution $p(z) = \mathcal{N}(0, I)$. Specifically, for each image-text pair, a random reference vector $z_{\text{ref}}^i \sim p(z)$ is sampled, and the objective $\mathcal{L}_{\text{RaFA}} = \frac{1}{2B}\sum_{i=1}^B \|z_{\text{img}}^i - z_{\text{ref}}^i\|_2^2 + \|z_{\text{txt}}^i - z_{\text{ref}}^i\|_2^2$ is minimized.
    - Design Motivation: Directly aligning positive pairs ($\mathcal{L}_{\text{align}}$) destroys uniformity. Incorporating a shared prior distribution as an intermediate bridge simultaneously achieves three goals: (i) indirectly narrowing the modality gap (via the shared $z_{\text{ref}}$), (ii) guiding the feature distribution toward uniformity (a standard Gaussian is approximately uniform on a hypersphere), and (iii) introducing randomness to prevent overfitting.

2. **Hybrid Contrastive Distillation (HyCD)**:
    - Function: Retains the prior knowledge of the pre-trained model while learning new knowledge.
    - Mechanism: Self-distillation is conducted using the pre-trained CLIP as a teacher model. The key improvement is blending the teacher's output with the ground-truth labels using an alpha parameter to generate "hybrid soft labels": $\hat{q}_{i,j}^{I \to T} = \alpha \mathbb{I}_{i=j} + (1-\alpha) q_{i,j}^{I \to T}$. Then, the KL divergence between the student and the hybrid labels is minimized.
    - Design Motivation: Pure self-distillation (Self-KD) over-constrains parameter updates to pre-trained values, hindering RaFA learning. Blending in ground-truth labels ($\alpha=0.5$) allows the model to learn new cross-modal alignment knowledge from correct image-text pairings while retaining the teacher's dark knowledge.

3. **Design of Post-Pre-Training Paradigm**:
    - Function: Defines a lightweight training stage.
    - Mechanism: Train off-the-shelf pre-trained models on small-scale image-text datasets (e.g., COCO Captions ~118K pairs) for only 1 epoch with a learning rate of $1.0 \times 10^{-6}$ and a batch size of 512.
    - Design Motivation: Pre-training is too expensive, and fine-tuning degrades zero-shot performance. Post-pre-training bridges this gap.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{\text{RaFA}} + \mathcal{L}_{\text{HyCD}}$, combined with equal weights (experiments show equal weighting is optimal).

- $\mathcal{L}_{\text{RaFA}}$: L2 distance to random reference vectors.
- $\mathcal{L}_{\text{HyCD}} = \frac{1}{2}(\mathcal{L}_{\text{HyCD}}^{I \to T} + \mathcal{L}_{\text{HyCD}}^{T \to I})$, in the form of KL divergence.
- Optimizer: AdamW, learning rate $1 \times 10^{-6}$, single epoch.
- Default prior $p(z) = \mathcal{N}(0, I)$ and $\alpha = 0.5$.

## Key Experimental Results

### Main Results (Zero-shot Classification, 12 Datasets, ViT-B/32)

| Method | Metric (Avg Top-1 Acc) | ImageNet | Aircraft | Bird | Description |
|--------|------|------|----------|------|------|
| Pre-trained | 52.74 | 59.04 | 18.81 | 49.37 | Baseline |
| Contrastive | 45.75 | 52.96 | 13.98 | 40.07 | Catastrophic forgetting, performance drops significantly |
| m²-mix | 46.48 | 53.58 | 14.64 | 40.76 | Also degenerated |
| Self-KD | 52.94 | 59.06 | 18.96 | 51.52 | Only retains old knowledge |
| **CLIP-Refine** | **54.69** | **60.93** | **20.77** | **52.72** | **Average improvement +1.95** |

### Zero-shot Retrieval (COCO2017-Val, ViT-B/32)

| Method | T→I R@1 | I→T R@1 | Description |
|------|---------|---------|------|
| Pre-trained | 30.56 | 33.26 | Baseline |
| Contrastive | 34.88 | 31.86 | T→I improves but I→T degenerates |
| **CLIP-Refine** | **37.64** | **38.78** | Significant improvement in both directions |

### Ablation Study

| Configuration | Avg ZS Cls. | Description |
|------|---------|------|
| HyCD only | 53.13 | Distillation only, minor improvement |
| HyCD + $\mathcal{L}_{\text{align}}$ | 45.61 | Direct alignment, severe degeneration |
| CLIP-Refine (RaFA + HyCD) | **54.69** | Optimal combination |
| RaFA without randomness ($\beta=0$) | 53.79 | Randomness is important |
| RaFA standard Gaussian ($\beta=1$) | **54.69** | Default optimal |
| RaFA high variance ($\beta=100$) | 53.59 | Degenerates if variance is too large |

### Key Findings

- **Contrastive learning severely degenerates during post-pre-training**: Small batch sizes (512 vs. 32,768 in pre-training) lead to insufficient negative samples, causing catastrophic forgetting. Image encoders are more prone to overfitting than text encoders.
- **Direct alignment + distillation is the worst combination**: $\mathcal{L}_{\text{align}}$ collapses the uniformity of the feature space, which cannot be rescued even with distillation.
- **CLIP-Refine simultaneously improves the modality gap (0.79 vs. 1.33), alignment (1.28 vs. 1.37), and uniformity (0.049 vs. 0.089) in feature space analysis.**
- Data quality is more important than data quantity: COCO Captions outperforms CC3M/CC12M despite being 10 times smaller in scale.

## Highlights & Insights

- **Post-pre-training is an elegant training paradigm innovation**: It bridges the gap between pre-training and fine-tuning. The concept is clean and practical, and can be integrated with any pre-trained model and fine-tuning method.
- **The indirect alignment concept of RaFA is highly intuitive**: By using shared random reference vectors as a bridge between the two modalities, it avoids the destructive effects of direct alignment and naturally maintains uniformity.
- **The label mixing strategy of HyCD is simple yet effective**: A single $\alpha$ parameter balances old and new knowledge without requiring complex curriculum learning or progressive training.

## Limitations & Future Work

- Currently only validated on CLIP (a contrastive learning model); applicability to Sigmoid loss models like SigLIP remains to be explored.
- Post-pre-training is sensitive to dataset quality, showing limited efficacy on noisy datasets (unless additional filtering is applied).
- Since it only uses 1 epoch, the exploration of potential optimal training epochs remains unaddressed.
- Future work could explore combining RaFA with more advanced distribution matching methods (such as MMD or Sinkhorn).

## Related Work & Insights

- Complementary to prompt tuning (CoOp, MaPLe): Post-pre-training improves the foundation model, while prompt tuning performs task adaptation on top of it.
- The idea of random feature regularization in RaFA is inspired by the single-modality fine-tuning field (R3F, Random feature regularization); Ours creatively extends it to cross-modal alignment.
- Theoretical analysis of the modality gap problem (Qian et al.) demonstrates that contrastive learning cannot fully eliminate the modality gap, providing theoretical support for the necessity of post-pre-training.

## Rating
- Novelty: ⭐⭐⭐⭐ "Post-pre-training" is a novel concept, and the design of RaFA is elegant, though the core techniques (distillation + regularization) are relatively mature.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 classification and 2 retrieval datasets, covering multiple pre-trained models, with comprehensive ablation and feature space analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, coherent logic from motivation to methods to experiments, with well-designed figures/tables.
- Value: ⭐⭐⭐⭐ Highly practical, directly applicable to improving existing CLIP models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] A Two-Stage Progressive Pre-training using Multi-Modal Contrastive Masked Autoencoders](multi-modal_contrastive_masked_autoencoders_a_two-stage_progressive_pre-training.md)
- [\[CVPR 2025\] Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves](skip_tuning_pre-trained_vision-language_models_are_effective_and_efficient_adapt.md)
- [\[CVPR 2025\] Multimodal Autoregressive Pre-training of Large Vision Encoders](multimodal_autoregressive_pre-training_of_large_vision_encoders.md)
- [\[CVPR 2025\] SmartCLIP: Modular Vision-language Alignment with Identification Guarantees](smartclip_modular_vision-language_alignment_with_identification_guarantees.md)
- [\[CVPR 2026\] PowerCLIP: Powerset Alignment for Contrastive Pre-Training](../../CVPR2026/multimodal_vlm/powerclip_powerset_alignment_for_contrastive_pre-training.md)

</div>

<!-- RELATED:END -->
