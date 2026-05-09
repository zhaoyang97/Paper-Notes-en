---
title: >-
  [Paper Note] VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models
description: >-
  [NeurIPS 2025][Model Compression][Visual Foundation Models] VESSA proposes an unsupervised adaptation method for visual foundation models using short object-centric videos. Through a self-distillation framework combined with LoRA parameter-efficient fine-tuning and an uncertainty-weighted loss, it significantly improves downstream classification performance in target domains without requiring any labeled data.
tags:
  - NeurIPS 2025
  - Model Compression
  - Visual Foundation Models
  - Self-Supervised Fine-Tuning
  - Video Adaptation
  - Self-Distillation
  - LoRA
date: 2026-05-08
content_hash: ec8f5e0bc1ae3d6c
---

# VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.20994](https://arxiv.org/abs/2510.20994)
**Code**: [GitHub](https://github.com/jesimonbarreto/VESSA)
**Area**: Model Compression
**Keywords**: Visual Foundation Models, Self-Supervised Fine-Tuning, Video Adaptation, Self-Distillation, LoRA

## TL;DR

VESSA proposes an unsupervised adaptation method for visual foundation models using short object-centric videos. Through a self-distillation framework combined with LoRA parameter-efficient fine-tuning and an uncertainty-weighted loss, it significantly improves downstream classification performance in target domains without requiring any labeled data.

## Background & Motivation

Visual foundation models (VFMs) such as DINO and DINOv2 acquire powerful general-purpose visual representations through large-scale self-supervised pretraining. However, when applied to domains with distribution shifts relative to pretraining data (e.g., remote sensing, medical imaging), their performance tends to degrade.

Existing adaptation methods primarily rely on supervised fine-tuning, which demands large amounts of labeled data and is infeasible in many practical settings. Although unsupervised continued pretraining has been widely adopted in NLP to adapt language models to new domains, this strategy has not been demonstrated to be effective in the visual domain. The authors find that naively applying self-distillation methods directly during the fine-tuning stage leads to model degradation — the model rapidly forgets pretrained knowledge and enters a degraded state.

This motivates three core questions: (1) How can pretrained visual models be adapted to specific domains without supervision? (2) What form of unlabeled data is most suitable for such adaptation? (3) What learning strategies can effectively adapt pretrained visual representations under these constraints?

## Method

### Overall Architecture

VESSA consists of three main modules: Frame Selection, Preprocessing & Augmentation, and Model Fine-tuning. The input is short object-centric videos, and the output is a visual model with representations adapted to the target domain.

### Key Designs

1. **Frame Selection Module**: $n$ pairs of frames are randomly sampled from each video. For each pair, a starting frame index $t \sim \mathcal{U}(1, T-\delta_{\max})$ is first selected, and the second frame is sampled at temporal offset $\delta \sim \mathcal{U}(1, \delta_{\max})$. This randomization strategy introduces temporal diversity, enabling the model to learn robust representations across different viewpoints. Experiments show that randomly sampling $\delta \in [5,10]$ yields the best results, indicating that moderate viewpoint differences facilitate representation learning.

2. **Progressive Unfreezing Strategy**: This is the core design for preventing fine-tuning degradation. Specifically:

    - **Phase 1**: The entire backbone is frozen; only the projection head is trained for several epochs to adapt it to the existing embedding space.
    - **Phase 2**: The backbone is progressively unfrozen. LoRA is applied to the first $H$ layers for low-rank adaptation (updating only low-rank matrices $\Delta W = AB$ in Q/K/V projections, where $r \ll \min(d,k)$), preserving low-level visual features; the last $L$ layers are fully unfrozen for standard updates to adapt high-level semantic representations.

   Experiments show that unfreezing the last 2 layers yields optimal performance (91.87%), and unfreezing more layers leads to degradation.

3. **Uncertainty-Weighted Self-Distillation Loss (UWSD)**: An uncertainty weighting scheme is introduced on top of the standard DINO loss. The entropy $\mathcal{H}(q)$ of the teacher network's output distribution is computed to modulate each sample's contribution to the loss:

   $$w(q) = 1 + \gamma \cdot \mathcal{H}(q)$$

   $$\mathcal{L}_{\text{UWSD}} = \frac{1}{N} \sum_{(q,s,s_{lc_i}) \in \mathcal{B}} w(q) \cdot \mathcal{L}_{\text{DINO}}(q, s, s_{lc_i})$$

   Teacher outputs with high entropy (uncertainty) receive larger weights, prioritizing learning from hard samples. $\gamma=1$ yields stable performance.

### Loss & Training

The base loss is DINO's cross-entropy self-distillation loss, with the teacher network updated via EMA of student weights. The overall strategy involves training the head for 10 epochs followed by full model training for 10 epochs. Batch size is 256, input resolution is $224 \times 224$, and 3 frame pairs are sampled per video. Local crops are also sampled in pairs from different frames to maintain temporal consistency.

## Key Experimental Results

### Main Results

| Dataset | Model | Pretrained | ExPLoRA+Video | VESSA | Gain |
|--------|------|-----------|---------------|-------|------|
| CO3D | DINO | 78.86% | 83.64% | **85.03%** | +1.39 |
| CO3D | DINOv2 | 87.86% | 89.64% | **91.85%** | +2.21 |
| CO3D | TIPS | 60.02% | — | **70.56%** | +10.54 |
| MVImageNet | DINO | 90.44% | 87.74% | **92.51%** | +4.77 |
| MVImageNet | DINOv2 | 95.75% | 96.15% | 96.01% | ≈on par |
| MVImageNet | TIPS | 78.65% | — | **80.54%** | +1.89 |

### Ablation Study

| Configuration | Accuracy | Notes |
|------|--------|------|
| Full VESSA | **91.87%** | All components enabled |
| w/o UWSD | 90.92% | UWSD contributes ~1% |
| w/o local crops | 90.53% | Local crops contribute ~1.3% |
| w/o head training | 80.87% | Head training is the most critical factor (+11%) |
| Images instead of video | 88.54% | Video input outperforms images by 3.3% |
| Unfreeze 1 layer | 87.14% | Sensitive to number of unfrozen layers |
| Unfreeze 3 layers | 90.80% | 2 layers is optimal |
| DINO from scratch (Image) | 33.86% | Insufficient data |
| DINO from scratch (Video) | 39.39% | Video outperforms images by 5.53% |

### Key Findings

- **Head training is the most critical component**: Skipping head training and fine-tuning directly causes performance to drop from 91.87% to 80.87%, providing direct evidence that a randomly initialized projection head leads to gradient instability.
- **Video data consistently outperforms images**: Across all configurations, video input uniformly outperforms corresponding image input, indicating that multi-view temporal information provides effective supervision signals beyond simple data augmentation.
- **Simulating video with image transformations fails**: Adding translation, rotation, and scale augmentations to images only achieves 81.49% vs. 91.85% (DINOv2), demonstrating that the advantage of video stems from genuine viewpoint variation rather than simple geometric transforms.

## Highlights & Insights

- Successfully transfers the concept of "unsupervised continued pretraining" from NLP to the visual domain, filling a gap in unsupervised adaptation of visual foundation models.
- The progressive unfreezing strategy is simple yet highly effective, avoiding the representation degradation commonly observed in self-supervised fine-tuning.
- Only short object-centric videos are required (no annotations), lowering the barrier for data collection.

## Limitations & Future Work

- Catastrophic forgetting is a significant concern: after adaptation, KNN accuracy on ImageNet drops from 82.1% to 17.15% (DINOv2), rendering the model unusable as a general-purpose model.
- The method depends on object-centric short videos; such structured multi-view data is not readily available in many scenarios.
- Cross-dataset generalization is limited: training on MVImageNet and evaluating on CO3D results in a 5–7 percentage point performance drop.

## Related Work & Insights

- The idea of applying LoRA to self-supervised continued learning originates from ExPLoRA; however, VESSA does not construct a new foundation model but instead adapts directly to downstream tasks.
- The self-distillation framework builds on DINO, but introduces critical training strategy improvements that make it viable in the fine-tuning setting.
- Compared to video-to-image knowledge transfer methods such as VITO and ViC-MAE, VESSA requires neither complex frame selection pipelines nor hybrid losses.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic exploration of unsupervised video-based adaptation for visual foundation models.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 33 foundation models × 22 datasets, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear logic with well-motivated progressive exposition.
- Value: ⭐⭐⭐⭐ Highly practical, providing a viable solution for scenarios lacking labeled data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Revisiting Semi-Supervised Learning in the Era of Foundation Models](revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[NeurIPS 2025\] C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)

</div>

<!-- RELATED:END -->
