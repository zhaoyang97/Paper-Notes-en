---
title: >-
  [Paper Note] Synthetic Data is an Elegant GIFT for Continual Vision-Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Continual learning] Using Stable Diffusion to generate synthetic images from class names, knowledge distillation is performed via contrastive distillation + image-text alignment constraints + adaptive weight consolidation. With only 1K synthetic images per task, this approach outperforms ZSCL, a continual learning method that uses 100K real ImageNet images.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Continual learning"
  - "synthetic data"
  - "knowledge distillation"
  - "catastrophic forgetting"
  - "VLM adaptation"
date: 2026-05-08
content_hash: ffc21086b993dc42
---

# Synthetic Data is an Elegant GIFT for Continual Vision-Language Models

**Conference**: CVPR 2025  
**arXiv**: [2503.04229](https://arxiv.org/abs/2503.04229)  
**Code**: [https://github.com/Luo-Jiaming/GIFT_CL](https://github.com/Luo-Jiaming/GIFT_CL)  
**Area**: Multimodal VLM  
**Keywords**: Continual learning, synthetic data, knowledge distillation, catastrophic forgetting, VLM adaptation

## TL;DR
Using Stable Diffusion to generate synthetic images from class names, knowledge distillation is performed via contrastive distillation + image-text alignment constraints + adaptive weight consolidation. With only 1K synthetic images per task, this approach outperforms ZSCL, a continual learning method that uses 100K real ImageNet images.

## Background & Motivation

**Background**: VLMs face catastrophic forgetting in continual learning—a sharp drop in performance on previous tasks after fine-tuning on new tasks. Existing methods such as ZSCL mitigate forgetting by utilizing real ImageNet images for knowledge distillation.

**Limitations of Prior Work**: (1) High cost of acquiring and storing real data (ZSCL requires 100K ImageNet images), along with privacy and copyright concerns. (2) Existing distillation methods rely on feature distance loss (MSE), but the teacher model itself might make mistakes, and blindly reducing feature distances propagates these errors. (3) Parameter regularization methods like EWC calculate Fisher information at the start of training, whereas requirements may dynamically evolve during training.

**Key Challenge**: Continual learning requires "replaying" past knowledge to prevent forgetting, but obtaining and storing data from previous tasks is highly expensive.

**Goal**: To replace real data with zero-cost synthetic data for continual learning, while simultaneously improving distillation and regularization strategies.

**Key Insight**: Directly generate images using class names as prompts via Stable Diffusion. These synthetic images are sufficient to cover key visual concepts. Combine this with contrastive distillation to maintain cross-modal alignment structure, hard image-text alignment to correct teacher mistakes, and adaptive Fisher updates for parameter regularization.

**Core Idea**: Implementing 1K synthetic images + contrastive distillation + hard image-text alignment + adaptive Fisher regularization to achieve better continual learning performance than using 100K real images.

## Method

### Overall Architecture
For each new task: Generate 1K synthetic images from the class name pool (downstream classes + random ImageNet classes) $\to$ Fine-tune CLIP on new task data while performing knowledge distillation using synthetic images (Teacher = model from the previous task) $\to$ Leverage three joint losses to prevent forgetting.

### Key Designs

1. **Contrastive Distillation (CD)**:

    - **Function**: Maintains the cross-modal alignment structure of the teacher model.
    - **Mechanism**: Constructs teacher/student image-text similarity matrices and aligns their row distributions (image-to-text) and column distributions (text-to-image) using KL divergence. This outperforms feature MSE because it preserves the global structural relationships of "which image-text pairs should be close" rather than local feature coordinates.
    - **Design Motivation**: Ablation shows that CD outperforms MSE feature distillation by over 5 percentage points.

2. **Image-Text Alignment Constraint (ITA)**:

    - **Function**: Corrects errors made by the teacher model.
    - **Mechanism**: Uses the identity matrix as a "hard target"—the similarity between a synthetic image and its corresponding class name should be 1, and 0 for other class names. The hard target is blended into the teacher's soft target with a weight of $\beta=0.25$: $(1-\beta) \cdot p_{teacher} + \beta \cdot I$.
    - **Design Motivation**: The teacher model may produce errors on synthetic data (e.g., the teacher might mistake a synthetic "cat" image as somewhat looking like a "dog"). ITA uses ground truth to correct these biases.

3. **Adaptive Weight Consolidation (AWC)**:

    - **Function**: Dynamically updates parameter importance estimation.
    - **Mechanism**: Unlike EWC which calculates Fisher information only once at the beginning of training, AWC updates the Fisher info in real-time at each training step using the gradient of the distillation loss. This allows the regularization to adaptively adjust throughout the training process.
    - **Design Motivation**: Static Fisher information fails to capture shifts in parameter importance during training.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{task} + \lambda_{CD} \cdot \mathcal{L}_{CD} + \lambda_{ITA} \cdot \mathcal{L}_{ITA} + \lambda_{AWC} \cdot \mathcal{L}_{AWC}$. Stable Diffusion v1.5 is used to generate images.

## Key Experimental Results

### Main Results

| Method | Data | Transfer↑ | Avg↑ | Last↑ |
|------|------|-----------|------|-------|
| Zero-shot | - | 69.4 | 65.3 | 65.3 |
| ZSCL | 100K ImageNet | 68.1 | 75.4 | 83.6 |
| MoE-Adapter | Real | 68.9 | 76.7 | 85.0 |
| **GIFT** | **1K Synthetic** | **69.3** | **77.3** | **86.0** |

### Ablation Study

| Component | Transfer Δ | Avg Δ | Last Δ |
|------|-----------|-------|--------|
| +CD only | +2.5 | +7.8 | +2.7 |
| +CD+ITA | +7.3 | +13.6 | +8.8 |
| +CD+ITA+AWC | **+8.3** | **+14.6** | **+10.1** |

### Key Findings
- **1K Synthetic Images > 100K Real Images**: GIFT outperforms ZSCL using only 1% of the data volume, proving that the concept coverage of synthetic data is sufficient for knowledge distillation.
- **ITA Contributes the Most**: Adding ITA improves Transfer by 4.8 percentage points (from +2.5 to +7.3), demonstrating that correcting teacher errors is crucial.
- **AWC > EWC**: Adaptive Fisher updates provide an improvement of over 1 percentage point on Avg compared to static computation.

## Highlights & Insights
- **"Continual learning with synthetic data"** challenges the conventional paradigm that "real-data replay is necessary"—the key lies in concept coverage rather than image realism.
- **The error-correcting mechanism of ITA** is highly practical—blending ground truth into soft labels when the teacher model makes errors is an elegant compromise.

## Limitations & Future Work
- The quality and diversity of synthetic images are bounded by the capabilities of Stable Diffusion.
- The design of the class name pool is manual (downstream classes + random ImageNet classes); an automatic selection of more representative classes might yield better results.
- Evaluated only on CLIP classification scenarios; performance on tasks like VQA/retrieval remains unexplored.

## Related Work & Insights
- **vs ZSCL**: ZSCL uses 100K real images for feature distillation. GIFT comprehensively outperforms it using only 1K synthetic images and superior distillation strategies.
- **vs MoE-Adapter**: MoE introduces extra parameters, whereas GIFT does not modify the model architecture.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel combination of synthetic data, contrastive distillation, and ITA error correction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed component ablations over two types of MTIL sequences.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation of methods.
- Value: ⭐⭐⭐⭐ Practical significance for privacy-sensitive or data-constrained continual learning scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[CVPR 2025\] SVLTA: Benchmarking Vision-Language Temporal Alignment via Synthetic Video Situation](svlta_benchmarking_vision-language_temporal_alignment_via_synthetic_video_situat.md)
- [\[ECCV 2024\] Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models](../../ECCV2024/multimodal_vlm/select_and_distill_selective_dual-teacher_knowledge_transfer_for_continual_learn.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[CVPR 2025\] Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](molmo_and_pixmo_open_weights_and_open_data_for_state-of-the-art_vision-language_.md)

</div>

<!-- RELATED:END -->
