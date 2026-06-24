---
title: >-
  [Paper Note] Pairwise Distance Distillation for Unsupervised Real-World Image Super-Resolution
description: >-
  [ECCV 2024][Image Restoration][Real-World Super-Resolution] This paper proposes a pairwise distance distillation framework that achieves degradation adaptation for unsupervised real-world image super-resolution by distilling the intra- and inter-model distance relationships between a specialized model and a generalized model.
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Real-World Super-Resolution"
  - "Unsupervised Learning"
  - "Knowledge Distillation"
  - "Pairwise Distance"
  - "Degradation Adaptation"
date: 2026-05-08
content_hash: 8753772b3f965f6e
---

# Pairwise Distance Distillation for Unsupervised Real-World Image Super-Resolution

**Conference**: ECCV 2024  
**arXiv**: [2407.07302](https://arxiv.org/abs/2407.07302)  
**Code**: Available  
**Area**: Image Restoration  
**Keywords**: Real-World Super-Resolution, Unsupervised Learning, Knowledge Distillation, Pairwise Distance, Degradation Adaptation

## TL;DR

This paper proposes a pairwise distance distillation framework that achieves degradation adaptation for unsupervised real-world image super-resolution by distilling the intra- and inter-model distance relationships between a specialized model and a generalized model.

## Background & Motivation

Single image super-resolution (SISR) is a classical problem in computer vision. Standard methods use known downsampling kernels (such as bicubic) to construct low-resolution (LR) and high-resolution (HR) training pairs. However, in real-world scenarios, the degradation process of LR images is unknown and far more complex than simple downsampling, involving noise, blur, compression artifacts, etc. This is referred to as the **Real-World SR (RWSR)** problem.

Currently, mainstream approaches for handling RWSR fall into two categories: (1) **Blind SR methods**: training a generalized model applicable to various degradations (e.g., Real-ESRGAN) using complex synthetic degradation augmentation, which sacrifices performance on specific degradations for generalization; (2) **Degradation estimation methods**: estimating degradation parameters first and then performing SR, which is inherently challenging.

The key challenge lies in the trade-off: generalized models pursue broad degradation coverage but underperform on specific degradations, whereas specialized models excel at known degradations but fail on unknown ones. This paper presents a novel perspective: through knowledge distillation, a specialized model trained on synthetic degradations is adapted to target real-world degradations while referencing a pre-trained generalized model.

## Method

### Overall Architecture

The pairwise distance distillation framework consists of three players: (1) a **Specialized Model** trained on synthetic degradations acting as the student model; (2) a pre-trained **Generalized Model** acting as the auxiliary teacher; and (3) the student model adapting to the target degradation by distilling two types of distance relationships—intra-model distance and inter-model distance.

### Key Designs

1. **Intra-model Distance Distillation**:
    - Function: Maintaining the relative structural relationships in the feature space.
    - Mechanism: For multiple LR images with the same degradation, the pairwise distance matrix between the model's output features is computed. By distilling the distance matrix of the generalized model to the specialized model, the specialized model learns the feature organization structure of the generalized model. This is more flexible than directly distilling absolute feature values because it allows for affine transformations in the feature space.
    - Design Motivation: In the absence of paired data, directly supervising SR output is impossible; however, the relative relationships between features can be exploited as indirect supervisory signals.

2. **Inter-model Distance Distillation**:
    - Function: Aligning the output relationship between the specialized model and the generalized model.
    - Mechanism: For a given input, the outputs of the specialized model and the generalized model are computed separately, and then the distance relationship between them is distilled. This enables the specialized model to learn the generalized model's processing strategies on real-world degradations while retaining its own knowledge of synthetic degradations.
    - Design Motivation: Although the generalized model is less precise on specific degradations, it contains valuable prior knowledge about real-world degradations.

3. **Degradation Adaptation Training Strategy**:
    - Function: Performing adaptation on unlabeled real-world data.
    - Mechanism: Unlabeled LR images from the target domain are used for adaptation training. The specialized model is fine-tuned on real-world data via distillation losses to progressively adapt to the target degradation. During training, the parameters of the generalized model are frozen, and only the specialized model is updated.
    - Design Motivation: Although real-world LR images lack paired HR ground-truths, they contain information about degradation patterns, which can be indirectly utilized through distillation.

### Loss & Training

- Intra-model distance loss: Minimizing the difference between the pairwise distance matrices of the specialized model and the generalized model on the same batch of samples.
- Inter-model distance loss: Constraining the output distance between the specialized model and the generalized model for the same input.
- Optional perceptual and adversarial losses are used to further enhance visual quality.
- Adaptation training uses a smaller learning rate to prevent excessive deviation from the pre-trained knowledge.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| RealSR | PSNR ↑ | SOTA | Real-ESRGAN | +0.5-1.0dB |
| RealSR | LPIPS ↓ | SOTA | BSRGAN | Significantly reduced |
| DRealSR | PSNR ↑ | SOTA | Multiple baselines | +0.3-0.8dB |
| Real-world images | Visual Quality | Best | Generalized Model | Clearer and more natural |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Intra-model distance only | Partial improvement | Learns feature structure but lacks cross-model information |
| Inter-model distance only | Partial improvement | Learns model alignment but has weak feature structure |
| Both combined | Best | Obvious complementary relationship |
| Different generalized models | Robust | Insensitive to the choice of the generalized model |

### Key Findings

- Pairwise distance distillation is more suitable for unsupervised scenarios than absolute feature distillation.
- Intra- and inter-model distances provide complementary supervisory signals.
- Through adaptation, the specialized model can outperform the generalized model on specific degradations.
- The method is effective against various types of degradations (noise, blur, JPEG compression).

## Highlights & Insights

- Solving unsupervised RWSR from a distillation perspective serves as a novel angle.
- The idea of using pairwise distances as unsupervised signals is highly generalizable.
- The combination of specialized and generalized models leverages the complementary strengths of both models.
- The method does not require degradation estimation or complex synthetic augmentation.

## Limitations & Future Work

- A pre-trained generalized model is required as a teacher; thus, the method's performance partially depends on the teacher's quality.
- Pairwise distance distillation requires computing distance matrices within each batch, which increases computational overhead.
- For extreme or rare degradation types, if the generalized model also performs poorly, the distillation performance may be limited.
- Strategies for online updating of the teacher model could be explored.

## Related Work & Insights

- **Real-ESRGAN**: A representative generalized RWSR method, trained through complex synthetic degradation augmentation.
- **CycleSR / DASR**: Unsupervised methods that utilize domain adaptation or cycle consistency.
- **Knowledge Distillation**: Works like FitNet and CRD provide the theoretical foundation for distillation.
- Insight: The concept of pairwise distance distillation can be generalized to other unsupervised image restoration tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The perspective of pairwise distance distillation is novel, bringing distillation into unsupervised SR.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive evaluations on multiple datasets and thorough ablation studies.
- Writing Quality: ⭐⭐⭐ The logical flow is clear, but some technical details could be more detailed.
- Value: ⭐⭐⭐ Provides practical contributions to real-world SR, though the applicable scenarios are relatively specific.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] A New Dataset and Framework for Real-World Blurred Images Super-Resolution](a_new_dataset_and_framework_for_real-world_blurred_images_super-resolution.md)
- [\[ICLR 2026\] VARestorer: One-Step VAR Distillation for Real-World Image Super-Resolution](../../ICLR2026/image_restoration/varestorer_one-step_var_distillation_for_real-world_image_super-resolution.md)
- [\[ECCV 2024\] DenoiSplit: A Method for Joint Microscopy Image Splitting and Unsupervised Denoising](denoisplit_a_method_for_joint_microscopy_image_splitting_and_unsupervised_denois.md)
- [\[ECCV 2024\] Towards Real-world Event-guided Low-light Video Enhancement and Deblurring](towards_real-world_event-guided_low-light_video_enhancement_and_deblurring.md)
- [\[ECCV 2024\] Rethinking Image Super-Resolution from Training Data Perspectives](rethinking_image_super-resolution_from_training_data_perspectives.md)

</div>

<!-- RELATED:END -->
