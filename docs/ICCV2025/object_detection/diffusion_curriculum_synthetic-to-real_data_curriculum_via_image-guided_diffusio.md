---
title: >-
  [Paper Note] Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion
description: >-
  [ICCV 2025][Object Detection][Diffusion Models] This paper leverages the image guidance strength of diffusion models to generate a continuous synthetic-to-real spectrum of data…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Diffusion Models"
  - "Curriculum Learning"
  - "Synthetic Data"
  - "Long-Tail Classification"
  - "Low-Quality Data"
date: 2026-05-08
content_hash: 61b0029141ba2c15
---

# Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion

**Conference**: ICCV 2025
**arXiv**: [2410.13674](https://arxiv.org/abs/2410.13674)  
**Code**: [tianyi-lab/DisCL](https://github.com/tianyi-lab/DisCL)  
**Area**: Object Detection / Data Augmentation
**Keywords**: Diffusion Models, Curriculum Learning, Synthetic Data, Long-Tail Classification, Low-Quality Data

## TL;DR

This paper leverages the image guidance strength of diffusion models to generate a continuous synthetic-to-real spectrum of data, and proposes a Diffusion Curriculum Learning (DisCL) strategy that adaptively selects synthetic data at optimal guidance levels across different training stages, effectively addressing long-tail classification and low-quality data learning challenges.

## Background & Motivation

Deep learning in real-world scenarios frequently faces dual challenges of data quality and quantity: images captured by wildlife cameras, traffic surveillance systems, and similar devices often suffer from poor illumination, motion blur, and occlusion; class imbalance further degrades model performance on tail classes.

Limitations of existing solutions:
- Traditional data augmentation (flipping, cropping, etc.) produces samples with limited diversity
- Although text-guided diffusion models can generate high-quality and diverse data, **text-only control cannot guarantee similarity between synthetic and original images**, causing out-of-distribution data to harm model performance
- Existing methods (e.g., ALIA, LDMLR) still exhibit deficiencies in controlling the synthetic-to-real distribution gap

Core insight: **The image guidance strength $\lambda$ in diffusion models naturally provides a continuous interpolation spectrum from synthetic to real.** Low $\lambda$ produces diverse, prototypical, and simple images, while high $\lambda$ produces images closely resembling the original but potentially inheriting its defects.

## Method

### Overall Architecture

DisCL consists of two stages:
1. **Synthetic-to-Real Data Generation**: Identifies hard samples and generates a full synthetic-to-real interpolation spectrum using image guidance levels $\lambda \in [0,1)$
2. **Generative Curriculum Learning**: Selects synthetic data at appropriate guidance levels for each training stage based on training dynamics

### Key Designs

1. **Synthetic-to-Real Interpolation via Image Guidance Control**:

    - Built upon Stable Diffusion XL; modifies the starting diffusion step in classifier-free guidance as $t(\lambda) = \lfloor(1-\lambda)T\rfloor$
    - Noise initialization formula: $z_{t(\lambda)} = \sqrt{\tilde{\alpha}_{t(\lambda)}} z_{real} + \sqrt{1-\tilde{\alpha}_{t(\lambda)}} \epsilon$
    - $\lambda = 0$ corresponds to pure text guidance (most diverse but furthest from original); $\lambda \rightarrow 1$ corresponds to near-original images (most similar but least diverse)
    - CLIPScore thresholding filters low-fidelity synthetic images where target objects are absent or occluded

2. **Non-Adaptive Curriculum for Long-Tail Classification (Diverse-to-Specific)**:

    - Core mechanism: Given data scarcity in tail classes, low-guidance-level data is used first to increase diversity and quantity, then guidance level is progressively increased to adapt the model to the real distribution
    - Early training employs $\lambda \rightarrow 0$ synthetic data (prototypical features, high diversity)
    - Later training gradually transitions to $\lambda \rightarrow 1$ data (closer to real distribution)
    - The synthetic-to-real gap is bridged incrementally to avoid abrupt distribution shifts

3. **Adaptive Curriculum for Low-Quality Data (Adaptive)**:

    - Hard samples are identified by the real-class probability of a pretrained classifier (lower probability indicates greater difficulty)
    - At each epoch, the guidance level $\lambda$ for the next round is adaptively selected based on "learning progress" (improvement in real-class confidence on a validation subset)
    - The $\lambda$ that maximizes progress at the current training stage is selected, ensuring the most informative data is used at each phase
    - This avoids negative transfer caused by premature introduction of out-of-distribution data in non-adaptive curricula for low-quality settings

### Loss & Training

- Hard sample identification: frequency-based for long-tail tasks; classifier confidence-based for low-quality tasks
- Synthetic data scale: 3–4× the original tail-class data yields optimal results for long-tail tasks
- Training backbones: ResNet-10 for ImageNet-LT; CLIP ViT-B/16 and ViT-L/14 for iWildCam
- DDIM is used as the noise scheduler

## Key Experimental Results

### Main Results: ImageNet-LT Long-Tail Classification

| Method | Curriculum | Many | Medium | Few | Overall |
|--------|-----------|------|--------|-----|---------|
| CE baseline | N/A | 57.70% | 26.60% | 4.40% | 35.80% |
| CE + CUDA | N/A | 57.49% | 28.16% | 6.58% | 36.30% |
| CE + LDMLR | N/A | 57.20% | 29.20% | 7.30% | 37.20% |
| BS + CUDA | N/A | 51.16% | 37.35% | 19.28% | 40.03% |
| **CE + DisCL** | Diverse-to-Specific | 56.78% | 30.73% | **23.64%** | 39.82% |
| **BS + DisCL** | Diverse-to-Specific | 52.68% | 37.68% | 21.36% | **41.33%** |

DisCL improves tail-class accuracy from 4.40% to 23.64% (+19.24%) and overall accuracy by 4.02%.

### Main Results: iWildCam Low-Quality Data Classification

| Method | Curriculum | OOD F1 | ID F1 |
|--------|-----------|--------|-------|
| FLYP | N/A | 35.5 | 52.2 |
| FLYP + ALIA | N/A | 36.9 | 52.6 |
| **FLYP + DisCL** | Adaptive | **38.2** | **54.3** |
| **FLYP + DisCL + WE** | Adaptive | **38.7** | **54.6** |

OOD and ID F1 scores improve by 2.7% and 2.1%, respectively.

### Ablation Study

| Configuration | Few-class (IN-LT) | Note |
|--------------|-------------------|------|
| Text-only Guidance ($\lambda=0$) | 17.90% | Text guidance only; diverse but large distributional gap |
| All-Level Guidance | 19.17% | All levels mixed; no curriculum strategy |
| DisCL Specific-to-Diverse | 18.36% | Reverse curriculum; prone to overfitting the real distribution |
| DisCL Adaptive | 16.78% | Adaptive strategy unsuitable here due to small validation set in long-tail setting |
| **DisCL Diverse-to-Specific** | **23.64%** | Optimal strategy; progressively bridges the distribution gap |

| Configuration | OOD F1 (iWildCam) | Note |
|--------------|-------------------|------|
| Easy-to-Hard (non-adaptive) | 35.2 | Fixed strategy from low to high guidance |
| Random | 35.9 | Random guidance level selection |
| **Adaptive** | **38.2** | Adaptively selected based on learning progress |

### Key Findings

- Synthetic data scaling experiments show that 3–4× tail-class synthetic data is the optimal trade-off; beyond this, head-class accuracy slightly decreases
- The Diverse-to-Specific and Adaptive strategies are best suited for long-tail and low-quality scenarios, respectively, demonstrating that **curriculum strategies must be matched to task characteristics**
- CLIPScore thresholding has a significant impact in the low-quality data setting (high variance in synthetic image quality) but a minor impact on ImageNet (Stable Diffusion produces consistently high-quality outputs for ImageNet categories)

## Highlights & Insights

- The idea of controlling synthetic-to-real interpolation via image guidance level is highly generalizable, elevating diffusion models from a "data augmentation tool" to a "controllable curriculum learning engine"
- The design of two curriculum strategies (non-adaptive vs. adaptive) reflects a deep understanding of the fundamental differences between task types
- The experimental design is systematic and comprehensive, covering four datasets: ImageNet-LT, CIFAR100-LT, iNaturalist2018, and iWildCam

## Limitations & Future Work

- Synthetic data quality is constrained by the capabilities of the diffusion model and CLIP alignment
- Text prompts are based solely on class names, without leveraging image caption information
- Discrepancies in object position and scale between synthetic and real images may amplify the distributional gap
- The method is validated only on classification tasks; more complex detection and segmentation tasks remain unexplored

## Related Work & Insights

- **CUDA** is among the earliest works to combine curriculum learning with data augmentation for long-tail learning, but relies solely on engineered augmentations
- **LDMLR** trains a dedicated long-tail sampler using diffusion models, whereas DisCL directly leverages image guidance from a pretrained diffusion model
- Compared to contrastive learning augmentation methods (e.g., DoCL), DisCL employs generative models rather than dynamic selection of real samples

## Rating

- Novelty: ⭐⭐⭐⭐ — The cross-cutting innovation of image guidance level × curriculum learning is concise and effective
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four datasets, extensive ablations, and validation across multiple loss functions
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, rigorous formal expression, and interpretable illustrations
- Value: ⭐⭐⭐⭐ — Provides a general framework for leveraging diffusion models for data augmentation with broad applicability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion](distil_data-free_inversion_of_suspicious_trojan_inputs_via_latent_diffusion.md)
- [\[ICCV 2025\] YOLOE: Real-Time Seeing Anything](yoloe_realtime_seeing_anything.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](../../CVPR2026/object_detection/invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [\[CVPR 2026\] Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](../../CVPR2026/object_detection/mitigating_memorization_in_texttoimage_diffusion_v.md)
- [\[ICCV 2025\] EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision](evrt-detr_latent_space_adaptation_of_image_detectors_for_event-based_vision.md)

</div>

<!-- RELATED:END -->
