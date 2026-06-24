---
title: >-
  [Paper Note] "CheXWorld: Image World Modeling for Radiograph Representation Learning"
description: >-
  [CVPR2025][Self-Supervised Learning][JEPA] Academic paper note for "CheXWorld: Image World Modeling for Radiograph Representation Learning".
tags:
  - CVPR2025
  - Self-Supervised Learning
  - JEPA
  - World Models
date: 2026-05-08
content_hash: af5920de4b1eb683
---
# CheXWorld: Image World Modeling for Radiograph Representation Learning

**Conference**: CVPR 2025  
**Institution**: Tsinghua University / PLA General Hospital  
**Keywords**: Chest X-ray, JEPA, World Model, Self-supervised Learning, Representation Learning  

## Background & Motivation

Chest X-ray (CXR) is the most common medical imaging examination globally, with billions of images generated annually. While self-supervised learning has achieved remarkable success in natural image domains (e.g., MAE, DINO, JEPA), direct transfer to the radiograph domain faces unique challenges:

**Invariance of Anatomical Structure**: Unlike natural images, the locations of organs in chest radiographs are relatively fixed (e.g., the heart in the lower left, lung fields on both sides). Models need to comprehend this consistent spatial layout.

**Locality of Pathological Lesions**: Most lesions (e.g., nodules, infiltrates) occupy only a small region of the image, and global feature learning may overlook these critical local details.

**Diversity of Domain Variation**: Chest radiographs of the same patient acquired using different devices or imaging parameters can exhibit substantial variations. The model must remain invariant to these domain variations.

Existing methods typically focus on only one of the aforementioned aspects. For instance, MAE emphasizes local reconstruction but ignores the global layout, whereas contrastive learning concentrates on augmentation invariance but potentially discards local details. The core motivation of CheXWorld is to **construct a model that understands the "world" of radiographs, simultaneously modeling local structure, global layout, and domain variation.**

The Joint-Embedding Predictive Architecture (JEPA) provides a natural framework for this, making predictions in the representation space rather than the pixel space, thus avoiding pixel-level reconstruction redundancy and supporting flexible designs for predictive tasks.

## Method

### Overall Architecture

CheXWorld is based on the I-JEPA framework, using ViT as the encoder, and designs three complementary prediction tasks to model different aspects of the radiograph world.

### Task 1: Local Anatomical Structure Modeling

Goal: To understand the local anatomical structures and lesion patterns in chest radiographs.

Mechanism: A masked-prediction paradigm similar to I-JEPA:
- Randomly mask several patches from the input radiograph.
- Encode visible patches using a context encoder.
- Predict the representations of masked patches from the representations of visible patches using a predictor.
- Loss function: L2 distance between the predicted representations of masked patches and the targets from the target encoder.

$$\mathcal{L}_{	ext{local}} = rac{1}{|M|} \sum_{i \in M} \|f_{	ext{pred}}(z_{	ext{ctx}}) - 	ext{sg}(f_{	ext{target}}(x_i))\|_2^2$$

where $M$ is the set of masked locations, and $	ext{sg}$ denotes the stop-gradient operation.

### Task 2: Global Layout Modeling

Goal: To understand the spatial relationships of organs and the overall thoracic structure.

Mechanism: Cross-crop Prediction + Relative Position Encoding:
- Generate two different cropped views (crop A, crop B) from the same radiograph.
- Predict the counterpart representations of crop B using the features of crop A.
- **Key Innovation**: Input the relative spatial position of the two crops as a condition into the predictor.

$$\mathcal{L}_{	ext{global}} = \|f_{	ext{pred}}(z_A, \Delta_{	ext{pos}}) - 	ext{sg}(f_{	ext{target}}(x_B))\|_2^2$$

where $\Delta_{	ext{pos}}$ encodes the relative position between crop A and crop B. This forces the model to understand: "If the heart is visible in the lower-left crop, what should appear in the right crop?"

### Task 3: Domain Variation Modeling

Goal: To learn invariance and sensitivity to different imaging conditions.

Mechanism: Augmentation-conditioned Prediction:
- Apply different data augmentations (brightness, contrast, noise, etc., to simulate different device parameters) to the same image.
- Predict the representation of the augmented image using the augmentation parameters as conditions.

$$\mathcal{L}_{	ext{domain}} = \|f_{	ext{pred}}(z_{	ext{orig}}, c_{	ext{aug}}) - 	ext{sg}(f_{	ext{target}}(T(x)))\|_2^2$$

where $c_{	ext{aug}}$ encodes the augmentation type and intensity, and $T(x)$ is the augmented image.

### Loss & Training

The four loss functions are jointly optimized:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{	ext{local}} + \lambda_2 \mathcal{L}_{	ext{global}} + \lambda_3 \mathcal{L}_{	ext{domain}} + \lambda_4 \mathcal{L}_{	ext{reg}}$$

where $\mathcal{L}_{	ext{reg}}$ is the Variance-Invariance-Covariance Regularization (VICReg) term, which prevents representation collapse.

## Key Experimental Results

### Chest Radiograph Classification

| Method | VinDr AUROC | ShenZhen Acc | CheXpert AUROC |
|------|------------|-------------|----------------|
| ImageNet Pre-trained | 88.45% | 93.12% | 86.72% |
| MAE (Radiograph Fine-tuned) | 91.86% | 96.44% | 89.13% |
| DINO v2 | 93.15% | 97.53% | 90.45% |
| I-JEPA | 93.89% | 97.81% | 91.02% |
| **CheXWorld** | **95.24%** | **98.88%** | **92.56%** |

### Segmentation Task

| Method | SIIM-ACR Dice |
|------|-------------|
| U-Net (ImageNet) | 78.32% |
| TransUNet | 81.45% |
| MAE + U-Net | 82.19% |
| **CheXWorld + U-Net** | **84.58%** |

### Few-shot Learning

| Method | 1-shot AUROC | 5-shot AUROC | 10-shot AUROC |
|------|-------------|-------------|--------------|
| DINO v2 | 58.34% | 72.15% | 79.82% |
| I-JEPA | 60.12% | 74.53% | 81.34% |
| **CheXWorld** | **64.60%** | **78.21%** | **84.15%** |

### Ablation Study

| Configuration | VinDr 1% AUROC | VinDr 100% AUROC |
|------|---------------|-----------------|
| Local Only | 84.71% | 93.89% |
| Local + Global | 87.43% | 94.56% |
| Local + Domain | 86.92% | 94.34% |
| **Local + Global + Domain (Full)** | **90.53%** | **95.24%** |

The gain from 84.71% to 90.53% (+5.82%) validates the complementarity of the three tasks, with a more pronounced impact particularly under data-scarce conditions (1%).

## Highlights & Insights

1. **World Model Perspective**: Formulates radiograph representation learning as a "world model" problem for the first time, unifying the modeling of structure, layout, and domain variation.
2. **Innovative Application of JEPA in Medical Imaging**: Designs three prediction tasks tailored specifically to medical imaging.
3. **Cross-crop Position Prediction**: Leverages the spatial consistency of radiograph anatomical structures to achieve global understanding through relative positional conditioning.

## Limitations & Future Work

- Validated only on chest radiographs; its applicability to other medical imaging modalities (e.g., CT, MRI) remains unexplored.
- The weights $\lambda$ for the three tasks require manual tuning.
- The impact of pre-training data scale has not been fully investigated.

## Summary

CheXWorld creatively applies the JEPA framework to radiograph representation learning, modeling different aspects of the radiograph world through a tri-fold task design. It consistently outperforms existing methods across classification, segmentation, and few-shot learning tasks, demonstrating particularly outstanding performance in low-data scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling](from_prototypes_to_general_distributions_an_efficient_curriculum_for_masked_imag.md)
- [\[CVPR 2025\] Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physical_systems.md)
- [\[ICML 2025\] AdaWorld: Learning Adaptable World Models with Latent Actions](../../ICML2025/self_supervised/adaworld_learning_adaptable_world_models_with_latent_actions.md)
- [\[CVPR 2026\] Suppressing Non-Semantic Noise in Masked Image Modeling Representations](../../CVPR2026/self_supervised/suppressing_non-semantic_noise_in_masked_image_modeling_representations.md)
- [\[ICLR 2026\] CARL: Camera-Agnostic Representation Learning for Spectral Image Analysis](../../ICLR2026/self_supervised/carl_camera-agnostic_representation_learning_for_spectral_image_analysis.md)

</div>

<!-- RELATED:END -->
