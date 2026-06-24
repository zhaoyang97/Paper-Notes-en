---
title: >-
  [Paper Note] Unsupervised Multi-modal Medical Image Registration via Invertible Translation
description: >-
  [ECCV 2024][Medical Imaging][Multi-modal registration] This paper proposes INNReg, which translates multi-modal medical images into a single modality using an invertible neural network, and then performs registration on the single-modality images. Combined with a barrier loss function based on normalized mutual information, it achieves registration accuracy superior to existing methods on MRI T1/T2 and MRI/CT datasets.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Multi-modal registration"
  - "invertible neural network"
  - "image translation"
  - "unsupervised learning"
  - "mutual information"
date: 2026-05-08
content_hash: ed17136476c7451e
---

# Unsupervised Multi-modal Medical Image Registration via Invertible Translation

**Conference**: ECCV 2024  
**PDF**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04467.pdf)
**Code**: [https://github.com/MeggieGuo/INNReg](https://github.com/MeggieGuo/INNReg)  
**Area**: Medical Images  
**Keywords**: Multi-modal registration, invertible neural network, image translation, unsupervised learning, mutual information

## TL;DR

This paper proposes INNReg, which translates multi-modal medical images into a single modality using an invertible neural network, and then performs registration on the single-modality images. Combined with a barrier loss function based on normalized mutual information, it achieves registration accuracy superior to existing methods on MRI T1/T2 and MRI/CT datasets.

## Background & Motivation

**Background**: Multi-modal medical image registration is crucial in clinical diagnosis and image-guided treatment, providing clinicians with complementary anatomical/functional information. Existing methods are mainly divided into two categories: direct registration methods based on traditional similarity metrics (such as mutual information, normalized cross-correlation), and indirect registration methods based on image translation.

**Limitations of Prior Work**: Direct registration methods face complex and unknown spatial relationships between multi-modal images, making it difficult to design effective similarity metrics. Although translation-based methods (e.g., CycleGAN, RegGAN) transform the problem into single-modality registration, they tend to destroy the geometric consistency of images during translation—the translated images may be structurally inconsistent with the original images, leading to distorted registration results.

**Key Challenge**: Image translation requires sufficient expressiveness to capture cross-modal appearance transformations, while strictly maintaining geometric consistency. The unidirectional mapping of existing translation networks (e.g., U-Net, ResNet) struggles to meet both needs simultaneously, and their training is unstable.

**Goal**: (1) How to strictly maintain geometric consistency during image translation? (2) How to design a more effective registration loss to improve multi-modal registration accuracy?

**Key Insight**: The authors observe that invertible neural networks (INNs) naturally possess bijective properties—the forward and inverse mappings are strictly reciprocal, ensuring that the geometric structure before and after translation is completely consistent. Meanwhile, the information-lossless nature of INNs allows them to preserve all structural information during the translation process.

**Core Idea**: Use an invertible neural network as an image translator to guarantee geometric consistency, combined with a barrier loss function based on normalized mutual information to constrain the registration network, achieving unsupervised multi-modal medical image registration.

## Method

### Overall Architecture

INNReg consists of two subnetworks: (1) an INN-based image translation network that translates images from different modalities (e.g., MRI T1 and T2) into a unified modality space; (2) a registration network that takes the translated single-modality image pairs to predict the deformation field for aligning the original multi-modal images. The input is a pair of multi-modal images (moving and fixed images), and the output is the aligned warped image along with the corresponding deformation field.

### Key Designs

1. **Invertible Neural Network Image Translator (INN Translator)**:

    - **Function**: Translate medical images from different modalities into a unified modality space while strictly maintaining geometric structures.
    - **Mechanism**: Build an invertible translation network based on affine coupling layers. The input features are split into two parts $x_1, x_2$, and invertible mapping is achieved through cross-affine transformations: $y_1 = x_1 \odot \exp(s_2(x_2)) + t_2(x_2)$, $y_2 = x_2 \odot \exp(s_1(y_1)) + t_1(y_1)$, where $s, t$ are arbitrary functions. The inverse transformation can precisely reconstruct the input by calculating in reverse, ensuring zero loss of geometric structure.
    - **Design Motivation**: Unlike the cycle consistency constraints of methods like CycleGAN, the invertibility of INN is mathematically guaranteed, eliminating the dependency on additional reconstruction losses to approximate geometric consistency.

2. **Dynamic Depthwise Separable Convolution Local Attention (DDC-Local Attention)**:

    - **Function**: Enhance the local feature extraction capability of the affine functions $s, t$ within the INN translator.
    - **Mechanism**: Introduce dynamic depthwise separable convolutions in the subnetworks of the affine coupling layer to dynamically generate convolution kernel weights based on the input content, while combining local attention mechanisms to capture modality-specific features within spatial neighborhoods. This enables the translation network to adaptively focus on modality discrepancies in different regions.
    - **Design Motivation**: Standard affine coupling layers use simple MLPs or CNNs as subnetworks, which have limited representation capacity and struggle to handle complex local modality discrepancies in medical images (e.g., contrast inversion of gray/white matter in MRI T1/T2).

3. **Normalized Mutual Information-based Barrier Loss (NMI-Barrier Loss)**:

    - **Function**: Constrain the optimization direction of the registration network to avoid local optima.
    - **Mechanism**: Convert Normalized Mutual Information (NMI) into a barrier-form loss function: $L_{barrier} = -\log(\text{NMI}(I_{fixed}, I_{warped}) - \tau)$, where $\tau$ is a threshold. When the NMI approaches $\tau$, the loss increases sharply, forming a "barrier" that forces the optimization process away from low-NMI regions.
    - **Design Motivation**: Traditional NMI loss suffers from flat gradients during optimization, making it prone to trapping in local optima. The barrier form amplifies the gradients of NMI near the target threshold via a logarithmic function, accelerating convergence and improving registration accuracy.

### Loss & Training

The total loss is a weighted sum of the translation loss and the registration loss: $L = L_{trans} + \lambda L_{reg}$. The translation loss includes adversarial loss and L1 reconstruction loss; the registration loss includes the NMI-barrier loss and a regularization term for the deformation field (to encourage a smooth deformation field). End-to-end training is adopted to jointly optimize the translation and registration networks.

## Key Experimental Results

### Main Results

| Dataset | Metric | INNReg | RegGAN | CycleGAN+VoxelMorph | Gain |
|--------|------|--------|--------|---------------------|------|
| MRI T1/T2 | Dice ↑ | **0.812** | 0.782 | 0.769 | +3.8% |
| MRI T1/T2 | HD95 ↓ | **2.34** | 2.71 | 2.89 | -13.7% |
| MRI/CT | Dice ↑ | **0.776** | 0.741 | 0.728 | +4.7% |
| MRI/CT | HD95 ↓ | **3.12** | 3.58 | 3.79 | -12.8% |

### Ablation Study

| Configuration | Dice ↑ | HD95 ↓ | Description |
|------|--------|--------|------|
| Full INNReg | **0.812** | **2.34** | Full Model |
| w/o INN (using ResNet) | 0.783 | 2.67 | INN contributes about 3.6% |
| w/o DDC-Attention | 0.795 | 2.52 | Dynamic attention contributes about 2.1% |
| w/o Barrier Loss (using standard NMI) | 0.798 | 2.48 | Barrier loss contributes about 1.7% |
| w/o translation network (direct multi-modal registration) | 0.752 | 3.11 | Translation strategy is crucial |

### Key Findings
- The INN translator is the most critical component; replacing it with a standard ResNet leads to the largest drop in Dice, proving the importance of maintaining geometric consistency.
- The barrier loss is about 30% faster in convergence speed compared to the standard NMI loss, and achieves higher final accuracy.
- In scenarios with larger modality discrepancies such as MRI/CT, the advantages of INNReg are even more pronounced.

## Highlights & Insights
- **INN guaranteeing geometric consistency** is the most ingenious design of this work: mathematically derived invertibility is used to replace the approximate constraint of cycle consistency, fundamentally solving the geometric distortion issue during translation. This concept can be transferred to any image translation task where structural consistency needs to be maintained.
- The design of the **barrier loss function** is inspired by the barrier method in optimization theory, transforming NMI from an evaluation metric into an optimization objective with strong gradient signals. This is worth referencing in other scenarios requiring mutual information optimization.
- End-to-end joint training of the translation and registration networks avoids the issue of error accumulation inherent in two-stage methods.

## Limitations & Future Work
- Experiments are only conducted on 2D slices without extension to 3D volume registration, whereas clinical applications typically require 3D registration.
- The dataset size is relatively small (BraTS + Harvard), and the generalization capability remains to be validated on larger-scale datasets.
- INN consumes a large amount of memory (as intermediate states need to be stored for backward computation), which limits its capacity to handle high-resolution images.
- Registration under large-deformation scenarios is not considered; cascaded or diffeomorphic constraints could be introduced to enhance the performance in handling large deformations.

## Related Work & Insights
- **vs RegGAN**: RegGAN uses a standard GAN for image translation, relying on cycle consistency to constrain geometric consistency, which is a soft constraint that cannot fully avoid structural distortion. INNReg replaces this soft constraint with the mathematical invertibility of INN.
- **vs VoxelMorph**: VoxelMorph is a classic unsupervised registration method but can only handle single-modality registration. INNReg transforms multi-modal registration into single-modality registration via translation, expanding the applicability of VoxelMorph.
- **vs SYMNet**: SYMNet uses a symmetric registration loss but does not involve cross-modal translation. The two can be combined—using INN translation in conjunction with symmetric registration.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using INN for image translation to maintain geometric consistency is novel, though INN itself is not a new technology.
- Experimental Thoroughness: ⭐⭐⭐ Only two datasets, no 3D experiments, and ablation studies are not detailed enough.
- Writing Quality: ⭐⭐⭐⭐ Clear chain of motivation and accurate description of the methodology.
- Value: ⭐⭐⭐⭐ Provides an effective and theoretically guaranteed framework for multi-modal medical image registration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Adaptive Correspondence Scoring for Unsupervised Medical Image Registration](adaptive_correspondence_scoring_for_unsupervised_medical_ima.md)
- [\[ECCV 2024\] Improving Medical Multi-modal Contrastive Learning with Expert Annotations](improving_medical_multi-modal_contrastive_learning_with_expert_annotations.md)
- [\[ECCV 2024\] NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration](textttnephi_neural_deformation_fields_for_approximately_diff.md)
- [\[ECCV 2024\] I-MedSAM: Implicit Medical Image Segmentation with Segment Anything](i-medsam_implicit_medical_image_segmentation_with_segment_anything.md)
- [\[CVPR 2025\] Multi-modal Vision Pre-training for Medical Image Analysis (BrainMVP)](../../CVPR2025/medical_imaging/multi-modal_vision_pre-training_for_medical_image_analysis.md)

</div>

<!-- RELATED:END -->
