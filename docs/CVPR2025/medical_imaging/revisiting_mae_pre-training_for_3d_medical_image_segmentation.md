---
title: >-
  [Paper Note] Revisiting MAE Pre-Training for 3D Medical Image Segmentation
description: >-
  [CVPR 2025][Medical Imaging][Self-Supervised Learning] This work systematically addresses three major pitfalls in 3D medical imaging SSL research (small datasets, non-SOTA architectures, and insufficient evaluation). By leveraging an optimized MAE to pre-train a ResEnc U-Net CNN on 39K brain MRI scans, it outperforms the nnU-Net baseline by an average of approximately 3 Dice points across 11 downstream segmentation datasets.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Self-Supervised Learning"
  - "Masked Autoencoders"
  - "3D Medical Segmentation"
  - "CNN Pre-training"
  - "nnU-Net"
date: 2026-05-08
content_hash: 303ee63d7c06ce9f
---

# Revisiting MAE Pre-Training for 3D Medical Image Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2410.23132](https://arxiv.org/abs/2410.23132)  
**Code**: Yes (public code and models mentioned in the paper)  
**Area**: Medical Imaging  
**Keywords**: Self-Supervised Learning, Masked Autoencoders, 3D Medical Segmentation, CNN Pre-training, nnU-Net

## TL;DR

This work systematically addresses three major pitfalls in 3D medical imaging SSL research (small datasets, non-SOTA architectures, and insufficient evaluation). By leveraging an optimized MAE to pre-train a ResEnc U-Net CNN on 39K brain MRI scans, it outperforms the nnU-Net baseline by an average of approximately 3 Dice points across 11 downstream segmentation datasets.

## Background & Motivation

**Background**: 3D medical image segmentation is currently dominated by from-scratch nnU-Net training. Although the community is interested in pre-training (supervised pre-training has already been adopted), self-supervised learning (SSL) pre-training has not yet been widely adopted in this field.

**Limitations of Prior Work**: The authors identify three major pitfalls in existing SSL research—(P1) Pre-training datasets are too small: most methods use fewer than 10K volumes, which is close to the scale of annotated datasets; (P2) Non-SOTA architectures: many works use Transformer architectures, whereas CNNs (especially the nnU-Net family) still significantly outperform Transformers in 3D medical segmentation; (P3) Insufficient evaluation: too few datasets, stacked multiple contributions that prevent isolating the effect of pre-training, weak baseline comparisons, or even evaluation on the pre-training data.

**Key Challenge**: MAE pre-training has been thoroughly validated in 2D natural images, but its true potential in 3D medical imaging has been underestimated due to the aforementioned three pitfalls. Furthermore, CNN architectures are not naturally compatible with mask pre-training (masking corrupts the spatial structure of convolutions), requiring specific adaptations.

**Goal**: To strictly avoid the three pitfalls and answer a core question—under the proper settings, how much improvement can MAE pre-training actually bring to 3D CNNs?

**Key Insight**: Utilize a large-scale dataset (39K), a SOTA architecture (ResEnc U-Net), and comprehensive evaluation (5 development + 8 test datasets) to systematically optimize each design choice of MAE.

**Core Idea**: A simple MAE, when properly configured, can significantly outperform all existing SSL methods and the nnU-Net baseline.

## Method

### Overall Architecture

The input consists of 3D MRI volumes ([160x160x160] patches, 1mm isotropic), which are randomly masked and fed into a ResEnc U-Net encoder-decoder. An L2 reconstruction loss is computed over the masked regions for pre-training. Once pre-training is complete, the encoder weights are transferred to downstream segmentation tasks for fine-tuning.

### Key Designs

1. **Sparsification Adaptation**:

    - **Function**: Enables CNN architectures to correctly process masked inputs.
    - **Mechanism**: Three components — (a) Sparse convolution + normalization: Re-applies the mask after each convolution and considers only non-masked values during normalization to avoid zero-value bias in statistics; (b) Mask Token: Fills masked regions in the feature maps with learnable tokens before feeding them into the decoder to simplify the decoding task; (c) Densification Convolution: Adds a [3x3x3] convolution at each resolution (except the highest resolution) after mask token filling and before the decoder to smooth features.
    - **Design Motivation**: The receptive field of CNNs causes zero values in masked regions to gradually erode non-masked features layer by layer, which must be isolated using sparse convolutions. These adaptations are adapted from the 2D methods of Tian et al. and Woo et al., and this work is the first to systematically validate their effectiveness in 3D medical scenarios. The combination of all three improves DSC by approximately 0.3.

2. **Dynamic Masking Ratio**:

    - **Function**: Determines how much content is masked during pre-training.
    - **Mechanism**: Randomly samples masks in the bottleneck [5x5x5] resolution (corresponding to [32x32x32] voxel patches in the input), evaluating static ratios of 30%-90% and a dynamic range of U[60%-90%]. Ultimately, a dynamic 60%-90% mask is selected.
    - **Design Motivation**: Static ratios of 60% and 75% perform almost identically to the dynamic range, but dynamic masking offers more variation in training difficulty. Masking ratios that are too low (30%) or too high (90%) result in performance degradation.

3. **Fine-Tuning Strategy Optimization**:

    - **Function**: Determines how pre-trained weights are transferred and fine-tuned.
    - **Mechanism**: Transfers both encoder + decoder weights, uses a two-stage warm-up (12.5K steps each), and reduces the peak learning rate to 1e-3. Key findings: warm-up is mandatory, the encoder must not be frozen, and the learning rate must be reduced.
    - **Design Motivation**: When training from scratch, nnU-Net uses a default LR of 1e-2. However, with pre-training initialization, the features already have a solid starting point, and an excessively high learning rate would destroy the pre-trained representation.

### Loss & Training

- Pre-training loss: L2 reconstruction loss calculated only on masked regions (z-score normalized voxel space).
- Retains skip connections (following the consensus from prior work such as FCMAE).
- Pre-training hyperparameters: SGD with Nesterov momentum 0.99, PolyLR schedule, 250K steps (equivalent to 1000 epochs in the nnU-Net framework), batch size 6.
- Data augmentation: Only mild spatial augmentation (affine scaling, rotation, mirroring).
- Ignores modality distribution differences and randomly samples MRIs of different modalities.

## Key Experimental Results

### Main Results

| Method | Avg DSC (11 datasets) | Avg Rank |
|------|----------------------|----------|
| S3D (Ours) | **72.37** | **2.00** |
| Models Genesis | 71.83 | 3.18 |
| VolumeFusion | 70.94 | 4.36 |
| No (Fixed baseline) | 70.40 | 4.55 |
| No (Dynamic nnU-Net) | 69.40 | 4.64 |
| VoCo | 69.41 | 6.27 |

| Typical Dataset | S3D | nnU-Net (Fixed) | Gain |
|-----------|-----|-----------------|------|
| Atlas22 (D4 stroke) | 66.95 | 65.52 | +1.43 |
| Brain Mets (D2) | 65.24 | 56.53 | +8.71 |
| T2 Aneurysms (D11) | 47.26 | 41.97 | +5.29 |
| HNTS-MRG24 (D9) | 68.62 | 65.90 | +2.72 |

### Ablation Study

| Configuration | Avg DSC (D1-D5) | Description |
|------|----------------|------|
| Base (No Sparsification) | 71.35 | Baseline MAE |
| + Sparse Conv + BN | 71.36 | Sparse convolution and normalization only |
| + Mask Token | 71.37 | Added learnable mask token |
| + Densification Conv | **71.66** | Added densification convolution, +0.3 gain |
| Mask 60% | 71.60 | Static 60% |
| Mask 75% | 71.66 | Static 75% |
| U[60%-90%] Dynamic | **71.65** | Dynamic range, offering better flexibility |

### Key Findings

- SSL pre-training is indeed effective: S3D outperforms the from-scratch baseline in 10 out of 11 test datasets, with an average improvement of +2 DSC.
- MAE-based methods (MG, S3D) comprehensively outperform contrastive learning (VoCo) and volume fusion (VF) methods, indicating that reconstruction-based pre-training is better suited for CNNs.
- Models Genesis (a legacy method from 2019) remains very strong under the correct backbone + large data setup, exposing the issue of subsequent methods evaluating on sub-optimal architectures.
- Fine-tuning strategies are critical: omitting warm-up leads to a drop of 1-2 DSC, keeping the learning rate high impairs performance, and freezing the encoder performs poorly.
- Densification convolution contributes the most (+0.3 DSC) among the sparsification components, addressing the issue of feature discontinuity between the encoder and decoder.

## Highlights & Insights

- **The methodological value of "simple methods + correct settings" is extremely high**: This work does not propose a new SSL paradigm but instead demonstrates that good engineering practices (large-scale data, SOTA architecture, and rigorous evaluation) can make a simple MAE achieve state-of-the-art (SOTA) results. This serves as a cautionary tale for the entire SSL field.
- **The three-pitfalls analysis framework (P1/P2/P3)** can serve as a checklist for evaluating any SSL method to avoid unfair comparisons.
- **Systematic ablation of fine-tuning strategies** is highly practical—the optimal combination of warm-up, learning rate, and weight transfer scope is not intuitive and requires experimental validation.
- The 39K MRI dataset is sourced from 44 clinical centers, ensuring high data diversity and strong generalization of the pre-trained representation.

## Limitations & Future Work

- Pre-training data is limited to brain MRI, and transferability to other anatomical regions such as abdominal CT and whole-body MRI needs to be validated.
- The pre-training dataset is private (clinical data), meaning that although the authors released the code and model weights, the dataset cannot be reproduced.
- Only one architecture (ResEnc U-Net) was tested, and applicability to hybrid architectures (such as UNesT, MedNeXt) remains unknown.
- The performance of the dynamic masking ratio is almost identical to that of the static 75% ratio, making the advantage of the dynamic strategy less obvious.
- Future directions: Extending to CT data, and exploring combinations of masked pre-training with other SSL paradigms.

## Related Work & Insights

- **vs Swin UNETR**: Performs masked pre-training based on a Transformer architecture, which in practice underperforms compared to CNNs in 3D medical segmentation, validating the P2 pitfall.
- **vs VoCo**: A contrastive learning-based pre-training approach that performs poorly on the CNN backbone (avg rank 6.27), indicating insufficient adaptability of this paradigm in CNN pre-training.
- **vs Models Genesis**: A "legacy" method from 2019 but with the right approach (masking + reconstruction), which becomes strong when combined with a SOTA architecture, contrasting with the unfair evaluation of subsequent methods.

## Rating

- Novelty: ⭐⭐⭐ MAE itself is not new; the contribution lies in systematic validation and engineering optimization.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation across 5+8 datasets, multi-dimensional ablation, and fair baseline comparisons set a gold standard for evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ The three-pitfalls framework is clear, and the experiments are well-structured.
- Value: ⭐⭐⭐⭐ Of significant reference value for the 3D medical SSL community, with direct practical benefit through the release of model weights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Multi-modal Vision Pre-training for Medical Image Analysis (BrainMVP)](multi-modal_vision_pre-training_for_medical_image_analysis.md)
- [\[CVPR 2025\] VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging](vista3d_a_unified_segmentation_foundation_model_for_3d_medical_imaging.md)
- [\[CVPR 2026\] Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification](../../CVPR2026/medical_imaging/revisiting_2d_foundation_models_for_scalable_3d_medical_image_classification.md)
- [\[CVPR 2025\] Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](multi-resolution_pathology-language_pre-training_model_with_text-guided_visual_r.md)
- [\[CVPR 2025\] Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](interactive_medical_image_segmentation_a_benchmark_dataset_and_baseline.md)

</div>

<!-- RELATED:END -->
