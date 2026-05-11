---
title: >-
  [Paper Note] Orochi: Versatile Biomedical Image Processor
description: >-
  [NeurIPS 2025][Medical Imaging][Biomedical image processing] This paper proposes Orochi—the first general-purpose foundation model for low-level biomedical image processing. Through Task-related Joint-embedding Pre-train…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Biomedical image processing"
  - "general-purpose foundation model"
  - "self-supervised pre-training"
  - "Mamba"
  - "image registration/fusion/restoration/super-resolution"
date: 2026-05-08
content_hash: 0e58d597c637bd05
---

# Orochi: Versatile Biomedical Image Processor

**Conference**: NeurIPS 2025
**arXiv**: [2509.22583](https://arxiv.org/abs/2509.22583)
**Code**: Not yet available (open-source promised in the paper)
**Area**: Medical Imaging
**Keywords**: Biomedical image processing, general-purpose foundation model, self-supervised pre-training, Mamba, image registration/fusion/restoration/super-resolution

## TL;DR

This paper proposes Orochi—the first general-purpose foundation model for low-level biomedical image processing. Through Task-related Joint-embedding Pre-training (TJP) and a Multi-head Hierarchy Mamba architecture, Orochi matches or surpasses task-specific state-of-the-art models across four tasks—registration, fusion, restoration, and super-resolution—with lightweight fine-tuning of fewer than 5% of parameters.

## Background & Motivation

Deep learning is increasingly important in life sciences, with low-level biomedical image processing (registration, fusion, restoration, super-resolution) being among the most critical applications. The field faces a fundamental tension:

**The triple dilemma of task-specific model paradigms**:

**Task perspective**: Real-world biomedical imaging pipelines typically require multiple sequential steps (e.g., registration followed by fusion), yet each step currently demands a different specialist model.

**Degradation perspective**: The underlying degradation causes across tasks are correlated—low SNR and low resolution both reflect information loss, while occlusion and deformation both represent spatial transformations.

**Data perspective**: Biomedical images are multi-channel, large-scale, and high-throughput; training and deploying multiple specialist models is highly inefficient.

Existing platforms (e.g., ImageJ/Fiji, napari) provide a variety of model plugins, but these are each confined to specific tasks and datasets, leaving biologists overwhelmed by a vast plugin ecosystem.

The paper's **Key Insight** is to build a general-purpose low-level image processing foundation model that handles all low-level tasks within a unified framework, while exploiting cross-task learning for more generalizable feature representations. The **Core Idea** is to use task-related degradations as self-supervised signals—rather than generic Masked Image Modelling—because the intrinsic correlations among different degradations correspond precisely to the correlations among different low-level tasks.

## Method

### Overall Architecture

Orochi is organized around four levels:

- **Data level**: Raw unlabeled data collected from 100+ public studies (totaling over 100 TB), converted into training patches/volumes via random multi-scale sampling.
- **Pre-training level**: Task-related Joint-embedding Pre-Training (TJP).
- **Model level**: Multi-head Hierarchy Mamba.
- **Post-training level**: A three-tier fine-tuning framework (Full / Normal / Light).

### Key Designs

#### 1. Random Multi-scale Sampling

Patches/volumes are extracted from raw images at multiple scales:

- **Multi-scale downsampling**: The original image $I$ is downsampled to three scales $1, 1/2, 1/4$: $I_s = \downarrow_s(I)$.
- **Random window sampling**: For each scaled image, a fixed-size window $K$ is used to randomly sample sub-patches: $x_s = I_s(i:i+W-1, j:j+H-1)$.

**Design Motivation**: The regions of interest (ROIs) differ in scale across low-level tasks. Multi-scale sampling expands data diversity and allows the model to learn cross-scale features during pre-training.

#### 2. Task-related Joint-embedding Pre-training (TJP)

The **Core Idea** of TJP is to use four types of degradation—each directly corresponding to a low-level task—as self-supervised signals, enabling the model to learn intrinsic correlations among different degradations.

**Dual-Masking Reconstructive Fusion**: Designed for the fusion task. Two independent masks $M_A, M_B$ are applied to the training data, and the model must jointly reconstruct the original image from two partially masked inputs:

$$x_A = x \odot M_A, \quad x_B = x \odot M_B$$
$$\hat{x} = f_\theta(x_A, x_B)$$

This forces the model to discover complementary information from two incomplete views and fuse them.

**Spatially Varying Gaussian Downsampling**: Designed for the super-resolution task. It combines noisy downsampling with spatially varying Gaussian filtering:

$$D_{\text{LR}}(x) = \mathbf{G}_{\sigma_{\text{var}}}(\uparrow_{1/s}(\downarrow_s(x + \eta)))$$

The standard deviation of the Gaussian kernel varies with spatial coordinates, simulating the non-uniform blur of real optical systems.

**Multi-scale Smooth Perlin Noise Deformation**: Designed for the registration task. It generates realistic multi-scale deformation fields:

$$D_{\text{def}}(x) = \mathbf{T}(x, \Phi), \quad \Phi = \mathbf{G}_\sigma(\mathbf{Per}(\mathbf{f}, \mathbf{p}))$$

Multi-octave Perlin noise is used to produce hierarchical deformations, with a $\tanh$ function constraining maximum displacement.

**Multi-stage Noise Simulation**: Designed for the restoration task. Gaussian, Poisson, and salt-and-pepper noise are applied sequentially:

$$D_{\text{noise}}(x) = \mathbf{Bi}_p(\mathbf{Poi}(\max(0, x + \eta)))$$

**Design Motivation**: Unlike generic MIM methods such as MAE, which only learn to reconstruct masked regions, TJP trains the model to directly learn degradation-to-restoration mappings that correspond to specific downstream tasks. Experiments confirm that MAE performs poorly on registration (Dice of only 71.22 vs. 83.62 for TJP), since masking and spatial deformation are fundamentally different degradation types.

#### 3. Three-tier Fine-tuning Framework

- **Full**: Full fine-tuning of all parameters.
- **Normal**: Fine-tuning only the replaced dense convolutional heads.
- **Light**: Uses depth-wise separable convolutions, updating only ~1–2% of parameters for parameter-efficient fine-tuning.

### Model Architecture

Multi-head Hierarchy Mamba leverages the linear computational complexity of Mamba, combined with the hierarchical design and patch merging mechanism of Swin-Transformer. Detailed architecture is provided in the appendix.

## Key Experimental Results

### Main Results (Comparison Across Four Tasks)

| Task | Dataset | Metric | Orochi (Full) | Orochi (Light) | Prev. SOTA | SOTA Method |
|------|---------|--------|--------------|---------------|-----------|-------------|
| Restoration | CARE | PSNR (XY)↑ | 28.31 | **29.77** | 27.12 | UniFMIR |
| Super-resolution | HBA (4mm) | PSNR↑ | **35.33** | 34.83 | 32.41 | LIIF |
| Registration | OASIS | Dice↑ | **83.62** | 79.61 | 82.22 | Transmorph-L |
| Fusion | VIFB | Qabf↑ | **0.41** | 0.34 | 0.39 | BSAFusion |

Across all four tasks, Orochi under either full or lightweight fine-tuning matches or exceeds the task-specific SOTA methods.

### Ablation Study (Pre-training Strategy Comparison)

| Pre-training Strategy | Registration Dice↑ | Fusion Qabf↑ | Restoration PSNR↑ | SR PSNR↑ |
|----------------------|-------------------|-------------|------------------|---------|
| MAE (single mask) | 71.22 | 0.36 | 26.67 | 29.17 |
| I-JEPA (dual mask) | 69.97 | 0.39 | 25.02 | 28.81 |
| **Orochi (TJP)** | **83.62** | **0.41** | **29.88** | **33.63** |

TJP outperforms MAE by 12.4 Dice points on registration, strongly validating the necessity of task-related degradation design.

### Key Findings

1. **Lightweight fine-tuning can outperform full fine-tuning**: In restoration tasks with limited data (<100 training patches), Light mode (~1–2% parameters) surpasses Full mode, as full fine-tuning is prone to overfitting.
2. **TJP vs. MIM**: Generic MIM completely fails on certain low-level tasks, demonstrating the necessity of task-related degradation design.
3. **In-domain zero-shot generalization**: Pre-trained Orochi exhibits strong zero-shot processing capability on unseen test images.
4. **Complementary fusion verification**: In a centromere counting case study, the model successfully fused complementary information from two partially occluded views rather than simply reconstructing either one.

## Highlights & Insights

- **First general-purpose model for low-level biomedical image processing**: Pioneering unification of registration, fusion, restoration, and super-resolution under a single framework.
- **Degradation as task**: Directly aligning self-supervised degradation design with downstream tasks is better suited to low-level image processing than generic MIM.
- **Practice-oriented**: The three-tier fine-tuning framework offers biologists flexible options—Light for scarce data, Full for abundant data.
- **Data engineering**: Collecting pre-training data from 100+ studies totaling over 100 TB is itself a significant contribution.

## Limitations & Future Work

- Architectural details of Multi-head Hierarchy Mamba are relegated to the appendix, leaving the main text informationally insufficient.
- Code has not yet been released, making reproduction difficult.
- Coverage is limited to low-level image processing tasks; high-level semantic tasks (detection, segmentation, etc.) are not addressed.
- The evaluation metrics for fusion and registration tasks are limited in variety.

## Related Work & Insights

- UniFMIR has demonstrated the generalization ability of pre-trained foundation models for biomedical image restoration; Orochi extends this to four tasks.
- The joint-embedding prediction idea of I-JEPA inspired TJP, but TJP replaces generic masking with task-related degradations.
- The linear complexity of Mamba makes it feasible to process large-scale biomedical images (2–5D).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First general-purpose biomedical image processing model spanning four low-level tasks; TJP design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comparison against 30+ baselines across four tasks is comprehensive, but quantitative analysis of computational efficiency is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Framework is clearly presented, but methodological details rely too heavily on the appendix.
- **Value**: ⭐⭐⭐⭐⭐ Fills a critical gap in general-purpose models for low-level biomedical image processing with substantial practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains](pancakes_consistent_multi-protocol_image_segmentation_across_biomedical_domains.md)
- [\[NeurIPS 2025\] Unpaired Image-to-Image Translation for Segmentation and Signal Unmixing](unpaired_image-to-image_translation_for_segmentation_and_signal_unmixing.md)
- [\[NeurIPS 2025\] Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation](mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)
- [\[NeurIPS 2025\] The Boundaries of Fair AI in Medical Image Prognosis: A Causal Perspective](the_boundaries_of_fair_ai_in_medical_image_prognosis_a_causal_perspective.md)
- [\[ICCV 2025\] MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance](../../ICCV2025/medical_imaging/multiverseg_scalable_interactive_segmentation_of_biomedical_imaging_datasets_wit.md)

</div>

<!-- RELATED:END -->
