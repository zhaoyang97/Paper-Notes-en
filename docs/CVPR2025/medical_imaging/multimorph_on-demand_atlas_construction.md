---
title: >-
  [Paper Note] MultiMorph: On-demand Atlas Construction
description: >-
  [CVPR 2025][Medical Imaging][atlas construction] This paper proposes MultiMorph, a feed-forward brain atlas construction model. By leveraging a linear-complexity GroupBlock feature-sharing layer and a Centrality Layer, it generates an unbiased group atlas in a single forward pass given an arbitrary number of 3D brain images. It operates over 100 times faster than traditional optimization methods and generalizes to unseen modalities and populations without any fine-tuning.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "atlas construction"
  - "group registration"
  - "feed-forward network"
  - "synthetic data"
  - "brain MRI"
date: 2026-05-08
content_hash: 284bc50128e27b92
---

# MultiMorph: On-demand Atlas Construction

**Conference**: CVPR 2025  
**arXiv**: [2504.00247](https://arxiv.org/abs/2504.00247)  
**Code**: [https://github.com/mabulnaga/multimorph](https://github.com/mabulnaga/multimorph)  
**Area**: Medical Imaging / Brain Atlas Construction  
**Keywords**: atlas construction, group registration, feed-forward network, synthetic data, brain MRI

## TL;DR

This paper proposes MultiMorph, a feed-forward brain atlas construction model. By leveraging a linear-complexity GroupBlock feature-sharing layer and a Centrality Layer, it generates an unbiased group atlas in a single forward pass given an arbitrary number of 3D brain images. It operates over 100 times faster than traditional optimization methods and generalizes to unseen modalities and populations without any fine-tuning.

## Background & Motivation

**Background**: Anatomical atlases serve as the fundamental reference coordinates for quantifying anatomical variation in brain imaging studies, widely used in segmentation, shape analysis, and longitudinal modeling. Traditional unbiased atlas construction requires iterative optimization taking days to weeks, and learning-based methods like AtlasMorph still require days of training.

**Limitations of Prior Work**: (1) Traditional iterative methods (such as ANTs SyGN) are computationally expensive, requiring 4345 minutes to construct an atlas for 319 volumes; (2) learning-based methods (AtlasMorph, Aladdin) still need to be retrained for each new population and depend on machine learning expertise; (3) different populations and modalities require different atlases, and the high computational cost of repeated construction forces most researchers to use mismatched pre-computed templates, compromising analysis quality.

**Key Challenge**: Group-specific atlases require population-specific fidelity, but the computational cost of atlas construction scales with the group size, and changing the population or modality requires recomputation from scratch.

**Goal**: To design a test-time-ready atlas construction framework that: (1) accepts an arbitrary number of input images; (2) generates high-quality atlases in a single forward pass; (3) generalizes to new modalities and populations without fine-tuning.

**Key Insight**: To reformulate atlas construction as a group registration problem—instead of learning a fixed atlas template, the goal is to learn a function that outputs the deformation fields mapping any given set of images to a central space.

**Core Idea**: Using a linear-complexity GroupBlock to aggregate in-group image features at each UNet layer, combined with a Centrality Layer to structurally guarantee unbiasedness, and training with synthetic data to achieve modality-agnostic generalization.

## Method

### Overall Architecture

Given $m$ 3D brain MRIs, MultiMorph processes all images through a weight-shared UNet, aggregating in-group features at each layer using a GroupBlock. The network outputs $m$ stationary velocity fields (SVFs), which are integrated into diffeomorphic deformation fields after neutralizing global shift via the Centrality Layer. These fields warp all images to the central space, where they are averaged to yield the atlas. During training, 50% of the groups use synthetic data to enhance cross-modality generalization.

### Key Designs

1. **GroupBlock Feature Sharing Layer**:
    - **Function**: Implements in-group feature interaction across images at each level of the UNet.
    - **Mechanism**: For the feature maps $c_i^{(l)}$ of all images at layer $l$, the mean statistic $\bar{c}^{(l)} = s(\{c_1^{(l)}, ..., c_m^{(l)}\})$ is computed. Then, the mean is concatenated with each image's feature map followed by a convolution: $c_i^{(l+1)} = \text{Conv}([c_i^{(l)} \| \bar{c}^{(l)}]; \theta^{(l)})$.
    - **Design Motivation**: Group registration requires information sharing across images to find the central common space, but cross-attention mechanisms incur prohibitive memory footprints for 3D volumes (quadratic complexity). GroupBlock only requires computing a single mean and doing concatenation, offering linear complexity to handle large-scale 3D data.
    - **Ablation Validation**: Removing GroupBlock drops the Dice score from 0.884 to 0.870.

2. **Centrality Layer**:
    - **Function**: Structurally guarantees that the output atlas is unbiased toward all inputs.
    - **Mechanism**: Subtracts the group mean from the network-predicted velocity fields: $v_i = v_i^{(L)} - \bar{v}^{(L)}$, enforcing a strict zero-mean constraint on the velocity fields.
    - **Design Motivation**: Traditional methods use regularization to softly constrain centrality (e.g., AtlasMorph), which has limited efficacy. Directly applying a hard structural constraint guarantees that the mean displacement is zero.
    - **Ablation Validation**: Removing the Centrality Layer worsens the centrality metric by 1000x (from 12.0 to 16125).

3. **Synthetic Data Augmentation Training**:
    - **Function**: Achieves modality-agnostic generalization by generating synthetic brain imaging training data through domain randomization.
    - **Mechanism**: Starting from brain anatomical segmentation maps, intensity values are randomly sampled for $K$ brain structures, and noise/artifacts are added. Synthetic data is used for 50% of the training groups.
    - **Design Motivation**: Different MRI modalities (T1-w, T2-w, PD-w, etc.) present massive variations in tissue contrast. Synthetic training biases the network to learn shapes rather than intensities, which enables generalization to unseen PD-w modalities during training.
    - **Ablation Validation**: Using synthetic data improves the Dice score by up to 1.8 percentage points on the IXI dataset.

### Loss & Training

$$\mathcal{L}(\phi_i) = \mathcal{L}_{sim}(\mathbf{t}, \mathbf{x}_i \circ \phi_i) + \lambda \mathcal{L}_{reg}(\phi_i) + \gamma \mathcal{L}_{struc}(\text{seg}[\mathbf{t}], \text{seg}[\mathbf{x}_i] \circ \phi_i)$$

Three loss terms are used: NCC image similarity + deformation field gradient regularization + Soft-Dice structural alignment. $\lambda=1.0$, $\gamma=0.5$. The auxiliary Dice loss boosts the Dice score by approximately 2 percentage points.

## Key Experimental Results

### Main Results: Atlas Construction on the IXI Dataset (Table 1, MultiMorph was not trained on this dataset)

| Method | Modality | Construction Time (min) | Dice↑ | Folds↓ | Centrality↓ |
|------|------|-------------|-------|--------|-------------|
| ANTs | T1-w | 4345.2 | 0.863 | 524.2 | 10.4 |
| AtlasMorph | T1-w | 1141.5 | 0.894 | 47.9 | 7.8 |
| Aladdin | T1-w | 325.2 | 0.885 | **0.0** | 106.8 |
| **MultiMorph** | T1-w | **10.5** | **0.913** | 1.1 | **1.4** |
| ANTs | PD-w | 4320.2 | 0.856 | 313.1 | 12.4 |
| **MultiMorph** | PD-w | **7.8** | **0.900** | 1.6 | **0.9** |

### Ablation Study (Table 4, OASIS-1 Dataset)

| Configuration | Dice↑ | Folds↓ | Centrality(×10⁻³)↓ |
|------|-------|--------|-------------------|
| No CL + GB(mean) | 0.892 | 0.0 | 16125 |
| CL + No GB | 0.870 | 0.1 | 9.9 |
| CL + GB(mean) | 0.884 | 1.1 | 12.0 |
| CL + GB(mean) + Dice | **0.919** | 5.4 | 18.6 |

### Key Findings

- MultiMorph outperforms all baselines on the **unseen IXI dataset**, including methods natively trained on it.
- Generalizing to the **unseen PD-w modality** during training, it still achieves a high Dice score of 0.900.
- In subgroup analysis, it only takes 1.5 minutes (on CPU) to generate an atlas, whereas baseline methods require 12 to 436 minutes.
- Age-conditioned atlases clearly capture ventricular enlargement and white matter degradation associated with normal aging and dementia.

## Highlights & Insights

1. **Simplicity of GroupBlock**: Using mean calculation + concatenation + convolution instead of cross-attention achieves linear complexity for processing large-scale 3D volumetric groups. The idea is simple and highly effective.
2. **Hard vs. Soft Constraints**: The Centrality Layer enforces centrality structurally rather than via regularization, improving centrality by 1000x. This represents a highly valuable design philosophy.
3. **Shape Bias Strategy via Synthetic Data**: Extreme domain randomization biases the network toward learning shape rather than intensity, achieving zero-shot modality generalization with high versatility.
4. **Outstanding Practical Value**: It democratizes atlas construction from "ML experts + GPU clusters" to "seconds of CPU inference," truly empowering general biomedical researchers.

## Limitations & Future Work

1. It assumes diffeomorphic deformation, making it unable to handle topologies altered by pathology (e.g., massive brain injuries).
2. It is currently trained only on neuroimaging; extending it to other organs would require training on anatomy-independent synthetic data.
3. All activations are stored in memory during inference, which may limit scalability for extremely large 3D volumes due to memory constraints.

## Related Work & Insights

- **DUSt3R/MUSt3R series**: These adopt a "dense prediction + backend alignment" paradigm. MultiMorph similarly reframes atlas construction as a dense prediction task.
- **SynthSeg/SynthMorph**: Pioneering works that utilize synthetic data training to achieve modality-independence; MultiMorph extends this paradigm.
- **TAG (VAE Mean Decoding)**: Linearly averaging VAE latent vectors typically fails to decode into valid atlases. MultiMorph circumvents this by directly constructing the atlas in the warped image space.

## Rating

- ⭐ Originality: 8/10 — Reformulates atlas construction into group registration and feed-forward prediction; GroupBlock and CL designs are simple and elegant.
- ⭐ Experimental Thoroughness: 9/10 — Comprehensive evaluations across multiple datasets, physical modalities, subgroups, and ablations.
- ⭐ Value: 9/10 — 100x acceleration combined with zero-shot generalization makes it immediately applicable.
- ⭐ Overall: 8.5/10 — A highly practical foundational tool for medical imaging; simple design with performance comprehensively outplaying the SOTA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Multi-modal Vision Pre-training for Medical Image Analysis (BrainMVP)](multi-modal_vision_pre-training_for_medical_image_analysis.md)
- [\[CVPR 2025\] OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection](openmibood_open_medical_imaging_benchmarks_for_out-of-distribution_detection.md)
- [\[CVPR 2025\] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation](sam_dpo_semi_supervised.md)
- [\[CVPR 2025\] vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation](vesselfm_a_foundation_model_for_universal_3d_blood_vessel_segmentation.md)
- [\[CVPR 2025\] MoEdit: On Learning Quantity Perception for Multi-Object Image Editing](moedit_on_learning_quantity_perception_for_multi-object_image_editing.md)

</div>

<!-- RELATED:END -->
