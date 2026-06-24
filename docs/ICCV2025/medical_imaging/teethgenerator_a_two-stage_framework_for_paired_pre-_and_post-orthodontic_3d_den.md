---
title: >-
  [Paper Note] TeethGenerator: A Two-Stage Framework for Paired Pre- and Post-Orthodontic 3D Dental Data Generation
description: >-
  [ICCV 2025][Medical Imaging][orthodontic data generation] This paper proposes TeethGenerator, a two-stage framework for generating paired pre- and post-orthodontic 3D dental point cloud models. Stage I employs a VQ-VAE combined with a diffusion model to generate post-treatment tooth morphology, while Stage II uses a Transformer conditioned on a style model to generate the corresponding pre-treatment dental arrangement.
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "orthodontic data generation"
  - "3D dental model"
  - "VQ-VAE"
  - "diffusion model"
  - "paired data synthesis"
date: 2026-05-08
content_hash: 7438139cca017930
---

# TeethGenerator: A Two-Stage Framework for Paired Pre- and Post-Orthodontic 3D Dental Data Generation

**Conference**: ICCV 2025
**arXiv**: [2507.04685](https://arxiv.org/abs/2507.04685)  
**Code**: [https://github.com/lcshhh/teeth_generator](https://github.com/lcshhh/teeth_generator)  
**Area**: Medical Imaging / 3D Generation
**Keywords**: orthodontic data generation, 3D dental model, VQ-VAE, diffusion model, paired data synthesis

## TL;DR

This paper proposes TeethGenerator, a two-stage framework for generating paired pre- and post-orthodontic 3D dental point cloud models. Stage I employs a VQ-VAE combined with a diffusion model to generate post-treatment tooth morphology, while Stage II uses a Transformer conditioned on a style model to generate the corresponding pre-treatment dental arrangement.

## Background & Motivation

- Tooth arrangement networks in digital orthodontics (e.g., TANet, OrthoGAN, TADPM) require large quantities of **paired pre- and post-orthodontic 3D dental models** for training.
- Acquiring such paired data is extremely time-consuming, as it requires multi-year longitudinal scans from real patients and involves privacy and commercial concerns.
- The only publicly available 3D orthodontic dataset contains merely 1,060 pairs, with an imbalanced distribution of malocclusion types (e.g., only 20 cases of anterior open bite).
- Existing 3D shape generation methods (GAN/VAE/diffusion) are primarily designed for single-object generation and **cannot handle the multi-instance structure of dental models** (24–32 segmented teeth must be generated simultaneously).
- Four core challenges in dental data synthesis: multi-instance generation, distribution matching, orthodontic consistency (identical tooth morphology before and after treatment), and pose diversity.

## Method

### Overall Architecture

TeethGenerator operates in two stages: Stage I generates post-treatment dental models (morphology generation), and Stage II generates the corresponding pre-treatment dental models (style transfer). The actual output is point clouds; meshes are used for visualization only.

### Key Designs

1. **Dental Spatial Structure Encoding**: Following the FDI numbering system, 32 teeth are organized into a $2 \times 2 \times 8$ structured grid that preserves critical spatial relationships:

    - Adjacency (neighboring teeth on the same side)
    - Bilateral symmetry (left–right symmetry within the same jaw)
    - Occlusal symmetry (correspondence between upper and lower jaws)
    - Each tooth is sampled with 128 points, then partitioned into $r^3$ voxels ($r=4$), yielding an overall voxel layout of $[2r, 2r, 8r]$.

2. **Stage I – Tooth Morphology Generation Module**:

    - **VQ-VAE Reconstruction**: Based on the PVCNN paradigm, intra-voxel features are extracted (PointNet + max-pooling). A 3D U-Net encoder–decoder reconstructs the aggregated dental point cloud $\hat{P}_{post} \in \mathbb{R}^{K \times N \times C}$, and an auxiliary MLP predicts a valid-tooth mask (since the number of teeth $K$ is fixed but the actual count varies).
    - **Latent Diffusion Model**: With VQ-VAE parameters frozen, a diffusion model is trained in the latent space to learn the encoding distribution. At inference, Gaussian noise is denoised to obtain latent codes, which are then decoded by the VQ-VAE decoder to produce post-treatment dental models.
    - Reconstruction loss uses Chamfer Distance: $L_{rec} = \sum_i \text{CD}(P_i, \hat{P}_{post_i})$

3. **Stage II – Tooth Style Generation Module**:

    - **Style Extractor ($E_{style}$)**: Voxel features are extracted independently for each tooth in the style model → mean pooling + standard deviation pooling → two MLPs → summation yields the style encoding for each tooth (capturing local detail and style characteristics).
    - **Shape Extractor ($E_{shape}$)**: Voxels are partitioned globally across the entire dental model (rather than per tooth) → max-pooling → global shape features are obtained (for global context used in collision avoidance).
    - **Transformer for Transformation Parameter Prediction**: Style features serve as Transformer input; shape features are injected into each layer via cross-attention → output $\mathbf{x} \in \mathbb{R}^{K \times 9}$ (3D translation + 6D rotation parameters per tooth).
    - The rotation matrix $R_i \in \mathbb{R}^{3 \times 3}$ is computed from the 6D rotation representation, ensuring morphological consistency between pre- and post-treatment teeth.

### Loss & Training

- **Stage I**:
    - VQ-VAE reconstruction loss: Chamfer Distance
    - Diffusion model loss: standard denoising objective $\mathcal{L} = \mathbb{E}\|\epsilon_\theta(\mathbf{x_t}, t) - \epsilon\|_2^2$, cosine noise schedule
    - Batch size 32, 500 epochs, AdamW lr=1e-3

- **Stage II**:
    - Distance loss: $\mathcal{L}_{dis} = \sum_i \|P_{pre} - P_{style}\|^2$ (L2 computed directly using point correspondences)
    - Collision avoidance loss: $\mathcal{L}_{ca} = \sum_{(a,b) \in \mathcal{K}}((\frac{1}{1+d/s})^{12} - 2(\frac{1}{1+d/s})^6)$ (Lennard-Jones potential applied to adjacent tooth pairs and occlusal pairs)
    - Total loss: $\mathcal{L} = \mathcal{L}_{dis} + \mathcal{L}_{ca}$
    - 12-layer Transformer, 8-head attention, batch size 64, 300 epochs, AdamW lr=1e-4

## Key Experimental Results

### Main Results

**Post-treatment tooth generation quality comparison (1-NNA + uniqueness, 720 samples):**

| Model | CD (%, ↓) | EMD (%, ↓) | $U_{CD}$ (%, ↑) |
|:---:|:---:|:---:|:---:|
| PointFlow | 97.62 | 83.88 | 62.22 |
| DPM | 89.25 | 74.50 | 75.69 |
| PVD | 84.87 | 78.12 | 49.58 |
| LION | 90.41 | 77.93 | 52.78 |
| DiT-3D | 95.75 | 82.01 | 34.03 |
| **TeethGenerator** | **69.50** | **71.88** | **96.25** |

**Improvement on the downstream tooth arrangement task with synthetic data** (TANet backbone):

Performance improves consistently as the amount of synthetic data ($n \times 720$) added to the training set increases, converging after $n=10$.

### Ablation Study

**Ablation on voxelization strategy and spatial structure (Stage I):**

| ID | Spatial Structure | Voxelization | CD ↓ | EMD ↓ | $U_{CD}$ ↑ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2×2×8 | 3³ | 82.63 | 81.50 | 92.64 |
| **2** | **2×2×8** | **4³** | **69.50** | **71.88** | **96.25** |
| 3 | 2×2×8 | 5³ | 77.75 | 83.23 | 95.56 |
| 4 | — | No voxel | 89.59 | 84.13 | 21.11 |
| 5 | — | Global 8×8×32 | 78.16 | 76.91 | 89.03 |
| 6 | 1×1×32 | 4³ | 74.48 | 72.82 | 94.17 |

### Key Findings

- **TeethGenerator substantially outperforms general-purpose 3D generation methods**: CD decreases from the best baseline of 84.87 to 69.50, and uniqueness improves from 75.69% to 96.25%.
- **General methods fail to generate segmented multi-tooth structures**: PointFlow, LION, and others produce holistic point clouds in which individual teeth cannot be distinguished.
- **Voxel-based feature extraction is critical**: Without voxelization (No. 4), uniqueness drops sharply to 21.11%, indicating a large number of repetitive outputs.
- **The $2 \times 2 \times 8$ spatial structure outperforms $1 \times 1 \times 32$**: it explicitly models bilateral and occlusal symmetry interactions.
- **The style extractor requires local detail**: swapping the voxelization strategies of $E_{style}$ and $E_{shape}$ prevents successful style replication.
- **Synthetic data significantly improves downstream performance**: adding 10× synthetic data leads to consistent improvement in TANet tooth arrangement metrics.

## Highlights & Insights

- This is the first **paired pre- and post-orthodontic 3D dental data generation framework**, directly addressing the critical data scarcity bottleneck in digital orthodontics.
- The design of Stage II is elegant: predicting transformation parameters (translation + rotation) rather than generating point clouds directly naturally ensures morphological consistency between pre- and post-treatment teeth.
- The $2 \times 2 \times 8$ grid encoding based on the FDI numbering system effectively leverages anatomical structural priors of the dentition.
- Using the Lennard-Jones potential as the collision avoidance loss—borrowed from molecular dynamics—is well-suited to densely packed 3D arrangement scenarios.
- The validation of synthetic data effectiveness demonstrates that high-quality synthetic 3D medical data can genuinely improve downstream task performance.

## Limitations & Future Work

- The dataset contains only 720 training samples; the model's ability to generalize to a broader range of malocclusion types remains to be verified.
- Stage I outputs a fixed set of 32 teeth with a masking strategy to handle variable tooth counts, which may be less effective in cases with many missing teeth.
- Mesh reconstruction relies on non-rigid registration matched against a real reference database, limiting the morphological diversity of generated meshes.
- Tooth roots, gingiva, and other anatomical structures are not modeled; only crown-level point clouds are generated.
- The style model in Stage II must be selected from existing data, which constrains style diversity.

## Related Work & Insights

- TeethGenerator is complementary to downstream tooth arrangement methods such as TANet and TADPM: TeethGenerator addresses the data problem, while the latter address the arrangement problem.
- The combination of VQ-VAE and latent diffusion (analogous to LION) demonstrates strong performance in multi-instance 3D generation.
- The two-stage strategy of "first generate the standard post-treatment state, then transfer style to obtain the pre-treatment state" for paired data synthesis is generalizable to other medical paired data generation scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First framework for paired orthodontic 3D data generation with a novel problem formulation; however, individual components (VQ-VAE, diffusion model, Transformer) are combinations of established techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers distributional similarity, uniqueness, downstream task improvement, voxelization ablation, and style transfer strategy comparison.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated; visualizations (generated vs. real dental model comparisons) are convincing.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a critical data bottleneck in digital orthodontics; synthetic data has been validated to improve downstream task performance, with significant clinical application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GuideGen: A Text-Guided Framework for Paired Full-Torso Anatomy and CT Volume Generation](../../AAAI2026/medical_imaging/guidegen_a_text-guided_framework_for_paired_full-torso_anatomy_and_ct_volume_gen.md)
- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](../../AAAI2026/medical_imaging/a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [\[ICCV 2025\] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)
- [\[ICML 2026\] CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support](../../ICML2026/medical_imaging/cascade_conformal_prediction_uncertainty-adaptive_prediction_intervals_for_two-s.md)
- [\[CVPR 2025\] Revisiting MAE Pre-Training for 3D Medical Image Segmentation](../../CVPR2025/medical_imaging/revisiting_mae_pre-training_for_3d_medical_image_segmentation.md)

</div>

<!-- RELATED:END -->
