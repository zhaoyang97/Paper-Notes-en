---
title: >-
  [Paper Note] From Pixels to Views: Learning Angular-Aware and Physics-Consistent Representations for Light Field Microscopy
description: >-
  [NeurIPS 2025][3D Vision][Light Field Microscopy] This paper proposes XLFM-Former, which learns angular–spatial priors of XLFM through **view-level Masked View Modeling (MVM-LF)** self-supervised pretraining…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Light Field Microscopy"
  - "XLFM"
  - "Masked View Modeling"
  - "ORC Loss"
  - "PSF"
  - "Self-supervised Pretraining"
date: 2026-05-08
content_hash: ac01ba341b41bf9b
---

# From Pixels to Views: Learning Angular-Aware and Physics-Consistent Representations for Light Field Microscopy

**Conference**: NeurIPS 2025
**arXiv**: [2510.22577](https://arxiv.org/abs/2510.22577)  
**Code**: [https://github.com/hefengcs/XLFM-Former](https://github.com/hefengcs/XLFM-Former)  
**Authors**: Feng He, Guodong Tan, Qiankun Li, Jun Yu, Quan Wen (University of Science and Technology of China)
**Area**: 3D Vision — Light Field Microscopy 3D Reconstruction
**Keywords**: Light Field Microscopy, XLFM, Masked View Modeling, ORC Loss, PSF, Self-supervised Pretraining

## TL;DR

This paper proposes XLFM-Former, which learns angular–spatial priors of XLFM through **view-level Masked View Modeling (MVM-LF)** self-supervised pretraining, and introduces an **Optical Rendering Consistency Loss (ORC Loss)** based on PSF differentiable rendering to constrain the physical plausibility of the reconstructed volume. On the first standardized XLFM-Zebrafish benchmark constructed by the authors, the method achieves an average PSNR of 54.04 dB, surpassing the best baseline ConvNeXt (50.16 dB) by **7.7%**.

## Background & Motivation

- **Background**: Extended Light Field Microscopy (XLFM) enables complete light field acquisition in a single exposure at 100 Hz, making it a critical tool for large-scale in vivo volumetric imaging in neuroscience (zebrafish, mice). However, deep learning-based XLFM 3D reconstruction lags significantly behind — lacking both standardized datasets with reproducible evaluation protocols, and methods capable of efficiently modeling angular–spatial structure with physical grounding.
- **Limitations of Prior Work**: ① Each XLFM frame encodes a densely angular-sampled 3D scene through a microlens array, producing highly entangled multi-view observations that conventional CNNs struggle to model across views; ② high-quality volumetric ground truth (generated via Richardson-Lucy deconvolution) is computationally expensive, making large-scale supervised learning costly; ③ existing methods (XLFMNet, FNet) either reconstruct only sparse neural signals while ignoring complete morphology, or suffer from memory explosion due to Fourier-domain convolutions requiring multiple GPUs.
- **Key Challenge**: Raw XLFM acquisition is cheap and abundant, but annotations are expensive and standardized benchmarks are absent — data-rich yet label-scarce. Models trained with purely pixel-level losses may generate visually plausible but optically inconsistent "hallucinated" structures, undermining scientific credibility.
- **Goal**: How can one efficiently learn angular priors of XLFM under label-scarce conditions while ensuring physical consistency of the reconstructed volume?
- **Key Insight**: XLFM reconstruction is reframed as a **structured prediction problem** — using "views" rather than "pixels" as the atomic modeling unit for self-supervised pretraining, and introducing differentiable forward rendering constraints based on the known PSF.
- **Core Idea**: View-level masked self-supervised pretraining to learn angular priors + PSF differentiable rendering loss to enforce physical consistency = data-efficient and physically credible full-volume XLFM reconstruction.

## Method

### Overall Architecture

XLFM-Former adopts a hierarchical **Swin Transformer encoder + CNN decoder** architecture for progressively reconstructing 3D volumes from XLFM light field data. The overall pipeline consists of two stages:

1. **Pretraining Stage**: MVM-LF self-supervised pretraining is performed on unannotated XLFM light field data. Among the 27 sub-aperture views, 70% are randomly masked and the model is trained to predict the masked views using a lightweight CNN decoder with $\ell_2$ loss for 250 epochs.
2. **Fine-tuning Stage**: The pretraining decoder is discarded; the encoder weights are retained to initialize XLFM-Former, which is then trained for supervised full-volume reconstruction using a combination of losses (MS-SSIM + Edge + PSNR + MSE + ORC).

### Key Designs

1. **Masked View Modeling for Light Fields (MVM-LF)**

    **Function**: Enables the encoder to learn angular priors and inter-view dependencies of XLFM without annotations, improving data efficiency and feature generalization.

    **Mechanism**: From the 27 sub-aperture views $\mathcal{U} = \{U_1, U_2, \dots, U_{27}\}$ of XLFM, a subset $\mathcal{U}_{\text{mask}}$ is randomly sampled at a ratio of $r_m = 0.7$ (70%). The corresponding views are zero-padded while retaining positional information, and the model is trained to reconstruct the masked views from the unmasked ones: $\hat{\mathcal{U}}_{\text{mask}} = f_\theta(\mathcal{U} \setminus \mathcal{U}_{\text{mask}})$, with MSE loss: $\mathcal{L}_{\text{MVM-LF}} = \sum_{U_i \in \mathcal{U}_{\text{mask}}} \|U_i - \hat{U}_i\|_2^2$.

    **Design Motivation**: XLFM views are not independent — they exhibit occlusion patterns, spatial redundancy, and angular continuity, analogous to dependencies in natural language or multi-view stereo systems. Using "views" (rather than pixels) as the masking unit naturally aligns with the physical sampling structure of XLFM, forcing the model to learn global scene structure and inter-view angular correlations. Compared to pixel-level MAE pretraining, view-level masking captures the essential structure of light field data (experiments show MVM-LF outperforms random masking by 1.07 dB).

2. **Optical Rendering Consistency Loss (ORC Loss)**

    **Function**: Ensures that the reconstructed 3D volume not only structurally matches the ground truth, but also remains physically consistent (optically credible) under the XLFM imaging forward model.

    **Mechanism**: Both the predicted volume $\mathcal{V}_{\text{pred}}$ and the GT volume $\mathcal{V}_{\text{GT}}$ are forward-rendered via 3D convolution with the known system PSF (point spread function), yielding synthesized light field images $\mathbf{I}_{\text{pred}} = h * \mathcal{V}_{\text{pred}}$ and $\mathbf{I}_{\text{GT}} = h * \mathcal{V}_{\text{GT}}$, and the MSE between them is minimized: $\mathcal{L}_{\text{ORC}} = \|h * \mathcal{V}_{\text{pred}} - h * \mathcal{V}_{\text{GT}}\|_2^2$.

    **Design Motivation**: Purely pixel-level losses (MSE/SSIM) only constrain pointwise matching in volume space and do not guarantee consistency of the reconstructed volume under the optical forward model. Directly using raw measurements as constraints is infeasible — raw data contains sensor noise, dark current, and scattering artifacts, which introduce non-physical gradients. ORC Loss uses the PSF forward projection of the GT volume as a clean supervision signal, bridging data-driven learning with wave-optics consistency. Experiments further confirm that ORC Loss is robust to PSF errors (±10% FWHM perturbation causes only ±0.12 dB variation).

3. **XLFM-Zebrafish Standardized Benchmark Dataset**

    **Function**: Fills the gap of lacking standardized datasets and reproducible evaluation protocols in the XLFM field, providing infrastructure for systematic progress in this direction.

    **Mechanism**: 22,581 light field images are collected, covering 3 freely swimming zebrafish + 13 fixed zebrafish (7 for training/validation + 6 unseen for testing), with dual sampling rates (10 fps for high temporal resolution + 1 fps for long-term tracking), along with standardized train/test splits and evaluation procedures.

    **Design Motivation**: Prior comparisons among XLFM reconstruction methods have been largely anecdotal, with the absence of reproducible benchmarks causing fragmented progress. The combination of different behavioral states (freely swimming vs. fixed) and sampling conditions ensures data diversity, while unseen test fish guarantee credible generalization evaluation.

### Loss & Training

**Pretraining Stage**: Only $\ell_2$ loss is used, with batch size = 8, initial lr = 1e-4 + ReduceLROnPlateau scheduler, trained for 250 epochs on 4×A100-80GB GPUs.

**Fine-tuning Stage**: Multi-loss combination:

$$\mathcal{L}_{\text{total}} = \frac{1}{\lambda_1}\mathcal{L}_{\text{MS\_SSIM}} + \frac{1}{\lambda_2}\mathcal{L}_{\text{Edge}} + \frac{1}{\lambda_3}\mathcal{L}_{\text{PSNR}} + \frac{1}{\lambda_4}\mathcal{L}_{\text{MSE}} + \frac{1}{\lambda_5}\mathcal{L}_{\text{ORC}}$$

Batch size = 1 (required by volumetric reconstruction), initialized from pretrained encoder weights. The five loss terms respectively constrain structural similarity, edge sharpness, peak signal-to-noise ratio, pointwise error, and physical consistency.

## Key Experimental Results

### Main Results

XLFM-Zebrafish test set (6 unseen samples), all methods using the same Swin-XLFM or corresponding standard architectures:

| Method | #1 PSNR | #2 PSNR | #3 PSNR | #4 PSNR | #5 PSNR | #6 PSNR | Avg PSNR↑ | Avg SSIM↑ |
|--------|---------|---------|---------|---------|---------|---------|-----------|-----------|
| ConvNeXt | 49.48 | 53.88 | 44.87 | 51.38 | 51.52 | 49.79 | 50.16 | 0.9876 |
| ViT | 49.38 | 52.67 | 45.29 | 51.09 | 51.35 | 45.90 | 49.28 | 0.9876 |
| PVT | 47.21 | 47.93 | 44.50 | 49.46 | 48.32 | 46.60 | 47.34 | 0.9829 |
| EfficientNet | 45.04 | 54.68 | 42.13 | 49.56 | 48.63 | 27.16 | 44.53 | 0.9296 |
| ResNet-50 | 46.46 | 54.89 | 41.46 | 49.47 | 48.82 | 39.98 | 46.85 | 0.9634 |
| ResNet-101 | 47.20 | 54.90 | 41.33 | 49.47 | 49.09 | 39.50 | 46.91 | 0.9554 |
| U-Net | 48.81 | 57.23 | 44.41 | 52.61 | 52.06 | 41.47 | 49.43 | 0.9847 |
| **XLFM-Former** | **53.97** | **59.83** | **49.31** | **54.55** | **54.65** | **51.95** | **54.04** | **0.9944** |

XLFM-Former achieves the highest PSNR and SSIM on all 6 test samples, surpassing the second-best baseline ConvNeXt by **3.88 dB** (7.7%) in average PSNR, with an SSIM improvement of 0.0068.

### Ablation Study

| Configuration | Description | PSNR↑ | SSIM↑ |
|---------------|-------------|-------|-------|
| Baseline | No ORC Loss, no MVM-LF | 52.14 | 0.9924 |
| + ORC Loss only | Physical loss only | 52.96 | 0.9931 |
| + MVM-LF only | View-level pretraining only | 53.38 | 0.9938 |
| **Full (Ours)** | **ORC + MVM-LF** | **54.04** | **0.9944** |
| ImageNet-1K pretrain | Visual domain weights | 52.70 | 0.9931 |
| ImageNet-22K pretrain | Large-scale visual weights | 52.38 | 0.9923 |
| Random mask pretrain | Pixel-level MAE | 52.97 | 0.9934 |
| MAE (ViT backbone) | Standard MAE | 46.55 | 0.9752 |
| **MVM-LF pretrain** | **View-level masking** | **54.04** | **0.9944** |

Robustness to missing views (using MVM-LF pretraining):

| Available View Ratio | PSNR↑ | SSIM↑ |
|----------------------|-------|-------|
| 100% (scratch, no pretrain) | 52.14 | 0.9924 |
| 90% (w/ MVM-LF) | 52.97 | 0.9933 |
| 80% | 53.26 | 0.9936 |
| 70% | 52.67 | 0.9928 |
| 60% | 52.54 | 0.9928 |

### Key Findings

1. **Strong complementarity between ORC Loss and MVM-LF**: Used individually, they yield gains of +0.82 dB and +1.24 dB respectively; combined, the gain is +1.90 dB, nearly additive, indicating that the two components constrain different dimensions (physical consistency vs. angular priors).
2. **View-level > pixel-level masking**: MVM-LF (54.04) outperforms random masking (52.97) by 1.07 dB and ImageNet-1K/22K pretraining by 1.34/1.66 dB, demonstrating the necessity of task-specific pretraining. Standard MAE achieves only 46.55 dB, showing severe mismatch with XLFM data.
3. **Pretraining significantly improves data efficiency**: With only 10% labeled data, MVM-LF pretraining (51.92 dB) already exceeds training from scratch with 100% labels (~52.14 dB), with the advantage being especially pronounced in low-annotation regimes.
4. **Robustness to missing views**: The pretrained model with only 60% of views as input (52.54 dB) still surpasses the scratch model with 100% of views (52.14 dB), demonstrating that MVM-LF teaches the model to infer global structure from incomplete views.
5. **ORC Loss robustness to PSF errors**: ±10% FWHM perturbation of the PSF causes only ±0.12 dB variation, tolerating minor calibration deviations in practical imaging systems.
6. **Cross-domain generalization**: On the H2B-Nemos dataset, XLFM-Former's zero-shot inference (53.72 dB) exceeds the supervised ResNet-101 baseline (51.42 dB) by +2.29 dB, demonstrating strong cross-domain capability.

## Highlights & Insights

1. **Conceptual leap: "Views as atomic units"**: The central insight of this paper is elevating the modeling granularity of light field data from pixels to views — the 27 sub-aperture views of XLFM are analogous to views in multi-view stereo systems, and the occlusion patterns, redundancy, and angular continuity among them constitute structured dependencies. This insight drives the MVM-LF design, which is better suited to the nature of light field data than generic MAE strategies.

2. **Scientific rigor through differentiable rendering constraints**: ORC Loss is not merely an auxiliary regularizer — it explicitly embeds the imaging physics of XLFM (PSF forward model) into the learning process, constraining the network to physically feasible solutions. This is critical for scientific imaging (reconstructions must not only look good but must be optically self-consistent). The key insight is using the PSF forward projection of the GT volume as clean supervision, thereby avoiding non-physical gradients from noisy raw measurements.

3. **Community contribution through benchmark construction**: As the first standardized XLFM benchmark, XLFM-Zebrafish not only supports the experiments in this paper but provides reproducible evaluation infrastructure for the entire field. The long-term value of such "paving" contributions should not be underestimated.

4. **Surprising zero-shot cross-domain results**: XLFM-Former's zero-shot performance on the H2B-Nemos dataset (53.72 dB) even exceeds supervised fine-tuning on the same dataset (52.34 dB), indicating that the angular priors learned through MVM-LF pretraining transfer strongly across domains.

5. **Full-volume reconstruction vs. sparse signal extraction**: Unlike methods such as XLFMNet that reconstruct only sparse neural signals, XLFM-Former reconstructs complete volumetric structures (functional signals + morphological information), which is highly significant for biological applications that require simultaneous analysis of neural activity and anatomical structure.

## Limitations & Future Work

1. **Limited biological diversity**: Validation is conducted only on zebrafish larvae; mice, Drosophila, and other larger or more complex tissues are not explored. Differences in PSF characteristics and imaging conditions across biological models may affect generalization.

2. **High computational resource requirements**: Training requires 4×A100-80GB GPUs, and the batch size for volume reconstruction is limited to 1, constraining accessibility and scalability to larger datasets.

3. **Functional tracking extraction not evaluated**: The paper focuses on 3D volume reconstruction quality (PSNR/SSIM), but neuroscience ultimately cares about neural activity trace extraction accuracy — the full pipeline from reconstructed volumes to functional analysis (registration → segmentation → clustering → trace extraction) is not evaluated end-to-end.

4. **Dependence of ORC Loss on PSF**: Although robust to ±10% FWHM perturbations, scenarios with larger PSF deviations or entirely unknown PSFs are not explored — optical system aging and temperature variation in real-world deployment may cause larger shifts.

5. **Hyperparameter sensitivity of the pretraining strategy**: The 70% masking ratio is determined through experimental search; whether different imaging systems or sample types require re-tuning is not discussed, and there is no theoretical guidance on the optimal epoch ratio between pretraining and fine-tuning.

## Related Work & Insights

| Method | Core Technique | Reconstruction Type | Full Volume? | Self-supervised? | Physical Constraint? | Key Limitation |
|--------|---------------|---------------------|-------------|-----------------|---------------------|----------------|
| XLFMNet | SLNet + XLFMNet sparse decomposition | Sparse neural signals | ✗ | ✗ | ✗ | Ignores complete morphological structure |
| CWFA | Conditional normalizing flow | Sparse activity | ✗ | ✗ | ✗ | Same; neural signals only |
| FNet | Fourier global convolution | Full volume | ✓ | ✗ | ✓ (end-to-end) | Memory explosion; requires multiple GPUs |
| MLFM | Transformer + pixel-level masking | Light field super-resolution | — | ✓ | ✗ | Pixel-level masking mismatches view structure |
| MAE | ViT + random patch masking | General vision | — | ✓ | ✗ | Incompatible with light field data (46.55 dB) |
| **XLFM-Former** | **Swin-T + MVM-LF + ORC** | **Full volume** | **✓** | **✓** | **✓** | Validated on zebrafish only |

**Directions for inspiration**:

- The view-level self-supervised paradigm of MVM-LF can be generalized to other multi-view imaging systems: light field cameras, multi-view CT, and sparse-view completion in NeRF data acquisition.
- The "known forward model + differentiable rendering constraint" framework of ORC Loss applies to any inverse problem — astronomical imaging, MRI reconstruction, ultrasound imaging, and beyond, as long as the forward model is differentiable.
- The finding that view/view-level self-supervision outperforms generic MAE in "data-cheap but label-scarce" scenarios motivates the design of pretraining tasks aligned with physical sampling structures in scientific imaging domains.

## Rating

- **Novelty**: ⭐⭐⭐⭐ View-level masked self-supervised pretraining represents a new paradigm for light field learning, and the PSF differentiable rendering design of ORC Loss is elegant; however, the individual components (Swin-T / MAE / differentiable rendering) are not entirely novel — the innovation lies in their combination and adaptation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ First standardized benchmark + comparison against 7 SOTA architectures + detailed ablation (components / pretraining strategy / masking ratio / annotation ratio / missing views / PSF robustness) + cross-domain generalization (H2B-Nemos) + qualitative visualization — comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ The narrative driven by four design insights is clear, and the correspondence between physical motivation and method design is explicit; placing some architectural details in supplementary materials is slightly inconvenient.
- **Value**: ⭐⭐⭐⭐ Significant infrastructure value for the computational neuroscience community (dataset + benchmark + method); full-volume reconstruction with physical consistency is critical for real scientific applications; the target audience is relatively niche, but the impact within this sub-field is substantial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LiTo: Surface Light Field Tokenization](../../ICLR2026/3d_vision/lito_surface_light_field_tokenization.md)
- [\[NeurIPS 2025\] MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics](mpmavatar_learning_3d_gaussian_avatars_with_accurate_and_robust_physics-based_dy.md)
- [\[NeurIPS 2025\] Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](concerto_joint_2d-3d_self-supervised_learning_emerges_spatial_representations.md)
- [\[NeurIPS 2025\] Motion4D: Learning 3D-Consistent Motion and Semantics for 4D Scene Understanding](motion4d_learning_3d-consistent_motion_and_semantics_for_4d_scene_understanding.md)
- [\[ICCV 2025\] Relative Illumination Fields: Learning Medium and Light Independent Underwater Scenes](../../ICCV2025/3d_vision/relative_illumination_fields_learning_medium_and_light_independent_underwater_sc.md)

</div>

<!-- RELATED:END -->
