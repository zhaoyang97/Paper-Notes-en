---
title: >-
  [Paper Note] CD-DPE: Dual-Prompt Expert Network Based on Convolutional Dictionary Feature Decoupling for Multi-Contrast MRI Super-Resolution
description: >-
  [AAAI 2026][Medical Imaging][Multi-contrast MRI super-resolution] This paper proposes CD-DPE, a network that employs an iterative Convolutional Dictionary Feature Decoupling Module (CD-FDM) to disentangle multi-contrast…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Multi-contrast MRI super-resolution"
  - "convolutional dictionary"
  - "feature decoupling"
  - "dual-prompt"
  - "expert network"
date: 2026-05-08
content_hash: 42c54971b21c5bb1
---

# CD-DPE: Dual-Prompt Expert Network Based on Convolutional Dictionary Feature Decoupling for Multi-Contrast MRI Super-Resolution

**Conference**: AAAI 2026
**arXiv**: [2511.14014](https://arxiv.org/abs/2511.14014)
**Code**: [Available](https://github.com/xianming-gu/CD-DPE)
**Area**: Medical Imaging
**Keywords**: Multi-contrast MRI super-resolution, convolutional dictionary, feature decoupling, dual-prompt, expert network

## TL;DR

This paper proposes CD-DPE, a network that employs an iterative Convolutional Dictionary Feature Decoupling Module (CD-FDM) to disentangle multi-contrast MRI features into cross-contrast shared and modality-specific components, followed by a Dual-Prompt Feature Fusion Expert Module (DP-FFEM) for adaptive fusion and reconstruction. CD-DPE surpasses existing state-of-the-art methods on multiple public benchmarks.

## Background & Motivation

MRI super-resolution (SR) aims to reconstruct high-resolution (HR) images from low-resolution (LR) scans to improve diagnostic accuracy. In clinical practice, multiple contrast sequences (T1W, T2W, PD, etc.) are typically acquired, and rapidly obtained HR reference images (e.g., T1W) can assist in enhancing LR target images that require longer scan times (e.g., T2W).

**Three major limitations of existing methods**:

**Naive fusion strategies**: Early CNN-based methods directly concatenate reference and target images, failing to capture complex cross-contrast dependencies and resulting in blurred reconstruction details.

**Limitations of Transformer-based methods**: Although attention mechanisms can model long-range dependencies, their ability to reconstruct high-frequency details from extremely low-resolution inputs is limited, while computational and memory costs are high.

**Lack of rigorous constraints in decomposition methods**: Existing approaches that decompose reference images into shared/specific components (e.g., Lei et al.) lack strict constraints on the decomposition and fusion mechanisms, causing shared features to be over-smoothed.

Core challenge: **How to effectively extract shared and modality-specific features from multi-contrast MRI while eliminating redundant information interference without sacrificing structural detail?**

## Method

### Overall Architecture

CD-DPE comprises two core modules:

1. **CD-FDM (Convolutional Dictionary Feature Decoupling Module)**: Extracts shared and modality-specific features.
2. **DP-FFEM (Dual-Prompt Feature Fusion Expert Module)**: Adaptively fuses features and reconstructs the HR image.

### Key Designs

#### 1. Convolutional Dictionary Feature Decoupling Module (CD-FDM)

CD-FDM is grounded in convolutional dictionary learning and decomposes multi-contrast MRI into three groups of sparse representations:

**Mathematical formulation**: Multi-contrast images are decomposed as:

$$I_x^s = \sum_j^J u_j^x \otimes \theta_d^x + c_j \otimes \theta_d^c, \quad I_y = \sum_j^J u_j^y \otimes \theta_d^y + c_j \otimes \theta_d^c$$

where $u_j^x, u_j^y$ are modality-specific sparse representations, $c_j$ is the shared sparse representation, and $\theta_d$ denotes dictionary filters.

**Iterative update process** (based on unfolding learning):

- **Specific feature extraction**: $U_x^l = \text{Prox}(U_x^{l-1} - \eta_x \Delta U_x)$, iteratively refined via residual updates through CDME (encoder) and CDMD (decoder).
- **Reference image alignment**: An OffNet (offset network) is introduced, employing a lightweight U-Net to learn a displacement field $\phi$ and feature representation $\mathcal{A}$, aligning reference features to the target image via spatial transformation.
- **Shared feature update**: Residuals of the reference- and target-specific features are subtracted from the common features, ensuring that shared features retain only the structural information genuinely common to both.
- Iteration is performed $L=3$ times, progressively improving decoupling quality.

**Key sub-structures**:

- CDM encoder/decoder: 3-level multi-scale structure with channel sizes 64/96/128.
- MFFN (Multi-scale Feed-Forward Network): Implements the proximal operator Prox.
- OffNet: Handles spatial misalignment between reference and target images.

#### 2. Dual-Prompt Feature Fusion Expert Module (DP-FFEM)

DP-FFEM guides feature fusion through two prompt mechanisms:

**Frequency Prompt**:

- Constructs a reference representation $F_r = [F_y^L, F_c^L]$ and a target representation $F_t = [F_x^L, F_c^L]$.
- Learns trainable frequency prototypes $\mathcal{P}_F$ to perform attention modulation on the Fourier transform of reference features.
- Generates an attention map $\mathcal{V}^y = f_{\phi_1}(\mathscr{F}(F_r), \mathcal{P}_F)$.
- Transfers the reference attention to the target representation: $\tilde{F}_t = F_t \otimes \mathcal{V}^y + F_t$.

**Adaptive Routing Prompt**:

- Learnable routing prompt $\mathcal{P}_R \in \mathbb{R}^{(C \times H \times W) \times E}$.
- Multiplied with target features to produce routing logits; Top-K selection identifies the $K$ most relevant expert branches.
- Softmax normalization yields routing weights $\mathcal{V}^x$.
- Final reconstruction: $\hat{I}_x = \sum_{i=1}^E \mathcal{V}^x \cdot \mathcal{E}_i(\tilde{F}_t \cdot \mathcal{V}^x)$.
- $E=4$ experts and $K=2$ are used.

### Loss & Training

The total loss consists of three terms: $\mathcal{L} = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{fc} + \lambda_2 \mathcal{L}_{mi}$

- **Reconstruction loss** $\mathcal{L}_{rec} = \|\hat{I}_x - I_x^{hr}\|_1$: L1 distance ensures content fidelity.
- **Consistency loss** $\mathcal{L}_{fc}$: Constrains the combination of specific and shared features to reconstruct the original image; $\lambda_y = 0.01$ balances the two terms.
- **Decoupling loss** $\mathcal{L}_{mi}$: Minimizes mutual information between shared and specific features to enforce feature independence.

Training configuration: Adam optimizer (lr=1e-4), batch size=4, 50 epochs, NVIDIA RTX A6000 48GB, $\lambda_1=1, \lambda_2=0.1$.

## Key Experimental Results

### Main Results

CD-DPE is compared against five state-of-the-art methods on two public datasets, BraTS2018 and IXI:

**BraTS2018 dataset (T1W→T2W reconstruction):**

| Method | 2× PSNR↑ | 2× SSIM↑ | 4× PSNR↑ | 4× SSIM↑ | Params(M) | FLOPs(G) |
|--------|----------|----------|----------|----------|-----------|----------|
| WavTrans | 39.79 | 0.9874 | 34.83 | 0.9677 | 10.0 | 216.2 |
| A2-CDic | 40.47 | 0.9883 | 35.70 | 0.9704 | 10.1 | 831.1 |
| **CD-DPE** | **40.70** | **0.9885** | **36.00** | **0.9716** | 11.7 | 426.1 |

**IXI dataset (PD→T2W reconstruction):**

| Method | 2× PSNR↑ | 4× PSNR↑ | 4× SSIM↑ |
|--------|----------|----------|----------|
| WavTrans | 42.88 | 38.51 | 0.9711 |
| A2-CDic | 41.59 | 37.91 | 0.9726 |
| **CD-DPE** | **43.22** | **38.59** | **0.9735** |

### Ablation Study

**Module ablation (BraTS2018 4× SR):**

| Configuration | PSNR Change | SSIM Change |
|---------------|-------------|-------------|
| w/o CD-FDM (replaced by CNN) | −13.48% | −1.92% |
| w/o DP-FFEM (replaced by CNN) | −5.93% | −0.55% |
| w/o dual prompts (DP-FFEM retained) | Below full model | Below full model |
| w/o $\mathcal{L}_{mi}$ | −3.05% | −0.22% |
| w/o $\mathcal{L}_{fc}$ | −2.90% | −0.31% |

**Generalization experiment (trained on IXI → tested on FastMRI Knee):**

| Method | PSNR↑ | SSIM↑ |
|--------|-------|-------|
| WavTrans | 28.07 | 0.7428 |
| A2-CDic | 25.21 | 0.7517 |
| **CD-DPE** | **29.41** | **0.8387** |

CD-DPE achieves a 4.8% PSNR gain and 11.6% SSIM gain on the unseen dataset, demonstrating substantially superior generalization over competing methods.

### Key Findings

- CD-FDM is the performance backbone of the model; removing it causes a 13.48% PSNR drop, demonstrating that convolutional dictionary decoupling is significantly superior to simple CNN-based decomposition.
- The mutual information loss $\mathcal{L}_{mi}$ is critical for preventing feature entanglement; without it, specific and shared features cannot be correctly separated.
- The two prompts in DP-FFEM are complementary: the frequency prompt guides feature selection, while the routing prompt determines the optimal fusion strategy.
- CD-DPE achieves a moderate parameter count (11.7M) and inference time (0.061s), with FLOPs (426G) lower than A2-CDic (831G).

## Highlights & Insights

1. **Rigor of convolutional dictionary decoupling**: By employing unfolding learning to transform the optimization problem into learnable network layers, the approach is more theoretically grounded than heuristic decomposition methods.
2. **Complementary design of dual prompts**: The frequency prompt captures structural patterns in the frequency domain (*what to fuse*), while the routing prompt determines the fusion pathway (*how to fuse*), with clearly defined roles for each.
3. **Introduction of MoE concepts**: The combination of expert networks and routing prompts represents a natural application of Mixture-of-Experts in image reconstruction, enhancing the flexibility of the fusion strategy.
4. **Outstanding generalization**: Large performance margins on the completely unseen FastMRI Knee dataset demonstrate that the decoupling-fusion paradigm possesses inherent generalization advantages.

## Limitations & Future Work

- **Sensitivity to contrast differences**: Performance degrades when the contrast mechanisms of reference and target images differ substantially; incorporating MRI physical priors (e.g., relaxation time mapping) could help.
- **Iterative computational overhead**: The iterative unfolding in CD-FDM introduces additional computation; more efficient decoupling mechanisms warrant exploration.
- **2D slices only**: Experiments are conducted on 2D slices and have not been extended to 3D volumetric data.
- **Single reference image**: Only one contrast modality is used as a reference; multi-reference, multi-contrast joint reconstruction is a promising direction.

## Related Work & Insights

- **A2-CDic**: Also based on convolutional dictionary but lacks dual-prompt fusion; CD-DPE extends it by introducing MoE concepts.
- **DiffMSR**: Employs diffusion models for multi-contrast SR, but with lower inference efficiency.
- **DANCE**: Uses neighborhood-guided aggregation strategies with hand-crafted designs that limit generalization.
- The framework of convolutional dictionary decoupling combined with MoE fusion is broadly applicable to other multimodal medical image reconstruction tasks (e.g., CT-MRI fusion).

## Rating

- **Novelty**: ★★★★☆ — The combination of convolutional dictionary decoupling with dual-prompt MoE is original.
- **Experimental Thoroughness**: ★★★★★ — Two datasets, detailed ablations, generalization validation, and feature visualizations.
- **Writing Quality**: ★★★★☆ — Mathematical derivations are rigorous, though the density of notation makes reading somewhat demanding.
- **Value**: ★★★★☆ — Fast inference (0.061s), open-source code, and strong generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI](pings-x_physics-informed_normalized_gaussian_splatting_with_axes_alignment_for_e.md)
- [\[AAAI 2026\] Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics](dual-path_knowledge-augmented_contrastive_alignment_network_for_spatially_resolv.md)
- [\[AAAI 2026\] DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis](dw-dgat_dynamically_weighted_dual_graph_attention_network_for_neurodegenerative_.md)
- [\[AAAI 2026\] DeepGB-TB: A Risk-Balanced Cross-Attention Gradient-Boosted Convolutional Network for Rapid, Interpretable Tuberculosis Screening](deepgb-tb_a_risk-balanced_cross-attention_gradient-boosted_convolutional_network.md)
- [\[CVPR 2026\] MRI Contrast Enhancement Kinetics World Model](../../CVPR2026/medical_imaging/mri_contrast_enhancement_kinetics_world_model.md)

</div>

<!-- RELATED:END -->
