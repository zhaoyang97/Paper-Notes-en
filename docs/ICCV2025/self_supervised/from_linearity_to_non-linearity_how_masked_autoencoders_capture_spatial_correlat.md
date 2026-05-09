---
title: >-
  [Paper Note] From Linearity to Non-Linearity: How Masked Autoencoders Capture Spatial Correlations
description: >-
  [ICCV 2025][Self-Supervised Learning][Masked Autoencoder] This paper theoretically analyzes how MAE learns spatial correlations in images. It derives a closed-form solution for linear MAE, reveals how masking ratio and patch size select short- or long-range spatial features, and extends the analysis to nonlinear MAE, providing theoretical guidance for hyperparameter selection in practice.
tags:
  - ICCV 2025
  - Self-Supervised Learning
  - Masked Autoencoder
  - spatial correlation
  - linear analysis
  - hyperparameter selection
  - ViT
date: 2026-05-08
content_hash: 873e44dbea4cf221
---

# From Linearity to Non-Linearity: How Masked Autoencoders Capture Spatial Correlations

**Conference**: ICCV 2025
**arXiv**: [2508.15404](https://arxiv.org/abs/2508.15404)
**Code**: None
**Area**: Self-Supervised Learning / Theory
**Keywords**: Masked Autoencoder, spatial correlation, linear analysis, hyperparameter selection, ViT

## TL;DR

This paper theoretically analyzes how MAE learns spatial correlations in images. It derives a closed-form solution for linear MAE, reveals how masking ratio and patch size select short- or long-range spatial features, and extends the analysis to nonlinear MAE, providing theoretical guidance for hyperparameter selection in practice.

## Background & Motivation

MAE (Masked Autoencoders) has become a core technique for pre-training visual foundation models (e.g., SAM, EVA, Unified IO), yet its working mechanism remains poorly understood. Key limitations include:

- MAE requires extensive tuning of hyperparameters such as masking ratio, patch size, and encoder/decoder depth.
- Exhaustive search is prohibitively expensive when adapting to new datasets or modalities (e.g., 1600 training epochs on ImageNet).
- **The relationship between hyperparameter choices and downstream task performance lacks theoretical explanation.**

The core hypothesis is that **MAE learns spatial correlations in the input image**, with masking ratio and patch size controlling the spatial scale of these correlations. MAE introduces a spatial locality inductive bias into ViT architectures that would otherwise be absent.

## Method

### Overall Architecture

The paper begins with an exact analytical solution for linear MAE and progressively extends the analysis to nonlinear MAE via Jacobian analysis, establishing a complete theoretical chain from "masking parameters → feature spatial scale → downstream task adaptation."

### Key Designs

1. **Closed-Form Solution for Linear MAE (Theorem 1)**:

   Linear MAE minimizes the objective $\ell_m(A,B) = \mathbb{E}_R[\|X - (R \odot X)AB\|^2]$, where $R$ is a random mask.

   By computing the expectation in closed form, the objective decomposes into two terms:
   $$\ell_m(A,B) = \underbrace{\|X - (1-m)XAB\|^2}_{\text{reconstruction term}} + m(1-m) \underbrace{\|GAB\|^2}_{\text{regularization term}}$$

   where $G^\top G = \text{blkdiag}_p(X^\top X)$ is the block-diagonal version of the data covariance matrix with block size equal to patch size $p$.

   **Key insights**:
   - $m=0$ degenerates to a standard autoencoder (no regularization term).
   - The masking ratio $m$ controls the regularization strength.
   - The patch size $p$ controls the block-diagonal structure.
   - The global optimum is: $B = CU_k$, $A = V^{-1}X^\top X B^\top (BB^\top)^{-1} C^{-1}$, where $V = (1-m)X^\top X + m \cdot \text{blkdiag}_p(X^\top X)$ and $U_k$ are the top-$k$ eigenvectors of $X^\top X V^{-1} X^\top X$.

   **Contrast with AE**: AE performs PCA to extract directions of maximum variance; MAE projects onto directions after weighted whitening by $V^{-1}X^\top X$, thereby **selecting features that appear as cross-patch redundancies** rather than high-variance directions. This is the core reason why MAE outperforms AE on downstream perception tasks.

2. **Ising Model Validation**:

   Data are generated using the Ising model (a probabilistic model with controllable local correlations), approximating the correlation function as $\langle x_i, x_j \rangle = \tanh(J)^r$.

   Key finding: The MAE encoder **preferentially attends to dimensions at patch boundaries**, since boundary positions exhibit the highest correlation with neighboring patches and are most informative for predicting masked patches. This directly validates the theoretical prediction that MAE learns cross-patch redundant features.

3. **Spatial Information Integration Analysis (Jacobian Analysis)**:

   The influence of input pixel $i$ on reconstructed output pixel $j$ is measured via the Jacobian $|(AB)_{ij}|$. Exponential fits are applied to models trained on CIFAR-10:

   - **AE**: learns highly localized kernels; influence decays rapidly with distance.
   - **DAE (denoising autoencoder)**: slightly less localized than AE.
   - **MAE**: integrates information from farther spatial distances; Jacobian decays more slowly.

   **Masking ratio effect**: Higher masking ratio → more diffuse average Jacobian → utilization of longer-range information.
   **Patch size effect**: Larger patch → more integration of information from outside the patch → higher spatial entropy.

4. **Nonlinear MAE Analysis (Adaptive Basis)**:

   Drawing on the diffusion model analysis of Kadkhodaie et al., the nonlinear MAE is approximated via a first-order Taylor expansion:
   $$h(\tilde{x}) \approx A(\tilde{x}) + b$$
   where $A = \nabla_{\tilde{x}} h(\tilde{x})$ is the Jacobian (adaptive basis matrix).

   Key findings:
   - The basis learned by nonlinear MAE **adapts to the input image** (unlike the fixed basis of linear MAE).
   - During training, the basis progressively transitions from highly localized to globally diffuse.
   - ViT exploits higher-order correlations (e.g., the association between a rider's shirt and a horse), surpassing the second-order statistical limitations of linear models.

### Practical Hyperparameter Guidelines

- **Encoder vs. decoder depth**: Increasing encoder depth consistently improves linear probe accuracy, while the optimal decoder requires only 2–4 layers. A single-layer decoder loses only 0.30% accuracy after fine-tuning (95.26% vs. 95.56%).
- **Masking ratio × patch size**: Small patch size combined with high masking ratio achieves the best linear probe accuracy, but at the cost of slower training.
- **Fine-tuning strategy**: Freezing all layers except the last Transformer block results in only ~2% accuracy loss while saving nearly half the training time and memory.
- Reconstruction error is a poor proxy for downstream performance.

## Key Experimental Results

### Main Results: Feature Spatial Scale and Downstream Tasks

| Method | Premise | Trend as Gabor $\sigma$ increases | Depth Prediction |
|--------|---------|----------------------------------|-----------------|
| AE | Max-variance features | Similar across $\sigma$ | Worse than MAE at low dim. |
| DAE ($\sigma=0.2$) | Denoising + L2 regularization | Slightly better than AE | Performance between AE and MAE |
| MAE ($m=0.8$) | Cross-patch redundant features | **Markedly better at high $\sigma$** | **Significantly better at low dim.** |

MAE with high masking ratio shows the greatest advantage on tasks requiring long-range spatial information integration (large-$\sigma$ Gabor filters).

### Ablation Study: Encoder/Decoder Depth (CIFAR-10)

| Encoder Layers | Decoder Layers | Linear Probe Acc. | Fine-tune Acc. | Note |
|---------------|---------------|-------------------|---------------|------|
| 12 | 1 | ~90% | 95.26% | Fastest training |
| 12 | 2 | ~92% | 95.48% | Good balance |
| 12 | 4 | ~92% | 95.56% | Best performance |
| 12 | 8 | ~91% | 95.40% | Oversized decoder hurts |

| Masking Ratio | Patch Size | Linear Probe Trend | Note |
|--------------|-----------|-------------------|------|
| 0.1 | 2 | Lower | Too much visible context; no cross-patch need |
| 0.5 | 2 | Medium | Moderate regularization |
| 0.8 | 2 | Highest | Forces use of long-range information |
| 0.8 | 4 | Slightly lower but faster | Larger patch reduces token count |

### Key Findings

- **MAE ≠ AE + regularization**: The regularization term $\|GAB\|^2$ in MAE has a data-aware block-diagonal structure rather than a simple L2 penalty, leading to fundamentally different learned features.
- **Masking introduces a spatial locality inductive bias**: ViT itself lacks locality; MAE indirectly introduces a CNN-like locality prior through masking.
- **Training dynamics**: The Jacobian of nonlinear MAE evolves from localized to global during training, resembling a "local-first, global-later" curriculum.
- **Redundancy feature hypothesis**: MAE succeeds because downstream perception tasks are themselves redundant functions of the input, and MAE precisely identifies these cross-patch redundant features.

## Highlights & Insights

- A complete theory-to-experiment chain is established, from exact solutions of linear models to Jacobian analysis of nonlinear models.
- The Ising model experiment concisely and powerfully reveals MAE's tendency to preferentially learn patch boundary features.
- The contrast between "MAE selects redundant features" and "AE selects high-variance features" is highly illuminating.
- Practical recommendations are clear: small patch size + high masking ratio + shallow decoder + freezing most layers during fine-tuning.

## Limitations & Future Work

- The linear analysis is restricted to rank-deficient or full-rank encoder/decoder settings, whereas ViT in practice uses overcomplete representations.
- Linear MAE captures only second-order statistics; higher-order analysis is limited to Jacobian approximations.
- Experiments are conducted primarily on CIFAR-10 (32×32); validation on higher-resolution images is limited.
- Guidance on selecting optimal MAE hyperparameters for specific downstream tasks (e.g., segmentation, detection) is absent.
- Only fully connected architectures are considered for linear MAE, leaving a gap with ViT's attention-based structure.

## Related Work & Insights

- **Baldi & Hornik (1989)**'s AE–PCA equivalence serves as the starting point for the linear analysis in this work.
- The analysis methodology from **Kadkhodaie et al.** on diffusion models is adapted for MAE.
- **Kong et al.** hypothesize that MAE learns a hierarchical latent variable model; this paper proposes the more concrete "spatial correlation learning" hypothesis.
- This work has direct practical implications: when applying MAE to a new dataset, practitioners should first analyze the spatial correlation scale of the data and select patch size and masking ratio accordingly.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First derivation of the closed-form solution for linear MAE and establishment of the masking parameter–spatial scale correspondence.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both theoretical validation and practical evaluation, from the Ising model to CIFAR-10/ImageNet.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations, excellent figure design, and natural transitions from theory to practice.
- Value: ⭐⭐⭐⭐ Provides a theoretical basis for MAE hyperparameter selection and offers lasting reference value to the community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Suppressing Non-Semantic Noise in Masked Image Modeling Representations](../../CVPR2026/self_supervised/suppressing_non-semantic_noise_in_masked_image_modeling_representations.md)
- [\[NeurIPS 2025\] Manifolds and Modules: How Function Develops in a Neural Foundation Model](../../NeurIPS2025/self_supervised/manifolds_and_modules_how_function_develops_in_a_neural_foundation_model.md)
- [\[NeurIPS 2025\] Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings](../../NeurIPS2025/self_supervised/hybrid_autoencoders_for_tabular_data_leveraging_model-based_augmentation_in_low-.md)
- [\[CVPR 2026\] LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency](../../CVPR2026/self_supervised/las-comp_zero-shot_3d_completion_with_latent-spatial_consistency.md)
- [\[ICCV 2025\] Scaling Language-Free Visual Representation Learning](scaling_languagefree_visual_representation_learning.md)

<!-- RELATED:END -->
