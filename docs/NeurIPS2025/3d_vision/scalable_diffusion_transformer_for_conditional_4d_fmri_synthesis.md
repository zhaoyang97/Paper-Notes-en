---
title: >-
  [Paper Note] Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis
description: >-
  [3D Vision] This paper proposes the first diffusion Transformer for voxel-level whole-brain 4D fMRI conditional generation, combining 3D VQ-GAN latent space compression, a CNN-Transformer hybrid backbone, and strong conditioning via AdaLN-Zero and cross-attention. The model achieves a task activation map correlation of 0.83, RSA of 0.98, and perfect condition specificity across seven cognitive tasks from the HCP dataset.
tags:
  - 3D Vision
date: 2026-05-08
content_hash: 19f4bc9d41c96009
---

# Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis

## Metadata
- **Conference**: NeurIPS 2025
- **arXiv**: [2511.22870](https://arxiv.org/abs/2511.22870)
- **Code**: Not yet available
- **Area**: 3D Vision
- **Keywords**: Diffusion Models, Transformer, fMRI Generation, Conditional Generation, Brain Imaging

## TL;DR
This paper proposes the first diffusion Transformer for voxel-level whole-brain 4D fMRI conditional generation, combining 3D VQ-GAN latent space compression, a CNN-Transformer hybrid backbone, and strong conditioning via AdaLN-Zero and cross-attention. The model achieves a task activation map correlation of 0.83, RSA of 0.98, and perfect condition specificity across seven cognitive tasks from the HCP dataset.

## Background & Motivation

Task-based fMRI provides a unique window into the spatiotemporal dynamics underlying cognitive processes, and building generative models that capture the cognition-to-brain-activity mapping is a frontier direction in cognitive neuroscience. However, voxel-level whole-brain 4D task fMRI generation faces severe challenges:

**Extreme dimensionality**: fMRI data is a four-dimensional tensor $x \in \mathbb{R}^{H \times W \times D \times T}$ with simultaneously high spatial and temporal dimensions.

**Variance dominated by nuisance factors**: PCA analysis reveals that individual differences explain the largest variance, phase-encoding direction the second, while task-evoked signals emerge only as a weak third component — task signals are effectively "submerged."

**Oversimplification in prior work**: Previous approaches avoid voxel-level dynamics, instead adopting simplified representations such as ROI time series, functional connectivity matrices, or static 3D activation maps, thereby discarding critical voxel-level spatiotemporal information.

**Lack of neuroscience-oriented evaluation**: Standard image metrics such as FID cannot assess whether generated fMRI preserves task-specific spatiotemporal dynamics.

**Historical gap**: No prior method has successfully generated task-conditioned whole-brain 4D fMRI data using modern generative architectures.

## Method

### Overall Architecture

The model adopts latent diffusion modeling: a pretrained 3D VQ-GAN compresses fMRI volumes into a latent space $z \in \mathbb{R}^{C \times (H/4) \times (W/4) \times (D/4) \times T}$, a conditional diffusion process is performed in the latent space, and a VQ-GAN decoder reconstructs the 4D fMRI from the generated latents.

### Key Designs

1. **Latent Space Compression (3D VQ-GAN)**: Direct diffusion in voxel space is computationally intractable. The pretrained 3D VQ-GAN from Kim et al. is fine-tuned to compress fMRI volume-by-volume, reducing spatial resolution by a factor of four and substantially lowering dimensionality while preserving spatial structure. Temporal frames are stacked along the channel dimension to jointly model spatiotemporal information.

2. **CNN-Transformer Hybrid Backbone**: This design balances efficiency, inductive bias, and scalability under limited data conditions:

    - **Early layers use convolutional residual blocks**: Provide strong local spatiotemporal inductive bias, reduce computational cost, and stabilize training with limited data.
    - **Later layers use Transformer blocks**: Capture long-range dependencies across space and time via global attention, leveraging the strong scalability of diffusion Transformers.
    - **UNet-style hierarchical structure**: Features at different resolutions are fused via concatenation, integrating local detail with global context.

3. **Dual Conditioning Mechanism**: Designed to overcome individual and acquisition variability and to amplify weak task-specific signals:

    - **Adaptive normalization**: Transformer blocks employ AdaLN-Zero (modulating LayerNorm scale and shift from condition $c$); convolutional residual blocks use FiLM for condition-dependent modulation.
    - **Cross-attention**: Directly exchanges information between condition embeddings and latent tokens, injecting stronger task-specific signals.

### Loss & Training

The forward process progressively adds Gaussian noise: $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$

Training minimizes the simple objective (MSE of predicted noise):
$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{z_0, \epsilon, t, c} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c) \|^2 \right]$$

- AdamW optimizer (lr=$1 \times 10^{-4}$, weight decay=0.01)
- 400k training steps, batch size 16
- Linear noise schedule ($\beta_{\text{start}}=0.0015$, $\beta_{\text{end}}=0.0195$), $T=1000$
- Class dropout rate of 0.05 for classifier-free guidance
- EMA decay of 0.9999 for sampling
- Single A100 (40 GB), bfloat16 mixed precision

## Key Experimental Results

### Main Results: Scaling Performance (Neuroscience Alignment Metrics)

| Model Size | Parameters | Corr(↑) | RSA(↑) | Top-1 Acc(↑) |
|---------|-------|---------|--------|-------------|
| 38.1M | 38.1M | ~0.55 | ~0.85 | ~0.60 |
| 85.4M | 85.4M | ~0.60 | ~0.92 | ~0.72 |
| 151.5M | 151.5M | ~0.63 | ~0.92 | ~0.73 |
| 236.5M | 236.5M | ~0.70 | ~0.95 | ~0.86 |
| 340.3M | 340.3M | ~0.80 | ~0.98 | **1.00** |
| 462.9M | 462.9M | **~0.83** | **~0.98** | **1.00** |
| MONAI Baseline | ~237M | ~0.50 | ~0.80 | ~0.40 |

Performance improves monotonically with parameter count, exhibiting clear scaling laws reminiscent of foundation models.

### Ablation Study: Architecture and Conditioning Mechanisms

| Model Variant | Parameters | Corr(↑) | Top-1 Acc(↑) | RSA(↑) |
|---------|-------|---------|-------------|--------|
| **Hybrid (CNN-early + Transformer-late)** | 236.5M | **0.7006** | **0.8571** | **0.9526** |
| All-CNN | 235.0M | 0.6289 | 0.7143 | 0.9195 |
| All-Transformer | 238.0M | 0.6734 | 0.7143 | 0.9448 |
| Full conditioning (AdaLN + CrossAttn) | 151.5M | **0.6267** | 0.5714 | **0.9207** |
| AdaLN-Zero only (no cross-attention) | 110.5M | 0.5066 | 0.7143 | 0.9001 |

### Key Findings

1. **Hybrid architecture is optimal**: The all-CNN variant performs weakest; the all-Transformer variant is marginally better; the CNN-Transformer hybrid achieves the best balance of performance and efficiency.
2. **Cross-attention is critical**: Removing cross-attention reduces Corr from 0.6267 to 0.5066, demonstrating that strong conditioning injection is indispensable for capturing weak task-evoked signals in fMRI.
3. **Clear scaling laws**: All three metrics improve consistently with parameter count; models above 340M achieve perfect condition specificity (Top-1 Acc = 1.0).
4. **Volume-wise 3D VQ-GAN compression is viable**: A dedicated 4D compression network is unnecessary; volume-wise compression with channel stacking suffices to support effective 4D synthesis.
5. **ROI time series validation**: ROI-averaged time series from synthesized data align with the hemodynamic responses of real data, whereas the MONAI baseline underestimates or distorts condition-specific responses.

## Highlights & Insights

1. **Filling a historical gap**: This is the first voxel-level whole-brain 4D fMRI conditional generative model, representing a qualitative leap from simplified representations to complete spatiotemporal modeling.
2. **Neuroscience-oriented evaluation framework**: The proposed three-dimensional evaluation combining Corr, RSA, and condition specificity is more informative than FID/IS for assessing generation fidelity in a neuroscientific sense.
3. **Four design principles**: (1) sufficient model capacity and a scalable backbone; (2) appropriate inductive bias under limited data; (3) strong conditioning injection to capture task signals; (4) 3D VQ-GAN as a practical substitute for 4D compression.
4. **Discovery of scaling laws**: Generative neuroimaging may benefit from scaling similarly to vision and language models, suggesting the feasibility of a foundation model for fMRI generation.

## Limitations & Future Work

- Training data are drawn exclusively from the HCP dataset; cross-site generalization remains unvalidated.
- Only one representative condition per paradigm is selected, leaving multi-condition interactions unexplored.
- Integration of multimodal signals (e.g., structural MRI, DTI) is not investigated.
- Downstream applications of the generative model (virtual experiments, data augmentation, etc.) are proposed only as future directions without empirical validation.
- Volume-wise VQ-GAN compression may discard inter-frame temporal continuity information.

## Related Work & Insights

- **Structural MRI diffusion generation**: Pinaya et al. (BrainDiffusion), Khader et al. — 3D brain anatomy synthesis.
- **DiT (Peebles & Xie)**: Foundational work on the scalability of diffusion Transformers.
- **VQ-GAN (Kim et al.)**: 3D medical image compression.
- **Simplified fMRI generation**: MindSimulator (3D activation maps), ROI time series generation.
- **Insight**: For generation tasks with extremely low signal-to-noise ratios (e.g., task-evoked signals in fMRI), the design of the conditioning injection mechanism may matter more than the choice of backbone architecture.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First conditional 4D fMRI diffusion Transformer; a pioneering contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ — Scaling study and ablations are comprehensive, but evaluation is limited to the HCP dataset.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clearly articulated, methodology is concisely presented, and the evaluation framework is elegantly designed.
- Value: ⭐⭐⭐⭐⭐ — Opens a practical path toward a foundation model for fMRI generation.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](../../ICCV2025/3d_vision/self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[ICCV 2025\] JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers](../../ICCV2025/3d_vision/jointdit_enhancing_rgb-depth_joint_modeling_with_diffusion_transformers.md)
- [\[ICCV 2025\] Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](../../ICCV2025/3d_vision/unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](../../ICCV2025/3d_vision/repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICLR 2026\] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](../../ICLR2026/3d_vision/ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)

<!-- RELATED:END -->
