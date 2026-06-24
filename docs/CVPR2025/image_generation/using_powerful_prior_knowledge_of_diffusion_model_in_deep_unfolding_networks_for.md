---
title: >-
  [Paper Note] Using Powerful Prior Knowledge of Diffusion Model in Deep Unfolding Networks for Image Compressive Sensing
description: >-
  [CVPR 2025][Image Generation][Compressive Sensing] By embedding the powerful prior knowledge of pre-trained diffusion models into Deep Unfolding Networks (DUNs), this work proposes the DMP-DUN method, enabling high-quality image compressive sensing reconstruction in only 2 steps.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Compressive Sensing"
  - "Deep Unfolding Networks"
  - "Diffusion Models"
  - "Image Reconstruction"
  - "Message Passing"
date: 2026-05-08
content_hash: 0856cce03982f613
---

# Using Powerful Prior Knowledge of Diffusion Model in Deep Unfolding Networks for Image Compressive Sensing

**Conference**: CVPR 2025  
**arXiv**: [2503.08429](https://arxiv.org/abs/2503.08429)  
**Code**: [GitHub](https://github.com/FengodChen/DMP-DUN-CVPR2025)  
**Area**: Image Generation  
**Keywords**: Compressive Sensing, Deep Unfolding Networks, Diffusion Models, Image Reconstruction, Message Passing

## TL;DR

By embedding the powerful prior knowledge of pre-trained diffusion models into Deep Unfolding Networks (DUNs), this work proposes the DMP-DUN method, enabling high-quality image compressive sensing reconstruction in only 2 steps.

## Background & Motivation

Image Compressive Sensing (CS) reconstructs original signals from measurements below the Nyquist sampling rate and is widely applied in fields such as MRI and Snapshot Compressive Imaging. Mathematically, the CS problem is represented as $\mathbf{y} = \mathbf{\Phi x} + \epsilon$, where $M \ll N$.

Deep Unfolding Networks (DUNs) map traditional iterative optimization algorithms into neural network layers, combining the interpretability of traditional algorithms with the efficiency of deep learning. However, the reconstruction quality of DUNs depends on the quality of the learned prior knowledge.

On the other hand, pre-trained diffusion models possess strong image priors, but directly applying them to CS reconstruction requires numerous iterative steps (e.g., DDNM requires 1000 steps) and performs poorly under low CS ratios.

**Core Motivation**: Can the strong priors of diffusion models be combined with the fast convergence of DUNs? Specifically, by embedding a single-step backward process of the diffusion model into each unfolding layer of DUNs to achieve high-quality reconstruction in a few steps.

## Method

### Overall Architecture

DMP-DUN consists of a three-step workflow: (1) designing the Diffusion Message Passing (DMP) iterative optimization algorithm to embed a pre-trained diffusion model into each iteration; (2) unfolding the DMP algorithm into a deep neural network named DMP-DUN; (3) learning timesteps and scaling parameters via end-to-end training, replacing manual hyperparameter tuning. In the overall structure, each DMP step contains three sub-modules: gradient descent, ResBlock, and diffusion model denoising, with a ResBlock at both the beginning and the end for input mapping and channel transformation.

### Key Design 1: Diffusion Message Passing (DMP) Algorithm

**Function**: Embeds a pre-trained diffusion model as a denoiser within the traditional AMP algorithm framework.

**Mechanism**: Inspired by the Onsager correction term in the AMP algorithm, DMP performs reconstruction using the following iterative formulas: $\mathbf{s}_t = \mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{\Phi}^T (\mathbf{\Phi x}_t - \mathbf{y})$ (gradient descent step), $\mathbf{r}_t = \mathcal{D}_t[\mathbf{s}_t + \sqrt{\bar{\alpha}_t} o_t]$ (Gaussian filtering + Onsager correction), and $\mathbf{x}_{t-1} = p_\theta(\mathbf{x}_{t-1}|\mathbf{r}_t)$ (diffusion model denoising). State evolution analysis demonstrates that $\mathbf{r}_t \sim \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}, \sigma_t^2 \mathbf{I})$, maintaining the data manifold constraint.

**Design Motivation**: Traditional methods (e.g., DDNM, MPGD) directly modify the diffusion process for reconstruction, which fails to preserve the data manifold and deviates severely at low CS ratios. Leveraging the error-decoupling properties of the Onsager term, DMP ensures that the variables at each iteration remain on the intermediate state manifold of the second-order diffusion model, thereby improving reconstruction quality.

### Key Design 2: Deep Unfolding and Lightweighting

**Function**: Unfolds DMP into an end-to-end trainable network to reduce computational overhead.

**Mechanism**: Lightweight convolutional residual blocks (ResBlocks) are utilized to replace two computationally expensive operations: (1) the divergence computation in the Onsager term $o_t$ (which requires Monte-Carlo SURE approximation in traditional methods, doubling the overhead); (2) the Gaussian filter $\mathcal{D}_t$. The head ResBlock directly maps the input $\mathbf{\Phi}^T \mathbf{y}$ to the distribution of the diffusion model's intermediate step $\mathbf{x}_K$, skipping the first $T-K$ reverse diffusion steps. The tail ResBlock converts the RGB output into a single-channel image.

**Design Motivation**: In the original DMP, calculating divergence requires an additional full forward pass of the diffusion model (via Monte-Carlo approximation), which is computationally intensive. Learning to replace the divergence computation with ResBlocks reduces the computation by several times. Meanwhile, the head mapping directly bypasses a large number of diffusion steps, making 2-step reconstruction feasible.

### Key Design 3: Learnable Timesteps and Scaling Parameters

**Function**: Automatically optimizes the hyperparameters in the diffusion model.

**Mechanism**: In traditional diffusion models, timesteps $t$ and scaling parameters $\bar{\alpha}_t$ are predefined hyperparameters. DMP-DUN treats them as trainable parameters, automatically discovering the optimal scheduling strategy through end-to-end optimization.

**Design Motivation**: Manually setting timestep schedules (such as uniform distribution) is suboptimal for few-step reconstruction. A learnable scheduling strategy allows the model to adaptively select the most effective combination of diffusion timesteps.

### Loss & Training

A standard MSE reconstruction loss is adopted. Joint training is performed across multiple CS ratios (1%, 4%, 10%, 25%, 50%), enabling a single model to adapt to various sampling rate scenarios.

## Key Experimental Results

### Main Results (Set11 Dataset, PSNR/SSIM)

| Method | CS=1% | CS=10% | CS=25% | CS=50% | FLOPs(G) |
|------|-------|--------|--------|--------|----------|
| ISTA-Net+ (CVPR18) | 17.45/0.413 | 26.49/0.804 | 32.44/0.924 | 38.08/0.968 | 56.2 |
| OCTUF (CVPR23) | 21.75/0.593 | 30.70/0.903 | 36.10/0.960 | 41.34/0.984 | 189.3 |
| CPP-Net (CVPR24) | 22.19/0.614 | 31.27/0.914 | 36.35/0.963 | 41.39/0.983 | 153.5 |
| DDNM (1000 steps) | 17.95/0.450 | 25.78/0.815 | 27.80/0.893 | 29.01/0.941 | 67039 |
| **DMP-DUN (2 steps)** | **23.18/0.629** | **32.63/0.921** | **37.58/0.965** | **42.06/0.984** | **157.0** |
| **DMP-DUN (4 steps)** | **23.32/0.631** | **33.22/0.928** | **38.29/0.968** | **42.82/0.985** | **303.4** |

### Ablation Study

| Configuration | Set11 Avg PSNR | Urban100 Avg PSNR |
|------|---------------|-------------------|
| No Unfolding (Direct DMP 10 steps) | 32.99 | — |
| DMP-DUN (2 steps+ResBlock) | 32.74 | — |
| DMP-DUN (4 steps+ResBlock) | 33.26 | — |

### Key Findings

1. **SOTA with Extremely Few Steps**: DMP-DUN outperforms all traditional DUN methods and the 1000-step DDNM in only 2 steps, achieving a PSNR over 8 dB higher on Set11.
2. **Significant Reduction in FLOPs**: Compared to 67039G FLOPs of DDNM, the 2-step DMP-DUN requires only 157G (a 427x reduction), while the 4-step version requires 303G.
3. **Clear Advantage at Low CS Ratios**: Under the extreme condition of CS=1%, DMP-DUN outperforms the second-best method by approximately 1 dB, indicating that diffusion priors are especially effective in severe undersampling scenarios.
4. **Importance of Manifold Preservation**: State evolution analysis proves that DMP maintains the data manifold throughout the reconstruction process, which is key to its superiority over other diffusion-based reconstruction methods.

## Highlights & Insights

- **Theoretical Elegance**: Starting from the AMP message passing framework, the diffusion model is naturally embedded into the iterative process, backed by complete theoretical derivation.
- **Both Efficiency and Quality**: By replacing high-cost computations with deep unfolding and lightweight ResBlocks, the core bottleneck of slow reconstruction in diffusion models is addressed.
- **Novel Perspective on Manifold Constraints**: Proving that DMP variables consistently conform to the intermediate state distribution of diffusion through state evolution equations provides a new perspective on understanding diffusion-based reconstruction.

## Limitations & Future Work

- It depends on the quality of the pre-trained diffusion model; if the diffusion model prior does not match the target data domain, performance may degrade.
- Only validated on block-based CS, without extension to specific application scenarios such as MRI or SCI.
- Although replacing divergence approximation with ResBlocks is highly efficient, it loses the theoretical guarantee of Onsager correction, and the actual recovered manifold constraint may be weaker than suggested by the theoretical analysis.
- Future work can explore extending the method to other inverse problems (e.g., deblurring, super-resolution).

## Related Work & Insights

- **Combination of AMP and DUN**: This work continues the line of thought of classic works like AMP-Net and ISTA-Net+, but introduces the prior of a pre-trained generative model for the first time.
- **Diffusion Models for Inverse Problems**: Methods like DDNM and MPGD that directly modify the diffusion process lack manifold preservation, which this work elegantly addresses using the AMP framework.
- **Insights**: The paradigm of combining strong pre-trained priors with traditional optimization frameworks is highly worth promoting to other signal processing tasks.

## Rating

⭐⭐⭐⭐ — Theoretically sound and experimentally significant, combining diffusion models with deep unfolding networks is a natural and elegant design. The efficiency improvement of reaching SOTA in just 2 steps is highly impressive. However, the extensibility to more application scenarios remains to be further validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] DDIS: When Model Knowledge Meets Diffusion Model](../../ICML2025/image_generation/when_model_knowledge_meets_diffusion_model_diffusion-assisted_data-free_image_synthesis.md)
- [\[CVPR 2025\] Navigating Image Restoration with VAR's Distribution Alignment Prior](navigating_image_restoration_with_vars_distribution_alignment_prior.md)
- [\[ICCV 2025\] DIIP: Diffusion Image Prior](../../ICCV2025/image_generation/diffusion_image_prior.md)
- [\[CVPR 2025\] Generation of Maximal Snake Polyominoes Using a Deep Neural Network](generation_of_maximal_snake_polyominoes_using_a_deep_neural_network.md)
- [\[CVPR 2025\] VideoWorld: Exploring Knowledge Learning from Unlabeled Videos](videoworld_exploring_knowledge_learning_from_unlabeled_videos.md)

</div>

<!-- RELATED:END -->
