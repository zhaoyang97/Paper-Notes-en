---
title: >-
  [Paper Note] Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks
description: >-
  [CVPR 2025][Image Restoration][Deep Unfolding Networks] This paper proposes VLU-Net, the first All-in-One Deep Unfolding Network (DUN) framework. It utilizes a fine-tuned CLIP model to automatically detect degradation types and guide the gradient descent module. Combined with a hierarchical feature unfolding structure, VLU-Net outperforms the state-of-the-art end-to-end method by 3.74dB on image dehazing.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Deep Unfolding Networks"
  - "Vision-Language Models"
  - "All-in-One Image Restoration"
  - "CLIP"
  - "Degradation-Awareness"
date: 2026-05-08
content_hash: 38d1f6fea6c4e78a
---

# Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks

**Conference**: CVPR 2025  
**arXiv**: [2503.16930](https://arxiv.org/abs/2503.16930)  
**Code**: [GitHub](https://github.com/xianggkl/VLU-Net)  
**Area**: Image Restoration  
**Keywords**: Deep Unfolding Networks, Vision-Language Models, All-in-One Image Restoration, CLIP, Degradation-Awareness

## TL;DR

This paper proposes VLU-Net, the first All-in-One Deep Unfolding Network (DUN) framework. It utilizes a fine-tuned CLIP model to automatically detect degradation types and guide the gradient descent module. Combined with a hierarchical feature unfolding structure, VLU-Net outperforms the state-of-the-art end-to-end method by 3.74dB on image dehazing.

## Background & Motivation

Image restoration aims to recover the original image from the degraded observation $\mathbf{y} = \Phi\mathbf{x} + \mathbf{n}$. Deep Unfolding Networks (DUNs) achieve a balance between interpretability and performance by unfolding iterative optimization algorithms (such as Proximal Gradient Descent, PGD) into deep networks. However, they suffer from the following limitations:

- **Degradation matrix must be manually specified**: The Gradient Descent Module (GDM) in existing DUNs requires manually selecting a specific degradation matrix $\Phi$ for each degradation type. Consequently, independent models must be trained separately for different restoration tasks.
- **Lack of All-in-One capability**: Although end-to-end All-in-One methods (e.g., PromptIR, InstructIR) can handle multiple degradations uniformly, they lack the interpretability of DUNs.
- **Information bottleneck**: Existing DUNs only pass 3-channel images between stages. High-dimensional features are repeatedly compressed and decompressed, which limits the utilization of multi-level information.
- **Homogeneous processing across stages**: Each GDM processes the same low-dimensional information, whereas different stages should handle degradations at different levels (e.g., different denoising strengths in different stages of PGD).

## Method

### Overall Architecture

VLU-Net consists of two processes: (1) CLIP fine-tuning, which uses degraded image-text pairs for contrastive learning to enhance CLIP's degradation recognition capability; and (2) the main Image Restoration (IR) pipeline, which is a $\mathbf{K}$-stage hierarchical DUN. Each stage comprises a degradation-guided GDM (D-GDM) and a proximal mapping module (PMM). The input is first projected into a feature space via a linear transformation, and the processed features are then projected back to the image space.

### Key Design 1: VLM-guided Degradation-aware Gradient Descent (D-GDM) — Automatic Selection of Degradation Transforms

**Function**: Utilize CLIP's degradation embeddings to automatically select the transform matrix suitable for the current degradation type, replacing manual specification.

**Mechanism**: Obtain the degradation vector $d_\mathbf{I} = \hat{E_I}(\mathbf{y})$ through a fine-tuned CLIP image encoder. Project it into a low-dimensional space and apply softmax to obtain a degradation retrieval vector. The corresponding degradation key is then extracted via a weighted sum from a learnable degradation key database $K_{(l)}$:

$$\text{key} = \text{sum}(\sigma(\text{Linear}(d_\mathbf{I})) * K_{(l)})$$

Then, multi-head deformable attention (MDTA) is used to perform interaction between the degradation key and the degradation input:

$$\tilde{\Phi}(\hat{\mathbf{x}}^{(k-1)}_{(l)}, d_\mathbf{I}) = \text{MDTA}(\hat{\mathbf{x}}^{(k-1)}_{(l)}, \text{key}, \hat{\mathbf{x}}^{(k-1)}_{(l)})$$

**Design Motivation**: The high-dimensional feature space of CLIP can effectively distinguish different degradation types (noise, blur, rain, haze, low-light). Fine-tuning strengthens its degradation identification capability. The learnable degradation key database allows the network to integrate information from other degradation modes for joint decision-making.

### Key Design 2: Hierarchical DUN Architecture — Feature-level Unfolding and Multi-level Information Propagation

**Function**: Eliminate the 3-channel information bottleneck between stages, and handle different levels of degradation information at different stages.

**Mechanism**: Use a linear transformation $\mathbf{W} \in \mathbb{R}^{3 \times C_{(1)}}$ to embed the degraded image into a high-dimensional feature space, performing unfolding optimization at the feature level (instead of the image level). The degraded input $\mathbf{y}$ generates degraded features $\hat{\mathbf{y}}_{(l)}$ at different levels through downsampling/upsampling, providing multi-level degradation information to the D-GDMs of different stages:

$$\hat{\mathbf{z}}^{(k)}_{(l)} = \hat{\mathbf{x}}^{(k-1)}_{(l)} - \rho \Phi^\mathbf{T}(\tilde{\Phi}(\hat{\mathbf{x}}^{(k-1)}_{(l)}, d_\mathbf{I}) - \hat{\mathbf{y}}_{(l)})$$

Finally, it is projected back to the image space via an approximate inverse transformation $\mathbf{W}^{-1}$.

**Design Motivation**: Repeated compression and decompression of 3 channels is the core bottleneck of DUNs. Feature-level unfolding allows the GDM to access high-dimensional degradation information and content features, while residual connections between stages preserve original degradation features. Stages at the same level share the degraded input, while different levels handle different granularities, achieving hierarchical information processing.

### Key Design 3: Efficient CLIP Fine-tuning Strategy — Enhanced Alignment of Degradation Features

**Function**: Enhance the degradation recognition ability of CLIP while preserving its general visual capacity.

**Mechanism**: Insert three-layer MLP adapters before the image and text encoders of CLIP. Perform contrastive learning using image-text pairs from $\mathbf{M}$ degradation datasets:

$$\mathcal{L}(\mathbf{Y}, \hat{\mathbf{T}}) = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{\tau \cos(\hat{E_I}(\mathbf{Y}_i), \hat{E_T}(\hat{\mathbf{T}}_i))}}{\sum_j e^{\tau \cos(\hat{E_I}(\mathbf{Y}_i), \hat{E_T}(\hat{\mathbf{T}}_j))}}$$

The fine-tuned CLIP is then frozen and used to provide degradation vectors for the IR process.

**Design Motivation**: Although the original CLIP already possesses some ability for degradation classification, fine-tuning improves the alignment of degradation features in the high-dimensional vector space, allowing it to better distinguish between different degradation types and levels.

### Loss & Training

The IR process is trained using the $L_1$ loss. VLU-Net consists of 8 non-shared stages distributed across 4 levels, with the number of Transformer blocks in the PMM configured as $\{4, 6, 6, 8\}$.

## Key Experimental Results

### Main Results: NHRBL Five-Task All-in-One (PSNR/dB)

| Method | Type | Dehazing SOTS | Deraining Rain100L | Denoising σ=25 | Deblurring GoPro | Low-light LOL | Average |
|------|------|----------|-------------|----------|------------|----------|------|
| DGUNet | one-by-one DUN | 24.78 | 36.62 | 31.10 | 27.25 | 21.87 | 28.32 |
| PromptIR | All-in-one | 26.54 | 36.37 | 31.47 | 28.71 | 22.68 | 29.15 |
| InstructIR | All-in-one | 27.10 | 36.84 | 31.40 | 29.40 | 23.00 | 29.55 |
| **VLU-Net** | **All-in-one DUN** | **30.84** | **38.54** | 31.43 | 27.46 | 22.29 | **30.11** |

### NHR Three-Task All-in-One

| Method | Dehazing | Deraining | Denoising σ=15 | Denoising σ=25 | Average |
|------|------|----------|----------|----------|------|
| PromptIR | 30.58 | 36.37 | 33.98 | 31.31 | 32.06 |
| InstructIR | 30.22 | 37.98 | 34.15 | 31.43 | 32.42 |
| **VLU-Net** | **31.07** | **38.93** | 34.00 | 31.47 | **32.69** |

### Key Findings

- VLU-Net outperforms InstructIR by **3.74dB** on dehazing and **1.70dB** on deraining.
- As the first All-in-One DUN, VLU-Net outperforms its counterpart DUN method, DGUNet, by an average of **1.79dB**.
- The performance advantage is most pronounced in dehazing and deraining tasks where degradation matrices differ significantly, validating the value of the degradation-aware GDM.
- Improvements in deblurring and low-light enhancement are less pronounced than in dehazing/deraining (deblurring is even lower than some end-to-end methods), likely due to the higher complexity of these degradations.
- Degradation classification accuracy improves significantly after fine-tuning CLIP, verifying the feasibility of using VLMs for degradation detection.

## Highlights & Insights

1. **Expanding the DUN framework to All-in-One for the first time**: The primary challenge of specifying the degradation matrix in DUNs is addressed via the VLM-guided degradation-aware GDM.
2. **CLIP degradation embedding as soft routing**: Instead of hard classification followed by branch selection, the model retrieves degradation keys via softmax weighting, enabling the joint handling of mixed degradations.
3. **Hierarchical feature unfolding breaks the information bottleneck of DUNs**: Performing unfolding at the feature level rather than the image level represents a significant improvement to the DUN framework.
4. **Interpretability advantage**: Compared to end-to-end methods, the DUN structure naturally possesses the interpretability of iterative optimization.

## Limitations & Future Work

- Performance on deblurring (27.46 vs 29.40) and low-light enhancement (22.29 vs 23.00) is inferior to InstructIR, potentially because these degradations do not conform to the simple $\Phi\mathbf{x} + \mathbf{n}$ formulation.
- Keeping the frozen CLIP model introduces an extra overhead of 88M parameters and 18GMACs, which is unfavorable for lightweight deployment.
- The 8-stage non-shared design results in a model size of 35M parameters, which is larger than some end-to-end methods.
- The size of the degradation key database and the scalability to other degradation types require further verification.
- Future work may explore lighter degradation detection modules to replace CLIP.

## Related Work & Insights

- **DGUNet**: The first DUN method to propose flexible manual selection of degradation matrices.
- **DA-CLIP**: Integrates degraded image-text pairs into CLIP and combines them with diffusion models for image restoration.
- **PromptIR / InstructIR**: End-to-end All-in-One restoration methods guided by prompts or natural language instructions.
- **Restormer**: A Transformer-based general-purpose image restoration method.

## Rating

⭐⭐⭐⭐ — Highly innovative. It organically combines VLMs and DUNs for the first time to address the All-in-One image restoration problem. The prominent improvements in dehazing and deraining tasks validate the utility of the degradation-aware GDM. The hierarchical unfolding design represents an important advancement for the DUN framework. However, relatively poor performance in deblurring/low-light tasks and the overhead of CLIP are limitations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Visual-Instructed Degradation Diffusion for All-in-One Image Restoration](visual-instructed_degradation_diffusion_for_all-in-one_image_restoration.md)
- [\[CVPR 2025\] Degradation-Aware Feature Perturbation for All-in-One Image Restoration](degradation-aware_feature_perturbation_for_all-in-one_image_restoration.md)
- [\[CVPR 2025\] One-Step Event-Driven High-Speed Autofocus](one-step_event-driven_high-speed_autofocus.md)
- [\[ICCV 2025\] AFUNet: Cross-Iterative Alignment-Fusion Synergy for HDR Reconstruction via Deep Unfolding Paradigm](../../ICCV2025/image_restoration/afunet_crossiterative_alignmentfusion_synergy_for_hdr_recons.md)
- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](../../ICCV2025/image_restoration/eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)

</div>

<!-- RELATED:END -->
