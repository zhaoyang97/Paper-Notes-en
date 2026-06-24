---
title: >-
  [Paper Note] SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation
description: >-
  [CVPR 2025][Medical Imaging][point cloud generation] This paper proposes SeaLion, a semantic part-aware latent point diffusion technique that jointly predicts noise and point-wise segmentation labels during the denoising process, and decodes point clouds conditioned on these segmentation labels. It generates 3D point clouds with high-quality inter-part coherence and precise segmentation labels. Additionally, a part-aware Chamfer distance (p-CD) evaluation metric is proposed…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "point cloud generation"
  - "diffusion models"
  - "semantic segmentation"
  - "part-aware"
  - "latent space"
date: 2026-05-08
content_hash: fed3c7e268914506
---

# SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation

**Conference**: CVPR 2025  
**arXiv**: [2505.17721](https://arxiv.org/abs/2505.17721)  
**Code**: [https://github.com/Dekai21/SeaLion](https://github.com/Dekai21/SeaLion)  
**Area**: Medical Images  
**Keywords**: point cloud generation, diffusion models, semantic segmentation, part-aware, latent space

## TL;DR

This paper proposes SeaLion, a semantic part-aware latent point diffusion technique that jointly predicts noise and point-wise segmentation labels during the denoising process, and decodes point clouds conditioned on these segmentation labels. It generates 3D point clouds with high-quality inter-part coherence and precise segmentation labels. Additionally, a part-aware Chamfer distance (p-CD) evaluation metric is proposed, achieving substantial improvements over DiffFacto on ShapeNet and IntrA datasets.

## Background & Motivation

**Background**: Significant progress has been made in 3D point cloud generation (e.g., Lion, PVD, DPM), but existing methods mainly focus on generating unlabeled point clouds. Point cloud generation with semantic segmentation labels remains under-explored: existing approaches (e.g., TreeGAN, EditVAE) can generate separable sub-parts, but these sub-parts **lack explicit semantic meanings**.

**Limitations of Prior Work**: DiffFacto is the only method capable of generating semantically labeled point clouds. However, it uses independent DDPMs to generate each part separately and then assembles them using pose prediction. This "generate parts independently, then assemble" strategy leads to **poor inter-part coherence**, as the assembled shapes may not conform to the real distribution. Existing evaluation metrics (1-NNA-P, SNAP) also fail to effectively measure inter-part coherence.

**Key Challenge**: Distributed generation (generating each part independently) inherently cannot guarantee inter-part coherence, while holistic generation (generating the entire point cloud at once) struggles to obtain precise point-wise semantic labels. Simultaneously generating high-quality point clouds with exact semantic labels while ensuring overall coherence is the core challenge.

**Goal**: Design a model that simultaneously generates high-quality point clouds and precise segmentation labels, while guaranteeing inter-part coherence. Concurrently, propose metrics that can properly evaluate this task.

**Key Insight**: Inspired by the finding that intermediate features of generative models can be used for semantic segmentation (i.e., intermediate representations of DDPM contain high-level semantic information), this work leverages intermediate features to simultaneously predict noise and segmentation labels during the diffusion denoising process, enabling semantically-aware generation.

**Core Idea**: Instead of generating each part separately, the latent points of all parts are diffused simultaneously in a unified latent space. By utilizing the shared downsampling path of a U-Net to learn a common representation for both noise prediction and segmentation prediction, and outputting noise and labels via two parallel upsampling paths, the point cloud coordinates are finally decoded conditioned on the predicted labels.

## Method

### Overall Architecture

SeaLion is based on the hierarchical latent space diffusion framework of Lion and is trained in two stages. The first stage trains a conditional VAE (incorporating a global encoder $\phi_z$, a point-level encoder $\phi_h$, and a conditional decoder $\xi_h$), which performs encoding and decoding conditioned on the segmentation labels $y$. The second stage trains two diffusion modules (global diffusion $\epsilon_z$ and point-level diffusion $\epsilon_h$), where $\epsilon_h$ jointly predicts noise and segmentation labels. During inference, a global latent variable $z_0$ is generated first, followed by the generation of latent points $h_0$ and segmentation labels $\hat{y}$. Finally, the point cloud is decoded conditioned on $\hat{y}$ and $z_0$.

### Key Designs

1. **Semantic Part-Aware Latent Point Diffusion**:

    - **Function**: Enables the generative model to acquire semantic part awareness during the diffusion process.
    - **Mechanism**: Two key innovations: (1) Introducing the segmentation encoding $y$ as a condition in the encoder and decoder of the VAE, training the model to reconstruct point clouds guided by $y$ to learn semantic-geometry correspondence; (2) The point-level diffusion model $\epsilon_h$ simultaneously outputs noise prediction $\hat{\epsilon}_t$ and segmentation prediction $\hat{y}_t$: $\hat{\epsilon}_t, \hat{y}_t \leftarrow \epsilon_h(h_t, t, z_0)$. During inference, the segmentation prediction is progressively smoothed via EMA (with a smoothing factor of 0.1) to yield the final label $\hat{y}$, and the conditional decoder $\xi_h$ decodes point clouds strictly aligned with $\hat{y}$.
    - **Design Motivation**: Compared to the traditional "generate unlabeled point clouds first, then assign pseudo-labels with a pre-trained segmentation model" pipeline, joint generation is more concise and robust—it does not rely on external models, and conditional decoding guarantees alignment between coordinates and labels.

2. **Dual-Path Upsampling U-Net Architecture**:

    - **Function**: Extract task-specific features for noise prediction and segmentation prediction from a shared representation.
    - **Mechanism**: The point-level diffusion model $\epsilon_h$ utilizes a modified PVCNN (Point-Voxel CNN) U-Net, which consists of a shared downsampling path to learn a common representation $r_c$, and two parallel upsampling paths to output the noise prediction feature $r_\epsilon$ and the segmentation prediction feature $r_y$, respectively. Each upsampling layer concatenates the task-specific features from the previous layer with the common representation from the corresponding downsampling layer before processing.
    - **Design Motivation**: Although noise prediction and segmentation prediction share underlying geometric information, they are essentially two distinct tasks. Shared downsampling ensures parameter efficiency, while separate upsampling paths guarantee task specificity.

3. **Part-Aware Chamfer Distance (p-CD)**:

    - **Function**: Evaluate the generation quality of point clouds with segmentation labels, especially regarding inter-part coherence.
    - **Mechanism**: For two point clouds $x^1$ and $x^2$ (each containing $|P|$ parts), the Chamfer Distance is calculated within each part and summed: $\text{p-CD}(x^1, x^2) = \sum_{p \in P} \text{CD}(x^1_p, x^2_p)$. If the part composition of the two point clouds differs, p-CD is defined as infinity. Based on p-CD, metrics such as 1-NNA (p-CD), COV (p-CD), and MMD (p-CD) can be computed.
    - **Design Motivation**: Existing 1-NNA-P (averaged by part) fails to capture inter-part incoherence—in extreme cases, randomly combining parts from different shapes can still obtain a high 1-NNA-P score. p-CD avoids this loophole by performing a holistic calculation, effectively detecting shapes with "anomaly assembly".

### Loss & Training

- **VAE Stage**: Maximizes the ELBO, which contains a reconstruction term and a KL regularization term, with $\lambda_z$ and $\lambda_h$ as balancing hyperparameters.
- **Diffusion Stage**: Global diffusion loss is defined as $\mathcal{L}(\epsilon_z) = E[\|\epsilon_z(z_t,t) - \epsilon\|^2_2]$; point-level diffusion loss is defined as $\mathcal{L}(\epsilon_h) = E[\|\hat{\epsilon}_t - \epsilon\|^2_2 + \lambda_{seg} H(y, \hat{y}_t)]$, where $H$ denotes the cross-entropy loss, and $\lambda_{seg}$ balances the two tasks.
- The VAE is trained for 8k epochs, and diffusion is trained for 24k epochs using the Adam optimizer with a learning rate of 1e-3.
- The VAE has 22.3M parameters, and the diffusion model has 98.1M parameters, trained on a single RTX 3090 GPU.

## Key Experimental Results

### Main Results

| Method | Airplane 1-NNA↓ | Car 1-NNA↓ | Chair 1-NNA↓ | Lamp 1-NNA↓ |
|------|-----------------|------------|-------------|------------|
| Lion + SPoTr | 67.13 | 77.36 | 65.27 | - |
| DiffFacto | 81.67 | 90.51 | 77.34 | 67.13 |
| **Ours** | **65.40** | **73.10** | **63.14** | **61.71** |

SeaLion outperforms DiffFacto by an average of 13.33% across four categories (1-NNA p-CD).

### Ablation Study

| Configuration | Description | Effect |
|------|------|------|
| Lion (Unlabeled) + PointNet++ | Two-step method: generate unlabeled point cloud first, then segment | 1-NNA 68.48 (airplane) |
| Lion + SPoTr (SOTA segmentation) | Two-step method using a better segmentation model | 1-NNA 67.13 |
| DiffFacto | Separate generation per part followed by assembly | 1-NNA 81.67 (inter-part incoherence) |
| SeaLion (full) | Joint generation + conditional decoding | 1-NNA **65.40** |
| SeaLion (Semi-supervised, 10% labels) | Only 10% of data is labeled | Performance remains acceptable |

### Key Findings

- DiffFacto exhibits a drastic performance drop from 1-NNA-P to 1-NNA (p-CD) (e.g., airplane: ~50 $\rightarrow$ 81.67), validating its poor inter-part coherence.
- SeaLion supports semi-supervised training (with only a small amount of annotated data), leveraging additional unlabeled data to boost performance and reduce annotation costs.
- The mIoU of segmentation predictions progressively improves as the diffusion timestep decreases from $T$ to 0, which is consistent with the noise removal process.
- Using the generated data to augment training sets for segmentation models effectively boosts downstream segmentation performance (especially under label scarcity such as the medical dataset IntrA).
- SeaLion can function as a part editing tool: by freezing the latent points of parts to be preserved and only performing diffusion-denoising on the remaining parts, localized shape variations can be achieved.

## Highlights & Insights

- **Joint Generation Paradigm for Labels and Point Clouds**: Instead of post-hoc segmentation, labels are generated synchronously during the generation process, and the point cloud is decoded conditioned on these labels to guarantee alignment. This paradigm can be extended to any scenario requiring the generation of labeled data (e.g., 2D image generation + semantic segmentation).
- **Elegant Design of the p-CD Metric**: With a simple modification (computing CD per part and summing them), p-CD overcomes the systemic flaw of existing metrics that fail to capture inter-part coherence. The paper intuitively demonstrates this flaw using a counterexample (the random assembly experiment in Figure 4).
- **Semi-Supervised Capability**: In fields where annotations are expensive, such as medical imaging, the ability to exploit unlabeled data holds substantial practical value.

## Limitations & Future Work

- The model parameter size is relatively large (VAE 22.3M + Diffusion 98.1M), and its training and inference efficiency require further optimization.
- A separate model must be trained for each category, lacking cross-category generalization.
- Currently, generating point clouds with a fixed number of parts is supported, but handling variable numbers of parts (such as chairs with different structures) remains to be explored.
- The p-CD metric assumes that the two compared point clouds have the same part composition; otherwise, the distance is defined as infinity. This definition lacks flexibility when dealing with novel part combinations.
- Conditional decoding heavily relies on the accuracy of the segmentation label predictions; incorrect label predictions may lead to geometric distortions.

## Related Work & Insights

- **vs DiffFacto**: DiffFacto generates each part separately and then assembles them, causing poor inter-part coherence. SeaLion diffuses the latent points of all parts simultaneously, naturally preserving holistic coherence with a 13.33% average improvement in 1-NNA (p-CD).
- **vs Lion**: Lion is a state-of-the-art unlabeled point cloud generative model. SeaLion integrates segmentation conditioning and joint prediction into Lion's framework, bringing improvements in generation quality as well (even without considering labels, achieving 65.40 vs 67.13 on airplane).
- **vs Two-Step Pipeline (Lion + SPoTr)**: Generate-then-segment methods are limited by the generalization capability of the segmentation model and cannot guarantee coordinate-label alignment. SeaLion's end-to-end approach is more concise and robust.

## Rating

- Novelty: ⭐⭐⭐⭐ The joint label generation workflow is innovative, and the p-CD metric fills a gap in validation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive demonstration on ShapeNet (6 classes) + IntrA (medical), along with semi-supervised, data augmentation, and editing application experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, and the counterexample illustration in Figure 4 is highly persuasive.
- Value: ⭐⭐⭐⭐ Labeled point cloud generation + evaluation metric = a complete problem definition and solution with direct downstream application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis](latent_drifting_in_diffusion_models_for_counterfactual_medical_image_synthesis.md)
- [\[CVPR 2025\] DiN: Diffusion Model for Robust Medical VQA with Semantic Noisy Labels](din_diffusion_model_for_robust_medical_vqa_with_semantic_noisy_labels.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](../../NeurIPS2025/medical_imaging/semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [\[CVPR 2026\] Sketch2CT: Multimodal Diffusion for Structure-Aware 3D Medical Volume Generation](../../CVPR2026/medical_imaging/sketch2ct_multimodal_diffusion_for_structure-aware_3d_medical_volume_generation.md)
- [\[CVPR 2025\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)

</div>

<!-- RELATED:END -->
