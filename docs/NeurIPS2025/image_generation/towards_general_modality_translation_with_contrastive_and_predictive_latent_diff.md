---
title: >-
  [Paper Note] Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge
description: >-
  [NeurIPS 2025][Image Generation][Diffusion Bridge Models] This paper proposes LDDBM (Latent Denoising Diffusion Bridge Model), which extends denoising diffusion bridge models into a shared latent space and incorporates c…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Diffusion Bridge Models"
  - "Modality Translation"
  - "Contrastive Learning"
  - "Latent Space"
  - "Transformer"
date: 2026-05-08
content_hash: dbb1d72abfb354da
---

# Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge

**Conference**: NeurIPS 2025
**arXiv**: [2510.20819](https://arxiv.org/abs/2510.20819)  
**Code**: [Project Page](https://sites.google.com/view/lddbm/home)  
**Area**: Diffusion Models / Cross-Modal Translation
**Keywords**: Diffusion Bridge Models, Modality Translation, Contrastive Learning, Latent Space, Transformer

## TL;DR

This paper proposes LDDBM (Latent Denoising Diffusion Bridge Model), which extends denoising diffusion bridge models into a shared latent space and incorporates contrastive alignment loss and predictive loss to achieve a general-purpose framework for arbitrary modality translation.

## Background & Motivation

Diffusion models have demonstrated remarkable performance in unimodal generation (e.g., images, audio), yet applying them to **Modality Translation (MT)**—converting information across different sensory modalities—remains an open challenge. Prior methods suffer from the following limitations:

**Shared Dimensionality Assumption**: Denoising diffusion bridge models (DDBM) require source and target distributions to share the same dimensional space $\mathbb{R}^d$, making them unable to handle heterogeneous modality translation such as 2D→3D.

**Gaussian Source Prior**: Standard diffusion models begin from a simple prior distribution, limiting flexibility for translation between arbitrary distributions.

**Modality-Specific Architectures**: Architectures such as U-Net are naturally suited to grid-structured data but perform poorly on abstract or unstructured modalities.

**Limitations of Latent Bridge Methods**: Existing latent-space diffusion bridges are designed primarily for computational efficiency and are not tailored for general MT tasks.

The authors observe that independently trained autoencoders lead to misaligned latent spaces (verified via t-SNE visualization), and that a naive bridge loss fails to preserve high-frequency details. These findings motivate the design of the contrastive and predictive losses.

## Method

### Overall Architecture

The LDDBM framework consists of three steps: (i) encoding source and target samples into a shared latent space using modality-specific encoders; (ii) performing a diffusion bridge in the latent space using a Transformer-based denoiser; and (iii) decoding the predicted latent variables back to the target modality using a modality decoder. At inference time, only the source encoder $E_y$, bridge model $B$, and target decoder $D_x$ are required.

### Key Designs

1. **Latent Variable Bridge Modeling**: Given target modality distribution $p(x)$ ($x \in \mathbb{R}^k$) and source modality distribution $p(y)$ ($y \in \mathbb{R}^s$, $k \neq s$), intermediate latent variables $z_0, z_T \in \mathbb{R}^d$ are introduced to factorize the conditional distribution as $p(x|y) = p(z_T|y) \cdot p(z_0|z_T) \cdot q(x|z_0)$. The bridge $p(z_0|z_T)$ is modeled via DDBM, eliminating the requirement for dimensional matching.

2. **Predictive Loss**: Unlike conventional two-stage reconstruction losses, the predictive loss directly constrains the full encode–bridge–decode pipeline:
    $\mathcal{L}_{\text{pred}} = d(D_x \circ B \circ E_y(y), x)$
   where $B$ denotes the bridge model. This loss encourages semantic consistency across the entire pipeline, replacing independent autoencoder losses for source and target, thereby reducing computational overhead while providing unidirectional supervision.

3. **Contrastive Alignment Loss**: Exploiting the structure of paired data, $(z_0, z_T)$ is treated as a positive pair, while other in-batch samples serve as negative pairs:
    $\mathcal{L}_{\text{infoNCE}} = \log \frac{\phi(z_0, z_T)}{\phi(z_0, z_T) + \sum_{j=1}^{M} \phi(z_0, z_T^j)}$
   where $\phi(u,v) = \exp(u^T v / \tau |u| |v|)$ and temperature $\tau = 0.5$. This loss pulls semantically related samples closer while pushing apart unrelated ones, effectively aligning cross-modal latent spaces.

4. **Encoder–Decoder Transformer Architecture**: A Transformer encoder processes source modality tokens $z_T$ to produce a memory representation, which is then used to condition the denoising process through cross-attention layers in a Transformer decoder. Learnable $[\text{MASK}]$ tokens serve as output tokens to enhance expressiveness. Timestep embeddings modulate the outputs of self-attention and feed-forward layers.

### Loss & Training

The overall objective is:
$$\mathcal{L} = \mathcal{L}_{\text{bridge}} + \mathcal{L}_{\text{pred}} + \mathcal{L}_{\text{infoNCE}}$$

Training employs an **alternating iteration strategy**—alternating optimization between reconstruction and bridge alignment—inspired by adversarial training. This design is motivated by the conflict between the bridge's assumption of fixed marginal distributions and the continuously evolving latent space structure induced by trainable encoders. The alternating strategy achieves the best balance between stability and final performance.

## Key Experimental Results

### Main Results: Multi-View → 3D Shape Generation (ShapeNet)

| Method | 1-NNA ↓ | IoU ↑ |
|--------|---------|-------|
| Pix2Vox-A (task-specific) | - | 0.697 |
| EDM | 0.532±0.013 | 0.631±0.006 |
| 3D-EDM | 0.575±0.009 | 0.602±0.003 |
| DiT | 0.548±0.004 | 0.613±0.011 |
| SiT | 0.563±0.007 | 0.604±0.003 |
| **LDDBM** | **0.508±0.005** | **0.664±0.002** |

### Zero-Shot Super-Resolution (FFHQ→CelebA-HQ, 16×16→128×128)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|--------|--------|---------|
| EDM | 23.1±0.7 | 0.58±0.05 | 0.41±0.02 |
| DiWa (task-specific) | 23.3 | 0.65 | 0.39 |
| DiT | 22.2±1.1 | 0.52±0.07 | 0.49±0.01 |
| SiT | 21.5±0.4 | 0.57±0.02 | 0.51±0.03 |
| **LDDBM** | **25.6±0.4** | **0.68±0.03** | **0.32±0.01** |

### Ablation Study

**Architecture Ablation:**

| Component | ShapeNet IoU ↑ | ShapeNet 1-NNA ↓ | CelebA PSNR ↑ |
|-----------|----------------|-------------------|----------------|
| U-Net | 0.635 | 0.518 | 23.2 |
| DiT | 0.613 | 0.548 | 22.2 |
| + Encoder-Decoder | 0.651 | 0.518 | 23.4 |
| + Spatial Embedding | 0.658 | 0.522 | 22.9 |
| + [MASK] (full model) | **0.664** | **0.508** | **25.6** |

**Loss Ablation:**

| Configuration | ShapeNet 1-NNA ↓ | CelebA LPIPS ↓ | Notes |
|---------------|-------------------|-----------------|-------|
| $\mathcal{L}_{\text{rec}}$ | 0.625 | 0.62 | Baseline reconstruction |
| $\mathcal{L}_{\text{pred}}$ | 0.522 | 0.41 | Predictive loss yields significant gains |
| $\mathcal{L}_{\text{pred}} + \mathcal{L}_{\text{infoNCE}}$ | **0.508** | **0.32** | Full configuration is optimal |

### Key Findings

- LDDBM consistently achieves state-of-the-art results over all general baselines, and inference is more than 2× faster than DDBM on edges→bags translation.
- In cross-modal speech↔face translation, LDDBM also leads among general baselines (Face→Voice: 71.2% vs. SiT 65.7%).
- The predictive loss is more effective than independent autoencoder losses while also reducing computational overhead.

## Highlights & Insights

- **Shift from computational efficiency to task generality**: Prior latent-space bridges were primarily motivated by reduced computation; this work is the first to systematically explore their potential for general MT.
- **Dual constraint of contrastive and predictive losses**: The contrastive loss handles global alignment while the predictive loss ensures end-to-end fidelity; together they form a complementary and powerful training signal.
- **Architecture selection from a "translator" perspective**: Inspired by sequence-to-sequence translation in NLP, the encoder–decoder Transformer is chosen over decoder-only DiT, a decision strongly validated by ablation experiments.

## Limitations & Future Work

- Currently limited to paired modality translation; unpaired cross-modal translation remains unexplored.
- Encoders and decoders remain modality-specific components, requiring new designs for unseen modalities.
- Extension to sequential or high-dimensional data (video, volumetric representations) has not been explored.
- The full framework incurs relatively high training costs due to the alternating training strategy.

## Related Work & Insights

- DDBM serves as the foundational framework but is restricted to same-dimensional translation; the proposed latent-space extension naturally eliminates this constraint.
- CrossFlow/FlowTok employ Flow Matching for text–image translation, but are tied to task-specific designs.
- CLIP-style contrastive learning is effectively transferred to latent space alignment within the bridge model setting.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of extending DDBM to latent space with contrastive and predictive losses is clear and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers diverse tasks including 3D generation, super-resolution, scene generation, and audio–image translation, with detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with a complete and coherent motivation-to-design logical chain.
- **Value**: ⭐⭐⭐⭐ Provides a strong baseline for general MT, though practical applicability remains constrained by the paired data requirement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] System-Embedded Diffusion Bridge Models](system-embedded_diffusion_bridge_models.md)
- [\[NeurIPS 2025\] CORAL: Disentangling Latent Representations in Long-Tailed Diffusion](coral_disentangling_latent_representations_in_longtailed_dif.md)
- [\[NeurIPS 2025\] Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation](predictive_feature_caching_for_training-free_acceleration_of_molecular_geometry_.md)
- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](../../ICCV2025/image_generation/contrastive_flow_matching.md)
- [\[NeurIPS 2025\] ThermalGen: Style-Disentangled Flow-Based Generative Models for RGB-to-Thermal Image Translation](thermalgen_style-disentangled_flow-based_generative_models_for_rgb-to-thermal_im.md)

</div>

<!-- RELATED:END -->
