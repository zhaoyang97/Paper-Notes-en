---
title: >-
  [Paper Note] Depth-Synergized Mamba Meets Memory Experts for All-Day Image Reflection Separation
description: >-
  [AAAI 2026][Image Restoration][Image Reflection Separation] This paper proposes DMDNet, which employs a depth-aware scanning strategy (DAScan) to guide Mamba toward salient structures, incorporates a depth-synergized state space model (DS-SSM) to suppress ambiguous feature propagation, and introduces a memory expert compensation module (MECM) to leverage cross-image historical knowledge, achieving all-day (daytime + nighttime) image reflection separation.
tags:
  - "AAAI 2026"
  - "Image Restoration"
  - "Image Reflection Separation"
  - "Mamba"
  - "Depth Awareness"
  - "Memory Experts"
  - "Nighttime Imaging"
date: 2026-05-08
content_hash: 814c80a84d1d7a60
---

# Depth-Synergized Mamba Meets Memory Experts for All-Day Image Reflection Separation

**Conference**: AAAI 2026
**arXiv**: [2601.00322](https://arxiv.org/abs/2601.00322)  
**Code**: [github.com/fashyon/DMDNet](https://github.com/fashyon/DMDNet)  
**Area**: Others
**Keywords**: Image Reflection Separation, Mamba, Depth Awareness, Memory Experts, Nighttime Imaging

## TL;DR

This paper proposes DMDNet, which employs a depth-aware scanning strategy (DAScan) to guide Mamba toward salient structures, incorporates a depth-synergized state space model (DS-SSM) to suppress ambiguous feature propagation, and introduces a memory expert compensation module (MECM) to leverage cross-image historical knowledge, achieving all-day (daytime + nighttime) image reflection separation.

## Background & Motivation

Image reflection separation aims to decompose a mixed image $\bm{I}$ captured through glass into a transmission layer $\bm{T}$ (the scene behind the glass) and a reflection layer $\bm{R}$ (the reflected content on the glass surface).

**Core challenges of existing methods**:

**Limited single-image information**: Existing methods rely on limited information from a single image and tend to confuse the two layers when $T$ and $R$ exhibit similar contrast.

**Nighttime scenes are more challenging**:
   - Daytime: Abundant natural light enhances $T$ while suppressing $R$, yielding large contrast differences between the two layers.
   - Nighttime: Artificial light sources are randomly distributed; $T$ is darkened by insufficient global illumination, while $R$ produces glare and scattered highlights from strong light on the glass surface, causing $T$ and $R$ to have similar contrast.

**Dependence on additional hardware**: Multi-view, polarization filter, and infrared camera methods require specialized equipment.

**Need for manual intervention**: Language prompt and manual annotation methods are time-consuming and labor-intensive.

**Key insight**: Depth estimation can provide physical cues without additional hardware or manual intervention. When depth estimation is performed on a mixed image, **the depth map naturally highlights the coherent, sharp structure of $T$** while suppressing the blurry, transparent overlay of $R$ (as shown in Figure 1). This implies that high proximity values tend to carry salient structures.

**Two limitations of Mamba**:

**Disruption of structural continuity**: Fixed sequential scanning breaks coherent contours and textures in the transmission scene.

**Error propagation**: States from early-scanned regions in the SSM continuously influence subsequent regions, causing uncertainty from ambiguous features to spread across the entire image.

## Method

### Overall Architecture

DMDNet consists of three branches:
- **Encoding branch**: Extracts multi-scale features of $T$ and $R$ using MuGI blocks.
- **Depth Semantic Modulation Branch (DSBranch)**: Modulates encoded features using depth semantic features.
- **Decoding branch**: Performs $T$ and $R$ separation via DMBlock (DSMamba + MECM + EFFN).

The channel configuration is $C_1,...,C_5 = [48, 96, 192, 384, 768]$.

### Key Designs

1. **Depth-Synergized Decoupled Mamba (DSMamba)**:

   Comprises two submodules: Depth-Aware Scanning (DAScan) and Depth-Synergized State Space Model (DS-SSM).

   **DAScan tailors distinct scanning strategies for $T$ and $R$**:
    - **DA-RScan (for $T$)**: Adopts a "large-region-first + near-to-far" scheme. The proximity map is partitioned into a region scan map $\bm{M}_{reg}$, scanning from the largest to the smallest region (larger regions = more salient semantics), with pixels within each region scanned in near-to-far order. This preserves semantic continuity among pixels belonging to the same object.
    - **DA-GScan (for $R$)**: Adopts a "global near-to-far" scheme, scanning from the globally nearest to the farthest pixel. This matches the sparse and discontinuous distribution characteristics of $R$.
    - A reverse DAScan is subsequently performed to supplement structural cues.

   **DS-SSM modulates the sensitivity of state updates**:
    $\bm{h}_t = \bm{A}\bm{h}_{t-1} + \bm{B}_{aware}\bm{x}_t, \quad \bm{y}_t = \bm{C}_{aware}\bm{h}_t + \bm{D}\bm{x}_t$
    $\bm{B}_{aware} = (1-\bm{\gamma}) \cdot \bm{B} + \bm{\gamma} \cdot \bm{B}_{depth}$
    $\bm{C}_{aware} = (1-\bm{\gamma}) \cdot \bm{C} + \bm{\gamma} \cdot \bm{C}_{depth}$

   where $\bm{\gamma}$ is a weight map in $[0,1]$ derived from the proximity map. In structurally salient regions, a larger $\gamma$ amplifies the influence of depth-guided matrices, accelerating the integration of sharp structures; in ambiguous regions, the intervention is suppressed to prevent blurry feature propagation.

   Design motivation: DAScan ensures the model encounters salient structures early in the sequence modeling process, while DS-SSM synergistically regulates state evolution according to structural saliency—the two are functionally complementary.

2. **Memory Expert Compensation Module (MECM)**:

   Leverages cross-image historical knowledge to dynamically activate the most relevant experts for targeted compensation. It comprises an **expert gate** (selecting $N_{Exp}^K$ most relevant experts from $N_{Exp}$ candidates) and **memory experts**.

   Each memory expert contains two streams:
    - **GPStream (Global Pattern Interaction Stream)**:
      - Global pattern adjustment: The input is pooled into a global representation $\bm{I}_G$, similarity is computed with the memory bank $\bm{Mem} \in \mathbb{R}^{M \times C}$, and memories are aggregated via weighted summation to produce global compensation.
      - Memory evolution: For each sample, the most responsive memory entries are selected and update vectors are generated via weighted multiplication, updating the memory bank in a residual manner.
    - **SCStream (Spatial Context Refinement Stream)**:
      - The memory bank is reshaped into convolutional kernels and convolved with the input to generate a spatial similarity map.
      - For each spatial position, the Top-$k$ most relevant memory entries are selected.
      - Weighted summation: $\bm{F}_{comp}[b,hw,d] = \sum_{k=1}^{K} \bm{W}_A[b,k,hw] \cdot \bm{Mem}_K[b,k,hw,d]$

   Design motivation: Single-image information is limited; the memory bank accumulates cross-image feature pattern knowledge—e.g., texture detail and structural contour experts for $T$, and sparse highlight and blurry ghost experts for $R$.

3. **NightIRS Dataset**:

   A dataset of 1,000 nighttime reflection image triplets $(I, T, R)$ is constructed using glass and acrylic panels of varying thickness to introduce reflections, covering diverse nighttime illumination conditions (streetlights, neon signs, illuminated buildings, low-light natural environments), and considering varying camera-to-glass distances and viewing angles. A high-resolution version (NightIRS-HR) is also provided.

### Loss & Training

- Adam optimizer with an initial learning rate of $10^{-4}$, decayed in stages (epoch 30 → $5\times10^{-5}$; epoch 50 → $10^{-5}$).
- Training for 60 epochs, batch size = 1, cropped to $352\times352$ patches.
- Training data: 7,643 PASCAL VOC pairs + 200 Nature pairs + 89 Real pairs.
- MECM configuration: $N_{Exp}=4$ experts, with $N_{Exp}^K=2$ selected.
- Single NVIDIA RTX 4090 GPU.

## Key Experimental Results

### Main Results

Average performance on the transmission layer across public datasets (daytime scenes):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| BDN (ECCV'18) | 20.55 | 0.800 | 0.202 |
| ERRNet (CVPR'19) | 22.77 | 0.837 | 0.141 |
| DSRNet (ICCV'23) | 24.03 | 0.861 | 0.119 |
| DSIT (NIPS'24) | 26.11 | 0.883 | 0.105 |
| RDNet (CVPR'25) | 26.21 | 0.885 | 0.094 |
| **DMDNet (Ours)** | **26.27** | **0.889** | **0.093** |

Performance on the NightIRS dataset:

| Method | T-PSNR↑ | T-SSIM↑ | T-LPIPS↓ | R-PSNR↑ | Params (M) | FLOPs (G) |
|------|---------|---------|----------|---------|---------|----------|
| DSIT (NIPS'24) | 24.61 | 0.827 | 0.168 | 27.18 | 131.76 | 74.18 |
| RDNet (CVPR'25) | 25.08 | 0.831 | 0.149 | 27.93 | 266.43 | 66.10 |
| **DMDNet (Ours)** | **25.24** | **0.832** | **0.144** | **28.37** | **87.22** | **39.33** |

### Ablation Study

**Component ablation of DSMamba (transmission layer on public datasets)**:

| T Scan | R Scan | SSM | SPE | PSNR↑ | SSIM↑ | LPIPS↓ |
|-------|-------|-----|-----|-------|-------|--------|
| DA-RScan | DA-GScan | DS-SSM | ✓ | **26.27** | **0.889** | **0.093** |
| DA-RScan | DA-RScan | DS-SSM | ✓ | 25.99 | 0.886 | 0.098 |
| DA-GScan | DA-GScan | DS-SSM | ✓ | 25.87 | 0.886 | 0.100 |
| DA-RScan | DA-GScan | DS-SSM | ✗ | 25.66 | 0.882 | 0.105 |
| DA-RScan | DA-GScan | Original | ✓ | 25.78 | 0.884 | 0.098 |
| Original | Original | DS-SSM | ✓ | 25.69 | 0.884 | 0.096 |

**Comparison of Mamba variants (public datasets)**:

| Method | T-PSNR↑ | T-SSIM↑ | R-PSNR↑ | Params (M) |
|------|---------|---------|---------|---------|
| MambaIR | 25.56 | 0.880 | 22.09 | 103.61 |
| VMambaIR | 25.89 | 0.884 | 22.06 | 83.76 |
| MambaIRv2 | 24.84 | 0.868 | 21.66 | 88.38 |
| **DSMamba (Ours)** | **26.27** | **0.889** | **22.31** | **87.22** |

### Key Findings

1. **DA-RScan for $T$ + DA-GScan for $R$ is the optimal pairing**: Each matches the respective characteristics of the transmission layer (coherent structures) and the reflection layer (sparse, discontinuous features).
2. **DS-SSM significantly outperforms the original SSM**: PSNR improvement of 0.49 dB (25.78 → 26.27).
3. **SPE spatial positional encoding contributes substantially**: PSNR improvement of 0.61 dB (25.66 → 26.27).
4. **DSMamba comprehensively outperforms MambaIR/VMambaIR/MambaIRv2**: In both $T$ and $R$ recovery quality.
5. **DMDNet's advantage is more pronounced in nighttime scenes**: Parameter count is only 1/3 of RDNet, with FLOPs at only 60%.
6. **Visualization validation**: $\bm{B}_{depth}$ and $\bm{C}_{depth}$ demonstrably amplify activations in salient structural regions and suppress them in ambiguous regions.

## Highlights & Insights

- **Depth estimation as a free physical cue**: Depth estimation models can "see through" reflection occlusions to extract underlying structures—this observation is highly elegant and constitutes the core insight of the paper.
- **Tailored scanning strategies for $T$ and $R$**: Rather than a one-size-fits-all approach, distinct scanning orders are designed according to the different characteristics of the two layers (coherent vs. sparse).
- **Synergistic design of DS-SSM and DAScan**: DAScan ensures that good structures are encountered first; DS-SSM ensures that good structures are amplified while poor ones are suppressed—forming a complete logical chain.
- **Memory mechanism compensates for single-image limitations**: Cross-image knowledge accumulation provides "experiential" compensation for single-image inference.
- **NightIRS dataset fills a gap**: The first dataset specifically designed for nighttime reflection separation.

## Limitations & Future Work

1. **Dependence on pretrained depth estimation models**: Errors in depth estimation propagate into the reflection separation process.
2. **Fixed memory bank size**: The choice of $M$ requires balancing storage and performance.
3. **Computational overhead**: Although lighter than RDNet, the model still has 87M parameters compared to some lightweight methods.
4. **Limited scale of the NightIRS dataset**: 1,000 triplets may be insufficient to cover all nighttime scenarios.
5. **Video scenarios not considered**: Exploiting temporal consistency could potentially further improve performance.

## Related Work & Insights

DMDNet cleverly integrates depth estimation, Mamba state space models, and memory-augmented MoE architectures. The depth-aware scanning strategy can be generalized to other sequence modeling tasks requiring structural awareness. The memory expert mechanism also offers inspiration for other single-image restoration tasks such as dehazing and deraining. The all-day design philosophy is worth emulating in the image enhancement community.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The synergistic design of depth-guided Mamba scanning and DS-SSM is highly novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (11 comparison methods, comprehensive ablation, new dataset)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rigorous formulations, well-articulated motivation)
- Value: ⭐⭐⭐⭐ (First work addressing nighttime reflection separation, with strong generalization potential)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation](../../CVPR2026/image_restoration/reflexsplit_single_image_reflection_separation_via_layer_fusion-separation.md)
- [\[CVPR 2026\] Reflection Separation from a Single Image via Joint Latent Diffusion](../../CVPR2026/image_restoration/reflection_separation_from_a_single_image_via_joint_latent_diffusion.md)
- [\[AAAI 2026\] SpatioTemporal Difference Network for Video Depth Super-Resolution](spatiotemporal_difference_network_for_video_depth_super-resolution.md)
- [\[ICLR 2026\] Content-Aware Mamba for Learned Image Compression](../../ICLR2026/image_restoration/content-aware_mamba_for_learned_image_compression.md)
- [\[CVPR 2026\] Multi-Scale Gradient-Guided Unrolling Architecture with Adaptive Mamba for Compressive Sensing](../../CVPR2026/image_restoration/multi-scale_gradient-guided_unrolling_architecture_with_adaptive_mamba_for_compr.md)

</div>

<!-- RELATED:END -->
