---
title: >-
  [Paper Note] ZigMa: A DiT-style Zigzag Mamba Diffusion Model
description: >-
  [ECCV 2024][Image Generation] ZigMa proposes a DiT-style Zigzag Mamba diffusion model. By employing a heterogeneous layer-wise zigzag scanning scheme, it maintains spatial continuity, achieving superior generation quality compared to Mamba baselines with zero parameter or memory overhead, while retaining the linear complexity advantage over Transformers.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: f6bd90c520491247
---

# ZigMa: A DiT-style Zigzag Mamba Diffusion Model

**Conference**: ECCV 2024  
**arXiv**: [2403.13802](https://arxiv.org/abs/2403.13802)  
**Area**: Image Generation

## TL;DR

ZigMa proposes a DiT-style Zigzag Mamba diffusion model. By employing a heterogeneous layer-wise zigzag scanning scheme, it maintains spatial continuity, achieving superior generation quality compared to Mamba baselines with zero parameter or memory overhead, while retaining the linear complexity advantage over Transformers.

## Background & Motivation

Diffusion models have made significant progress in various vision tasks, but they face two core bottlenecks:

**Quadratic Complexity of Transformers**: Transformer-based diffusion backbones such as DiT suffer from an $O(M^2)$ complexity bottleneck due to the self-attention mechanism. Despite optimizations like Flash Attention, they remain limited when handling long token sequences.

**Spatial Continuity Issue of Mamba**: As a State Space Model (SSM) with linear complexity, Mamba excels in 1D sequence modeling. However, existing vision Mamba designs directly flatten 2D tokens into 1D sequences in a row/column-major order, which ignores the spatial continuity of adjacent patches in images.

**Parameter Overhead of Multi-Directional Scanning**: Approaches like VisionMamba incorporate multi-directional scanning within a single Mamba block to compensate for the lack of spatial awareness, but this introduces additional parameters and GPU memory overhead.

The core insight of ZigMa is that **the complexity of multi-directional scanning can be amortized across different layers of the network**, where each layer employs a distinct zigzag scanning scheme to ensure spatial continuity at zero extra parameter cost.

## Method

### Overall Architecture

ZigMa adopts a DiT-style architecture with Adaptive LayerNorm (adaLN), stacking $L$ layers of Zigzag Mamba blocks. Each layer consists of:
- A Mamba scanning module (for long-sequence modeling)
- A cross-attention module (for multimodal reasoning, such as text conditioning)
- AdaLN modulation (for injecting timesteps and conditions)

The training framework is based on Stochastic Interpolant, which unifies diffusion models, Flow Matching, and Normalizing Flows.

### Key Designs

**Zigzag Scanning Scheme**:
- Traditional sweep scan flattens 2D tokens row by row, resulting in spatial jumps from the end of one row to the beginning of the next, which destroys the continuity of adjacent patches.
- Zigzag scanning alternates the scanning direction in each row (similar to a serpentine path), ensuring that tokens adjacent in the 1D sequence are also spatially adjacent in the 2D space.
- Eight distinct zigzag space-filling schemes $\mathbf{S}_j$ ($j \in [0,7]$) are designed, covering different start positions and paths in both horizontal and vertical directions.

**Heterogeneous Layer-wise Scanning**:
- Each layer utilizes a different scanning scheme: $\Omega_i = \mathbf{S}_{\{i \% 8\}}$
- The token order is rearranged before scanning and restored to the original order afterward:
$$z_{\Omega_i} = \text{arrange}(z_i, \Omega_i), \quad \bar{z}_{\Omega_i} = \text{scan}(z_{\Omega_i}), \quad z_{i+1} = \text{arrange}(\bar{z}_{\Omega_i}, \bar{\Omega}_i)$$
- Key advantage: Unlike using $k$ directions within the same block (which requires $k$ times the parameters), the heterogeneous layer-wise scheme scatters the scanning diversity across different layers, resulting in **zero extra parameter and memory overhead**.

**Cross-Attention for Text Conditioning**:
- A cross-attention block with skip connections is appended to the Mamba block.
- Conditioning signals (timesteps + text prompts) modulate both Mamba scanning and cross-attention via MLPs.

**Factorized Scanning for 3D Video**:
- Decomposes 3D Zigzag into a 2D spatial Zigzag + 1D temporal scan.
- Employs an "sst" scheme (two spatial scans and one temporal scan), assuming redundancy in the temporal dimension.

### Loss & Training

Velocity field estimation loss based on Stochastic Interpolant:

$$\mathcal{L}_v(\theta) = \int_0^T \mathbb{E}[\|v_\theta(x_t, t) - \dot{\alpha}_t x_* - \dot{\sigma}_t \varepsilon\|^2] dt$$

A linear path is adopted: $\alpha_t = 1-t$, $\sigma_t = t$.

## Key Experimental Results

### Main Results

**Table 1: FacesHQ 1024×1024 high-resolution generation (4096 tokens)**

| Method | FID↓ | FDD↓ |
|------|------|------|
| VisionMamba | 51.1 | 66.3 |
| **ZigMa** | **37.8**| **50.5** |
| ZigMa (bs×2) | **26.6** | **31.2** |

**Table 2: MS-COCO 256×256 text-conditioned generation**

| Method | FID↓ |
|------|------|
| Sweep | 195.1 |
| Zigzag-1 | 73.1 |
| VisionMamba | 60.2 |
| **Zigzag-8** | **41.8** |

**Table 3: UCF101 video generation**

| Method | Frame-FID↓ | FVD↓ |
|------|-----------|------|
| Bidirection | 256.1 | 320.2 |
| 3D Zigzag | 238.1 | 282.3 |
| **Factorized ZigMa** | **216.1** | **210.2** |
| Bidirection (bs×4) | 146.2 | 201.1 |
| **ZigMa (bs×4)** | **121.2** | **140.1** |

### Ablation Study

**Table 4: Ablation on the number of scanning schemes (MultiModal-CelebA)**

| Scheme | FID↓ (256) | KID↓ (256) | FID↓ (512) | KID↓ (512) |
|------|-----------|-----------|-----------|-----------|
| Sweep | 158.1 | 0.169 | 162.3 | 0.203 |
| Zigzag-1 | 65.7 | 0.051 | 121.0 | 0.113 |
| Zigzag-2 | 54.7 | 0.041 | 96.0 | 0.079 |
| **Zigzag-8** | **45.5** | **0.011** | **34.9** | **0.023** |

**Table 5: Ablation on positional encoding (CelebA 256)**

| Method | Without PE | Cosine PE | Learnable PE |
|------|------|-----------|-------------|
| VisionMamba | 21.33 | 18.47 | 16.38 |
| **ZigMa** | **14.27** | **14.04** | **13.32** |

**Table 6: Efficiency comparison with Transformers (CelebA 256)**

| Method | FID↓ | Memory (GB)↓ | FLOPs (G)↓ |
|------|------|-------------|-----------|
| U-ViT | 14.50 | 35.10 | 12.5 |
| DiT | 14.64 | 29.20 | 5.5 |
| **ZigMa** | **14.27** | **17.80** | **5.2** |

### Key Findings

1. Moving from Sweep to Zigzag-8, the FID drops from 158.1 to 45.5 (at 256 resolution), with an even more substantial gain at 512 resolution (162.3 to 34.9). This validates the significance of spatial continuity in long sequences.
2. Even without positional encoding, ZigMa (FID=14.27) outperforms VisionMamba with Cosine PE (18.47), demonstrating that the zigzag scanning itself encodes an implicit spatial inductive bias.
3. Compared to U-ViT, ZigMa reduces GPU memory usage by 49% (from 35.1 to 17.8 GB) while housing comparable generation quality.
4. In video generation, factorized 3D Zigzag significantly outperforms direct 3D Zigzag, highlighting that processing spatial and temporal information separately is more effective.

## Highlights & Insights

- **Minimalist Yet Efficient Design Philosophy**: Significant gains are achieved simply by altering the token traversal order without introducing any extra parameters, serving as a "free lunch."
- **Core Insight of Heterogeneous Layer-wise Scanning**: Dispersing scanning diversity across layers rather than inside a multi-directional single block cleverly circumvents the $k$-fold overhead of $k$-way Mamba.
- **Inductive Bias of Spatial Continuity**: It explicitly identifies and quantifies the importance of spatial continuity when scaling Mamba from 1D to 2D, offering fundamental guidance for future vision applications of SSMs.
- **Large-scale Validation of Stochastic Interpolant**: This work scales the framework to 1024×1024 resolution image and video generation for the first time.

## Limitations & Future Work

1. The experimental scale is constrained by GPU resources; thus, the model was not fully trained on FacesHQ 1024.
2. More complex space-filling curves beyond Zigzag-8 (such as Hilbert curves) yield poor performance, and the theoretical foundation for the optimal scanning scheme remains unclear.
3. There is still significant room for improvement in FID on complex datasets like MS-COCO.
4. Currently, only class and text conditioning are supported, leaving other conditional control modes unexplored.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [\[ECCV 2024\] ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](ziplora_any_subject_in_any_style_by_effectively_merging_loras.md)
- [\[ECCV 2024\] Memory-Efficient Fine-Tuning for Quantized Diffusion Model](memory-efficient_fine-tuning_for_quantized_diffusion_model.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)

</div>

<!-- RELATED:END -->
