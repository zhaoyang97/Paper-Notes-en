---
title: >-
  [Paper Note] MambaIRv2: Attentive State Space Restoration
description: >-
  [CVPR 2025][Image Restoration][Mamba] This work proposes MambaIRv2, which injects learnable prompts into the output matrix $\mathbf{C}$ of Mamba via the Attentive State-space Equation (ASE) to enable attention-like non-causal global querying. It also introduces Semantic Guided Neighboring (SGN) to rearrange sequences according to semantic labels, alleviating long-range decay. Requiring only a single-direction scan, it outperforms multi-directional methods…
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Mamba"
  - "state space model"
  - "attentive state-space equation"
  - "semantic guided neighboring"
  - "non-causal modeling"
  - "super-resolution"
  - "denoising"
date: 2026-05-08
content_hash: e41d96090e61f735
---

# MambaIRv2: Attentive State Space Restoration

**Conference**: CVPR 2025  
**arXiv**: [2411.15269](https://arxiv.org/abs/2411.15269)  
**Code**: [GitHub](https://github.com/csguoh/MambaIR)  
**Area**: Image Restoration  
**Keywords**: Mamba, state space model, attentive state-space equation, semantic guided neighboring, non-causal modeling, super-resolution, denoising

## TL;DR

This work proposes MambaIRv2, which injects learnable prompts into the output matrix $\mathbf{C}$ of Mamba via the Attentive State-space Equation (ASE) to enable attention-like non-causal global querying. It also introduces Semantic Guided Neighboring (SGN) to rearrange sequences according to semantic labels, alleviating long-range decay. Requiring only a single-direction scan, it outperforms multi-directional methods, surpassing SRFormer by 0.35dB on lightweight SR with 9.3% fewer parameters.

## Background & Motivation

**Background**: Mamba has been introduced to image restoration (e.g., MambaIR) due to its linear complexity and global receptive field, achieving promising results. However, its causal modeling nature acts as an inherent bottleneck for non-causal tasks like image restoration.

**Limitations of Prior Work**:
1. **Causal Limitation**: The $i$-th token in Mamba can only attend to the previous $i-1$ tokens, preventing the utilization of subsequent pixels in the image.
2. **Redundancy in Multi-directional Scanning**: To compensate for causal deficiencies, existing methods (e.g., MambaIR) employ 4-directional scanning. However, experiments show that the cosine similarity between sequences of different directions is $> 0.7$, leading to significant redundancy and increased computational overhead.
3. **Long-range Decay**: The control matrix $\bar{\mathbf{A}}$ is statistically less than 1, causing $\bar{\mathbf{A}}^k$ to decay exponentially with distance $k$, resulting in extremely weak interaction between long-distance pixels.

**Key Challenge**: The fundamental mismatch between the causal nature of Mamba and the non-causal requirements of image restoration.

**Key Insight**: Starting from the mathematical connection between attention and state-space models, the authors discover that the output matrix $\mathbf{C}$ corresponds to the Query in attention. Consequently, injecting global semantic prompts into $\mathbf{C}$ enables non-causal querying.

## Method

### Overall Architecture

Input low-quality image $\rightarrow$ 3x3 conv for shallow feature extraction $\rightarrow$ multiple Attentive State Space Groups (ASSG), each containing multiple ASSBs $\rightarrow$ task-specific reconstruction (pixel-shuffle for super-resolution / conv for denoising). Each ASSB adopts local-to-global progressive modeling: window MHSA (local) + ASSM (global).

### Key Designs

**1. Bridging Analysis between Attention and State Space Models**
- **Function**: Unifies causal linear attention and state-space equations into a general form for comparison.
- **Key Findings**:
    - Hidden state $h_i \sim \mathbf{S}_i$ (cumulative KV in attention)
    - Input matrix $\mathbf{B} \sim \mathbf{K}^\top$ (similar to Key)
    - Output matrix $\mathbf{C} \sim \mathbf{Q}$ (similar to Query)
    - Control matrix $\bar{\mathbf{A}} \sim \mathbf{I}$ (attention is identity, while SSM has decay)
- **Significance**: Since $\mathbf{C}$ plays the role of Query, it can be utilized to "query" unscanned pixel information.

**2. Attentive State-space Equation (ASE)**
- **Function**: Adds a learnable prompt $\mathbf{P}$ to the output matrix $\mathbf{C}$ of the original state-space equation.
- **Mechanism**:
    - Construct a prompt pool $\mathcal{P} \in \mathbb{R}^{T \times d}$ using low-rank decomposition $\mathcal{P} = \mathbf{M}\mathbf{N}$ ($\mathbf{N}$ is shared across blocks, $\mathbf{M}$ is block-specific).
    - Routing strategy: Linear projection + LogSoftmax to predict probabilities $\rightarrow$ Gumbel-Softmax for differentiable selection $\rightarrow$ obtain one-hot routing matrix $\mathbf{R}$ $\rightarrow$ $\mathbf{P} = \mathbf{R}\mathcal{P}$.
    - Modified state-space equation: $y_i = (\mathbf{C} + \mathbf{P})h_i + \mathbf{D}x_i$.
- **Design Motivation**: The prompt represents a set of semantically similar pixels across the entire image. After injection, $\mathbf{C}$ can "see" unscanned pixels. This allows global information retrieval with only a single-direction scan, eliminating the redundancy and overhead of multi-directional scanning.

**3. Semantic Guided Neighboring (SGN)**
- **Function**: Rearranges the image according to semantic labels before feeding it into ASE, making semantically similar pixels spatially adjacent in the 1D sequence.
- **Mechanism**:
    - Reuse the routing matrix $\mathbf{R}$ from ASE (which already assigns semantic labels to each pixel).
    - SGN-unfold: Aggregates pixels of the same prompt category into groups, and concatenates each group in the order of category values to form a semantic neighborhood sequence.
    - After processing with ASE, SGN-fold performs the inverse transform to restore spatial arrangement.
- **Design Motivation**: Alleviates Mamba's long-range decay—pixels that are originally far in space but share similar semantics become sequence neighbors after rearrangement, meaning $\bar{\mathbf{A}}^k$ no longer needs to span long distances.

### Loss & Training

- Super-resolution: $L_1$ loss
- Denoising/JPEG CAR: Charbonnier loss
- Initial learning rate $2 \times 10^{-4}$ with milestone decay
- Adam optimizer with $\beta_1=0.9, \beta_2=0.999$
- Training patch size: SR 64×64, denoising 128×128; batch size SR=32, denoising=8
- Use $2\times$ pre-trained weights to initialize $3\times$/$4\times$ models, with learning rate and training iterations halved
- Three variants: MambaIRv2-S/B/L (Small/Base/Large)

## Key Experimental Results

### Main Results — Lightweight Super-Resolution ($\times$2)

| Method | #Param | Urban100 PSNR | Manga109 PSNR |
|---|---|---|---|
| SwinIR-light | 910K | 32.76 | 39.12 |
| MambaIR-light | 905K | 32.85 | 39.20 |
| SRFormer-light | 853K | 32.91 | 39.28 |
| **MambaIRv2-light** | **774K** | **33.26** | **39.35** |

Outperforms SRFormer by 0.35dB on Urban100 with 9.3% fewer parameters.

### Main Results — Lightweight Super-Resolution ($\times$4)

| Method | #Param | Urban100 PSNR | Manga109 PSNR |
|---|---|---|---|
| SwinIR-light | 930K | 26.47 | 30.92 |
| SRFormer-light | 873K | 26.67 | 31.17 |
| MambaIR-light | 925K | 26.75 | 31.26 |
| **MambaIRv2-light** | **794K** | **26.92** | **31.37** |

### Ablation Study

**Component Effectiveness (Lightweight 2$\times$ SR, 250K iter)**:

| MHSA | ASE | SGN | Urban100 PSNR | Manga109 PSNR |
|---|---|---|---|---|
| ✔ | | | 32.89 | 39.11 |
| ✔ | ✔ | | 32.94 | 39.20 |
| ✔ | ✔ | ✔ | **32.97** | **39.24** |

**Ablation on Prompt Injection Location**:

| Location | Urban100 PSNR | Manga109 PSNR |
|---|---|---|
| $\mathbf{B}$ (Input matrix) | 32.96 | 39.23 |
| $\Delta$ | 32.93 | 39.19 |
| $y$ (Output) | 32.94 | 39.21 |
| **$\mathbf{C}$ (Output matrix)** | **32.97** | **39.24** |

### Key Findings

1. **Causal modeling is the core bottleneck of Mamba in image restoration**: The cosine similarity between 4-direction scans is $> 0.7$, indicating significant redundancy.
2. **$\mathbf{C}$ is the optimal location for prompt injection**: Consistent with the theoretical analysis, $\mathbf{C}$ corresponds to the Query, making it the most effective location for semantic prompt injection.
3. **Single-direction scanning can outperform multi-directional scanning**: The non-causal capability of ASE eliminates the need for 4-direction scans, delivering both higher efficiency and superior performance.
4. **SGN semantic rearrangement effectively alleviates long-range decay**: This is achieved with virtually zero parameter overhead (reusing the routing matrix).
5. **Even the strong Transformer baseline HAT is outperformed**: It surpasses HAT by 0.29dB on classic SR $\times$2 on Manga109.

## Highlights & Insights

- The design is derived from the mathematical equivalence between attention and state-space models, providing a solid theoretical foundation.
- The prompt pool + Gumbel-Softmax routing design is elegant, and SGN reuses routing information with zero overhead.
- Eliminating redundancy via single-direction scanning offers a highly appealing efficiency advantage.
- "Making Mamba non-causal like Attention" is a clear narrative and a valuable research direction.

## Limitations & Future Work

- The prompt pool size $T$ and intrinsic rank $r$ require hyperparameter tuning.
- The semantic grouping of SGN is based on simple prompt routing, which may lack fine-grained grouping granularity.
- The paper only evaluates classic SR, denoising, and JPEG CAR tasks, leaving tasks like deblurring and dehazing unexplored.
- The temperature parameter of the Gumbel-Softmax might affect training stability.
- A direct comparison with concurrent Mamba-based restoration methods (e.g., MaIR) is lacking.

## Related Work & Insights

- MambaIR pioneered the use of Mamba for image restoration but left the causality issue unresolved; this work serves as an official "v2" upgrade.
- ATD (Adaptive Token Dictionary) similarly introduces external knowledge into attention, though in a different manner.
- The mathematical connection between linear attention and SSMs (e.g., in Mamba-2) provides the theoretical foundation for this study.
- Insight: Architectural design should focus on "how to modify modules to solve specific problems" rather than simply "which modules to use"—injecting non-causal capabilities into causal models is a highly generalizable concept.

## Rating

⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Efficient Visual State Space Model for Image Deblurring](efficient_visual_state_space_model_for_image_deblurring.md)
- [\[ECCV 2024\] MambaIR: A Simple Baseline for Image Restoration with State-Space Model](../../ECCV2024/image_restoration/mambair_a_simple_baseline_for_image_restoration_with_state-space_model.md)
- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](../../ICCV2025/image_restoration/eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)
- [\[CVPR 2025\] MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration](mair_a_locality-_and_continuity-preserving_mamba_for_image_restoration.md)
- [\[CVPR 2025\] QMambaBSR: Burst Image Super-Resolution with Query State Space Model](qmambabsr_burst_image_super-resolution_with_query_state_space_model.md)

</div>

<!-- RELATED:END -->
