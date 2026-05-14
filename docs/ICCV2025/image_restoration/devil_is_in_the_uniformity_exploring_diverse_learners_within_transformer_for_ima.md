---
title: >-
  [Paper Note] Devil is in the Uniformity: Exploring Diverse Learners within Transformer for Image Restoration
description: >-
  [ICCV 2025][Image Restoration][Multi-Head Attention] Targeting the redundancy caused by uniform subspace allocation across heads in standard Multi-Head Attention (MHA), this paper proposes HINT…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "Multi-Head Attention"
  - "Transformer"
  - "Low-Light Enhancement"
  - "Dehazing"
  - "Desnowing"
date: 2026-05-08
content_hash: e3a1a2bef7c88f32
---

# Devil is in the Uniformity: Exploring Diverse Learners within Transformer for Image Restoration

**Conference**: ICCV 2025
**arXiv**: [2503.20174](https://arxiv.org/abs/2503.20174)
**Code**: [https://github.com/joshyZhou/HINT](https://github.com/joshyZhou/HINT)
**Area**: Image Restoration
**Keywords**: Image Restoration, Multi-Head Attention, Transformer, Low-Light Enhancement, Dehazing, Desnowing

## TL;DR
Targeting the redundancy caused by uniform subspace allocation across heads in standard Multi-Head Attention (MHA), this paper proposes HINT, which introduces Hierarchical Multi-Head Attention (HMHA) and Query-Key Cache Updating (QKCU) to enhance inter-head diversity and interaction, achieving state-of-the-art results on 12 benchmarks across 5 image restoration tasks.

## Background & Motivation
- **Background**: Transformers have achieved remarkable success in image restoration, with MHA as the core component, enabling parallel attention computation across multiple heads to capture diverse features.
- **Limitations of Prior Work**: Standard MHA allocates equal-sized ($C/h$) subspaces to each head, causing different heads to attend to the same regions and produce redundant representations — NLP research has shown that only a few heads are critical to final decisions, while the rest can be pruned.
- **Key Challenge**: Uniform subspace partitioning leads to similar information across heads, and the absence of inter-head collaboration further aggravates redundancy from two complementary dimensions.
- **Key Insight**: MHA is improved along two axes — (1) enabling heads to learn diverse features from subspaces of varying sizes; (2) introducing intra-layer and cross-layer head interaction mechanisms.
- **Core Idea**: Diverse learning is achieved via channel similarity reranking combined with hierarchical subspace partitioning, while head collaboration is enhanced through Query-Key caching that propagates information both within and across layers.

## Method

### Overall Architecture
HINT adopts an encoder-decoder architecture with $N_1=4$ scales. The degraded input image is first processed by a convolutional layer for shallow feature extraction, then passed through $N_1$ restoration stages to produce deep features. The encoder contains only FFN layers (asymmetric design), while the decoder includes HMHA + FFN blocks. Skip connections are used between the encoder and decoder. A refinement stage consisting of $N_3=4$ blocks follows, and a residual image is produced as the final output.

### Key Designs
1. **Hierarchical Multi-Head Attention (HMHA)**:

    - **Function**: Enables different heads to learn from subspaces of varying sizes, capturing diverse contextual information.
    - **Mechanism**: The channel space is partitioned hierarchically as $C = [C_1, C_2, \ldots, C_h]$ with $C_1 \leq C_2 \leq \cdots \leq C_h$. Prior to partitioning, channels are reranked based on Pearson correlation to ensure each subspace contains semantically independent information. The dimension ratio is set to [1, 2, 2, 3] with 4 heads.
    - **Design Motivation**: Uniform partitioning causes subspaces to carry similar information, leading heads to attend to the same regions. Differentiated subspace sizes combined with reranking force different heads to learn hierarchical and complementary representations.

2. **Query-Key Cache Updating (QKCU) — Intra-Layer Modulation**:

    - **Function**: Enhances information interaction among heads within the same layer.
    - **Mechanism**: An IntraCache is maintained by adding the concatenation of Q and K to the HMHA output, followed by a gating mechanism and a compress-reconstruct transformation:
      $$\mathbf{F}_{gated} = \text{GELU}(\text{Conv}(\mathbf{F}_{intra}^s)) \odot \mathbf{F}_{intra}^s$$
      $$\mathbf{F}_{Intra}^o = \text{Conv}_{up}(\text{Conv}_{down}(\mathbf{F}_{gated}))$$
    - **Design Motivation**: The gating mechanism selectively retains the most informative elements, while the compress-reconstruct step forces the model to focus on critical features.

3. **Query-Key Cache Updating (QKCU) — Cross-Layer Modulation**:

    - **Function**: Leverages historical attention scores to modulate attention computation in the current layer.
    - **Mechanism**: An InterCache stores historical $QK^T$ attention matrices. The current-layer attention scores are modulated via scale-shift:
      $$\mathbf{F}_m = \mathbf{F}_{scale} \odot \mathbf{F}_{att} + \mathbf{F}_{shift}$$
      The cross-layer cache is updated progressively as: $\mathbf{F}_{inter} = \alpha \mathbf{F}_{inter} + (1-\alpha) \mathbf{F}_{inter}^l$, with $\alpha=0.9$.
    - **Design Motivation**: In conventional MHA, heads across different layers of the same network operate independently. The cross-layer cache enables the model to leverage historical attention patterns to guide current-layer attention allocation, achieving dynamic (input-dependent) modulation.

### Loss & Training
- AdamW optimizer is used.
- Embedding dimension $C=48$, 4 heads, dimension ratio [1, 2, 2, 3].
- Encoder-decoder block counts: [4, 6, 6, 6], with the 4th scale as the bottleneck.
- 4 blocks in the refinement stage.
- $\alpha=0.9$ controls cross-layer cache information flow.

## Key Experimental Results

### Main Results — Low-Light Enhancement (LOL-v2)

| Method | LOL-v2-real PSNR | LOL-v2-syn PSNR | Avg. PSNR | Avg. SSIM |
|--------|-----------------|-----------------|-----------|-----------|
| Restormer (CVPR'22) | 19.94 | 21.41 | 20.68 | 0.829 |
| MambaIR (ECCV'24) | 21.25 | 25.55 | 23.40 | 0.880 |
| Retinexformer (ICCV'23) | 22.80 | 25.67 | 24.24 | 0.885 |
| MambaLLIE (NeurIPS'24) | 22.95 | 25.87 | 24.41 | 0.894 |
| **HINT** | **23.11** | **27.17** | **25.14** | **0.917** |

HINT outperforms Retinexformer by 0.9 dB and surpasses general restoration methods by at least 1.74 dB.

### Main Results — Other Tasks

**Desnowing (Snow100K)**:

| Method | PSNR | SSIM |
|--------|------|------|
| AST (CVPR'24) | 32.50 | 0.96 |
| ConvIR-S (TPAMI'24) | 33.79 | 0.95 |
| **HINT** | **34.14** | 0.94 |

**Dehazing (SOTS)**:

| Method | PSNR | SSIM |
|--------|------|------|
| PromptIR (NeurIPS'23) | 31.31 | 0.973 |
| AdaIR (ICLR'25) | 31.80 | 0.981 |
| **HINT** | **32.24** | **0.981** |

### Ablation Study

**Attention Mechanism Ablation** (LOL-v2-syn):

| Configuration | PSNR | SSIM |
|---------------|------|------|
| W-MSA [Uformer] | 24.19 | 0.941 |
| MDTA [Restormer] | 26.42 | 0.948 |
| **HMHA (Ours)** | **27.17** | **0.950** |

**HMHA Reranking Strategy Ablation**:

| Configuration | Params (M) | PSNR | SSIM |
|---------------|-----------|------|------|
| No reranking [Restormer] | 24.76 | 26.42 | 0.948 |
| Random shuffle | 24.87 | 26.54 | 0.949 |
| **HMHA (Pearson reranking)** | **24.87** | **27.17** | **0.950** |

**QKCU Module Ablation**:

| IntraCache | InterCache | Params (M) | PSNR |
|------------|------------|-----------|------|
| ✗ | ✗ | 21.34 | 26.47 |
| ✓ | ✗ | 23.82 | 26.67 |
| ✗ | ✓ | 22.39 | 26.72 |
| **✓** | **✓** | **24.87** | **27.17** |

### Key Findings
- HMHA outperforms W-MSA by 2.98 dB and MDTA by 0.75 dB, validating the effectiveness of hierarchical subspace partitioning.
- Pearson reranking surpasses random shuffling by 0.63 dB and no reranking by 0.75 dB — the reranking strategy is critical to performance gains.
- The intra-layer and cross-layer QKCU modules each contribute approximately 0.2–0.25 dB, totaling 0.7 dB improvement with only a 16.5% parameter increase.
- Feature visualizations clearly demonstrate that MDTA heads attend to the same regions, while HMHA heads attend to distinct regions.
- HINT also leads on the MANIQA metric on real-world datasets without ground truth (DICM/MEF/NPE/VV).
- Model efficiency: 126.92G FLOPs and 24.87M parameters, comparable to Restormer (144.25G + 26.13M).

## Highlights & Insights
- The problem diagnosis is precise: visualizations directly demonstrate head redundancy in standard MHA — different heads attend to the same regions (red boxes) while missing degraded regions (yellow boxes).
- The combination of hierarchical subspace partitioning and channel reranking is elegant and effective, delivering significant performance gains at negligible computational cost.
- The exponential moving average update ($\alpha=0.9$) for the cross-layer QK cache enables gradual integration of historical information.
- Comprehensive evaluation across 12 benchmarks and 5 task categories (low-light enhancement, dehazing, desnowing, denoising, deraining) demonstrates strong generalizability.

## Limitations & Future Work
- The dimension ratio [1, 2, 2, 3] and $\alpha=0.9$ are manually configured; adaptive determination of these hyperparameters could yield further improvements.
- The Pearson correlation reranking is static (based on global channel statistics); dynamic, input-dependent reranking may be more effective.
- The asymmetric design of excluding attention from the encoder limits global modeling capacity on the encoding side.
- Systematic comparisons with recent Mamba-based architectures (e.g., MambaIR) across more tasks are lacking.
- The memory and computational overhead of the cross-layer InterCache at very large network depths is not fully discussed.

## Related Work & Insights
- Restormer's MDTA (computing attention along the channel dimension) serves as the key baseline; HINT improves upon the internal MHA mechanism of this design.
- Head redundancy has been studied in NLP (head pruning), and this paper is among the first to systematically transfer this perspective to image restoration.
- The scale-shift modulation in QKCU is conceptually aligned with FiLM-style conditioning, but uniquely incorporates historical caching of QK attention information.
- The hierarchical subspace partitioning idea is generalizable to other Transformer architectures (e.g., object detection, segmentation).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of hierarchical subspace partitioning and QK cache interaction is creative, though individual components build on prior work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 12 benchmarks, 5 task categories, comprehensive ablations, efficiency analysis, real-world evaluation, and downstream task assessment.
- **Writing Quality**: ⭐⭐⭐⭐ Problem diagnosis supported by clear visualizations; method description is detailed and well-structured.
- **Value**: ⭐⭐⭐⭐ Strong generalizability; components can serve as plug-and-play modules to improve existing Transformer-based restoration models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [\[ICCV 2025\] UniRes: Universal Image Restoration for Complex Degradations](unires_universal_image_restoration_for_complex_degradations.md)
- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](../../CVPR2026/image_restoration/sat_selective_aggregation_transformer_for_image_super_resolution.md)
- [\[ICCV 2025\] Exploiting Diffusion Prior for Task-driven Image Restoration](exploiting_diffusion_prior_for_task-driven_image_restoration.md)
- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)

</div>

<!-- RELATED:END -->
