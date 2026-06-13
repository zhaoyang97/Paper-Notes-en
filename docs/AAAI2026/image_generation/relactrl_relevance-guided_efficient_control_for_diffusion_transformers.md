---
title: >-
  [Paper Note] RelaCtrl: Relevance-Guided Efficient Control for Diffusion Transformers
description: >-
  [AAAI 2026][Image Generation][Controllable Generation] This paper proposes the RelaCtrl framework, which quantifies the sensitivity of each DiT layer to control information via a ControlNet Relevance Score…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Controllable Generation"
  - "Diffusion Transformer"
  - "ControlNet"
  - "Parameter Efficiency"
  - "Channel-Token Shuffling"
date: 2026-05-08
content_hash: 7ec85f95debab5d3
---

# RelaCtrl: Relevance-Guided Efficient Control for Diffusion Transformers

**Conference**: AAAI 2026
**arXiv**: [2502.14377](https://arxiv.org/abs/2502.14377)  
**Code**: None  
**Area**: Image Generation
**Keywords**: Controllable Generation, Diffusion Transformer, ControlNet, Parameter Efficiency, Channel-Token Shuffling

## TL;DR

This paper proposes the RelaCtrl framework, which quantifies the sensitivity of each DiT layer to control information via a ControlNet Relevance Score, and uses this analysis to guide the placement and modeling capacity of control blocks. A Two-Dimensional Shuffle Mixer (TDSM) is introduced to replace self-attention and FFN, achieving controllable generation quality superior to PixArt-δ with only 15% of its parameters and computational cost.

## Background & Motivation

### State of the Field

Diffusion Transformers (DiT) have achieved remarkable progress in text-to-image/video generation owing to their strong scalability (e.g., PixArt-α, Flux, SD3, Sora). Controllable generation is an important application of DiT, typically realized by attaching auxiliary control branches (e.g., ControlNet) to condition generation on edges, depth maps, segmentation masks, and other signals.

### Limitations of Prior Work

**Issue 1: Excessive parameter and computational overhead**
- PixArt-δ directly replicates the first 13 Transformer blocks, increasing parameter count and computation by **50%**
- OminiControl doubles the token sequence by concatenating control tokens, increasing computational complexity by approximately **70%**

**Issue 2: Uneven resource allocation**
- Differences in the relevance of control information across DiT layers are neglected
- Shallow-to-middle layers are more sensitive to control signals, while deeper layers exhibit weaker relevance
- Applying a uniform control block configuration to all layers introduces substantial redundancy in deeper layers

### Root Cause

How can the parameter count and computational cost of the control branch be drastically reduced while maintaining or even improving the quality and precision of controllable generation?

### Starting Point

The authors first conduct **systematic experiments** to quantify the importance of each layer to the control effect (ControlNet Relevance Score), and then **differentially allocate** the placement, parameter scale, and modeling capacity of control blocks according to these relevance scores.

## Method

### Overall Architecture

RelaCtrl comprises three core designs:
1. **ControlNet Relevance Prior**: quantifies the importance of control information at each layer
2. **Relevance-Guided Control Block Placement**: places control blocks at high-relevance positions
3. **Relevance-Guided Lightweight Control Block (RGLC)**: replaces the original Transformer block with TDSM

### Key Designs

#### 1. **ControlNet Relevance Score (CRS)**

**Mechanism**: A complete ControlNet with all 27 control blocks is first trained; during inference, each control block is skipped one at a time, and FID (generation quality) and HDD (control precision) are used to measure the impact of omitting each layer.

The scoring formula is:

$$CRS_i = \frac{1}{2}\left(\frac{F_i - F_{min}}{F_{max} - F_{min}} + \frac{H_i - H_{min}}{H_{max} - H_{min}}\right)$$

where $F_i$ and $H_i$ denote the FID and HDD rankings obtained after skipping the $i$-th control block, respectively.

**Key Findings**:
- Relevance follows a **rise-then-fall** trend across layers
- The most critical layers concentrate in **shallow-to-middle** positions (e.g., blocks 5, 6, 7)
- Removing control blocks from the last few layers causes only **marginal performance degradation**
- This distribution differs from layer importance patterns in LLMs (which tend to be monotonically decreasing or U-shaped)

**Design Motivation**: These findings suggest that PixArt-δ's strategy of directly copying the first 13 layers is suboptimal—it may omit critical intermediate layers while retaining unnecessary deep-layer control blocks.

#### 2. **Relevance-Guided Control Block Placement and Modeling**

The top-11 positions ranked by CRS are selected for control block placement (vs. PixArt-δ's 13 consecutive front layers), reducing the number of control blocks by approximately 15% while maintaining comparable performance.

A further strategy (Prior 2) adjusts the modeling capacity at each position according to relevance: high-relevance positions reduce the number of channel groups (expanding the attention feature dimension) to enhance modeling capability, while low-relevance positions increase the number of groups to reduce computation.

#### 3. **Two-Dimensional Shuffle Mixer (TDSM)**

**Core Idea**: From a MetaFormer perspective, the two core components of a Transformer are the token mixer (self-attention) and the channel mixer (FFN). TDSM **unifies both into a single operation**.

**Procedure**:
1. **Random channel selection**: The input $c_{in} \in \mathbb{R}^{H \times W \times D}$ is randomly partitioned along the channel dimension into $n$ groups $c_{rs}^i \in \mathbb{R}^{H \times W \times d_i}$
2. **Random 3D shuffle**: Token positions in 3D space are randomly permuted within each group
3. **Local self-attention**: Attention is computed within fixed-size local windows of size $s \times s \times d$
4. **Inverse restoration**: Inverse operations are applied to the token and channel dimensions to recover the original arrangement

**Theoretical Guarantee**:

$$d(t_j) \geq \frac{\sqrt{2}}{4}(H + Wd_i)$$

The lower bound on the average interaction distance of the grouped attention in TDSM is $\Omega(\frac{\sqrt{2}}{4}(H+Wd_i))$, guaranteeing non-local interaction modeling capability.

**Design Motivation**:
- The $O(N^2)$ complexity of standard self-attention is prohibitively expensive for a control branch
- FFN layers are highly redundant (as demonstrated in prior work)
- Random shuffling breaks the locality constraint of grouped attention, enabling non-local modeling at low computational cost

#### Complete RGLC Block Pipeline

$$c_{cond} = ZC(TDSM(c_{in}) + c_{in})$$

where $c_{in}$ = control condition input $c$ + zero convolution($x$) ($x$ is from the corresponding frozen backbone block), and $ZC$ denotes zero convolution.

### Loss & Training

- The PixArt-α backbone is frozen
- The control branch (RGLC blocks + zero convolutions) is trained from scratch
- Exactly the same training settings as PixArt-δ are used for fair comparison

## Key Experimental Results

### Main Results

Quantitative comparison on the COCO validation set:

| Method | Condition | HDD↓ | FID↓ | C-Ae↑ | C-SC↑ |
|--------|-----------|------|------|-------|-------|
| PixArt-δ | Canny | 96.26 | 21.38 | 5.508 | 0.279 |
| **RelaCtrl** | Canny | **94.04** | **20.34** | **5.584** | **0.282** |
| PixArt-δ | HED | 98.91 | 29.22 | 5.243 | 0.275 |
| **RelaCtrl** | HED | **96.11** | **27.73** | **5.451** | **0.276** |
| PixArt-δ | Depth | 99.69 | 35.21 | 5.723 | 0.283 |
| **RelaCtrl** | Depth | **99.11** | **33.93** | **5.887** | **0.285** |
| PixArt-δ | Seg. | 0.379(mIoU) | 35.50 | 5.668 | 0.282 |
| **RelaCtrl** | Seg. | **0.405** | **33.76** | **5.702** | **0.287** |

RelaCtrl **comprehensively outperforms** PixArt-δ across all four conditional control tasks, using only 15.3% of its parameters.

### Ablation Study

Effect of the number of control blocks (ranked by relevance):

| Configuration | HDD↓ | FID↓ | Parameter Ratio |
|---------------|------|------|-----------------|
| ControlNet-top13 (baseline) | 96.26 | 21.38 | 100% |
| Relevance-top13 | 94.57 | 20.31 | 100% |
| Relevance-top12 | 95.88 | 20.79 | 92.5% |
| **Relevance-top11** | 95.57 | 21.28 | 84.6% |
| Relevance-top10 | 96.36 | 22.24 | 76.9% |

Effect of RGLC and Prior 2:

| Configuration | HDD↓ | FID↓ | Parameter Ratio |
|---------------|------|------|-----------------|
| **RelaCtrl (full)** | **94.04** | **20.34** | **15.3%** |
| w/o RGLC (original copied blocks) | 95.57 | 21.28 | 84.6% |
| w/o Prior 2 (uniform TDSM) | 97.30 | 22.47 | 17.1% |
| Baseline (PixArt-δ) | 96.26 | 21.38 | 100% |

### Efficiency Analysis

| Method | Params (M) | Computation (GFLOPs) | Inference Time (s) |
|--------|-----------|---------------------|--------------------|
| PixArt-α (baseline) | 611.15 | 542.56 | 3.81 |
| +ControlNet | +294.34 (+48.16%) | +270.57 (+49.87%) | +0.51 |
| **+RelaCtrl** | **+45.15 (+7.38%)** | **+46.71 (+8.61%)** | **+0.24** |

### Key Findings

- **Relevance-guided placement > sequential copying**: Even with the same 13 control blocks, relevance-ranked placement (Relevance-top13) outperforms sequential front-13 placement (FID 20.31 vs. 21.38)
- **11 blocks ≈ 13 blocks**: Under relevance guidance, 11 control blocks suffice to match the performance of 13
- **RGLC blocks outperform original copied blocks**: Replacing self-attention and FFN with TDSM improves performance while reducing parameters by 85%
- **Prior 2 is critical**: Removing relevance-guided TDSM channel adjustment leads to significant performance degradation
- **Effective across all four conditions**: Consistent improvements on Canny, HED, Depth, and Segmentation

## Highlights & Insights

1. **Analysis-driven design**: Rather than relying on intuition, the method systematically quantifies each layer's control contribution via layer-wise ablation—this "analyze first, then design" methodology is broadly instructive
2. **Counter-intuitive finding on relevance distribution**: Control information relevance in DiT follows a "rise-then-fall" pattern rather than a monotonic trend, differing from patterns observed in LLMs, suggesting that layer importance distributions may vary substantially across tasks
3. **Theoretical guarantee for TDSM**: Beyond proposing an efficient replacement module, the authors formally prove a lower bound on its non-local modeling capability
4. **Extreme efficiency**: 7.38% additional parameters + 8.61% additional computation outperforms PixArt-δ with 48%+ additional parameters, yielding an efficiency ratio of approximately **6.5×**

## Limitations & Future Work

- CRS requires training a complete ControlNet (27 control blocks) upfront; the computational overhead of this preliminary analysis is not discussed
- The relevance analysis is conducted on PixArt-α; whether the conclusions transfer to other DiT architectures such as Flux and SD3 remains to be verified
- The random shuffling in TDSM may introduce noise; its long-term effect on training stability is not thoroughly discussed
- Validation is limited to 512 resolution; the efficiency advantage at higher resolutions (e.g., 1024, 2048) may be even greater but is unexplored
- Controllable generation in video generation models (e.g., CogVideoX) faces similar efficiency challenges, which are not addressed in this work

## Related Work & Insights

- **Relationship to ControlNet-XS**: ControlNet-XS improves interaction bandwidth from a feedback control perspective, while RelaCtrl optimizes resource allocation from a layer importance perspective—the two approaches are complementary
- **Influence of MetaFormer**: Decomposing the Transformer into a token mixer and a channel mixer provides the theoretical grounding for TDSM's design
- **Broader implications**: Relevance analysis could be generalized to other scenarios requiring auxiliary modules (e.g., layer-wise LoRA allocation, adapter placement selection)
- Future directions for DiT ControlNet design may include **adaptive relevance estimation** that does not require pre-training a full ControlNet

## Rating

- Novelty: ⭐⭐⭐⭐ (The relevance analysis and TDSM design are novel, though the overall paradigm of "analyze + prune" is incremental)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 conditional tasks, multiple baselines, comprehensive ablations, efficiency analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, rigorous theoretical proofs, rich visualizations)
- Value: ⭐⭐⭐⭐⭐ (Addresses a critical efficiency problem in DiT controllable generation with strong practical utility)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Laytrol: Preserving Pretrained Knowledge in Layout Control for Multimodal Diffusion Transformers](laytrol_preserving_pretrained_knowledge_in_layout_control_fo.md)
- [\[AAAI 2026\] DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models](dogfit_domain-guided_fine-tuning_for_efficient_transfer_learning_of_diffusion_mo.md)
- [\[CVPR 2026\] EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation](../../CVPR2026/image_generation/edgedit_hardware-aware_diffusion_transformers_for_efficient_on-device_image_gene.md)
- [\[ICML 2026\] Spectral Guidance for Flexible and Efficient Control of Diffusion Models](../../ICML2026/image_generation/spectral_guidance_for_flexible_and_efficient_control_of_diffusion_models.md)
- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](../../ICCV2025/image_generation/edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)

</div>

<!-- RELATED:END -->
