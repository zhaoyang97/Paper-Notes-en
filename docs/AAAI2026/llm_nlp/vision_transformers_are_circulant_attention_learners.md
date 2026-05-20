---
title: >-
  [Paper Note] Vision Transformers are Circulant Attention Learners
description: >-
  [AAAI 2026][LLM/NLP][Vision Transformer] This paper discovers that self-attention matrices in ViTs inherently learn Block Circulant with Circulant Blocks (BCCB) patterns, and proposes Circulant Attention…
tags:
  - "AAAI 2026"
  - "LLM/NLP"
  - "Vision Transformer"
  - "Circulant Attention"
  - "BCCB"
  - "FFT"
  - "Efficient Attention"
date: 2026-05-08
content_hash: cfa07d82157e0163
---

# Vision Transformers are Circulant Attention Learners

**Conference**: AAAI 2026
**arXiv**: [2512.21542](https://arxiv.org/abs/2512.21542)  
**Code**: N/A  
**Area**: LLM NLP
**Keywords**: Vision Transformer, Circulant Attention, BCCB, FFT, Efficient Attention

## TL;DR
This paper discovers that self-attention matrices in ViTs inherently learn Block Circulant with Circulant Blocks (BCCB) patterns, and proposes Circulant Attention, which achieves $O(N\log N)$ complexity via 2D FFT, yielding consistent improvements on ImageNet classification, COCO detection, and ADE20K segmentation.

## Background & Motivation
**Background**: While ViT self-attention is highly expressive, its $O(N^2)$ complexity is prohibitively expensive at high resolutions (N=H×W). PVT reduces cost via K/V downsampling, Swin via local windows, and BiFormer via sparse routing, but all sacrifice global modeling capacity.

**Limitations of Prior Work**: Existing efficient attention methods impose efficiency through external constraints (locality/sparsity), suppressing the long-range modeling capability of self-attention. Such "external patch" efficiency optimizations are fundamentally at odds with the design goals of self-attention.

**Key Observation (Core Contribution)**: Visualization of DeiT attention matrices reveals that they frequently approximate BCCB (Block Circulant matrix with Circulant Blocks) structure — i.e., 2D circulant matrices. Attention distributions of neighboring queries exhibit translation invariance (analogous to 2D global convolution). Moreover, BCCB matrix multiplication can be performed in $O(N\log N)$ via 2D FFT.

**Key Challenge**: Self-attention, at $O(N^2)$ cost, effectively learns patterns that can be computed in $O(N\log N)$ — indicating substantial computational redundancy in standard self-attention.

**Key Insight**: Since ViTs naturally learn BCCB structure, explicitly enforcing the attention matrix to be BCCB is more principled than imposing external constraints.

**Core Idea**: Orthogonally project the self-attention matrix onto the BCCB subspace and implement $O(N\log N)$ global attention via 2D FFT, substantially reducing computation while preserving expressiveness.

## Method

### Overall Architecture
Standard self-attention is replaced by BCCB-based Circulant Attention, which can be seamlessly integrated into architectures such as DeiT, PVT, and Swin. The pipeline closely mirrors standard self-attention, differing only in the use of a BCCB attention matrix and DFT operations in place of dense matrix multiplication.

### Key Designs

1. **Orthogonal Projection onto the BCCB Subspace**:

    - Function: Projects the raw attention matrix $A = QK^T/\sqrt{d}$ onto the nearest BCCB matrix.
    - Mechanism: The BCCB subspace admits an orthogonal basis $\{B_0,...,B_{N-1}\}$ (where $B_k$ is the unit BCCB matrix with a 1 at position $k$); the projection is given by $\tilde{A} = \frac{1}{N}\sum_{k=0}^{N-1}\langle A, B_k\rangle B_k$. A BCCB matrix is fully determined by its first row, and matrix-vector multiplication is equivalent to 2D circular cross-correlation (i.e., depth-wise separable convolution with 2D circular padding), computable via FFT: $Bx = \mathcal{F}_{2D}^{-1}(\overline{\mathcal{F}_{2D}(b)} \odot \mathcal{F}_{2D}(x))$.
    - Design Motivation: Projection onto the BCCB subspace is the mathematically optimal way to obtain the "nearest BCCB matrix," guaranteeing minimal deviation from the original attention.

2. **Token Reweighting Module**:

    - Function: Compensates for inherent limitations of the BCCB structure.
    - Mechanism: $T = \text{SiLU}(xW_T)$, reweighting the output of each token.
    - Design Motivation: BCCB matrices have equal row/column sums, which constrains different queries from obtaining different total attention. Token reweighting breaks this constraint and is responsible for a +1.2% accuracy gain.

3. **Efficient Computation Pipeline**:

    - Standard self-attention: $O = \sigma(\tilde{A})V$, where $\tilde{A}$ is the BCCB matrix.
    - $\tilde{A}V$ is computed via 2D DFT with complexity $O(N\log N)$.
    - At resolution 1536²: FLOPs are reduced by 8×, with 7× inference speedup.

## Key Experimental Results

### ImageNet Classification

| Model | Resolution | Params | FLOPs | Top-1 Acc |
|-------|------------|--------|-------|-----------|
| DeiT-T | 224² | 5.7M | 1.2G | 72.2% |
| **CA-DeiT-T** | 224² | 6.1M | 1.2G | **75.0% (+2.8)** |
| DeiT-S | 224² | 22.1M | 4.6G | 79.8% |
| **CA-DeiT-S** | 224² | 23.8M | 4.8G | **81.0% (+1.2)** |
| DeiT-B | 224² | 86.6M | 17.6G | 81.8% |
| **CA-DeiT-B** | 224² | 93.6M | 18.9G | **82.3% (+0.5)** |
| PVT-T | 224² | 13.2M | 1.9G | 75.1% |
| **CA-PVT-T** | 224² | 12.2M | 2.0G | **78.1% (+3.0)** |

### Object Detection and Semantic Segmentation

| Task | Model | Performance | Notes |
|------|-------|-------------|-------|
| COCO Detection | CA-PVT-S | 44.2 AP | Matches PVT-L with 30% fewer parameters |
| ADE20K Segmentation | CA-PVT-S | 42.3% mIoU | +2.5% vs. baseline |

### Ablation Study (DeiT-S)

| Configuration | Top-1 Acc | Notes |
|---------------|-----------|-------|
| + Circulant only | 79.7% | Near-lossless (−0.1%), validating BCCB structural fidelity |
| + head dim=1 | 80.2% | A single head dimension suffices for full expressiveness |
| + token reweight | 81.0% | Key gain; compensates for BCCB equal-row-sum constraint |

### High-Resolution Efficiency Comparison

| Resolution | Standard Attn FLOPs | Circulant FLOPs | Speedup |
|------------|---------------------|-----------------|---------|
| 224² | Baseline | ~same | ~1× |
| 1536² | Very high | 8× lower | **7× inference speedup** |

### Key Findings
- BCCB projection incurs near-zero accuracy loss (only −0.1%), confirming that ViT self-attention intrinsically learns BCCB structure.
- The token reweighting module contributes +1.3%, serving as the key mechanism to compensate for the equal-row-sum constraint of BCCB matrices, which limits different queries from obtaining different total attention.
- At resolution 1536², FLOPs are reduced by 8× and inference is accelerated by 7×, yielding substantial benefits in high-resolution scenarios.
- CA-PVT-S matches PVT-L accuracy with 30% fewer parameters and 40% fewer FLOPs, demonstrating significantly improved parameter efficiency.
- Smaller models benefit more: DeiT-T gains +2.8% while DeiT-B gains only +0.5%, suggesting that attention in smaller models more closely approximates BCCB.

## Highlights & Insights
- The observation that ViTs intrinsically learn BCCB structure reveals a structural preference of self-attention and provides a theoretical basis for designing more efficient attention mechanisms.
- The $O(N\log N)$ global attention mechanism achieves efficiency without sacrificing long-range modeling capacity, offering a rare solution that reconciles efficiency and expressiveness.
- BCCB projection is equivalent to 2D global circular convolution, unifying attention and convolution mechanisms from a novel perspective.
- The finding that head dim=1 suffices for full expressiveness is intriguing, suggesting that the channel dimension of attention may be overestimated in vision tasks.

## Limitations & Future Work
- Circulant Attention is applied only to early-stage attention layers rather than as a global replacement.
- The specific requirement of head dim=1 lacks a thorough theoretical explanation.
- The improvement mechanism of the token reweighting module remains limited in scope.

## Related Work & Insights
- **vs. Swin**: Swin reduces complexity via local windows at the cost of global information, whereas Circulant Attention preserves a global receptive field.
- **vs. Linear Attention**: Linear attention achieves $O(N)$ complexity but at the cost of reduced expressiveness; Circulant Attention achieves $O(N\log N)$ with near-lossless quality.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The BCCB observation is novel and provides a mathematically grounded perspective on attention.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across three vision tasks and multiple architectures.
- Writing Quality: ⭐⭐⭐⭐ Visualization analysis is clear and mathematical treatment is rigorous.
- Value: ⭐⭐⭐⭐⭐ Directional contribution establishing a new paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Learning Spatial Decay for Vision Transformers](learning_spatial_decay_for_vision_transformers.md)
- [\[NeurIPS 2025\] CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers](../../NeurIPS2025/llm_nlp/cat_circular-convolutional_attention_for_sub-quadratic_transformers.md)
- [\[NeurIPS 2025\] Strassen Attention, Split VC Dimension and Compositionality in Transformers](../../NeurIPS2025/llm_nlp/strassen_attention_split_vc_dimension_and_compositionality_in_transformers.md)
- [\[ICLR 2026\] AP-OOD: Attention Pooling for Out-of-Distribution Detection](../../ICLR2026/llm_nlp/ap-ood_attention_pooling_for_out-of-distribution_detection.md)
- [\[ICLR 2026\] Trapped by simplicity: When Transformers fail to learn from noisy features](../../ICLR2026/llm_nlp/trapped_by_simplicity_when_transformers_fail_to_learn_from_noisy_features.md)

</div>

<!-- RELATED:END -->
