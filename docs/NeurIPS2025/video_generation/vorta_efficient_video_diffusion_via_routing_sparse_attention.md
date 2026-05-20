---
title: >-
  [Paper Note] VORTA: Efficient Video Diffusion via Routing Sparse Attention
description: >-
  [NeurIPS 2025][Video Generation][Video diffusion model acceleration] This paper proposes VORTA, a framework that achieves end-to-end 1.76× acceleration of video diffusion Transformers without quality degradation…
tags:
  - "NeurIPS 2025"
  - "Video Generation"
  - "Video diffusion model acceleration"
  - "sparse attention"
  - "routing mechanism"
  - "coreset selection"
date: 2026-05-08
content_hash: e04c1c175fdd5c7a
---

# VORTA: Efficient Video Diffusion via Routing Sparse Attention

**Conference**: NeurIPS 2025
**arXiv**: [2505.18809](https://arxiv.org/abs/2505.18809)  
**Code**: [GitHub](https://github.com/wenhao728/VORTA)  
**Area**: Video Generation
**Keywords**: Video diffusion model acceleration, sparse attention, routing mechanism, coreset selection, video generation

## TL;DR

This paper proposes VORTA, a framework that achieves end-to-end 1.76× acceleration of video diffusion Transformers without quality degradation, through bucketed coreset attention (for modeling long-range dependencies) and a signal-aware routing mechanism (for adaptively selecting sparse attention branches). Combined with caching and distillation methods, it achieves up to 14.41× acceleration.

## Background & Motivation

### Efficiency Bottleneck of Video Diffusion Transformers

Video Diffusion Transformers (VDiT) have achieved remarkable progress in high-quality video generation, but at an extremely high computational cost. For instance, HunyuanVideo requires approximately 1,000 seconds (500 PFLOPS) to generate a 5-second 720p video, with attention operations accounting for over 75% of the total computation.

The complexity of 3D self-attention is $\mathcal{O}(L^2 d)$, where the sequence length $L = F \times H \times W$. Even after VAE compression and patchification, HunyuanVideo still operates on sequences of approximately 100K tokens.

### Limitations of Prior Work

**Local sparse methods** (e.g., STA): These exploit the concentration of attention scores within local neighborhoods to restrict interaction range, but perform poorly on long-range attention heads. While the nearest 4% of keys contribute over 80% of attention weights, the remaining 96% remain important during early sampling steps.

**Online analysis methods** (e.g., ARnR): These dynamically detect sparsity patterns but introduce $\mathcal{O}(L^2)$ similarity computation overhead and require re-tuning when sampling configurations change.

**Key Challenge**: VDiT simultaneously exhibits both local and long-range attention patterns, which dynamically switch throughout the sampling process, making simple static strategies insufficient.

### Three Categories of Attention in VDiT

The authors classify attention in VDiT into three types:
- **Local attention**: Focuses on short-range interactions, responsible for fine-grained details.
- **Long-range attention**: Distributed across the entire sequence, capturing high-level semantics (layout and motion), with high tolerance for minor perturbations (e.g., merging similar tokens).
- **Critical attention**: Maintains both global awareness and local detail simultaneously, and is highly sensitive to perturbations.

Key finding: Long-range attention heads exhibit high intra-sequence redundancy — tokens are highly similar, and a small number of representative tokens suffice to summarize their information.

## Method

### Overall Architecture

VORTA consists of two core components: (1) sparse attention variants tailored to different attention types; and (2) a signal-to-noise ratio-based routing mechanism that adaptively selects the optimal sparse strategy for each attention head.

### Key Designs

#### 1. Sliding Window Attention (for Local Attention)

A 3D Sliding Tile Attention is adopted, converting jagged attention masks into block-dense masks to improve GPU hardware efficiency:

$$\mathbf{M} = \{m_{i,j}\} = \{j \in (\tau(i)-w, \tau(i)+w]\}$$

where $\tau(i) = \lfloor i/t \rfloor \cdot t + \lceil t/2 \rceil$ denotes the tile center. The window size is set to $(18, 27, 24)$.

#### 2. Bucketed Coreset Attention (for Long-Range Attention)

**Core Idea**: For long-range attention heads, redundant tokens are first pruned via coreset selection, followed by attention computation on the compressed sequence.

$$\text{coreset-attn}(\mathbf{H}) = \text{unpool} \circ \text{attn} \circ \text{pool}(\mathbf{H})$$

**Bucketed Coreset Selection (BCS)**: Tokens are partitioned into buckets of size $(t, h, w) = (2, 3, 2)$. Within each bucket, the similarity between the center token and its neighbors is computed, and the top-$k$ most similar tokens are pruned (coreset ratio $r_{\text{core}} = 0.5$).

**Complexity advantage**: BCS requires only $\mathcal{O}(L)$ complexity ($\mathcal{O}(thw)$ per bucket × $L/(thw)$ buckets), in contrast to $\mathcal{O}(L^2)$ for global pairwise methods. Retaining 50% of tokens reduces attention computation to 25% of the original (due to the quadratic relationship).

Compared to standard average pooling, BCS offers a critical advantage in selectivity: when neighboring tokens differ significantly, simple averaging causes information loss leading to mosaic artifacts or blurring, whereas BCS preserves diversity by pruning the most similar tokens.

#### 3. Signal-Aware Attention Routing

**Design Motivation**: Attention behavior is strongly correlated with the signal-to-noise ratio of input features — long-range attention dominates in early steps (constructing high-level semantics), while local attention prevails in later steps (refining details).

The router consists of one linear layer per attention layer, taking the diffusion timestep embedding $\mathbf{T}$ as input:

$$\boldsymbol{\alpha}^{(n)} = \text{softmax}(\mathbf{T} \mathbf{W}_R^{(n)})$$

During inference, hard selection is applied by choosing the branch with the highest gating value:

$$\mathbf{H}^{(n+1)} = \begin{cases} \text{sliding-attn}(\mathbf{H}^{(n)}) & \text{if } \alpha_2 > \alpha_1, \alpha_3 \\ \text{coreset-attn}(\mathbf{H}^{(n)}) & \text{if } \alpha_3 > \alpha_1, \alpha_2 \\ \text{attn}(\mathbf{H}^{(n)}) & \text{otherwise} \end{cases}$$

This adds only 0.1% additional parameters and introduces no inference-time overhead. In practice, the full attention branch is selected in only approximately 0.2% of cases.

### Loss & Training

The router is trained using a self-supervised distillation strategy, with the original VDiT parameters frozen and only the router weights updated:

$$\mathcal{L} = \mathcal{L}_{\text{CFM}} + \lambda_{\text{distill}} \cdot \text{MSE}(\mathbf{H}_{\text{org}}^{(N)}, \mathbf{H}^{(N)}) + \lambda_{\text{reg}} \cdot \sum_{n=1}^{N} \|\alpha_1^{(n)}\|^2$$

where $\lambda_{\text{distill}} = 20$ and $\lambda_{\text{reg}} = 0.02$. The $L_2$ regularization encourages sparse selection. Training requires only 100 steps on the Mixkit dataset and completes in approximately one day on 2× H100 GPUs.

## Key Experimental Results

### Main Results on HunyuanVideo

| Method | Type | VBench↑ | LPIPS↓ | Latency (s) | Speedup | Memory (GB) |
|--------|------|---------|--------|-------------|---------|-------------|
| HunyuanVideo | - | 82.26 | - | 1043.85 | 1.00× | 47.64 |
| + ARnR | Sparse | 82.39 | 0.211 | 790.55 | 1.32× | 78.15 |
| + STA | Sparse | 82.33 | 0.201 | 676.39 | 1.54× | 51.79 |
| + PAB | Cache | 82.40 | 0.186 | 815.51 | 1.28× | >80 |
| **+ VORTA** | **Sparse** | **82.59** | **0.185** | **594.23** | **1.76×** | **51.15** |
| + VORTA & PAB | Combined | 82.56 | 0.195 | 444.19 | 2.35× | >80 |
| + PCD | Distill | 81.17 | 0.564 | 125.98 | 8.29× | 47.64 |
| + VORTA & PCD | Combined | 81.49 | 0.575 | 72.46 | **14.41×** | 51.15 |

### Ablation Study (Wan 2.1 1.3B, 480p)

| Configuration | VBench↑ | Latency (s) | Speedup |
|---------------|---------|-------------|---------|
| Wan 2.1 baseline | 81.20 | 73.24 | 1.00× |
| w/o sliding attention | 80.25 | 65.14 | 1.12× |
| w/o coreset attention | 79.89 | 66.10 | 1.11× |
| w/o full attention (fallback branch removed) | 77.14 | 59.34 | 1.23× |
| w/o timestep conditioning | 81.03 | 65.00 | 1.13× |
| Average pooling AP(2,1,1) | 77.08 | 57.53 | 1.27× |
| Average pooling AP(1,2,1) | 76.01 | 57.64 | 1.27× |
| **VORTA (full)** | **81.06** | **58.42** | **1.25×** |

### Key Findings

1. VORTA slightly outperforms the original model on VBench (82.59 vs. 82.26), possibly due to a regularization effect from pruning redundant attention.
2. Removing the full attention branch causes a 4-point VBench drop with no additional speedup, confirming the existence of critical attention heads.
3. Removing timestep conditioning causes the router to uniformly select the same branch across all steps, resulting in degraded performance and reduced speedup.
4. BCS significantly outperforms simple average pooling (81.06 vs. 75.94–77.08), validating the necessity of selective pruning.
5. Routing patterns exhibit clear temporal regularity: coreset attention dominates in early steps, transitioning to sliding attention in later steps.

## Highlights & Insights

1. **Theoretically grounded design**: The three-category attention taxonomy provides a clear theoretical basis for assigning distinct sparse strategies.
2. **Linear complexity of BCS**: The bucketed strategy reduces coreset selection from $\mathcal{O}(L^2)$ to $\mathcal{O}(L)$, which is key to practical acceleration.
3. **Strong composability**: VORTA is orthogonal to caching (PAB) and distillation (PCD) methods, achieving 14.41× acceleration when combined.
4. **Scheduler generalization**: Adapts to different ODE solvers and step counts without re-analysis.
5. **Backbone generalization**: Validated on both MMDiT (HunyuanVideo) and DiT (Wan 2.1) architectures.

## Limitations & Future Work

- Primarily targets attention acceleration; limited speedup for short sequences (images or low-resolution video).
- Only supports bidirectional generation paradigms; autoregressive video generation requires substantial adaptation.
- When the pretrained model itself produces artifacts (e.g., distortions or physics violations), VORTA inherits and may amplify these issues.
- Router training requires approximately one day, which, while far less than pretraining, is still non-trivial.
- The coreset ratio is fixed at 50%; adaptive ratios could further optimize the efficiency–quality trade-off.

## Related Work & Insights

- **Sparse attention**: STA (Sliding Tile Attention), ARnR (online analysis), SVG
- **Video diffusion acceleration**: PAB (feature caching), PCD (consistency distillation)
- **Conditional computation**: MoE (Mixture of Experts), BlockDrop
- **Video generation**: HunyuanVideo, Wan 2.1, CogVideoX
- **Insights**: The routing mechanism design can be generalized to other conditional computation scenarios; BCS's linear-complexity coreset selection is applicable to long-context LLM inference.

## Rating
- Novelty: ⭐⭐⭐⭐☆ — The combination of coreset attention and signal-aware routing is a novel design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across multiple backbones, schedulers, and combinations, with detailed runtime analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Taxonomically clear with excellent figure design.
- Value: ⭐⭐⭐⭐⭐ — Significant practical speedup with open-sourced code and weights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VSA: Faster Video Diffusion with Trainable Sparse Attention](vsa_faster_video_diffusion_with_trainable_sparse_attention.md)
- [\[NeurIPS 2025\] Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation](radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)
- [\[NeurIPS 2025\] S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation](s2q-vdit_accurate_quantized_video_diffusion_transformer_with_salient_data_and_sp.md)
- [\[NeurIPS 2025\] Training-Free Efficient Video Generation via Dynamic Token Carving](training-free_efficient_video_generation_via_dynamic_token_carving.md)
- [\[NeurIPS 2025\] LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation](lemica_lexicographic_minimax_path_caching_for_efficient_diffusion-based_video_ge.md)

</div>

<!-- RELATED:END -->
