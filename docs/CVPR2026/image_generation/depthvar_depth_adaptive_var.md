---
title: >-
  [Paper Note] Depth Adaptive Efficient Visual Autoregressive Modeling
description: >-
  [CVPR 2026][Image Generation][Visual Autoregressive] Reveals the fundamental limitations of the frequency-driven hard pruning paradigm in VAR models and proposes DepthVAR, a training-free inference acceleration framework. By adaptively allocating the Transformer computation depth for each token (rather than binary keep/prune), it achieves $2.3\times$-$3.1\times$ speedup with minimal quality loss.
tags:
  - CVPR 2026
  - Image Generation
  - Visual Autoregressive
  - Inference Acceleration
  - Dynamic Depth
  - Training-free
  - Token-level Computation Allocation
date: 2026-05-08
content_hash: af9ad1970d296a39
---

# Depth Adaptive Efficient Visual Autoregressive Modeling

**Conference**: CVPR 2026  
**arXiv**: [2604.17286](https://arxiv.org/abs/2604.17286)  
**Code**: [https://github.com/STOVAGtz/DepthVAR](https://github.com/STOVAGtz/DepthVAR)  
**Area**: Image Generation  
**Keywords**: Visual Autoregressive, Inference Acceleration, Dynamic Depth, Training-free, Token-level Computation Allocation

## TL;DR

Reveals the fundamental limitations of the frequency-driven hard pruning paradigm in VAR models and proposes DepthVAR, a training-free inference acceleration framework. By adaptively allocating the Transformer computation depth for each token (rather than binary keep/prune), it achieves $2.3\times$-$3.1\times$ speedup with minimal quality loss.

## Background & Motivation

**Background**: Visual Autoregressive (VAR) models significantly reduce sequence lengths in text-to-image generation by replacing traditional "next-token" prediction with "next-scale" prediction. However, as resolution increases, the number of tokens per scale grows quadratically, leading to severe computational waste by applying full-layer calculations uniformly to all tokens.

**Limitations of Prior Work**: Methods like FastVAR and SparseVAR utilize frequency features for hard pruning of tokens—estimating high-frequency distributions and discarding "unimportant" low-frequency tokens. However, this approach has fundamental issues: even with a perfect frequency mask (oracle experiment), hard pruning still leads to significant quality degradation; more precise frequency estimation does not guarantee better generation quality (Pearson $r = 0.138$).

**Key Challenge**: Hard pruning binarizes tokens into "keep/discard," but in reality, low-frequency regions do not entirely lack a need for computation; they simply require less—the problem lies in the "all-or-nothing" coarse-grained decision.

**Goal**: Shift from the hard pruning paradigm to continuous computation depth allocation, allowing each token to receive a number of Transformer layers matched to its complexity.

**Key Insight**: The authors find that pre-trained VAR models naturally possess depth redundancy due to the use of LayerDrop regularization during training—generation quality peaks before reaching the final layer, and representations of different tokens saturate at different depths.

**Core Idea**: Replace hard pruning with per-token dynamic depth allocation. A cyclic rotation scheduler generates non-static depth scores, which are converted into layer masks via bit-reversal mapping to achieve balanced layer utilization.

## Method

### Overall Architecture

During the multi-scale prediction process of VAR, the first few (small) scales follow the standard procedure. From a certain scale onwards, dynamic depth inference is applied. For each subsequent scale $i$, an adaptive depth score scheduler first generates depth scores $\mathcal{S}_i \in [0,1]^{h_i \times w_i}$ based on layer-wise change information from the previous scale. These are then converted into layer masks $\mathcal{M}_i$ to selectively skip certain Transformer blocks during inference. Finally, the code is output by blending based on the depth scores.

### Key Designs

1.  **Bit-reversal Depth Map**:
    -   Function: Evenly disperses per-token depth scores across various Transformer layers.
    -   Mechanism: Given a depth map $\mathcal{D}_i = \lfloor \mathcal{S}_i \cdot L \rfloor$, instead of simply selecting the first $d$ layers, a bit-reversal permutation $\pi_L$ is used to disperse $d$ layers of computation uniformly. For example, if $L=32, d=5$, layers $\{0, 16, 8, 24, 4\}$ are selected instead of $\{0,1,2,3,4\}$. This generates a layer mask $\mathcal{M}_i(\ell, m, n) = \mathbf{1}\{\ell \in \mathcal{L}_i(m,n)\}$.
    -   Design Motivation: If "shallow" tokens only pass through the first few layers, some layers will be heavily pruned while others bear uneven loads. Bit-reversal ensures each layer is utilized relatively equally, similar to index rearrangement in FFT.

2.  **Layer Behavior Approximation and Code Blending**:
    -   Function: Guarantees spatial integrity of feature maps and depth proportionality of output during sparse inference.
    -   Mechanism: At each layer $\ell$, Transformer blocks are computed only for active positions. Masked positions are recovered using inter-layer residuals cached from the previous scale (after upsampling) as proxies: $r_i^\ell = \text{Layer}_\ell(r_i^{\ell-1} \odot \mathcal{M}_i(\ell)) + \text{up}(r_{i-1}^\ell - r_{i-1}^{\ell-1}) \odot (1 - \mathcal{M}_i(\ell))$. Finally, codebook lookup results are weighted by depth scores: $z_i = \mathcal{S}_i \cdot \text{lookup}(p_i)$.
    -   Design Motivation: Cached proxy recovery exploits local stability between scales to ensure subsequent layers receive spatially complete feature maps. Code blending makes the contribution proportional to the computational investment.

3.  **Adaptive Depth Score Scheduler**:
    -   Function: Generates dynamic, non-static depth scores for each token position.
    -   Mechanism: Aggregates absolute feature changes of each layer from the previous scale to form a "decision rank map" $\mathcal{B}_i$, normalized to percentiles $\rho_i$, and then mapped to depth scores via a scheduler function $\mathcal{G}$. A key innovation is cyclic percentile rotation $\mathcal{G}'(\rho)$, preventing the same set of tokens from being repeatedly updated or skipped. A reference scale $\mathcal{R}$ is used to constrain the computation of larger scales.
    -   Design Motivation: Directly reusing decisions from the previous scale causes certain regions to be repeatedly treated with low priority. Cyclic rotation ensures every region has the opportunity for sufficient computation across different scales.

### Loss & Training

A completely training-free framework that does not modify model parameters. All mechanisms are applied during inference, with the speedup ratio controlled by adjusting the reference scale $\mathcal{R}$, scheduler function types, and parameters.

## Key Experimental Results

### Main Results

| Method | GenEval↑ | Latency (ms)↓ | HPSv2↑ | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Infinity Baseline | 0.7237 | 2706 | 30.47 | 1× |
| SparseVAR-0.7 | 0.7208 | 1281 | 29.76 | 2.1× |
| FastVAR | 0.7238 | 1080 | 29.93 | 2.5× |
| **Ours (R=9)** | **0.7318** | 1622 | **30.29** | 1.7× |
| **Ours (R=7)** | 0.7207 | 869 | 29.98 | **3.1×** |

### Ablation Study

| Config | GenEval | Description |
| :--- | :--- | :--- |
| Standard Inference (Full Depth) | 0.7237 | Baseline |
| Hard Pruning + Oracle Frequency | Quality Drop | Proves the fundamental limits of the hard pruning paradigm |
| Ours w/o Cyclic Rotation | Slightly Lower | Fixed ranking leads to long-term under-computation of some regions |
| Ours w/o Code Blending | Lower | Shallow tokens contribute too much, leading to uneven quality |

### Key Findings

-   The correlation between frequency estimation accuracy and generation quality is extremely weak ($r=0.138$); even an oracle mask cannot save hard pruning.
-   Generation quality of VAR models peaks before the final layer (early exit is feasible), but saturation depths vary greatly between tokens.
-   Experiments on HART show that DepthVAR is effective across different VAR architectures, demonstrating good generality.

## Highlights & Insights

-   The fundamental questioning of the frequency-driven hard pruning paradigm is very compelling—the oracle experiment directly proves that the issue lies not in frequency estimation accuracy, but in the "all-or-nothing" decision paradigm itself. This finding has important guiding significance for subsequent VAR acceleration research.
-   The analogy for bit-reversal layer allocation comes from FFT, elegantly migrating a classic signal processing technique to the layer selection problem in deep learning.
-   The training-free design allows the method to be plug-and-play for any trained VAR model, offering high practicality.

## Limitations & Future Work

-   While training-free is an advantage, it also means the model has no chance to adapt to sparse computation patterns, leaving room for further optimization.
-   Cached proxy recovery assumes feature changes between scales are locally stable, which may introduce errors in rapidly changing regions.
-   Experiments were only validated on two VAR models, Infinity and HART; applicability to newer VAR architectures remains to be verified.
-   Future directions: Introduce depth-aware regularization during training, or combine intermediate results during the autoregressive process for adaptive scheduling.

## Related Work & Insights

-   **vs FastVAR/SparseVAR**: Also VAR acceleration but uses a hard pruning paradigm; DepthVAR achieves better quality at the same speedup ratio.
-   **vs MoD (Mixture-of-Depths)**: Also a dynamic depth method but requires training a router; DepthVAR is entirely training-free.
-   **vs Early Exit**: Early exit involves all tokens exiting uniformly; DepthVAR uses per-token dynamic depth.

## Rating

-   Novelty: ⭐⭐⭐⭐ Insightful questioning of hard pruning; adaptive depth allocation is a meaningful paradigm shift.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Oracle experiments and multi-dimensional comparisons are persuasive.
-   Writing Quality: ⭐⭐⭐⭐ Clear analytical logic; the derivation from observation to method is natural.
-   Value: ⭐⭐⭐⭐ Opens a new path for VAR acceleration; the training-free characteristic enhances practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation](evatok_adaptive_length_video_tokenization_for_eff.md)
- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](../../ICLR2026/image_generation/visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [\[AAAI 2026\] HACK: Head-Aware KV Cache Compression for Efficient Visual Autoregressive Modeling](../../AAAI2026/image_generation/head-aware_kv_cache_compression_for_efficient_visual_autoreg.md)
- [\[NeurIPS 2025\] InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](../../NeurIPS2025/image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)
- [\[CVPR 2026\] Test-Time Instance-Specific Parameter Composition: A New Paradigm for Adaptive Generative Modeling](test-time_instance-specific_parameter_composition_a_new_paradigm_for_adaptive_ge.md)

</div>

<!-- RELATED:END -->
