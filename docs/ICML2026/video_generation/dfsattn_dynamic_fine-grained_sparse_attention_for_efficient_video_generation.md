---
title: >-
  [Paper Note] DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation
description: >-
  [ICML 2026][Video Generation][Sparse Attention] DFSAttn achieves a **2.1× end-to-end speedup** with quality comparable to full attention through **3D Hilbert curve reordering** + **hierarchical block scoring** + **adapti…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Sparse Attention"
  - "Hilbert Curve"
  - "Dynamic Masking"
date: 2026-05-08
content_hash: 2ea6f81f0c88aa90
---

# DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation

**Conference**: ICML 2026  
**arXiv**: [2605.23445](https://arxiv.org/abs/2605.23445)  
**Code**: To be confirmed  
**Area**: Video Generation / Diffusion Models / Model Compression  
**Keywords**: Sparse Attention, Video Generation, Hilbert Curve, Dynamic Masking

## TL;DR
DFSAttn achieves a **2.1× end-to-end speedup** with quality comparable to full attention through **3D Hilbert curve reordering** + **hierarchical block scoring** + **adaptive mask caching**—addressing the core issue of quality degradation in block-sparse attention at high sparsity rates (>80%).

## Background & Motivation

**Background**: Video Diffusion Transformers (DiT) achieve high-fidelity video generation via 3D full attention, but its quadratic complexity creates a severe computational bottleneck—generating 129 frames of 720p video takes approximately 30 minutes on an H100 GPU for HunyuanVideo. Block-sparse attention is a common direction for complexity reduction and naturally aligns with GPU-efficient kernels like FlashAttention.

**Limitations of Prior Work**: Current block-sparse attention methods (static like radial sparsity, dynamic like XAttention) suffer from severe quality degradation at high sparsity rates (80%), failing to maintain generation quality while providing significant acceleration. The fundamental cause is that the **coarse-grained block-level representation** used by existing methods **mismatches the dynamic, fine-grained attention sparsity patterns** inherent in DiT.

**Key Challenge**: On one hand, GPU-efficient computation requires block-level sparsity for alignment with FlashAttention; on the other hand, DiT attention patterns exhibit dynamic and fine-grained sparse features, with numerous local important interactions scattered across the attention map. Applying coarse-grained block operations directly to these fine-grained patterns inevitably loses critical dependencies.

**Goal**: To capture and utilize the fine-grained, dynamic sparse patterns within DiT while maintaining the efficiency of GPU block-level execution.

**Key Insight**: Derived from two key observations—(1) the sparsity patterns in DiT attention maps are highly heterogeneous across layers and heads, rendering static patterns ineffective; (2) the effectiveness of block-sparse attention improves monotonically as diffusion steps evolve (early steps are noise-dominated, while late steps highlight structures), suggesting different sparsity budgets for different steps.

**Core Idea**: Through a three-layer progressive design—**Global Hilbert Reordering to amplify inter-block similarity differences** + **Hierarchical Block Scoring to refine semantic heterogeneity** + **Adaptive Mask Caching to dynamically adapt to the diffusion process**—the method preserves block-level execution efficiency while implicitly inducing fine-grained sparsity.

## Method

### Overall Architecture
(1) Encode 3D latent video representations as 1D token sequences with text conditions; (2) Reorder tokens using 3D Hilbert curves to bring spatio-temporally adjacent tokens closer in the sequence; (3) Estimate block importance via hierarchical block scoring to compute sparse masks (cached and reused at fixed intervals); (4) Apply sparse masks to SparseFlashAttention and restore the original order for the output.

### Key Designs

1. **3D Hilbert Curve Token Reordering**:

    - **Function**: Maps spatio-temporally adjacent tokens from the 3D video tensor to adjacent positions in a 1D sequence, amplifying similarity differences between blocks.
    - **Mechanism**: Utilizes the locality-preserving property of Hilbert space-filling curves to project $(f, h, w)$ dimension tokens to 1D via a Hilbert mapping $\mathcal{P}$. When two tokens are close in the original 3D space, their distance remains small in the 1D sequence after reordering. tokens in the same block thus tend to come from coherent video regions, while different blocks capture distinct regions, significantly increasing block-level consistency. Experiments show reordering reduces intra-block variance for queries and keys by approximately 20%.
    - **Design Motivation**: Standard row-major flattening destroys 3D locality. Applying block-level sparsity to reordered sequences induces fine-grained, interconnected sparse patterns in the original space. The overhead is extremely low (approx. 2% runtime for 120K tokens).

2. **Hierarchical Block Scoring Mechanism**:

    - **Function**: Replaces single block-level representations with multi-granularity aggregation to generate more accurate block importance estimates.
    - **Mechanism**: Blocks are first decomposed into smaller sub-blocks (size $B_s$). A sub-block attention score matrix $\hat{A}$ is computed and aggregated into block-level scores: $\hat{S}_{uv} = \sum_{i' \in \mathcal{B}_u} \sum_{j' \in \mathcal{B}_v} \hat{A}_{i' j'}$. Through this hierarchical aggregation, each block-level score captures not just average features but contributions from multiple semantic centers. For a query block $\mathcal{B}_u$, the $\gamma M$ highest-scoring key blocks are selected ($\gamma$ being the sparsity rate) to construct the sparse mask $\mathcal{M}$.
    - **Design Motivation**: Coarse-grained averaging assumes semantic uniformity within a block, but DiT blocks often contain multiple semantic clusters. Hierarchical scoring avoids the bottleneck of single representations; sub-block size 16 achieves optimal quality ($PSNR$ 29.378) without additional overhead.

3. **Adaptive Sparse Mask Caching + Dynamic Budget Allocation**:

    - **Function**: Reuses sparse masks across diffusion steps and dynamically adjusts the sparsity rate.
    - **Mechanism**: Observations show that block-sparse attention effectiveness rises monotonically as diffusion steps progress—early steps are noise-dominated with diffuse attention, while later steps approach the data manifold with concentrated attention. The sparsity budget is adapted accordingly: initialize $\gamma_0 = 0.3$, decreasing by 0.1 every 25% of steps, leading to an average sparsity rate of ~80% in the remaining 75% of steps. Masks are recalculated and cached at fixed intervals (every 25% of steps). Although masks are cached, sparse attention outputs are recalculated at every step to ensure dynamic evolution of token representations.
    - **Design Motivation**: Avoids computation overhead of per-step mask calculation. Dynamic budget allocation ensures sufficient attention range in early steps. Figure 6 shows the adaptive scheme achieves 3-4 points higher PSNR than fixed schemes at the same latency.

## Key Experimental Results

### Main Results

| Dataset | Metric | Standard | RadialAttention | SVG | SVG2 | **DFSAttn** |
|--------|------|------|----------|------|------|----------|
| Wan2.1 | PSNR ↑ | — | 17.405 | 17.393 | 18.034 | **22.370** |
| Wan2.1 | SSIM ↑ | — | 0.624 | 0.612 | 0.640 | **0.764** |
| Wan2.1 | LPIPS ↓ | — | 0.357 | 0.362 | 0.338 | **0.183** |
| Wan2.1 | Sparsity | 0% | 73.78% | 65.71% | 68.19% | **78.51%** |
| Wan2.1 | Gain ↑ | 1.00× | 1.72× | 1.75× | 1.90× | **1.79×** |
| HunyuanVideo | PSNR ↑ | — | 20.897 | 26.825 | 28.577 | **29.381** |
| HunyuanVideo | SSIM ↑ | — | 0.750 | 0.853 | 0.864 | **0.898** |
| HunyuanVideo | Gain ↑ | 1.00× | 1.74× | 1.92× | 2.20× | **2.10×** |

Ours surpasses SVG by 29% on Wan2.1 (PSNR 22.37 vs 17.39) and SVG2 by 3% on HunyuanVideo (PSNR 29.38 vs 28.58).

### Ablation Study

| Configuration | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Description |
|------|--------|--------|--------|------|
| Raster Scan | 27.794 | 0.874 | 0.124 | Baseline |
| 2D Hilbert (per frame) | 29.265 | 0.893 | 0.090 | Ignores inter-frame coherence |
| 3D Block (Block3D) | 29.156 | 0.897 | 0.090 | Block-level recursive, destroys global locality |
| **3D Hilbert (Ours)** | **29.378** | **0.901** | **0.087** | Global spatio-temporal preservation, optimal |

### Key Findings
- Global 3D Hilbert reordering outperforms other strategies, demonstrating the necessity of preserving both spatial and temporal locality simultaneously.
- DFSAttn significantly outperforms baselines at high sparsity rates (> 80%) in PSNR, SSIM, and LPIPS, achieving 1.79× / 2.10× acceleration while maintaining quality.
- VBench composite scores are close to full attention, indicating the overall video quality is fully preserved.

## Highlights & Insights
- **Theory-Practice Integration**: A theoretical lower bound for block-sparse attention effectiveness (Theorem 4.4) is derived, explicitly linking block selection accuracy to inter-block similarity differences and semantic heterogeneity, guiding the design of the three core components.
- **Clever Spatial Transformation**: Using the locality-preserving property of Hilbert curves for global reordering not only amplifies block differences and refines representations but also implicitly induces fine-grained sparsity in the original space—block-level sparsity applied to the reordered sequence manifests as interconnected fine-grained patterns in the original 2D/3D space.
- **Dynamic Cross-step Adaptation**: Leveraging the progressive evolution of attention structures during diffusion to design an adaptive sparsity budget provides a general reference for other work using block sparsity to accelerate diffusion models.

## Limitations & Future Work
- Fixed block size (128) lacks adaptive adjustment for varying video resolutions or frame counts; exploring dynamic adjustments based on content or resolution is a potential direction.
- While cross-head heterogeneity is mentioned, the mask is shared across all attention heads, potentially losing unique sparse characteristics of specific heads.
- Synergy with other acceleration techniques (e.g., coordination ratio with AdaCache) is not detailed and warrants deeper exploration.

## Related Work & Insights
- **vs Static Sparsity (RadialAttention)**: Fixed patterns struggle to adapt to dynamic attention; DFSAttn's dynamically constructed masks result in a 4.5 point higher PSNR.
- **vs Coarse Dynamic Methods (SVG2)**: Coarse block averaging is hindered by semantic mixing within blocks; DFSAttn's hierarchical aggregation refines estimates for a 1.2 point higher PSNR, further enhanced by Hilbert reordering.
- **vs Fine-Grained Sparse Kernels (FG-Attn)**: FG-Attn designs custom fine-grained sparse CUDA kernels; DFSAttn adopts a kernel-free approach, providing better portability and fewer dependencies.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Starting from a theoretical lower bound to guide a three-layer progressive design, the combination of Hilbert reordering and hierarchical aggregation is both innovative and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on two SOTA models with multi-dimensional metrics, detailed ablations, and comparisons against three strong baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, tight integration of theory and methodology, and highly informative figures.
- Value: ⭐⭐⭐⭐⭐ Directly addresses practical bottlenecks in video generation; 2.1× acceleration while maintaining quality has immediate engineering value; the theoretical bound serves as a major reference for other diffusion acceleration research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] VEDA: Scalable Video Diffusion via Distilled Sparse Attention](veda_scalable_video_diffusion_via_distilled_sparse_attention.md)
- [\[ICML 2026\] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)
- [\[ICML 2026\] Lightning Unified Video Editing via In-Context Sparse Attention](lightning_unified_video_editing_via_in-context_sparse_attention.md)
- [\[ICML 2026\] Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)
- [\[NeurIPS 2025\] VORTA: Efficient Video Diffusion via Routing Sparse Attention](../../NeurIPS2025/video_generation/vorta_efficient_video_diffusion_via_routing_sparse_attention.md)

</div>

<!-- RELATED:END -->
