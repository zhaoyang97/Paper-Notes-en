---
title: >-
  [Paper Note] MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting
description: >-
  [ICLR 2026][3D Gaussian Splatting] This paper proposes MoE-GS, the first framework to introduce a Mixture-of-Experts architecture into dynamic Gaussian Splatting. Through a Volume-aware Pixel Router, it adaptively fuses multiple heterogeneous deformation priors (HexPlane / per-Gaussian / polynomial / interpolation), consistently surpassing state-of-the-art methods on the N3V and Technicolor datasets, while maintaining efficiency via single-pass rendering, gate-aware pruning, and knowledge distillation.
tags:
  - ICLR 2026
  - 3D Gaussian Splatting
  - dynamic scene
  - mixture of experts
  - novel view synthesis
  - knowledge distillation
date: 2026-05-08
content_hash: 0bda90d0b4a01eaf
---

# MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting

**Conference**: ICLR 2026
**arXiv**: [2510.19210](https://arxiv.org/abs/2510.19210)
**Code**: [https://cvsp-lab.github.io/MoE-GS](https://cvsp-lab.github.io/MoE-GS)
**Area**: 3D Vision / Dynamic Scene Reconstruction
**Keywords**: 3D Gaussian Splatting, dynamic scene, mixture of experts, novel view synthesis, knowledge distillation

## TL;DR
This paper proposes MoE-GS, the first framework to introduce a Mixture-of-Experts architecture into dynamic Gaussian Splatting. Through a Volume-aware Pixel Router, it adaptively fuses multiple heterogeneous deformation priors (HexPlane / per-Gaussian / polynomial / interpolation), consistently surpassing state-of-the-art methods on the N3V and Technicolor datasets, while maintaining efficiency via single-pass rendering, gate-aware pruning, and knowledge distillation.

## Background & Motivation

**Background**: Novel view synthesis for dynamic scenes has extended from NeRF to 3DGS, giving rise to a variety of dynamic Gaussian methods: MLP-based deformation networks (4DGaussians, E-D3DGS), polynomial motion models (STG), and interpolation-based approaches (Ex4DGS).

**Limitations of Prior Work**: Through empirical analysis, the authors identify inconsistencies at three levels: (a) **Scene-level** — different methods exhibit large performance variation across scenes, with no universally optimal approach; (b) **Spatial-level** — within a single scene, different regions are best reconstructed by different methods; (c) **Temporal-level** — the optimal method for a given video changes dynamically across frames.

**Key Challenge**: Each deformation model embeds a specific inductive bias — HexPlane favors low-motion regions, per-Gaussian embeddings suit fast and consistent optical flow, polynomials handle globally smooth motion, and interpolation addresses locally diverse motion. Real-world scenes typically contain mixed motion patterns that no single method can comprehensively cover.

**Goal**: To adaptively fuse multiple heterogeneous dynamic Gaussian experts so that the model automatically selects the most appropriate deformation prior for different spatial and temporal regions.

**Key Insight**: Drawing inspiration from the MoE architecture, each dynamic GS method is treated as an expert, and a router is designed to adaptively fuse experts at the pixel level. The key challenge is that the router must simultaneously perceive 3D volumetric information and 2D pixel information.

**Core Idea**: By splatting per-Gaussian 3D routing weights into pixel space via differentiable weight splatting, the method achieves volume-aware adaptive expert fusion.

## Method

### Overall Architecture
Stage 1: Each expert is trained independently → Stage 2: Expert parameters are frozen and the Volume-aware Pixel Router is trained → Inference: the router adaptively fuses the rendered outputs of $N$ experts. Optional post-processing: pruning or distillation.

### Key Designs

1. **Volume-aware Pixel Router**:

    - **Function**: Adaptively assigns expert weights at the pixel level while perceiving 3D volumetric information.
    - **Mechanism**: Per-Gaussian weights $\bm{w}_i^{per} = [w_i, w_i^{dir}, (t \cdot w_i^{time})]^T$ (encoding view and time dependency) are learned for each Gaussian, splatted into 2D pixel space to obtain $w_{2D}(u)$, then refined by a lightweight MLP and normalized via softmax to produce gating weights $G'_k(u)$.
    - **Design Motivation**: A Pixel Router (pure 2D MLP) lacks volumetric awareness and produces overly smoothed results; a Volume Router (directly adjusting opacity in 3D space) suffers from unstable optimization. The Volume-aware Pixel Router optimizes in 2D space (stable) while leveraging 3D features (volumetric context).
    - **Comparison**: PSNR — Pixel Router 31.12 < Volume Router 32.05 < **VA Pixel Router 33.23**

2. **Single-Pass Multi-Expert Rendering**:

    - **Function**: Merges all experts' Gaussians into a single batch, performing projection and rasterization only once.
    - **Mechanism**: Each Gaussian is augmented with a one-hot expert identity $e_j \in \mathbb{R}^K$; during alpha blending, colors are separated by expert identity: $C_k(u) = \sum_j T_j(u) \alpha_j(u) c_j \cdot (e_j)_k$.
    - **Effect**: FPS increases from 40 to 68 (Table 5).

3. **Gate-Aware Pruning**:

    - **Function**: Removes Gaussians that contribute little to the MoE output.
    - **Mechanism**: The gradient of gating weights with respect to per-Gaussian weights is accumulated: $\mathcal{E}_i = \frac{1}{|\mathcal{D}|} \sum_v \|\frac{\partial G'_k(v)}{\partial \bm{w}_i^{per}(v)}\|$; Gaussians below a threshold are pruned.
    - **Effect**: At 55% pruning, PSNR drops by only 0.02 dB, FPS increases from 44 to 83, and memory is reduced from 878 to 351 MB.

4. **Knowledge Distillation**:

    - **Function**: Transfers MoE performance to a single expert for lightweight deployment.
    - **Mechanism**: $\mathcal{L}_k^{KD} = \lambda \cdot \mathcal{L}(G'_k \cdot I_{E_k}, G'_k \cdot I_{GT}) + (1-\lambda) \cdot \mathcal{L}((1-G'_k) \cdot I_{E_k}, (1-G'_k) \cdot I_{MoE})$ — regions with high router weights are supervised by ground truth, while regions with low weights use the MoE output as pseudo-labels.
    - **Design Motivation**: When $N \geq 4$, multi-expert inference incurs significant overhead; distilling into a single expert preserves near-MoE performance.

### Loss & Training
- Training loss: L1 + SSIM (standard 3DGS loss).
- Two-stage training: Stage 1 trains each expert independently; Stage 2 freezes the experts and trains only the router.
- Experts require a smaller training budget: MoE-GS with 20% of the training budget still outperforms any single expert trained with 100%.

## Key Experimental Results

### Main Results

| Method | N3V Avg. PSNR↑ | Technicolor Avg. PSNR↑ |
|--------|----------------|------------------------|
| 4DGaussians | 31.43 | 30.79 |
| E-D3DGS | 32.33 | 33.06 |
| STG | 31.92 | 33.69 |
| Ex4DGS | 32.10 | 33.45 |
| **MoE-GS (N=3)** | **33.23** | **34.55** |
| **MoE-GS (N=4)** | **33.27** | — |

MoE-GS (N=3) outperforms the strongest single expert, E-D3DGS, by 0.9 dB PSNR.

### Ablation Study

| Router Variant | PSNR↑ | SSIM↑ |
|----------------|-------|-------|
| Pixel Router | 31.12 | 0.952 |
| Volume Router | 32.05 | 0.951 |
| **Volume-aware Pixel Router** | **33.23** | **0.954** |

| Efficiency Strategy | PSNR | FPS | Memory (MB) |
|---------------------|------|-----|-------------|
| w/o both | 32.54 | 36 | 747 |
| Full MoE-GS (N=3) | 33.23 | 68 | 270 |

### Key Findings
- **Expert diversity matters**: The gain from N=2→3 is significant (+0.69 dB), while N=3→4 yields a marginal improvement (+0.04 dB).
- **Low training budgets remain effective**: MoE-GS with 20% of the training budget (32.60) still outperforms any single expert trained with 100%.
- **Router visualization** shows that routing weights semantically correspond to motion patterns — high-motion regions tend to favor the per-Gaussian deformation expert.
- A distilled single expert can achieve performance close to the full MoE (detailed figures are provided in the appendix).

## Highlights & Insights
- **Splatting as routing**: The method cleverly reuses the 3DGS splatting mechanism for routing weight propagation — learning 3D weights while optimizing and fusing in 2D space — simultaneously achieving volumetric awareness and optimization stability.
- **Complementary heterogeneous experts**: Different deformation priors (embedding / polynomial / interpolation) each excel in distinct motion regimes; the MoE architecture is naturally suited to exploit such complementarity.
- **Complete efficiency toolbox**: From single-pass rendering and gate-aware pruning to full knowledge distillation, the framework provides a complete deployment path spanning high quality to high efficiency.

## Limitations & Future Work
- The MoE framework inherently increases parameter count and training cost ($N$ experts = $N\times$ training time, though reducible to 20%).
- The two-stage training (experts first, then router) is not joint end-to-end optimization and may not reach the global optimum.
- The expert combination is a manually selected fixed set; automated expert selection or construction is not explored.
- Validation is limited to multi-view video datasets; extension to monocular dynamic scenes remains unexplored.

## Related Work & Insights
- **vs. 4DGaussians**: 4DGaussians uses HexPlane embeddings for deformation, performing well in low-motion scenes but poorly in high-motion ones; MoE-GS can automatically select the appropriate expert.
- **vs. STG**: STG models trajectories with polynomials, yielding globally smooth results but insufficient local detail; as one MoE expert, it contributes its global prior.
- **vs. E-D3DGS**: E-D3DGS is the strongest single baseline (32.33), yet MoE-GS achieves 33.23 by fusing multiple experts.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First to introduce MoE into dynamic GS; the Volume-aware Pixel Router is an elegant design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Two standard benchmarks, multiple $N$ configurations, comprehensive ablations, efficiency analysis, and distillation evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-motivated (three-level analysis), with clear method descriptions.
- **Value**: ⭐⭐⭐⭐ — MoE + GS is a promising direction, though generalizability warrants further validation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](../../NeurIPS2025/moe/more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)

<!-- RELATED:END -->
