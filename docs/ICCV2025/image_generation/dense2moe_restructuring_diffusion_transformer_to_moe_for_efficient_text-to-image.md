---
title: >-
  [Paper Note] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation
description: >-
  [ICCV 2025][Image Generation][Diffusion Models] Dense2MoE is the first paradigm for converting dense Diffusion Transformers (DiT) into sparse MoE structures. By replacing FFN layers with MoE layers and grouping Transformer blocks into Mixture of Blocks (MoB), combined with a multi-stage distillation pipeline, it compresses FLUX.1's 12B parameters to 5.2B activated parameters while preserving original performance, comprehensively outperforming pruning-based methods.
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Diffusion Models"
  - "Mixture of Experts"
  - "Model Compression"
  - "Knowledge Distillation"
  - "FLUX"
date: 2026-05-08
content_hash: 4aa119b05088af0a
---

# Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation

**Conference**: ICCV 2025
**arXiv**: [2510.09094](https://arxiv.org/abs/2510.09094)  
**Code**: N/A  
**Area**: Image Generation
**Keywords**: Diffusion Models, Mixture of Experts, Model Compression, Knowledge Distillation, FLUX

## TL;DR

Dense2MoE is the first paradigm for converting dense Diffusion Transformers (DiT) into sparse MoE structures. By replacing FFN layers with MoE layers and grouping Transformer blocks into Mixture of Blocks (MoB), combined with a multi-stage distillation pipeline, it compresses FLUX.1's 12B parameters to 5.2B activated parameters while preserving original performance, comprehensively outperforming pruning-based methods.

## Background & Motivation

Diffusion Transformers (DiT) have achieved remarkable performance in text-to-image generation, but model scale has grown dramatically—FLUX.1 has 12 billion parameters, 13.8× that of SD1.5—imposing substantial inference overhead. Existing model compression relies primarily on pruning, yet at high compression ratios pruning causes severe performance degradation, as it reduces total parameter count and fundamentally limits model capacity.

The core insight of this paper is: **can the number of activated parameters per inference be reduced without decreasing total parameter count?** The MoE architecture allows different inputs to activate different subsets of parameters, simultaneously reducing computation while preserving model capacity. Key observations are: FFN layers account for nearly 50% of total DiT parameters—making them well-suited for MoE conversion; and the importance of DiT blocks varies significantly across timesteps and prompts—making dynamic activation appropriate.

## Method

### Overall Architecture

Dense2MoE comprises two levels of sparsification and a three-stage distillation pipeline:
- **MoE Layer**: Replaces FFN with an MoE layer (shared experts + multiple ordinary experts), reducing activated parameters within each block.
- **MoB Group**: Groups consecutive Transformer blocks and dynamically routes to activate only a subset, reducing model depth.
- **Distillation Pipeline**: Enhanced initialization → MoE distillation → MoB distillation.

### Key Designs

1. **FFN→MoE Replacement**: The original FFN is replaced with a shared expert (expansion ratio $r_s$) and $n$ ordinary experts (expansion ratio $r_n$). The constraint $r = r_s + n \cdot r_n$ preserves total parameter count. At inference, each token activates only the top-$k$ ordinary experts, yielding an activated expansion ratio $r_a = r_s + k \cdot r_n$. Default configuration: $r_s=1, r_n=0.25, n=12, k=2$, giving an FFN activation compression ratio of $r_a/r = 1.5/4 = 37.5\%$ (62.5% reduction). The forward pass is:
    $y^{(t)} = \text{MLP}_s(x^{(t)}) + \sum_{i=1}^k g(x^{(t)},i) \cdot \text{MLP}_n^{(i)}(x^{(t)})$

2. **Mixture of Blocks (MoB)**: Consecutive $m$ Transformer blocks are grouped into a MoB group, with only $\kappa$ adjacent blocks activated at inference. The router leverages the global embedding $y$ from AdaLN (which fuses text and timestep conditions):
    $\text{TopK}(\alpha W_x[x^{(p)}, c^{(p)}] + (1-\alpha)W_y y, \kappa)$
   This enables routing to be explicitly conditioned on text and timestep, achieving dynamic depth compression.

3. **Three-Stage Distillation Pipeline**:

    - **Enhanced Initialization**: Weight importance is assessed via first-order Taylor criterion $\mathcal{I}_i = |\frac{\partial\mathcal{L}}{\partial w_i}w_i|$; important weights are assigned to the shared expert and the remainder distributed equally among ordinary experts. KD is then applied to the shared-expert-only model to improve initialization quality.
    - **MoE Distillation**: Shared experts are frozen; ordinary experts and gating networks are activated and trained using output distillation loss + block feature loss + load balancing loss.
    - **MoB Distillation**: A group feature loss is designed, using the output of the last block within the corresponding MoB group in the original model as teacher features, aligned with the MoB group output.

### Loss & Training

- **Output Distillation Loss**: $\mathcal{L}_{distill} = \mathbb{E}\|f_{tea} - f_{stu}\|_2^2$
- **Block Feature Loss**: $\mathcal{L}_{feature} = \sum_{l=1}^L w_l \|f_{tea}^{(l)} - f_{stu}^{(l)}\|_2^2$, with normalized feature weights ensuring stable learning across layers.
- **Load Balancing Loss**: Prevents routing collapse to a small number of experts; $\lambda_{balance} = 10^{-2}$.
- Training setup: 32× A100 GPUs, global batch size 64.
- Training data: Laion-5B + Coyo-700M + JourneyDB.
- Four sparsity levels are constructed on FLUX.1 [dev]: L (5.2B) / M (4B) / S (3.2B) / XS (2.6B).

## Key Experimental Results

### Main Results (Comparison with Pruning Methods)

| Model | Activated Params (B) | FLOPs (T) | GenEval↑ | DPG↑ | CLIP↑ | IR↑ |
|------|-----------|---------|---------|------|-------|-----|
| FLUX.1 [dev] | 11.90 | 66.00 | 0.6595 | 83.42 | 32.24 | 0.9656 |
| FLUX.1-Lite (pruning) | 8.16 | 53.15 | 0.5229 | 79.00 | 31.79 | 0.8380 |
| FLUX-Mini (pruning) | 3.18 | 17.37 | 0.3209 | 69.34 | 29.94 | 0.2151 |
| **FLUX.1-MoE-L** | **5.15** | **43.42** | **0.5702** | **81.63** | **31.39** | **0.8011** |
| **FLUX.1-MoE-S** | **3.19** | **26.43** | **0.4441** | **75.61** | **30.67** | **0.5942** |
| **FLUX.1-MoE-XS** | **2.64** | **20.26** | **0.4036** | **73.66** | **30.40** | **0.5076** |

MoE-L uses 3B fewer activated parameters than FLUX.1-Lite while achieving superior performance; MoE-S/XS substantially outperform FLUX-Mini at comparable parameter counts.

### Ablation Study (MLP Pruning vs. MoE / Depth Pruning vs. MoB)

**MLP Pruning vs. FFN-to-MoE**:

| Method | Activated Expansion Ratio | GenEval | DPG |
|------|----------|---------|-----|
| Diff-Pruning (r=1.5) | 1.5 | 0.4113 | 72.23 |
| Diff-Pruning (r=2.0) | 2.0 | 0.4888 | 77.53 |
| **MoE (r=1.5)** | **1.5** | **0.5728** | **81.24** |

MoE at 62.5% compression still outperforms pruning at 50% compression.

**Depth Pruning vs. MoB (same number of activated blocks)**:

| Method | D Blocks | S Blocks | GenEval | DPG |
|------|-----|-----|---------|-----|
| Lite | 9 | 26 | 0.0926 | 41.62 |
| BK-SDM | 9 | 26 | 0.3450 | 66.86 |
| **MoB** | **9** | **26** | **0.4956** | **76.51** |

MoB's advantage becomes more pronounced at high compression ratios—the Lite method nearly collapses under aggressive depth reduction.

### Key Findings

- **Taylor-based initialization significantly improves shared expert quality**: Consistent gains across all metrics compared to random splitting.
- **Two-stage separate training (shared → full MoE) outperforms joint training**: Independent optimization yields better results.
- **More MoB groups = better distillation**: More group feature alignment supervision layers are provided.
- **Expert specialization analysis**: The MoE in the image branch of double-stream blocks exhibits spatial specialization (different experts handle different spatial regions); expert selection patterns are similar across the same prompt category; expert selection is more concentrated during high-noise stages.
- **Supports dynamic Top-K**: The distillation pipeline design allows dynamic adjustment of activated expert count without additional training. Using only shared experts (Top-K=0) still generates reasonable images; increasing K improves detail and realism.
- HyperFLUX acceleration is compatible with MoE: MoE-L with 8-step sampling still maintains strong performance.

## Highlights & Insights

- **Paradigm Innovation**: The first work to introduce the Dense-to-MoE paradigm into diffusion models. The philosophy is "preserve capacity, reduce computation" versus pruning's "reduce capacity, reduce computation."
- **Novel MoB Concept**: Unlike token-level MoE routing, MoB performs block routing at the feature level, naturally suited to the multi-step denoising process of diffusion models.
- **Elegant Three-Stage Distillation**: Taylor initialization → shared expert distillation → MoE distillation → MoB distillation, built incrementally.
- **Practical for FLUX**: 12B → 5.2B activated parameters with no performance degradation, offering direct value for real-world deployment.

## Limitations & Future Work

- Although activated parameters are reduced by 60%, total parameter count (9B) remains large, so memory footprint may not decrease significantly.
- MoE inference requires specialized grouped GEMM kernel support (e.g., MegaBlocks); speedup on general-purpose GPUs may fall short of expectations.
- Latency improvement is relatively modest (21.2s → 17.8s, approximately 16%), as the attention component is not compressed.
- Validation is limited to FLUX.1; experiments on other DiT architectures (e.g., SD3, PixArt) are absent.
- Distillation training requires 32× A100 GPUs, incurring non-trivial cost.

## Related Work & Insights

- Dense-to-MoE works in the LLM domain (e.g., MoE-fying LLMs) provide direct inspiration for this paper.
- Mixture of Depths methods allow tokens to adaptively skip layers, conceptually related to MoB but implemented differently.
- DiT-MoE employs MoE during training to scale to 16B parameters, whereas this work converts an existing dense model.
- Step distillation methods (e.g., HyperFLUX) and model compression can be orthogonally combined.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First Dense-to-MoE approach for diffusion models; MoB design is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparisons against multiple pruning methods, detailed ablations, and expert specialization analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rich visualizations.
- Value: ⭐⭐⭐⭐⭐ Opens a new paradigm for efficient diffusion models with both practical and academic significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DICE: Staleness-Centric Optimizations for Parallel Diffusion MoE Inference](dice_staleness-centric_optimizations_for_parallel_diffusion_moe_inference.md)
- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[CVPR 2025\] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](../../CVPR2025/image_generation/dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)
- [\[CVPR 2025\] Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis](../../CVPR2025/image_generation/exploring_sparse_moe_in_gans_for_text-conditioned_image_synthesis.md)
- [\[NeurIPS 2025\] SparseDiT: Token Sparsification for Efficient Diffusion Transformer](../../NeurIPS2025/image_generation/sparsedit_token_sparsification_for_efficient_diffusion_transformer.md)

</div>

<!-- RELATED:END -->
