---
title: >-
  [Paper Note] MVAR: Visual Autoregressive Modeling with Scale and Spatial Markovian Conditioning
description: >-
  [ICLR 2026][LLM Efficiency][Visual Autoregressive] This paper proposes MVAR (Markovian Visual AutoRegressive), which introduces a scale Markov assumption (conditioning only on the adjacent preceding scale rather than all…
tags:
  - "ICLR 2026"
  - "LLM Efficiency"
  - "Visual Autoregressive"
  - "Next-Scale Prediction"
  - "Markov Assumption"
  - "Attention Optimization"
  - "Image Generation"
  - "Memory Efficiency"
date: 2026-05-08
content_hash: e07770256ffad7ed
---

# MVAR: Visual Autoregressive Modeling with Scale and Spatial Markovian Conditioning

**Conference**: ICLR 2026
**arXiv**: [2505.12742](https://arxiv.org/abs/2505.12742)  
**Code**: [Project Page](https://nuanbaobao.github.io/MVAR)  
**Area**: LLM Efficiency
**Keywords**: Visual Autoregressive, Next-Scale Prediction, Markov Assumption, Attention Optimization, Image Generation, Memory Efficiency

## TL;DR

This paper proposes MVAR (Markovian Visual AutoRegressive), which introduces a scale Markov assumption (conditioning only on the adjacent preceding scale rather than all prior scales) and spatial Markov attention (restricting neighborhood size to $k$), reducing VAR's attention complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(Nk)$. MVAR achieves comparable or superior performance on ImageNet 256×256 while reducing inference memory by 3.0–4.2×, and requires only 8 RTX 4090 GPUs for training.

## Background & Motivation

### Next-Scale Prediction Paradigm

VAR (Visual AutoRegressive modeling) reframes traditional next-token prediction as next-scale prediction: images are encoded as multi-scale residual token maps $\mathcal{R} = (r_1, r_2, \dots, r_L)$ and generated autoregressively from coarse to fine. This better preserves the 2D structure of images compared to raster-scan order, significantly improving generation efficiency and quality.

### Redundancy in Existing VAR

Through attention weight analysis, the authors identify two key sources of redundancy:

**Scale Redundancy**: Attention weights at each scale are heavily concentrated on the **immediately preceding scale**, with distant scales receiving negligible attention. Nevertheless, VAR conditions each scale on all prior scales.

**Spatial Redundancy**: Inter-scale attention exhibits a **diagonal-dominant pattern** (resembling the local connectivity of convolutions), where each token primarily attends to spatially neighboring tokens rather than all tokens.

These two redundancies lead to unnecessary GPU memory consumption and computational waste.

## Method

### Overall Architecture

MVAR is built upon two Markov assumptions:

1. **Scale Markov**: The current scale $r_l$ depends only on the immediately preceding scale $r_{l-1}$ (rather than all $r_1, \dots, r_{l-1}$).
2. **Spatial Markov**: Each token attends only to a local neighborhood of size $k$ in the adjacent scale.

### Scale Markovian Conditional Modeling

The conventional full-dependency conditional probability:

$$p(r_1, \dots, r_L) = \prod_{l=1}^{L} p(r_l | r_1, \dots, r_{l-1})$$

is simplified to adjacent-scale dependency:

$$p(r_1, \dots, r_L) = p(r_1) \prod_{l=2}^{L} p(r_l | \eta_k(r_{l-1}))$$

where $\eta_k(\cdot)$ denotes the spatial neighborhood restriction of size $k$.

This decouples inter-scale dependencies and enables **parallel training**: for 256×256 image generation, $r_1$ through $r_8$ are trained in parallel (via diagonal causal masks), while $r_9$ and $r_{10}$ are handled separately using a custom CUDA kernel.

### Spatial Markov Attention

For each token $i$, attention scores are computed over only $k$ nearest neighbors:

$$\mathbf{S}_i^l = [\mathbf{Q}_i^l (\mathbf{K}_{\eta_k^i(1)}^l)^T, \dots, \mathbf{Q}_i^l (\mathbf{K}_{\eta_k^i(k)}^l)^T]$$

The final output is:

$$\text{SA}_i^l = \text{SoftMax}(\mathbf{S}_i^l / \sqrt{d}) \, \mathbf{V}_i^l$$

reducing computational complexity from $\mathcal{O}(N_l^2)$ to $\mathcal{O}(N_l k)$.

### Key Designs

- **No KV cache is required at inference** (since only the preceding scale is needed, it can be discarded immediately after generation).
- During training, $r_1$–$r_8$ are processed in parallel via diagonal causal masks.
- $r_9$ and $r_{10}$ (highest resolution) are handled separately using a Neighborhood Attention CUDA kernel.
- A neighborhood size of $k = 7 \times 7$ is selected as the optimal trade-off between performance and efficiency.

### Loss & Training

Standard cross-entropy loss is computed independently at each scale as $\text{loss}_l$, consistent with VAR.

## Key Experimental Results

### Main Results: Class-Conditional Generation on ImageNet 256×256

| Model Type | Model | FID↓ | IS↑ | Precision↑ | Recall↑ | Params |
|-----------|-------|------|-----|-----------|---------|--------|
| GAN | StyleGAN-XL | 2.30 | 265.1 | 0.78 | 0.53 | 166M |
| Diffusion | DiT-XL/2 | 2.27 | 278.2 | 0.83 | 0.57 | 675M |
| Token-wise AR | VQGAN-re | 5.20 | 280.3 | — | — | 1.4B |
| Scale-wise | VAR-d16 | 3.55 | 280.4 | 0.84 | 0.51 | 310M |
| **Scale-wise** | **MVAR-d16** | **3.09** | **285.5** | **0.85** | 0.51 | 310M |

Trained from scratch, MVAR-d16 reduces FID by 0.46 and improves IS by 5.1 over VAR-d16.

### Pre-trained Fine-tuning Comparison

| Model | Inference Time↓ | KV Cache↓ | Inference Memory↓ | Training Speedup | FID↓ | IS↑ |
|-------|----------------|----------|------------------|-----------------|------|-----|
| VAR-d16 | 0.34s | 5704M | 10882M | — | 3.55 | 280.4 |
| MVAR-d16† | 0.27s | **0** | **3846M (2.8×)** | 1.6× | 3.40 | 297.2 |
| VAR-d20 | 0.52s | 8500M | 16244M | — | 2.95 | 302.6 |
| MVAR-d20† | 0.45s | **0** | **5432M (3.0×)** | 1.7× | 2.87 | 295.3 |
| VAR-d24 | 0.81s | 12240M | 23056M | — | 2.33 | 312.9 |
| MVAR-d24† | 0.71s | **0** | **7216M (3.2×)** | — | 2.23 | 300.1 |

Fine-tuned MVAR achieves 2.8–3.2× memory reduction across all model sizes while consistently improving FID.

### Ablation Study

**Effect of Number of Scale Conditioning Prefixes**:

| # Prefix Scales | KV Cache | Memory | GFLOPs | FID↓ | IS↑ |
|----------------|----------|--------|--------|------|-----|
| All (VAR) | 5704M | 10882M | 43.61 | 4.84 | 227.1 |
| 3 | 3565M | 9518M | 41.54 | 4.86 | 220.3 |
| 2 | 2147M | 9262M | 40.15 | 5.01 | 208.8 |
| **1 (MVAR)** | **0** | **4199M (2.6×)** | 37.84 | **4.35** | **240.6** |

Conditioning on only the single adjacent scale yields the best results, improving IS by 13.5, eliminating KV cache, and reducing memory by 2.6×.

**Effect of Neighborhood Size $k$**:
- $k = 3\times3$: Higher FID due to information loss from an overly small neighborhood.
- $k = 7\times7$: Optimal balance point.
- $k = 9\times9$: Diminishing returns.

### Key Findings

1. Scale redundancy is real and can be safely removed — conditioning on only the adjacent scale actually improves generation quality.
2. Spatial Markov attention yields the greatest gains at high-resolution scales ($r_9$, $r_{10}$), which account for 60% of total computation.
3. Completely eliminating KV cache simplifies inference and removes the need for complex cache management.

## Highlights & Insights

1. **From empirical observation to principled design**: The Markov properties are not assumed a priori but are motivated by detailed attention weight analysis that reveals redundancy.
2. **Democratizing training**: Only 8 RTX 4090 GPUs are required for training (compared to more expensive hardware for VAR), substantially lowering the barrier to visual generation research.
3. **Significance of eliminating KV cache**: Beyond memory savings, this simplifies the engineering complexity of inference systems.
4. **Less is more**: Reducing scale dependencies focuses the model on locally refined, more critical information, ultimately improving generation quality.
5. **Correspondence with sparse attention in NLP**: Spatial Markov attention validates the effectiveness of local attention mechanisms in the visual domain.

## Limitations & Future Work

1. **Validation limited to ImageNet 256×256**: Evaluation at higher resolutions and on more complex datasets remains limited.
2. **Engineering burden of custom CUDA kernels**: High-resolution scales ($r_9$, $r_{10}$) require hand-written kernels.
3. **Fixed neighborhood size $k$**: The same $k$ is applied across all scales, whereas the optimal $k$ may differ per scale.
4. **Text-conditional generation not explored**: Only class-conditional generation is demonstrated; effectiveness in text-to-image scenarios remains to be verified.

## Related Work & Insights

- **VAR** (Tian et al., 2024): The direct baseline for MVAR; validates the next-scale paradigm but exhibits redundancy.
- **MaskGIT** (Chang et al., 2022): An alternative approach that generates multiple tokens in parallel.
- **Neighborhood Attention** (Hassani & Shi, 2022): The local attention CUDA implementation directly adopted by MVAR.
- **Insight**: The Markov assumption framework is generalizable to other autoregressive visual models, such as simplifying inter-frame dependencies in video generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Elegantly introduces the Markov assumption into the next-scale paradigm.
- **Technical Depth**: ⭐⭐⭐⭐ — Balances theoretical analysis and engineering implementation with clear complexity analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers from-scratch training, fine-tuning, and multi-dimensional ablations, though dataset variety is limited.
- **Value**: ⭐⭐⭐⭐⭐ — Substantially reduces training and inference costs; trainable on 8×4090.
- **Overall Recommendation**: ⭐⭐⭐⭐ — Excellent practically oriented work that maintains theoretical elegance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration](../../CVPR2026/llm_efficiency/sparvar_exploring_sparsity_in_visual_autoregressive_modeling_for_training-free_a.md)
- [\[ICML 2026\] Proxy Compression for Language Modeling](../../ICML2026/llm_efficiency/proxy_compression_for_language_modeling.md)
- [\[ICML 2026\] OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](../../ICML2026/llm_efficiency/oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)
- [\[ACL 2026\] Native Hybrid Attention for Efficient Sequence Modeling](../../ACL2026/llm_efficiency/native_hybrid_attention_for_efficient_sequence_modeling.md)
- [\[CVPR 2026\] StoryTailor: A Zero-Shot Pipeline for Action-Rich Multi-Subject Visual Narratives](../../CVPR2026/llm_efficiency/storytailora_zero-shot_pipeline_for_action-rich_multi-subject_visual_narratives.md)

</div>

<!-- RELATED:END -->
