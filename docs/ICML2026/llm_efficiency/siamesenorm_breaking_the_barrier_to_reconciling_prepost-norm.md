---
title: >-
  [Paper Note] SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm
description: >-
  [ICML 2026][LLM Efficiency][SiameseNorm] Addressing the structural contradiction where Pre-Norm and Post-Norm cannot coexist within a single-stream architecture, the authors propose SiameseNorm…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "SiameseNorm"
  - "Pre-Norm"
  - "Post-Norm"
  - "Dual-stream residual"
  - "Training stability"
date: 2026-05-08
content_hash: 860054373cbb8329
---

# SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm

**Conference**: ICML 2026  
**arXiv**: [2602.08064](https://arxiv.org/abs/2602.08064)  
**Code**: https://github.com/Qwen-Applications/SiameseNorm  
**Area**: LLM Efficiency / Transformer Architecture / Normalization  
**Keywords**: SiameseNorm, Pre-Norm, Post-Norm, Dual-stream residual, Training stability

## TL;DR
Addressing the structural contradiction where Pre-Norm and Post-Norm cannot coexist within a single-stream architecture, the authors propose SiameseNorm, a dual-stream residual architecture. It maintains an unnormalized stream as the Pre-Norm identity gradient highway and a normalized stream for Post-Norm representation control. By coupling both streams via shared residual blocks, it consistently outperforms Pre-Norm baselines across 400M~15B dense/MoE language models, ViT, and DiT with negligible overhead.

## Background & Motivation

**Background**: Modern Transformers (GPT-3, LLaMA, DeepSeek-V3, Qwen3, ViT) almost exclusively use Pre-Norm because it places LayerNorm inside the residual branch, leaving the main path as a clean identity connection. This naturally provides a "gradient highway," enabling stable training for networks with hundreds of layers. Post-Norm places LN after the residual addition, periodically normalizing the main path representation, which offers higher per-layer expressiveness and performance but suffers from extreme training instability.

**Limitations of Prior Work**: While Pre-Norm enables stable training, recent studies have identified a "deep layer collapse" issue—removing several deep layers barely impacts performance. This reflects that the Pre-Norm main path representation $\|X_i\|_2$ grows nearly exponentially with depth (Paper Fig. 2(a) shows a 1.3B model reaching magnitudes of $\sim 10^3$ at the deepest layer), while each layer $F_i$ receives normalized inputs (constant magnitude). Residual updates in deep layers become increasingly "diluted" relative to the massive main path, leading to low utilization and limited effective depth. Post-Norm, however, requires multiplying by the Jacobian of LN $\mathbf{J}_{\mathrm{LN}}$ at each layer, making it prone to gradient explosion or disappearance during backpropagation, leading to divergence under high learning rates.

**Key Challenge**: The two paradigms demand conflicting properties on the **same residual main path**: Pre-Norm requires an "unnormalized identity path for gradient stability," while Post-Norm requires a "normalized main path for representation scale control." Existing hybrid schemes (HybridNorm, Mix-LN, SpanNorm) assign different paradigms to different layers, yet all updates accumulate on the same path, failing to satisfy both requirements simultaneously. These methods typically diverge under high learning rates ($\eta=10^{-3}$ or $2\times 10^{-3}$).

**Goal**: To design an architecture that enjoys both the optimization stability of Pre-Norm and the representation control of Post-Norm, while remaining fully compatible with existing Pre-Norm training recipes (learning rate, warm-up, initialization) without requiring re-tuning.

**Key Insight**: Since the two requirements are irreconcilable in a single stream, they are **structurally decoupled into two streams**. By maintaining two independently evolving residual states $X_i$ and $Y_i$, one acts as a Post-Norm normalized main path and the other as a Pre-Norm identity path. They share the same residual block $F_i$, allowing $F_i$ to receive gradient signals from both paths with zero parameter overhead.

**Core Idea**: Replacing the "normalization position debate in single-stream" with "Siamese dual-stream"—where two streams share computation modules, each fulfilling a specific normalization semantic.

## Method

### Overall Architecture
SiameseNorm maintains two coupled residual streams. Following Embedding, two streams are initialized as $X_0=Y_0=h$. For each layer $i=0,\dots,N-1$, the forward pass is (see Paper Algorithm 1):

1. Sum the two streams after aligning them in normalized space to serve as input for the shared residual block: $O = F_i(X_i + \mathrm{LN}_i^Y(Y_i))$
2. Update the Post-Norm stream (normalized main path): $X_{i+1} = \mathrm{LN}_i^X(X_i + O)$
3. Update the Pre-Norm stream (identity main path): $Y_{i+1} = Y_i + O$

The final output is $X_N + \mathrm{LN}_{\mathrm{final}}(Y_N)$. Both streams share the same $F_i$ (Attention/MLP), adding only two lightweight normalization operators $\mathrm{LN}_i^X$ and $\mathrm{LN}_i^Y$. The increase in parameters and FLOPs is $<0.1\%$. On a 15B MoE, training speed decreases by only 0.5%, and activation memory increases by only 2%.

### Key Designs

1. **Dual-stream Coupled Residual Topology (Core Innovation)**:
    - **Function**: Physically separates the Pre-Norm identity gradient path and Post-Norm normalized main path into two streams $X$ and $Y$, coupled via a shared residual block $F_i$.
    - **Mechanism**: Let $S_i=[X_i,Y_i]^\top$. The dual-stream transition Jacobian $\partial S_{j+1}/\partial S_j$ has diagonal blocks exactly equal to the pure Pre-Norm transition $\mathbf{I}+\mathbf{J}_{F_j}\mathbf{J}_{\mathrm{LN}_j^Y}$ and the pure Post-Norm transition $\mathbf{J}_{\mathrm{LN}_j^X}(\mathbf{I}+\mathbf{J}_{F_j})$. This means $F_i$ simultaneously receives "identity highway" gradients from the $Y$ stream and "normalized main path" gradients from the $X$ stream.
    - **Design Motivation**: Resolves the fundamental contradiction of conflicting normalization semantics on a single path. This topology can also revert to Pre-Norm ($\mathrm{LN}^X=0$), Post-Norm ($\mathrm{LN}^Y=0$), or layer-wise hybrid (Mix-LN) through simple parameter configuration, covering a broader design space.

2. **Depth-wise Scaling**:
    - **Function**: Multiplies updates injected into the Post-Norm stream ($X$ stream) by $1/\sqrt{l+1}$ (where $l$ is the layer index) to balance the relative contributions of the two streams.
    - **Mechanism**: Inspired by DeepNorm. In deep layers, the Pre-Norm stream $\|Y_i\|_2$ grows naturally while the Post-Norm stream $\|X_i\|_2$ remains bounded by LN. Consequently, the same shared update $O$ is too small for $Y$ and too large for $X$. Decaying the update amplitude into $X$ reduces optimization sensitivity for the Post-Norm stream in deep layers, ensuring compatibility with Pre-Norm training recipes.
    - **Design Motivation**: Allows SiameseNorm to be a "drop-in" replacement for Pre-Norm, avoiding the burden of new hyperparameter tuning—a key aspect of its practical value.

3. **Normalized Input**:
    - **Function**: Applies another LN to the aggregated representation $X_i + \mathrm{LN}_i^Y(Y_i)$ before it enters the shared residual block $F_i$.
    - **Mechanism**: Ensures that Attention/MLP modules always receive stable, normalized input distributions, aligning with standard Transformer training conventions. Ablations (Table 3) show removing this step increases PPL from 10.43 up to 10.51~10.88.
    - **Design Motivation**: Acts as necessary "glue" to maintain compatibility with modern Transformer training habits, even though individual sub-streams are already normalized.

### Loss & Training
The architecture follows the Pre-Norm training recipe exactly: standard AdamW, cosine learning rate, 2K-step warm-up, and no additional hyperparameters. All $\mathrm{LN}$ scales are initialized to 1.0. Language modeling is based on OLMo + FineWeb-Edu trained from scratch, and MoE experiments use OLMoE, totaling over 60,000 A100 hours.

## Key Experimental Results

### Main Results: 1.3B Dense Model, Comparison with 8 Normalization Schemes at Different LRs

| Learning Rate $\eta$ | Training Tokens | Pre-Norm PPL | HybridNorm PPL | SpanNorm PPL | SiameseNorm PPL | Avg. Downstream Score |
|----------------------|-----------------|--------------|----------------|--------------|-----------------|-----------------------|
| $4\times 10^{-4}$ Conservative | 100B | 11.21 | 10.91 | 11.00 | **10.57** | 52.26 |
| $1\times 10^{-3}$ High | 100B | 10.84 | **diverge** | 10.86 | **10.43** | 53.53 |
| $2\times 10^{-3}$ Aggressive | 100B | 10.89 | **diverge** | **diverge** | **10.48** | 55.63 |
| $2\times 10^{-3}$ Aggressive | 350B | 9.67 | — | — | **9.42** | 58.70 |
| MoE 15A2B $\eta=10^{-3}$ | 100B | 7.92 | — | — | **7.76** | 63.07 |

Key Observation: HybridNorm and SpanNorm are competitive at conservative LRs but diverge when the LR increases. SiameseNorm is the only method that converges stably at all LRs while maintaining the lowest PPL. At an aggressive LR with 100B tokens, SiameseNorm's Arithmetic accuracy reaches 39.6%, a 41% relative gain over Pre-Norm's 27.0%, reflecting the reasoning benefits of Post-Norm representation control.

### Cross-depth and Cross-modal Generalization (390M Parameters, 12B tokens, $\eta=10^{-3}$)

| Configuration | Pre-Norm | SiameseNorm | Gain |
|---------------|----------|-------------|------|
| 10 Layers / d=1280 | 17.47 PPL | 16.15 | -1.32 |
| 17 Layers / d=1024 | 17.23 | 15.69 | -1.54 |
| 33 Layers / d=768 | 17.29 | **15.64** | -1.65 |
| 80 Layers / d=512 | 18.02 | 15.98 | **-2.04** |
| DeiT-S (ImageNet) | 79.8 Acc | **81.3** | +1.5 |
| DiT-L/4 (FID) | 45.21 | **41.34** | -3.87 |

Pre-Norm begins degrading at 33 layers, while SiameseNorm achieves its best PPL at that depth. The gain increases with depth, validating that SiameseNorm mitigates Pre-Norm's "deep dilution" problem.

### Ablation Study (Table 3, $\eta=10^{-3}$)

| Normalized Input | Depth-Scaling | Topology | Avg. PPL |
|------------------|---------------|----------|----------|
| ✓ | × | Original (HybridNorm) | **diverge** |
| ✓ | ✓ | Original | 10.65 |
| ✓ | × | ResiDual | 11.68* |
| × | × | Siamese | 10.88 |
| ✓ | × | Siamese | 10.68 |
| × | ✓ | Siamese | 10.51 |
| ✓ | ✓ | Siamese | **10.43** |

### Key Findings
- **Siamese topology is the core of stability**: HybridNorm diverges without Depth-Scaling, while the Siamese topology reaches 10.68 PPL even without it and can train with 0 warm-up.
- **Synergy between Depth-wise Scaling and Siamese**: While scaling is a known technique, it further drops PPL from 10.68 to 10.43 within the SiameseNorm framework.
- **Gradient statistics validate the mechanism**: Under high LR, HybridNorm gradient norm spikes exceed 100, whereas SiameseNorm and Pre-Norm remain stable below 0.5.
- **Improved deep layer utilization**: Pruning deep layers in Pre-Norm results in minimal performance drops, whereas doing so in SiameseNorm causes significant degradation, proving deep layers are effectively utilized.

## Highlights & Insights
- **Methodological value of "Structural Decoupling"**: When an open problem (Pre vs Post-Norm) is irreconcilable in a single-stream framework, it is better to split conflicting requirements into two physical paths than to optimize normalization positions. This "coupled dual-stream" approach can be applied to other trade-offs like BatchNorm vs LayerNorm.
- **Jacobian derivation as a "Compass"**: The block matrix derivation of $\partial S_{j+1}/\partial S_j$ allows the architecture to be "configured as needed"—re-zeroing $\mathrm{LN}^X$ or $\mathrm{LN}^Y$ covers the entire hybrid normalization design space.
- **Product value in drop-in compatibility**: In industrial-scale models, not needing to retune hyperparameters is a hard requirement for adoption. The $1/\sqrt{l+1}$ scaling delivers this by ensuring the Post-Norm stream works with standard Pre-Norm recipes.

## Limitations & Future Work
- SiameseNorm increases activation memory by ~2%, which might be a bottleneck in VRAM-constrained ultra-large models, necessitating activation checkpointing.
- The largest scale tested is 15B MoE / 100B tokens; performance on 70B+ dense models or trillion-token scales remains to be verified.
- Depth-wise Scaling ($1/\sqrt{l+1}$) is empirical; a theoretical analysis of optimal scaling factors is absent.
- Experiments on ViT/DiT are relatively small-scale (DeiT-S, DiT-L/4); scalability to SD3-level DiTs or ViT-22B is not yet proven.

## Related Work & Insights
- **vs HybridNorm / SpanNorm / Mix-LN**: These stack different normalization semantics on a single path and fail at high LRs; SiameseNorm avoids this via physical separation.
- **vs ResiDual**: ResiDual uses dual branches but adds them only at the end; SiameseNorm couples both streams at every layer via shared $F_i$.
- **vs Hyper-Connections**: Hyper-Connections relies on Pre-Norm-biased initialization for stability, whereas SiameseNorm achieves stability through topology even with standard initialization.
- **vs DeepNorm**: DeepNorm uses residual scaling for single-stream Post-Norm; SiameseNorm incorporates this into a dual-stream framework to make Post-Norm-style flows usable under Pre-Norm recipes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reframes normalization as a topological design rather than a local improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 400M / 1.3B / 15B MoE, ViT, DiT, multiple LRs, and depths (60K A100 hours).
- Writing Quality: ⭐⭐⭐⭐ Clear Jacobian derivations and Algorithm 1; however, some figure descriptions (Fig. 2, 4) are fragmented.
- Value: ⭐⭐⭐⭐⭐ Fully compatible with Pre-Norm recipes, <2% overhead, already engineered by the Qwen team; extremely low barrier for industrial adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)
- [\[ACL 2026\] Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](../../ACL2026/llm_efficiency/breaking_block_boundaries_anchor-based_history-stable_decoding_for_diffusion_lar.md)
- [\[NeurIPS 2025\] Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search](../../NeurIPS2025/llm_efficiency/jet-nemotron_efficient_language_model_with_post_neural_architecture_search.md)
- [\[ICML 2026\] Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](efficient_training-free_multi-token_prediction_via_embedding-space_probing.md)
- [\[ICML 2026\] L$^3$: Large Lookup Layers](l3_large_lookup_layers.md)

</div>

<!-- RELATED:END -->
