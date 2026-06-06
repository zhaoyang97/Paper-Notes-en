---
title: >-
  [Paper Note] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention
description: >-
  [ICML 2026][Video Generation][Autoregressive Video Generation] Light Forcing is the first sparse attention scheme customized for autoregressive (AR) video diffusion models—Chunk-Aware Growth (CAG) dynamically allocates s…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Autoregressive Video Generation"
  - "Sparse Attention"
  - "Chunk-Aware Growth"
  - "Hierarchical Sparsity"
date: 2026-05-08
content_hash: 018b3dd629bea812
---

# Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention

**Conference**: ICML 2026  
**arXiv**: [2602.04789](https://arxiv.org/abs/2602.04789)  
**Code**: To be confirmed  
**Area**: Video Generation / Diffusion Models / Model Acceleration  
**Keywords**: Autoregressive Video Generation, Sparse Attention, Chunk-Aware Growth, Hierarchical Sparsity

## TL;DR
Light Forcing is the first sparse attention scheme customized for autoregressive (AR) video diffusion models—Chunk-Aware Growth (CAG) dynamically allocates sparsity by quantifying the cumulative error contribution of each generated chunk, and Hierarchical Sparse Attention (HSA) flexibly captures historical dependencies through frame-level → chunk-level two-stage mask selection. It achieves 1.30× end-to-end / 3.79× attention acceleration on Self Forcing, with a VBench total score of 84.5 > dense baseline 84.1.

## Background & Motivation

**Background**: Autoregressive video generation combines frame-by-frame generation + few-step diffusion, making it more suitable for real-time and interactive scenarios compared to bidirectional video diffusion. However, as with all Transformer models, the quadratic complexity of attention is a deployment bottleneck—in Self Forcing, attention accounts for approximately 75% of the total latency when generating the final chunks of a 480p video.

**Limitations of Prior Work**: Directly applying sparse attention designed for bidirectional models (such as STA / VMoBA / SLA) to AR leads to severe quality degradation:
- Errors in AR accumulate along the generation chain. Sparse attention exacerbates this accumulation, yet previous schemes ignore the **heterogeneous contribution of different chunks to global error**;
- Historical critical information is not fully utilized—demands for historical frames vary significantly across different layers / heads / timesteps, and sliding windows cannot cover all key information.

**Key Challenge**: In AR, the current chunk is predicted at the next noise level conditioned on past clean chunks. Subsequent chunks are prone to inheriting quality issues from preceding ones, while simultaneously requiring flexible access to complex and diverse historical context patterns (e.g., diagonals, attention sinks), which contradicts fixed sparsification strategies.

**Goal**: Design a sparse attention framework dedicated to AR video that reduces computation while preserving long-term consistency and rich motion.

**Key Insight**: Reduce cumulative error through chunk-level differentiated sparsity allocation (dense early, sparse later); use hierarchical mask selection to flexibly capture diverse historical dependencies under a fixed budget.

**Core Idea**: Quantify the cumulative error of each chunk via a theoretical framework and allocate sparsity accordingly (CAG), combined with a coarse-to-fine two-stage mask selection (HSA) to maintain global and local awareness.

## Method

### Overall Architecture
Two complementary modules:
- **Chunk-Aware Growth (CAG)**: Allocates different sparsity levels for each chunk at the macro level—theoretical analysis suggests early chunks should retain higher attention budgets as "visual anchors," while later chunks can tolerate higher sparsity.
- **Hierarchical Sparse Attention (HSA)**: Selects key blocks for each query chunk at the micro level—a two-stage coarse-to-fine process (frame-level → chunk-level) that maintains computational efficiency while capturing necessary history.

### Key Designs

1. **Chunk-Aware Growth (CAG)**:
    - **Function**: Dynamically allocates sparsity based on the cumulative global error of each chunk, achieving progressive sparsification.
    - **Mechanism**: Diffusion theory provides an upper bound for the error distance of the $i$-th chunk: $\text{TV}(q_t, p) \leq C_1 \frac{d^2 \log^3 T}{\sqrt{T}} + C_2 \sqrt{d} \varepsilon_{\text{score}} \log^2 T$. Early chunks retain dense attention as "visual anchors," while remaining chunks are assigned sparsity $s_i = s_{\text{base}} - \alpha_i \beta$ (where $\alpha_i$ is the noise level reached by the $i$-th chunk). Solve for $\beta$ via $\sum_{i=2}^n (1 - s_{\text{base}} + \alpha_i \beta) l_i^q l_i^k d = (1 - s_{\text{target}}) \sum_{i=2}^n l_i^q l_i^k d$ to satisfy the target total FLOPs.
    - **Design Motivation**: Experiments demonstrate that sparsity in early chunks leads to irreversible quality loss (over-saturation), while sparsity in later chunks is nearly lossless—reflecting the chain propagation of error in AR; differentiated chunk-level strategies effectively control error accumulation.

2. **Hierarchical Sparse Attention (HSA)**:
    - **Function**: Flexibly captures diverse historical dependencies within a fixed computational complexity.
    - **Mechanism**: Given query $q^{(i)} \in \mathbb{R}^{(f \times n) \times d}$ and historical KV $\{k^{:i}, v^{:i}\} \in \mathbb{R}^{(i \times f \times n) \times d}$. (1) **Token Compression**: Perform mean pooling at chunk and frame levels to obtain $\tilde{q}^{(i)}, \tilde{k}^{:i}, \hat{k}^\mathcal{M}$. (2) **Frame-level Mask Selection**: Calculate frame-level relevance $p_r^{(i)} = \langle \tilde{q}_r^{(i)}, \hat{k}^\mathcal{M} \rangle$ for the $r$-th query chunk, and select Top-K frames $\mathcal{T}_r = \text{TopK}_{\text{idx}}(p_r^{(i)}) \cup \mathcal{F}^{(i)}$ (where $\mathcal{F}^{(i)}$ includes always-selected frames: initial sink + most recent frames + intra-chunk frames). (3) **Chunk-level Mask Selection**: Calculate chunk-level relevance $o_r^{(i)}(\tau, j) = \langle \tilde{q}_r^{(i)}, \tilde{k}_j^{(\tau)} \rangle$ within the selected frames, and select Top-K chunk pairs $\mathcal{J}_r$ from $\mathcal{B}_r = \{(\tau, j) \mid \tau \in \mathcal{T}_r\}$. The final chunk mask is $B_r^{(i)}(\tau, j) = \mathbb{1}[\tau \in \mathcal{T}_r, j \in \mathcal{J}_r(\tau)]$.
    - **Design Motivation**: Historical attention patterns vary significantly across layers / heads / timesteps (visualized in Fig. 4); fixed sliding windows lose critical information. The hierarchical strategy flexibly handles diverse patterns within a fixed budget, with frame-level retrieval overhead of only ~2%. CAG and HSA are complementary—CAG macroscopically controls chunk-level sparsity, while HSA microscopically determines historical information per chunk.

## Key Experimental Results

### Main Results (Self Forcing 1.3B, VBench, 5s Video)

| Method | Latency (s) | Speedup | Aesthetic | Imaging | Motion Smooth | Dynamic | Subj. Consist. | Bg. Consist. | Total |
|------|---------|------|--------|--------|--------|--------|--------|--------|------|
| FlashAttention2 | 9.61 | 1.00× | 67.4 | 70.0 | 98.3 | 63.1 | 95.3 | 96.5 | 84.1 |
| STA | 8.27 | 1.16× | 64.5 | 71.7 | 98.5 | 48.9 | 96.3 | 96.9 | 83.6 |
| Radial | 7.39 | 1.30× | 45.8 | 66.1 | 96.0 | 88.6 | 90.2 | 93.6 | 73.7 |
| VMoBA | 7.42 | 1.29× | 65.2 | 69.9 | 97.3 | 84.2 | 92.8 | 95.5 | 83.6 |
| SLA | 7.71 | 1.25× | 66.7 | 69.8 | 98.3 | 44.2 | 95.6 | 96.7 | 83.2 |
| **Light Forcing** | **7.39** | **1.30×** | **67.2** | **71.0** | **98.3** | **66.7** | **96.2** | **96.5** | **84.5** |

Total score of 84.5 exceeds the dense baseline of 84.1, while achieving 1.30× end-to-end / 3.79× attention acceleration.

### Ablation Study

| Config | Subj. Consist. | Aesthetic | Imaging | Dynamic | Total | Description |
|------|--------|------|------|------|------|------|
| FlashAttention2 | 95.3 | 67.4 | 70.0 | 63.1 | 84.1 | Full Dense |
| +1D Sparse Attn | 86.9 | 51.4 | 66.0 | 52.8 | 73.0 | Direct application causes collapse |
| +Fine-tuning | 94.9 | 65.1 | 69.8 | 46.4 | 82.8 | Partial recovery but insufficient |
| +CAG | 96.1 | 67.7 | 71.0 | 37.5 | 83.2 | CAG improves aesthetic/imaging but dynamics drop |
| **+CAG & HSA** | **96.2** | **67.2** | **71.0** | **66.7** | **84.5** | HSA improves dynamics, overall exceeds baseline |

### Key Findings
- Direct application of sparsity to AR leads to severe collapse (73.0 vs 84.1), and fine-tuning only partially recovers performance.
- Using CAG alone improves aesthetic and imaging quality but harms dynamics—excessive sparsity results in over-reliance on preceding chunk priors; HSA improves dynamics via flexible historical access.
- For long videos (Infinite-Forcing 15s): 84.1 vs 83.6 dense baseline, with an improvement in dynamic degree (64.7 vs 54.7).
- HSA is robust to Top-K hyperparameters: Total scores remain between 84.3-84.5 for Top-K = {6, 9, 12}.
- Efficient Deployment: Combined with FP8 + RoPE / RMSNorm fusion, it reaches 27.4 FPS on RTX 5090 (3.08× end-to-end speedup) and 33.9 FPS on H100—achieving real-time AR video generation on consumer GPUs for the first time.

## Highlights & Insights
- **Ingenuity of Differentiated Chunk Strategy**: The derivation chain from observation (early sparsity irreversible vs. later sparsity tolerable) → theory (AR cumulative error) → solution (quantified allocation) is highly convincing.
- **Flexibility of Hierarchical Mask Selection**: Unlike fixed sliding windows, the two-stage selection (frame → chunk) preserves global awareness while maintaining local precision; the visualized attention patterns (diagonal, sink) in Fig. 4 demonstrate its necessity.
- **Quality Surpassing Baseline**: Not only does it accelerate, but the generation quality (84.5) also exceeds the dense version (84.1), suggesting that standard dense attention contains significant redundancy and that appropriate sparsification can actually improve generalization.
- **Transferable Methodology**: The CAG / HSA logic can be extended to other sequential generations (autoregressive text, audio-video synchronization); the chunk-aware cumulative error analysis framework serves as a reference for understanding error propagation in AR models.

## Limitations & Future Work
- Focuses on few-step (T=4) AR models like Self Forcing; generalization to more steps is yet to be determined.
- Frame-level retrieval in HSA adds a 2% overhead, leaving room for further optimization.
- Experiments rely primarily on VBench; cross-task evaluations (specific semantic consistency, multi-object tracking) would be more persuasive.
- The theoretical derivation of CAG is based on a symmetry assumption (identical T per chunk), which may be heterogeneous in practice.

## Related Work & Insights
- **vs. Bidirectional Sparse Attention (STA / VMoBA / SLA)**: These were designed for bidirectional models using chunk aggregation to identify key blocks; this paper finds they cause severe quality loss in AR due to ignoring heterogeneous error contributions between AR chunks and complex historical dependency patterns.
- **vs. Sliding Window Attention (LongFormer)**: Fixed windows lead to historical forgetting and long-term inconsistency; HSA flexibly expands the receptive field via dynamic frame retrieval while maintaining linear complexity.
- **vs. Other AR Accelerations (Self Forcing / LongLive optimizations)**: These primarily improve the denoising process or KV cache strategies; Light Forcing is orthogonal and stackable (a 2-3× combined speedup has been verified).

## Rating
- Novelty: ⭐⭐⭐⭐⭐  First to systematically analyze why AR sparse attention fails and design a dedicated solution; both chunk-aware and hierarchical selection are original.
- Experimental Thoroughness: ⭐⭐⭐⭐  Covers multiple models, lengths, and benchmarks with complete ablations and hyperparameter sensitivity; drawback is that quality evaluation still primarily relies on automated metrics.
- Writing Quality: ⭐⭐⭐⭐  Clear motivation, rigorous derivation, and well-organized; some mathematical notation is slightly complex but overall smooth.
- Value: ⭐⭐⭐⭐⭐  Directly addresses the practical bottleneck of AR video generation, achieving real-time generation (27.4 FPS) on consumer GPUs for the first time with open-source code; the theoretical framework holds academic value for AR error analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] VEDA: Scalable Video Diffusion via Distilled Sparse Attention](veda_scalable_video_diffusion_via_distilled_sparse_attention.md)
- [\[ICML 2026\] Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)
- [\[ICML 2026\] DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)
- [\[ICML 2026\] Lightning Unified Video Editing via In-Context Sparse Attention](lightning_unified_video_editing_via_in-context_sparse_attention.md)
- [\[NeurIPS 2025\] Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](../../NeurIPS2025/video_generation/self_forcing_bridging_the_train-test_gap_in_autoregressive_video_diffusion.md)

</div>

<!-- RELATED:END -->
