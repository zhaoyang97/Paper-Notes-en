---
title: >-
  [Paper Note] Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering
description: >-
  [ICML 2026][Video Generation][Sparse Attention] SVOO discovers that the attention sparsity of video DiT layers is an intrinsic property that is "input-independent within layers and significantly heterogeneous across laye…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Sparse Attention"
  - "DiT"
  - "Video Generation Acceleration"
  - "Co-Clustering"
  - "Hierarchical Sparsity"
date: 2026-05-08
content_hash: 274f98c9d72f501a
---

# Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering

**Conference**: ICML 2026  
**arXiv**: [2603.18636](https://arxiv.org/abs/2603.18636)  
**Code**: https://github.com/Mutual-Luo/SVOO  
**Area**: Video Generation / Diffusion Models / Model Efficiency  
**Keywords**: Sparse Attention, DiT, Video Generation Acceleration, Co-Clustering, Hierarchical Sparsity

## TL;DR
SVOO discovers that the attention sparsity of video DiT layers is an intrinsic property that is "input-independent within layers and significantly heterogeneous across layers." Based on this, it performs offline layer-wise sparsity calibration followed by online bidirectional QK co-clustering for block partitioning. It achieves training-free acceleration of up to 1.93× across 7 models (e.g., Wan/HunyuanVideo) while maintaining PSNR at 29 dB.

## Background & Motivation

**Background**: 3D DiT has become the mainstream backbone for high-fidelity video generation (Wan, HunyuanVideo, Sora). However, the cost of dense 3D self-attention grows quadratically with the number of tokens; running Wan2.1-1.3B for a 720p×81 frame video takes 417s on a single H200. Mainstream acceleration paths focus on "sparse attention," divided into trainable methods (VMoBA, VSA, DSV, BSA) and training-free methods (SVG, SVG2, STA, Radial, SpargeAttn, XAttention, DraftAttention, etc.). The latter do not require retraining and are deployment-friendly, but generally underperform compared to the former.

**Limitations of Prior Work**: The authors precisely summarize the bottlenecks of training-free sparse attention into two points: (L1) **Neglecting hierarchical heterogeneity**—using the same sparsity rate for all transformer layers ignores the functional differences of each layer; (L2) **Neglecting Q-K coupling**—independent k-means for query and key during block partitioning splits block-level salient patterns that arise from the Q-K joint relationship, fragmenting high-quality attention regions.

**Key Challenge**: Selecting "how much sparsity" requires considering layer structural heterogeneity, while "how to partition blocks" must treat Q and K as a coupled system rather than independent items. Existing methods use naive uniform/independent assumptions in both dimensions, capping the quality potential under the same compute budget.

**Goal**: Simultaneously remove the above two limitations without retraining the DiT to achieve a superior quality-speed trade-off.

**Key Insight**: The authors provide a key observation and theoretical support: "Layer-wise sparsity is actually an intrinsic property determined by $\mathbf{W}_Q\mathbf{W}_K^\top$ and is almost insensitive to input." Thus, one can "calibrate once offline and reuse throughout online inference." This property allows the cost of layer-wise sparsity rates to be amortized to nearly zero.

**Core Idea**: Replace "uniform sparsity + independent Q/K partitioning" with **"Offline layer-wise sparsity profiling + Online bidirectional QK co-clustering,"** feeding the individual sparsity budget of each layer into a block partitioning algorithm that truly accounts for Q-K coupling.

## Method

### Overall Architecture
SVOO is a two-stage pipeline: (i) Offline stage—Use a small calibration set (a few random prompts from VBench), run a forward pass on the original model, calculate attention density for each (layer, head), and take a conservative high percentile as the sparsity rate $s_{\ell,h}$. (ii) Online stage—During inference, for each layer and head, perform bidirectional co-clustering to simultaneously partition queries and keys into blocks, then select the top-K block pairs for dense computation based on the offline schedule, skipping all other blocks. The two stages are independent; the schedule is just a set of coefficients, cacheable within sub-1MB.

### Key Designs

1.  **Offline Layer-Wise Sparsity Profiling**:
    - **Function**: Estimates an intrinsic and input-robust sparsity rate $s_{\ell,h}\in[0,1)$ for each (layer, head).
    - **Mechanism**: For a prompt $x^k$, sort each row of the attention matrix for head $h$ in descending order and find the smallest proportion of elements $d_{\ell,h}^{(k)}$ covering a cumulative mass of $\tau{=}0.95$. Fit a univariate Gaussian $\mathcal{N}(\mu_{\ell,h},\sigma_{\ell,h}^2)$ across $m$ calibration inputs, and use the $\alpha{=}0.95$ percentile $\hat d_{\ell,h}=\mu+z_\alpha\sigma$ for conservative density estimation. Finally, $s_{\ell,h}=1-\hat d_{\ell,h}$.
    - **Design Motivation**: The authors prove Theorem 4.2 (under Bounded Token Representation, $|V(\mathbf{X})-V(\hat{\mathbf{X}})|$ is controlled by both $\|\mathbf{M}\|_2^2$ and $1/\sqrt n$), showing that in large-token video scenarios, layer-wise sparsity is naturally stable. This indicates that "profiling once, reusing for life" is theoretically valid and removes the need for per-prompt re-calibration.

2.  **Online Bidirectional Co-Clustering**:
    - **Function**: Simultaneously partitions queries and keys into $K_q$ and $K_k$ semantically aligned blocks without calculating the dense $QK^\top$.
    - **Mechanism**: Iterates between two steps: Step A uses the previous query centroids $\mathbf{C}_q^{(i-1)}$ as anchors to calculate the affinity vector for each key $\mathbf{P}_k=\mathcal{K}(\mathbf{C}_q)^\top$, then assigns the key to the key-block whose centroid $\bar{\mathbf{P}}_k[j]$ is closest in affinity. Step B symmetrically re-partitions queries based on the updated key centroids $\mathbf{C}_k^{(i)}$. The process only involves matrix multiplications of the scale (tokens × blocks), which is much cheaper than $n\times n$ attention.
    - **Design Motivation**: Traditional methods like SVG2 perform independent k-means on Q and K, assuming "optimal key partitioning is independent of queries." However, the authors derive that $\mathbf{q}^\top(\mathbf{k}_1-\mathbf{k}_2)\approx 0$ is the true condition for two keys to belong to the same block, which is clearly query-dependent. Co-clustering uses cross-affinity instead of Euclidean distance, ensuring tokens within a block share similar cross-attention preferences.

3.  **Block Selection + Sparse Attention Assembly**:
    - **Function**: Given $s_{\ell,h}$ and coupled partitions $(\mathcal{L}_q,\mathcal{L}_k)$, decides which block pairs undergo dense computation.
    - **Mechanism**: Uses block centroids for coarse block-level attention estimation $\hat A_{ij}=\mathbf{C}_q[i]\mathbf{C}_k[j]^\top$. For each query block, select the top-$\lceil(1-s_{\ell,h})K_k\rceil$ key blocks according to $\hat A$. Only these pairs undergo exact attention calculation; others are treated as 0. Softmax is normalized over the remaining logits.
    - **Design Motivation**: After co-clustering, block-level estimation aligns high-quality attention into a few block pairs, so PSNR remains high even with low retention ratios, closing the loop between L1 and L2.

### Loss & Training
Entirely training-free. All changes occur in the inference path: calibration is done once with 5 VBench prompts to obtain $s_{\ell,h}$. During inference, each transformer layer performs $I_{\max}$ rounds of co-clustering (only a few iterations needed in practice) before sparse attention. The schedule file size is negligible and can be distributed with the original checkpoint.

## Key Experimental Results

### Main Results
Evaluated on 7 mainstream video DiTs (Wan2.1-T2V 1.3B/14B, Wan2.1-I2V-14B, Wan2.2-T2V-A14B, Wan2.2-I2V-A14B, HunyuanVideo-T2V/I2V) against SpargeAttn, SVG1, SVG2, and Radial, standardized on H200, 720p, 81 frames.

| Model | Method | PSNR↑ | LPIPS↓ | ImgQual↑ | Latency | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Wan2.1-1.3B-T2V | Origin | — | — | 66.58 | 417s | 1.00× |
| Wan2.1-1.3B-T2V | SVG2 | 29.27 | 0.127 | 61.83 | 241s | 1.73× |
| Wan2.1-1.3B-T2V | **SVOO** | **29.99** | **0.125** | **66.57** | **216s** | **1.93×** |
| Wan2.1-14B-T2V | SVG2 | 27.34 | 0.111 | 68.29 | 1261s | 1.57× |
| Wan2.1-14B-T2V | **SVOO** | **27.79** | 0.111 | **68.92** | **1203s** | **1.64×** |
| Wan2.2-14B-T2V | SVG2 | 24.48 | 0.142 | 71.51 | 1061s | 1.52× |
| Wan2.2-14B-T2V | **SVOO** | **24.85** | 0.144 | **72.92** | **984s** | **1.63×** |

**Highlights**: SVOO achieves the highest quality (PSNR/LPIPS) and fastest speed across nearly all metrics. ImageQuality remains comparable to the original model (66.57% vs 66.58%), whereas cluster-based methods like SVG2 drop 4.7 points in ImgQual.

### Ablation Study

| Configuration | Key Metrics | Note |
| :--- | :--- | :--- |
| Full SVOO | PSNR 29.99 / 1.93× | Both profiling + co-clustering enabled |
| w/o Offline Profiling (Uniform Sparsity) | Significant PSNR drop | Degrades to baseline ignoring hierarchical heterogeneity (L1) |
| w/o Co-Clustering (Independent Q/K) | Near SVG2 performance | Replicates L2 flaw; quality-speed trade-off deteriorates |
| Varying $\tau, \alpha$ | Stable | Quality is insensitive to calibration thresholds within reasonable ranges |

### Key Findings
- The empirical observation that "layer sparsity is nearly input-independent" is directly proven by Theorem 4.2: $\mathbf{M}=\mathbf{W}_Q\mathbf{W}_K^\top$ determines inter-layer differences, while the $1/\sqrt n$ term suppresses intra-layer variance; this effectively comes for free in high-token video scenarios.
- The gains of co-clustering are most evident over cluster-based baselines (SVG2)—using "Q looks at K centroids, K looks at Q centroids" significantly improves block selection accuracy with the same clustering budget.
- Speed gains are more pronounced on the 1.3B model versus the 14B model (1.93× vs 1.64×), as the FFN ratio increases in larger models, leading to diminishing returns for attention-only acceleration.

## Highlights & Insights
- Transforming the "sparsity selection" into a **one-time offline calibration** problem avoids both training and the overhead of online dynamic searching—a rare design combining theoretical and engineering closure in training-free acceleration.
- Co-clustering highlights the overlooked detail of Q-K decoupling, solving it with cross-affinity iterations rather than heavier attention approximations. It requires minimal engineering yet yields clear benefits and high transferability.
- Theorem 4.2 upgrades the empirical trick "5 prompts are enough for calibration" into a provable conclusion, alleviating concerns about needing re-calibration for different prompt domains.

## Limitations & Future Work
- Experiments only cover 720p×81 frames; whether the number of co-clustering iterations remains single-digit at higher resolutions or longer sequences requires validation.
- $\tau{=}0.95$ and $\alpha{=}0.95$ are empirical values; aggressive or conservative needs may require tuning. There is currently no differentiable path from the schedule to downstream quality metrics.
- Acceleration is capped by the proportion of attention in total costs. On FFN-heavy models (HunyuanVideo-13B), gains are weaker than on attention-heavy Wan2.1-1.3B; orthogonal combinations with FFN acceleration remain unexplored.

## Related Work & Insights
- **vs SVG2**: Both use clustering for partitioning, but SVG2 clusters Q and K independently; SVOO aligns blocks via bidirectional cross-affinity and replaces uniform sparsity with a layer-wise schedule, making it a direct upgrade to SVG2.
- **vs XAttention / SpargeAttn**: These use antidiagonal or aggregated activation to estimate block importance ("estimate then select"); SVOO emphasizes "partition correctly then select," ensuring quality starts from the partition stage.
- **vs Trainable BSA/DSV**: BSA jointly sparsifies Q and KV during training, a concept similar to co-clustering but requiring retraining; SVOO brings this coupling back to inference-time algorithms, proving there is still significant room for training-free DiT methods.

## Rating
- Novelty: ⭐⭐⭐⭐ Empirical + theoretical combination of "intrinsic layer sparsity" refreshes training-free sparse attention intuition.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 7 open-source DiTs, both T2V/I2V tasks, and 5 mainstream baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear logical flow from Sec 3 motivation to Sec 4 method derivation; Theorem 4.2 integrates tightly with the method.
- Value: ⭐⭐⭐⭐⭐ High deployment value; extremely simple to implement and orthogonal to training-based methods, with near-zero cost for integration into existing video DiT inference services.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)
- [\[ICML 2026\] DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)
- [\[ICML 2026\] VEDA: Scalable Video Diffusion via Distilled Sparse Attention](veda_scalable_video_diffusion_via_distilled_sparse_attention.md)
- [\[ICML 2026\] Lightning Unified Video Editing via In-Context Sparse Attention](lightning_unified_video_editing_via_in-context_sparse_attention.md)
- [\[CVPR 2026\] When to Lock Attention: Training-Free KV Control in Video Diffusion](../../CVPR2026/video_generation/when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)

</div>

<!-- RELATED:END -->
