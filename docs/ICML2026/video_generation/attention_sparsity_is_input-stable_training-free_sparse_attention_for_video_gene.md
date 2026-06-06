---
title: >-
  [Paper Note] Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering
description: >-
  [ICML 2026][Video Generation][Sparse Attention] SVOO discovers that the attention sparsity of each layer in video DiT is an intrinsic property that is "input-invariant within layers and significantly heterogeneous across…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Sparse Attention"
  - "DiT"
  - "Video Generation Acceleration"
  - "Co-Clustering"
  - "Hierarchical Sparsity"
date: 2026-05-08
content_hash: bdb488edec76972c
---

# Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering

**Conference**: ICML 2026  
**arXiv**: [2603.18636](https://arxiv.org/abs/2603.18636)  
**Code**: https://github.com/Mutual-Luo/SVOO  
**Area**: Video Generation / Diffusion Models / Model Efficiency  
**Keywords**: Sparse Attention, DiT, Video Generation Acceleration, Co-Clustering, Hierarchical Sparsity

## TL;DR
SVOO discovers that the attention sparsity of each layer in video DiT is an intrinsic property that is "input-invariant within layers and significantly heterogeneous across layers." Based on this, it first performs offline layer-wise sparsity calibration, then conducts online QK bidirectional co-clustering. Without retraining, it achieves up to 1.93× acceleration on 7 models including Wan/HunyuanVideo while maintaining PSNR at 29 dB.

## Background & Motivation

**Background**: 3D DiT has become the mainstream backbone for high-fidelity video generation (Wan, HunyuanVideo, Sora share the same architecture), but the cost of dense 3D self-attention grows quadratically with token count. Running Wan2.1-1.3B for a 720p×81-frame video on a single H200 GPU takes 417s. The main acceleration approach is "sparse attention," which includes training-based (VMoBA, VSA, DSV, BSA) and training-free (SVG, SVG2, STA, Radial, SpargeAttn, XAttention, DraftAttention, etc.) methods. The latter requires no retraining and is deployment-friendly, but generally underperforms the former.

**Limitations of Prior Work**: The authors precisely summarize the bottlenecks of training-free sparse attention into two points: (L1) **Ignoring hierarchical heterogeneity**—all transformer layers share the same sparsity rate, disregarding functional differences between layers; (L2) **Ignoring Q-K coupling**—when partitioning blocks, query and key are clustered independently via k-means, but block-level salient patterns arise from Q-K joint behavior, so independent clustering fragments high-quality attention regions.

**Key Challenge**: "How much sparsity to use" requires layer-wise consideration of structural heterogeneity, while "how to partition blocks" must treat Q and K as a coupled system rather than independent entities. Existing methods use the simplest uniform/independent assumptions in both dimensions, capping the quality ceiling under the same compute budget.

**Goal**: To simultaneously overcome the above two limitations without retraining DiT, achieving a better quality–speed trade-off.

**Key Insight**: The authors make a key observation, supported by theory: "Layer-wise sparsity is actually an intrinsic property determined by $\mathbf{W}_Q\mathbf{W}_K^\top$ of the layer and is almost input-insensitive." Therefore, "offline calibration once, reuse throughout online inference" is feasible. This property amortizes the cost of layer-wise sparsity to nearly zero.

**Core Idea**: Replace "uniform sparsity rate + independent Q/K clustering" with **"offline layer-wise sparsity profiling + online bidirectional QK co-clustering"**, feeding each layer's individual sparsity budget into a block partitioning algorithm that truly considers Q-K coupling.

## Method

### Overall Architecture
SVOO is a two-stage pipeline: (i) Offline stage—using a small calibration set (just a few random prompts from VBench suffice), run a forward pass on the original model, record the attention density for each (layer, head), and take a conservative high quantile as the sparsity rate $s_{\ell,h}$ for that layer. (ii) Online stage—during inference, for each layer and head, perform bidirectional co-clustering to partition query and key simultaneously, then, according to the offline schedule, select the top-K block pairs for dense computation and skip the rest. The two stages are independent; the schedule is just a set of coefficients, and can be cached in a sub-1MB file.

### Key Designs

1. **Offline Layer-Wise Sparsity Profiling**:

    - **Function**: Estimate for each (layer, head) an intrinsic, input-robust sparsity rate $s_{\ell,h}\in[0,1)$.
    - **Mechanism**: For a prompt $x^k$, sort each row of the attention matrix for head $h$ in descending order, and find the minimal proportion $d_{\ell,h}^{(k)}$ covering cumulative quality $\tau{=}0.95$. Fit a univariate Gaussian $\mathcal{N}(\mu_{\ell,h},\sigma_{\ell,h}^2)$ over $m$ calibration inputs, and use the upper $\alpha{=}0.95$ quantile $\hat d_{\ell,h}=\mu+z_\alpha\sigma$ as a conservative density estimate, finally setting $s_{\ell,h}=1-\hat d_{\ell,h}$.
    - **Design Motivation**: The authors prove Theorem 4.2 (under Bounded Token Representation, $|V(\mathbf{X})-V(\hat{\mathbf{X}})|$ is controlled by both $\|\mathbf{M}\|_2^2$ and $1/\sqrt n$), showing that in large-token video scenarios, intra-layer sparsity is naturally stable; thus, "profile once, reuse forever" is theoretically justified, with no need to recalibrate for each new prompt.

2. **Online Bidirectional Co-Clustering**:

    - **Function**: Partition query and key into $K_q$ and $K_k$ semantically aligned blocks without computing dense $QK^\top$.
    - **Mechanism**: Alternately iterate two steps—Step A: use previous query centers $\mathbf{C}_q^{(i-1)}$ as anchors, compute each key's affinity vector $\mathbf{P}_k=\mathcal{K}(\mathbf{C}_q)^\top$, and assign keys to the closest key-block center $\bar{\mathbf{P}}_k[j]$; Step B: symmetrically, use the updated key centers $\mathbf{C}_k^{(i)}$ to repartition queries. The process only requires matrix multiplications of token count × block count, far less than $n\times n$ attention.
    - **Design Motivation**: Traditional methods like SVG2 cluster Q and K independently, effectively assuming "optimal key partitioning is query-independent"; but the authors derive that $\mathbf{q}^\top(\mathbf{k}_1-\mathbf{k}_2)\approx 0$ is the true condition for two keys to belong to the same block, which is clearly query-dependent. Co-clustering uses cross-affinity instead of Euclidean distance, ensuring tokens in the same block genuinely share "similar cross-attention preferences," which is the implicit assumption required for sparsity.

3. **Block Selection + Sparse Attention Assembly**:

    - **Function**: Given each layer's $s_{\ell,h}$ and coupled block partitions $(\mathcal{L}_q,\mathcal{L}_k)$, decide which block pairs undergo dense computation.
    - **Mechanism**: Use block centers for coarse block-level attention estimation $\hat A_{ij}=\mathbf{C}_q[i]\mathbf{C}_k[j]^\top$, for each query block select the top-$\lceil(1-s_{\ell,h})K_k\rceil$ key blocks by $\hat A$, and compute exact attention only for these block pairs, treating others as zero; softmax is normalized over the remaining logits.
    - **Design Motivation**: After co-clustering, block-level estimation has already concentrated high-quality attention into a few block pairs, so even a low retention ratio preserves PSNR; this closes the loop with the L1 and L2 motivations.

### Loss & Training
Completely training-free. All modifications occur only in the inference path: calibration is performed once with 5 VBench prompts to obtain $s_{\ell,h}$; during inference, each transformer layer performs $I_{\max}$ rounds of co-clustering (only a few iterations needed in practice), then applies sparse attention. The schedule file is negligible in size and can be distributed with the original checkpoint.

## Key Experimental Results

### Main Results
On 7 mainstream video DiT models (Wan2.1-T2V 1.3B/14B, Wan2.1-I2V-14B, Wan2.2-T2V-A14B, Wan2.2-I2V-A14B, HunyuanVideo-T2V/I2V), SVOO is compared with SpargeAttn, SVG1, SVG2, and Radial, all evaluated on H200, 720p, 81 frames.

| Model | Method | PSNR↑ | LPIPS↓ | ImgQual↑ | Latency | Speedup |
|-------|--------|-------|--------|----------|---------|---------|
| Wan2.1-1.3B-T2V | Origin | — | — | 66.58 | 417s | 1.00× |
| Wan2.1-1.3B-T2V | SVG2 | 29.27 | 0.127 | 61.83 | 241s | 1.73× |
| Wan2.1-1.3B-T2V | **SVOO** | **29.99** | **0.125** | **66.57** | **216s** | **1.93×** |
| Wan2.1-14B-T2V | SVG2 | 27.34 | 0.111 | 68.29 | 1261s | 1.57× |
| Wan2.1-14B-T2V | **SVOO** | **27.79** | 0.111 | **68.92** | **1203s** | **1.64×** |
| Wan2.2-14B-T2V | SVG2 | 24.48 | 0.142 | 71.51 | 1061s | 1.52× |
| Wan2.2-14B-T2V | **SVOO** | **24.85** | 0.144 | **72.92** | **984s** | **1.63×** |

Highlights: SVOO achieves the highest (quality) and fastest (speed) metrics in almost all cases. ImageQuality remains on par with the original model (66.57% vs 66.58%), while cluster-based methods like SVG2 lose 4.7 points in ImgQual.

### Ablation Study

| Configuration | Key Metrics | Description |
|---------------|------------|-------------|
| Full SVOO | PSNR 29.99 / 1.93× | Both profiling and co-clustering enabled |
| w/o Offline Profiling (Uniform Sparsity) | Significant PSNR drop | Degrades to baseline ignoring hierarchical heterogeneity, reproducing L1 limitation |
| w/o Co-Clustering (Independent Q/K Partition) | Close to SVG2 performance | Reproduces L2 limitation, quality–speed trade-off worsens |
| Different $\tau$, $\alpha$ | Stable | Calibration thresholds within reasonable range do not affect final quality, confirming schedule input-independence |

### Key Findings
- The empirical observation "layer sparsity is nearly input-invariant" is directly proven by Theorem 4.2: $\mathbf{M}=\mathbf{W}_Q\mathbf{W}_K^\top$ determines inter-layer differences, and the $1/\sqrt n$ term suppresses intra-layer variance; for video with large $n$, this is almost free.
- The benefit of co-clustering is most pronounced over cluster-based baselines (SVG2)—with clustering-based sparsity, adding "Q looks at K centers, K looks at Q centers" significantly improves block selection accuracy.
- Speedup is more significant on small 1.3B models than large 14B models (1.93× vs 1.64×), as FFN accounts for a larger proportion in big models, reducing the relative gain from attention acceleration.

## Highlights & Insights
- Transforms "how much sparsity to use" into a **one-time offline calibration** problem, avoiding both retraining and the overhead of online dynamic search—a rare "theory + engineering" dual-closure design in training-free acceleration.
- Co-clustering addresses the often-overlooked Q-K decoupling detail, solving it with a cross-affinity iteration instead of heavier attention approximations, yielding clear benefits with minimal engineering effort and high transferability.
- Theorem 4.2 elevates "why only 5 prompts suffice for calibration" from an empirical trick to a provable conclusion, alleviating concerns about needing to recalibrate when switching prompt domains.

## Limitations & Future Work
- Experiments only cover 720p×81 frames; whether co-clustering iteration count remains low at higher resolutions/longer sequences remains to be validated.
- $\tau{=}0.95$, $\alpha{=}0.95$ are empirical values; for extremely aggressive/conservative sparsity needs, retuning is required; lacks a differentiable optimization path from schedule to downstream quality metrics.
- Speedup is limited by the proportion of attention in total compute; in FFN-heavy models (HunyuanVideo-13B), gains are weaker than in attention-heavy models (Wan2.1-1.3B); orthogonal combination with FFN acceleration methods is unexplored.

## Related Work & Insights
- **vs SVG2**: Both use clustering for block partitioning, but SVG2 clusters Q and K independently; SVOO uses bidirectional cross-affinity for block alignment and replaces uniform sparsity with a layer-wise schedule, directly upgrading SVG2's approach.
- **vs XAttention / SpargeAttn**: These estimate block importance via antidiagonal/aggregated activation, still "estimate then select"; SVOO emphasizes "partition correctly first, then select," maximizing quality from the partition stage.
- **vs Training-based BSA/DSV**: BSA jointly sparsifies Q and KV during training, sharing the co-clustering philosophy but requiring retraining; SVOO brings the same coupling idea back to inference-time algorithms, demonstrating that the train-free approach still has significant potential for DiT.

## Rating
- Novelty: ⭐⭐⭐⭐ The empirical + theoretical combination of "layer sparsity as an intrinsic property" refreshes the design intuition for train-free sparse attention
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 7 open-source DiTs, both T2V/I2V, compared with 5 mainstream baselines
- Writing Quality: ⭐⭐⭐⭐ Clear logical flow from Sec 3 motivation to Sec 4 method, Theorem 4.2 closely tied to method
- Value: ⭐⭐⭐⭐⭐ Extremely simple to deploy, orthogonal to training-based methods, nearly zero-cost deployment to existing video DiT inference services

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Lightning Unified Video Editing via In-Context Sparse Attention](lightning_unified_video_editing_via_in-context_sparse_attention.md)
- [\[CVPR 2026\] When to Lock Attention: Training-Free KV Control in Video Diffusion](../../CVPR2026/video_generation/when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)
- [\[NeurIPS 2025\] Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation](../../NeurIPS2025/video_generation/radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)
- [\[CVPR 2026\] SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](../../CVPR2026/video_generation/switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [\[NeurIPS 2025\] VORTA: Efficient Video Diffusion via Routing Sparse Attention](../../NeurIPS2025/video_generation/vorta_efficient_video_diffusion_via_routing_sparse_attention.md)

</div>

<!-- RELATED:END -->
