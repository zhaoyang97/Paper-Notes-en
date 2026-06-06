---
title: >-
  [Paper Note] Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression
description: >-
  [ICML 2026][Model Compression][Mixture-of-Experts] Addressing the MoE LLM trilemma of "load imbalance – parameter redundancy – communication overhead…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Mixture-of-Experts"
  - "Dynamic Expert Clustering"
  - "Low-rank Residual"
  - "Hierarchical Routing"
  - "Heterogeneous Precision"
date: 2026-05-08
content_hash: 39500448a8c34499
---

# Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression

**Conference**: ICML 2026  
**arXiv**: [2510.02345](https://arxiv.org/abs/2510.02345)  
**Code**: https://github.com/szdtzpj/Breaking_the_moe_trilemma (Available)  
**Area**: Model Compression / MoE LLM / System Optimization  
**Keywords**: Mixture-of-Experts, Dynamic Expert Clustering, Low-rank Residual, Hierarchical Routing, Heterogeneous Precision

## TL;DR
Addressing the MoE LLM trilemma of "load imbalance – parameter redundancy – communication overhead," this paper proposes a unified framework: online expert grouping using "parameter + activation" dual-similarity clustering, structured compression (~5×) within groups via "shared base matrix + low-rank residual," two-level hierarchical routing ("select group then select expert"), FP16/INT4 heterogeneous precision, and offline offloading of idle groups. On GLUE/WikiText-103, it matches standard MoE performance with ~80% parameter reduction, 10–20% throughput gain, and a 3× reduction in expert load variance.

## Background & Motivation

**Background**: MoE has become the critical path for scaling LLMs (e.g., Switch, GShard, Mixtral)——theoretically increasing parameter capacity without significantly increasing FLOPs.

**Limitations of Prior Work**: Deploying MoE on A100/H100 hardware encounters an "optimization trilemma": (i) **Load imbalance**—top-$k$ gating results in a few overloaded experts and many idle ones; (ii) **Parameter redundancy**—the linear increase in parameters with the number of experts exhausts HBM capacity; (iii) **All-to-all communication overhead**—dispatching tokens to experts across different devices often becomes the dominant latency, especially for long sequences.

**Key Challenge**: Existing methods address these issues in isolation. Load balancing losses (Switch loss) are reactive and fail under distribution shifts; compression methods like MoE-Lite treat experts as independent entities, ignoring structural similarities; communication-aware routing (Tutel, SmILE) optimizes data paths on fixed architectures without addressing redundancy or imbalance. Worse, these three goals often conflict—optimizing one variable frequently degrades another.

**Goal**: Establish a unified framework to simultaneously reduce total storage, minimize activation parameters per token, maintain model quality, decrease cross-device traffic, and keep re-clustering overhead controllable.

**Key Insight**: The core observation is that "experts activated by semantically similar inputs also exhibit parameter redundancy." This hypothesis enables the co-optimization of architecture (grouping) and systems (routing/storage/communication).

**Core Idea**: Use dynamic clustering to group functionally similar experts → use a shared base + low-rank residual within groups to compress parameters → route first to groups then to experts, reducing all-to-all communication to a two-level process.

## Method

### Overall Architecture
The unified objective (Eq. 1) is $\min L_{\text{task}}+A_1 I_{\text{load}}+A_2 R_{\text{red}}+A_3 C_{\text{comm}}$, where the four learnable/designable variables are: Grouping, Parameterization (Param, Factors), and Routing strategy. The pipeline executes four tasks: (1) Online dual-similarity clustering → dividing $E$ experts into $G$ groups of $K=E/G$ each; (2) Shared base + low-rank residual compression within groups; (3) Two-level hierarchical routing; (4) Heterogeneous precision (FP16 base + INT4 residual) + dynamic offloading of idle groups.

### Key Designs

1. **Online Dual-Similarity Clustering**:
    - **Function**: Re-groups experts every $T$ steps using "parameter + activation" similarity as a foundation for compression, routing, and memory strategies.
    - **Mechanism**: Maintains two features for each expert $\mathcal{E}_i$: weight vector $\text{vec}(W_i)$ and activation centroid $\mu_i$ (updated via EMA: $\mu_i\leftarrow(1-\beta)\mu_i+\beta\bar{x}_i$, default $\beta=0.05$). Parameter similarity $S_{\text{param}}$ and task similarity $S_{\text{task}}$ are both cosine-based, fused by weight $\alpha$: $S=\alpha S_{\text{param}}+(1-\alpha)S_{\text{task}}$ (default $\alpha>0.5$). Every $T$ steps: use threshold $\tau$ (default 0.1) to prune low-similarity pairs for a neighbor graph to reduce $O(E^2)$ comparisons; perform K-means++ on distance $D=1-S$ to get $G$ groups; greedily relocate boundary experts if imbalanced. Cache $S_{\text{param}}$ for $m$ steps, recalculating only if weights change beyond $\epsilon$.
    - **Design Motivation**: Parameter similarity reflects structural likeness, while activation similarity reflects input processing. Fusing them ensures groups can share parameters and be activated by similar tokens simultaneously. EMA + periodic re-clustering makes the grouping robust to distribution shifts compared to static grouping.

2. **Structured Compression with Shared Base + Low-Rank Residual**:
    - **Function**: Significantly reduces parameter storage within each group while preserving "fine-grained specialization."
    - **Mechanism**: For each group $g$, compute the shared base $W_{\text{base}}^g=\frac{1}{|\mathcal{G}_g|}\sum_{i\in\mathcal{G}_g}W_i$. Each expert is represented as $\tilde W_i=W_{\text{base}}^g+A_i B_i^\top$, where $A_i\in\mathbb{R}^{d_{in}\times r}, B_i\in\mathbb{R}^{d_{out}\times r}$, with $r\ll \min(d_{in},d_{out})$ (default $r=16$). During the forward pass, $\tilde W_i x=W_{\text{base}}^g x+A_i(B_i^\top x)$. The base computation $W_{\text{base}}^g x$ can be reused for all experts in the group processing the same tokens. Compression ratio $CR=\frac{K d_{in} d_{out}}{d_{in} d_{out}+K r(d_{in}+d_{out})}$, approximately 6.6× for $d=4096, K=8, r=16$. Initialization uses $\text{TSVD}(W_i-W_{\text{base}}^g)$ for a warm start.
    - **Design Motivation**: Since intra-group experts are functionally similar, their unique "expertise" likely resides in a low-rank subspace. Extracting the commonality into a base matrix and compressing specializations into rank-16 residuals achieves high compression with minimal diversity loss. Frobenius reconstruction error is kept under 1.5%.

3. **Two-Level Hierarchical Routing + Heterogeneous Precision + Dynamic Offloading**:
    - **Function**: Addresses system-side communication and memory bottlenecks.
    - **Mechanism**: (a) **Hierarchical Routing**: The router first assigns tokens to groups ($O(G)$ instead of $O(E)$), then to specific experts within the group. This reduces cross-device all-to-all traffic as group-level dispatch acts as a coarse load balancer. (b) **Heterogeneous Precision**: $W_{\text{base}}^g$ uses FP16 (shared and precision-sensitive), while residuals $A_i, B_i$ use INT4 quantization (small magnitude, error-absorbable). (c) **Dynamic Offloading**: Inactive expert groups are offloaded from GPU to CPU/NVMe as needed, keeping peak VRAM close to dense models.
    - **Design Motivation**: All three system optimizations leverage the same group structure. Clustering serves not just compression but also communication (group granularity), memory (group offloading), and precision (heterogeneous base/residual).

### Loss & Training
Optimizes task loss plus three regularizers $I_{\text{load}}, R_{\text{red}}, C_{\text{comm}}$ as per Eq. 1. Parameters such as $T$, $m$, $\tau$, $\alpha$, $\beta$, $r$, $G$, and quantization bits are configurable. Default values ensure stable convergence on GLUE/WikiText-103.

## Key Experimental Results

### Main Results

| Metric | Standard MoE | Ours |
|---|---|---|
| Total Parameters (Relative) | 1.0× | ≈ 0.20× (~80% reduction) |
| Inference Throughput | 1.0× | 1.10–1.20× |
| Expert Load Variance | 1.0× | < 0.33× (> 3× reduction) |
| GLUE / WikiText-103 Quality | baseline | Comparable |
| Peak VRAM | High (Linear with experts) | Near dense model |

### Ablation Study

| Configuration | Observation |
|---|---|
| Low-rank residual only (no grouping) | Shared base fails, poor intra-group correlation, reconstruction error spikes |
| Clustering only (no compression) | Improved communication and load variance, but no parameter reduction |
| Hierarchical routing only (fixed experts) | Reduced communication, but redundancy and load shift persist |
| Full Framework | Simultaneously reaches Pareto front for all three system metrics |
| $r=4$ | High CR but reconstruction error > 1.5%, quality drops |
| $r\in\{16, 32\}$ | Reconstruction error plateaus; $r=16$ offers best cost-performance |

### Key Findings
- $r=16$ is the sweet spot: higher values (32) barely reduce reconstruction error while linearly increasing VRAM/latency; lower values (4/8) lack sufficient residual capacity.
- Both similarity components are essential: removing $S_{\text{param}}$ leads to large intra-group weight variance and residual failure; removing $S_{\text{task}}$ causes fragmented activation patterns, making hierarchical routing essentially random.
- Using router logits as token semantic embeddings provides a cheap, LLM-native semantic signal for clustering, enabling the "online learning" of functional grouping.

## Highlights & Insights
- Elevates grouping from a "post-hoc compression trick" to a "first-class architectural citizen"—a dynamic grouping mechanism driving compression, routing, and memory policies.
- The shared base + low-rank residual approach ("centralize commonality + retain individuality") is conceptually aligned with LoRA/MoLE/PERFT but applied internally to experts and maintained dynamically during training.
- Heterogeneous precision (FP16 base + INT4 residual) exploits the fact that residuals have small magnitudes, avoiding the "accuracy cliff" of uniform INT4 quantization. This can be extended to any "backbone + adapter" compression scenario.

## Limitations & Future Work
- Online clustering involves $O(E^2)$ comparisons; while mitigated by neighbor graphs and caching, it may become an overhead for thousands of experts.
- Evaluation is limited to GLUE/WikiText-103, which is small compared to modern MoE LLMs (e.g., Mixtral, DeepSeek-V3), leaving scalability evidence limited.
- Re-clustering causes momentary "oscillations" in low-rank residual warm starts; while SVD helps, stability in long training runs requires further validation.
- The interaction between dynamic offloading and expert parallelism in multi-node training is complex and not fully discussed relative to ZeRO-3/FSDP.

## Related Work & Insights
- **vs MoE-Lite**: Treats experts as independent; Ours uses clustering to discover inter-expert similarity for shared bases, achieving higher compression while retaining specialization.
- **vs Sub-MoE / Expert-Fusion**: These perform static/permanent merging, losing specialization; Ours uses dynamic clustering + residuals to preserve individuality without permanent information loss.
- **vs Tutel / SmILE / MoE-Lightning**: These optimize communication for fixed architectures; Ours restructures expert organization to provide group-level granularity for system optimization.
- **vs StableMoE / Switch-loss**: These modify router behavior to suppress imbalance; Ours provides structural suppression (group dispatch is a coarse balancer) without depending solely on auxiliary losses.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Dynamic clustering as a unified carrier for compression, routing, and memory is a rare "unified prescription" in MoE co-design.
- Experimental Thoroughness: ⭐⭐⭐ GLUE/WikiText-103 are small; lacks validation on massive MoE LLMs, though ablations and hyperparameter sweeps are relatively complete.
- Writing Quality: ⭐⭐⭐⭐ The trilemma narrative is clear, Eq. 1 explicitly defines targets and variables, and the methodology is well-structured.
- Value: ⭐⭐⭐⭐ ~80% parameter reduction, 10–20% throughput gain, and 3× lower load variance is a highly attractive combination for MoE deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs](gemq_global_expert-level_mixed-precision_quantization_for_moe_llms.md)
- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](../../ICLR2026/model_compression/steering_moe_llms_via_expert_deactivation.md)
- [\[AAAI 2026\] CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis](../../AAAI2026/model_compression/camera_multi-matrix_joint_compression_for_moe_models_via_mic.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](../../ICLR2026/model_compression/sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [\[ICML 2026\] DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts](dag-moe_from_simple_mixture_to_structural_aggregation_in_mixture-of-experts.md)

</div>

<!-- RELATED:END -->
