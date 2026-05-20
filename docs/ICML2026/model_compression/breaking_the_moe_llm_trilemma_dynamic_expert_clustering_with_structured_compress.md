---
title: >-
  [Paper Note] Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression
description: >-
  [ICML 2026][Model Compression][Mixture-of-Experts] To address the MoE LLM "load imbalance–parameter redundancy–communication overhead" trilemma…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Mixture-of-Experts"
  - "Dynamic Expert Clustering"
  - "Low-Rank Residual"
  - "Hierarchical Routing"
  - "Heterogeneous Precision"
date: 2026-05-08
content_hash: 6878632f79b3cb3c
---

# Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression

**Conference**: ICML 2026  
**arXiv**: [2510.02345](https://arxiv.org/abs/2510.02345)  
**Code**: https://github.com/szdtzpj/Breaking_the_moe_trilemma (available)  
**Area**: Model Compression / MoE LLM / System Optimization  
**Keywords**: Mixture-of-Experts, Dynamic Expert Clustering, Low-Rank Residual, Hierarchical Routing, Heterogeneous Precision

## TL;DR
To address the MoE LLM "load imbalance–parameter redundancy–communication overhead" trilemma, this paper proposes a unified framework: experts are grouped online via dual "parameter + activation" similarity clustering; within each group, "shared base matrix + low-rank residual" structured compression (~5×) is applied; then, a two-level hierarchical routing ("group selection then expert selection") is performed, combined with FP16/INT4 heterogeneous precision and offline unloading of idle groups. On GLUE/WikiText-103, this achieves about 80% parameter reduction, 10–20% throughput improvement, and a 3× reduction in expert load variance, while matching standard MoE performance.

## Background & Motivation

**Background**: MoE has become a key path for scaling LLMs (Switch, GShard, Mixtral, etc.)—in theory, it increases parameter capacity without significantly increasing FLOPs.

**Limitations of Prior Work**: Deploying MoE on A100/H100 hardware faces the "optimization trilemma": (i) **Load imbalance**—top-$k$ gating causes a few experts to overload while most remain idle; (ii) **Parameter redundancy**—the number of experts linearly increases parameters, exhausting HBM capacity; (iii) **All-to-all communication overhead**—dispatching tokens to experts across devices often dominates latency, especially for long sequences.

**Key Challenge**: Existing methods address only one aspect. Load balancing loss (Switch loss) is reactive and fails under distribution drift; MoE-Lite compresses each expert independently, ignoring structural similarity among experts; communication-aware routing (Tutel, SmILE) optimizes data paths for fixed architectures, but cannot address redundancy or imbalance. Worse, optimizing one variable often exacerbates another.

**Goal**: To simultaneously reduce total parameter storage, compress per-token activation parameters, maintain model quality, lower cross-device traffic, and keep reclustering overhead manageable within a single framework.

**Key Insight**: The core observation is—"experts activated by semantically similar inputs also exhibit parameter redundancy." This hypothesis enables joint optimization of architecture (grouping) and system (routing/storage/communication).

**Core Idea**: Dynamically cluster functionally similar experts → apply shared base + low-rank residual compression within groups → hierarchical routing first to group, then to expert, reducing all-to-all to inter-group + intra-group two levels.

## Method

### Overall Architecture
The unified objective (Eq. 1) is $\min L_{\text{task}}+A_1 I_{\text{load}}+A_2 R_{\text{red}}+A_3 C_{\text{comm}}$, with four learnable/designable variables: Grouping, Compression Parameterization (Param, Factors), Routing Strategy. The overall pipeline consists of four steps: (1) Online dual-similarity clustering → partition $E$ experts into $G$ groups, each with $K=E/G$ experts; (2) Within-group shared base + low-rank residual compression; (3) Two-level hierarchical routing (group selection, then expert selection within group); (4) Heterogeneous precision (FP16 base + INT4 residual) + dynamic offloading of idle groups.

### Key Designs

1. **Online Dual-Similarity Clustering**:

    - **Function**: Every $T$ steps, experts are regrouped based on "parameter + activation" dual similarity, forming the basis for subsequent compression/routing/memory strategies.
    - **Mechanism**: For each expert $\mathcal{E}_i$, maintain two features—weight vector $\text{vec}(W_i)$ and activation centroid $\mu_i$ (updated via EMA: $\mu_i\leftarrow(1-\beta)\mu_i+\beta\bar{x}_i$, default $\beta=0.05$). Parameter similarity $S_{\text{param}}$ and task similarity $S_{\text{task}}$ are both cosine similarities, fused by weight $\alpha$: $S=\alpha S_{\text{param}}+(1-\alpha)S_{\text{task}}$ (default $\alpha>0.5$, favoring parameter side as it directly reflects function). Every $T$ steps: first, prune low-similarity pairs with threshold $\tau$ (default 0.1) to obtain a neighbor graph, reducing $O(E^2)$ comparisons; then, run K-means++ on distance $D=1-S$ to obtain $G$ groups; if slightly imbalanced, greedily reassign boundary experts. Cache $S_{\text{param}}$ for $m$ steps, recomputing similarity only for experts with weight changes exceeding $\epsilon$ to amortize cost.
    - **Design Motivation**: Parameter similarity alone reflects "looking alike," activation similarity alone reflects "seeing similar inputs"; combining both ensures groups can share parameters and be co-activated by similar tokens. EMA + periodic reclustering allows grouping to adapt to distribution drift, making it much more robust than static grouping.

2. **Structured Compression with Shared Base + Low-Rank Residual**:

    - **Function**: Significantly reduces parameter storage within each group while preserving "fine-grained specialization" among experts.
    - **Mechanism**: For each group $g$, compute shared base $W_{\text{base}}^g=\frac{1}{|\mathcal{G}_g|}\sum_{i\in\mathcal{G}_g}W_i$ (group weight mean); each expert is then represented as a low-rank residual: $\tilde W_i=W_{\text{base}}^g+A_i B_i^\top$, where $A_i\in\mathbb{R}^{d_{in}\times r}, B_i\in\mathbb{R}^{d_{out}\times r}$, $r\ll \min(d_{in},d_{out})$ (default $r=16$). Inference: $\tilde W_i x=W_{\text{base}}^g x+A_i(B_i^\top x)$; $W_{\text{base}}^g x$ can be reused for all experts in the group processing the same token batch. Compression ratio $CR=\frac{K d_{in} d_{out}}{d_{in} d_{out}+K r(d_{in}+d_{out})}$; for $d=4096, K=8, r=16$, about 6.6×. Initialize $A_i B_i^\top$ via $\text{TSVD}(W_i-W_{\text{base}}^g)$ for warm start; after reclustering, immediately perform SVD again.
    - **Design Motivation**: Since experts within a group are functionally similar, their true "expertise" likely lies in a low-rank subspace—extracting the common part into the base matrix and compressing the specialized part into a rank-16 residual enables both significant parameter reduction and diversity retention. Frobenius reconstruction error is controlled within 1.5%, with negligible performance impact.

3. **Two-Level Hierarchical Routing + Heterogeneous Precision + Dynamic Offloading**:

    - **Function**: Simultaneously addresses communication and memory system bottlenecks.
    - **Mechanism**: (a) **Hierarchical Routing**—first-level router sends tokens to groups ($O(G)$ instead of $O(E)$), second-level selects specific expert within group; thus, all-to-all communication is first coarse-grained at group level, then fine-grained within group, significantly reducing cross-device traffic. Group-level dispatch also acts as a coarse load balancer. (b) **Heterogeneous Precision**—shared base $W_{\text{base}}^g$ uses FP16 (shared, precision-sensitive), low-rank residuals $A_i, B_i$ use INT4 quantization (small magnitude, quantization error absorbable). (c) **Dynamic Offloading**—entire inactive expert groups are offloaded from GPU to CPU/NVMe as needed, making peak GPU memory usage close to dense models.
    - **Design Motivation**: All three components leverage the same "group structure"—clustering is not just for compression, but also benefits communication (group-level granularity), memory (group-level offloading), and precision (base/residual heterogeneity), embodying the true meaning of a "unified framework."

### Loss & Training
Eq. 1 jointly optimizes task loss plus three regularization terms $I_{\text{load}}, R_{\text{red}}, C_{\text{comm}}$, with $A_1, A_2, A_3$ as hyperparameters. Clustering period $T$, cache lifetime $m$, similarity threshold $\tau$, fusion weight $\alpha$, EMA rate $\beta$, low rank $r$, number of groups $G$, and quantization bits are all configurable; default values provided in the paper ensure stable convergence on GLUE/WikiText-103.

## Key Experimental Results

### Main Results

| Metric | Standard MoE | Ours |
|---|---|---|
| Total Parameters (relative) | 1.0× | ≈ 0.20× (about 80% reduction) |
| Inference Throughput | 1.0× | 1.10–1.20× |
| Expert Load Variance | 1.0× | < 0.33× (over 3× reduction) |
| GLUE / WikiText-103 Quality | baseline | on par |
| Peak GPU Memory | High (scales linearly with experts) | Close to dense model |

(Core numbers are mainly reported in the Abstract/Introduction; detailed tables are in the appendix.)

### Ablation Study

| Configuration | Observation |
|---|---|
| Low-rank residual only (no grouping) | Shared base ineffective, poor within-group correlation, reconstruction error spikes |
| Clustering only (no compression) | Communication and load variance improve, but parameter count unchanged |
| Hierarchical routing only (fixed experts) | Communication drops, but parameter redundancy and load drift persist |
| Full framework | All three system metrics reach Pareto frontier simultaneously |
| $r=4$ | High CR but reconstruction error > 1.5% threshold, quality drops |
| $r\in\{16, 32\}$ | Reconstruction error plateaus, $r=16$ offers best trade-off |

### Key Findings
- $r=16$ is the sweet spot: increasing to 32 barely reduces reconstruction error but linearly increases memory/latency; decreasing to 4/8 lacks residual capacity, hurting performance.
- Both dual similarities are necessary: removing $S_{\text{param}}$ leads to large within-group weight differences and ineffective low-rank residuals; removing $S_{\text{task}}$ causes fragmented activation patterns and makes hierarchical routing essentially random.
- Using router logits as token semantic embeddings (as observed by Li & Zhou 2024) provides a cheap, LLM-native semantic signal for clustering, enabling online learning of functional groupings.

## Highlights & Insights
- Elevates grouping from a "post-hoc compression trick" to a "first-class architectural citizen"—a single dynamic grouping simultaneously drives compression, routing, and memory strategies, representing a new paradigm for MoE co-design.
- The "shared base + low-rank residual" approach of "centralizing commonality + reserving individuality" is fundamentally related to LoRA / MoLE / PERFT, but is applied within experts and maintained dynamically during training for the first time.
- Heterogeneous precision (FP16 base + INT4 residual) leverages the physical fact that "residuals are small and errors are absorbable," avoiding the accuracy cliff of indiscriminate INT4 quantization for all experts; this idea can be directly applied to any "backbone + adapter" compression scenario.

## Limitations & Future Work
- Online clustering involves $O(E^2)$ comparisons; the authors reduce this with neighbor graphs and caching, but for $E$ in the thousands, it remains explicit overhead. The real impact on training throughput needs validation at larger scales (e.g., DeepSeek-MoE scale).
- Evaluation datasets are mainly GLUE / WikiText-103, which are small compared to modern MoE LLM (Mixtral / DeepSeek-V3 / Qwen3-MoE) training scales; scalability evidence is limited.
- Reclustering can cause "oscillation" in low-rank residual warm starts; the paper mitigates this with SVD warm restarts, but stability over long training runs requires larger-scale experiments.
- Dynamic offloading interacts subtly with expert parallelism in cross-node training; the paper does not discuss coupling with memory optimizations like ZeRO-3 / FSDP.

## Related Work & Insights
- **vs MoE-Lite**: Treats experts as independent for compression; this work discovers similarity among experts via clustering, enabling within-group sharing—preserving specialization while achieving higher compression.
- **vs Sub-MoE / Expert-Fusion**: Their static/permanent merging loses specialization; this work uses dynamic clustering + residuals to retain specialization, with no permanent information loss.
- **vs Tutel / SmILE / MoE-Lightning**: These optimize communication for fixed architectures; this work restructures expert organization at the architectural level, providing "group" as a first-class granularity for communication optimization.
- **vs StableMoE / Switch-loss**: These modify router behavior to suppress imbalance; this work suppresses imbalance structurally (group-level dispatch is a natural coarse balancer), not relying on auxiliary losses.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elevates dynamic clustering to a shared foundation for compression, routing, and memory—a rare "unified prescription" in MoE co-design.
- Experimental Thoroughness: ⭐⭐⭐ Datasets (GLUE/WikiText-103) are small, lacking validation on larger MoE LLMs; however, ablations and hyperparameter sweeps are relatively complete.
- Writing Quality: ⭐⭐⭐⭐ The trilemma is clearly articulated, Eq. 1 lays out objectives and variables, and the method is well-structured.
- Value: ⭐⭐⭐⭐ The combination of ~80% parameter reduction + 10–20% throughput + 3× load variance reduction is highly attractive, offering an engineering-ready direction for MoE deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](../../ICLR2026/model_compression/steering_moe_llms_via_expert_deactivation.md)
- [\[AAAI 2026\] CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis](../../AAAI2026/model_compression/camera_multi-matrix_joint_compression_for_moe_models_via_mic.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](../../ICLR2026/model_compression/sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [\[ICML 2026\] RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression](rq-moe_residual_quantization_via_mixture_of_experts_for_efficient_input-dependen.md)
- [\[ICLR 2026\] MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE](../../ICLR2026/model_compression/mone_replacing_redundant_experts_with_lightweight_novices_for_structured_pruning.md)

</div>

<!-- RELATED:END -->
