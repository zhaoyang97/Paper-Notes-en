---
title: >-
  [Paper Note] HASTE: Hardware-Aware Dynamic Sparse Training for Large Output Spaces
description: >-
  [ICML 2026][Extreme Multi-label Classification (XMC)] For Extreme Multi-label Classification (XMC) with millions of labels, HASTE transforms "per-label independent fan-in sampling" into "group-shared fan-in based on sema…
tags:
  - "ICML 2026"
  - "Extreme Multi-label Classification (XMC)"
  - "fixed fan-in sparsity"
  - "group sharing"
  - "Tensor Core"
  - "head-tail split"
date: 2026-05-08
content_hash: 7e7a1ba4b569a34f
---

# HASTE: Hardware-Aware Dynamic Sparse Training for Large Output Spaces

**Conference**: ICML 2026  
**arXiv**: [2606.01117](https://arxiv.org/abs/2606.01117)  
**Code**: https://github.com/xmc-aalto/haste  
**Area**: Model Compression / Extreme Multi-label Classification / Hardware-Aware Sparse Training  
**Keywords**: Extreme Multi-label Classification (XMC), fixed fan-in sparsity, group sharing, Tensor Core, head-tail split  

## TL;DR
For Extreme Multi-label Classification (XMC) with millions of labels, HASTE transforms "per-label independent fan-in sampling" into "group-shared fan-in based on semantic clustering." Combined with a small dense head for high-frequency labels, this allows sparse training to achieve wall-clock gains proportional to FLOPs on GPUs—reaching up to $4.4\times$ speedup in the forward pass and $25\times$ in the backward pass compared to existing sparse baselines, while nearly closing the accuracy gap with dense models.

## Background & Motivation

**Background**: The bottleneck in Extreme Multi-label Classification (XMC) lies in the output layer—when the label count $L \sim 10^6$, the weight matrix $W \in \mathbb{R}^{L \times H}$ consumes significant memory and compute. Over the past decade, two main approaches have emerged: one focuses on label trees or nearest neighbor sampling (LightXML, CascadeXML, Renee series) to reduce **computation** while keeping **VRAM** usage high; the other directly sparsifies the output layer (Spartex, etc.) to compress both computation and memory.

**Limitations of Prior Work**: Direct sparsification seems elegant but is inefficient on GPUs. Unstructured sparsity is often anti-optimized for modern Tensor Cores due to random, uncoalesced memory access and low utilization, resulting in minimal wall-clock gains despite 90% FLOPs reduction. Recent "semi-structured fixed fan-in" methods (Spartex) assign $F$ input connections per label for load balancing; however, fan-in indices are **sampled independently and randomly** for each label. This causes adjacent labels to read completely different features, leading to poor cache hits and memory bandwidth bottlenecks.

**Key Challenge**: To achieve real wall-clock speedups, sparsity must provide **regular memory access patterns** (to enable coalescing) and **cross-output feature reuse** (to tile $H_k$ into shared memory). While block sparsity (BLOCK-SPARSE) achieves this, it severely limits expressivity by forcing all labels onto the same contiguous feature block, causing a 5–10 point drop in accuracy. Furthermore, sparse connections provide weak gradient signals to the encoder for long-tail labels, forcing Spartex to use an auxiliary loss that complicates hyperparameter tuning.

**Goal**: (i) Find an intermediate structure between "per-label independent fan-in" and "strict block sparsity" to balance memory regularity and expressivity; (ii) provide stable gradients to the encoder in long-tail scenarios via a data-driven approach rather than auxiliary supervision.

**Key Insight**: XMC labels are naturally clustered by semantics—in Amazon recommendation tasks, "wireless headphones" and "Bluetooth speakers" naturally utilize similar feature subsets. Therefore, letting **semantically similar labels share the same set of fan-in indices** aligns with the task structure and amortizes the cost of redundant feature reads across a group of labels.

**Core Idea**: Replace per-label fan-in with **group-shared fixed fan-in sparsity**. The output layer is split into a "dense head" for high-frequency labels and a "group-shared sparse tail" for massive long-tail labels. High-performance CUDA kernels are implemented to leverage Tensor Cores for this specific structure.

## Method

### Overall Architecture
Input $x$ passes through a shared encoder to obtain $h = f_\theta(x) \in \mathbb{R}^H$. The output layer is explicitly split into two branches:

- **Dense head**: For the top 2–5% high-frequency labels $\mathcal{H}$, using a lightweight projection $h_{\text{head}} = P_{\text{head}}h$ followed by dense weights $W_{\text{head}}$.
- **Sparse tail**: For the remaining $\mathcal{T}$ long-tail labels, using $h_{\text{tail}} = P_{\text{tail}}h$ followed by a group-shared fixed fan-in sparse layer.

The logit for label $\ell \in \mathcal{G}_k$ is $z_\ell(x) = \langle w_\ell, h_{\mathcal{I}_{g(\ell)}} \rangle$, where $w_\ell \in \mathbb{R}^F$ is the weight exclusive to that label, and $\mathcal{I}_{g(\ell)} \subseteq [H]$ is the **shared fan-in index set** for its group ($|\mathcal{I}_k| = F$). Training uses BCE, alternating between a "continuous phase" (parameter optimization) and a "discrete phase" (rewiring via dynamic sparse training protocols to update $\mathcal{I}_k$).

### Key Designs

1. **Group-Shared Fixed Fan-in Sparsity**:
    - **Function**: Partitions labels $\{1, \dots, L\}$ into $K$ groups $\{\mathcal{G}_k\}$ of size $G$, where labels in a group share the same fan-in index set $\mathcal{I}_k$ but maintain independent weights $w_\ell$. Index memory is reduced by a factor of $G$.
    - **Mechanism**: Grouping is defined as $\{\mathcal{G}_k\} = \arg\max_{\text{partition}} \sum_k \sum_{\ell \in \mathcal{G}_k} \mathrm{sim}(e_\ell, \mu(\mathcal{G}_k))$, where label embeddings $e_\ell = \mathrm{Normalize}(\frac{1}{|\mathcal{P}_\ell|} \sum_{i \in \mathcal{P}_\ell} h_i)$ are the means of encoder representations of positive samples. Since millions of labels make exact solving infeasible, a two-stage approximation is used: mini-batch spherical $k$-means to form coarse clusters, followed by greedy inner-cluster grouping.
    - **Design Motivation**: Transitioning from random fan-in to group sharing reduces index memory, allows for gathered feature tiles to be reused by multiple labels, and introduces a task-aligned inductive bias.

2. **Tensor Core Aware Gather-Once + Dense MMA Kernel**:
    - **Function**: Translates the group-shared structure into hardware-friendly CUDA operations. Each thread block uses a multiple of $G$ as the tile size in the label dimension.
    - **Mechanism**: The forward pass becomes $Z_k = H_k W_k^\top \in \mathbb{R}^{B_t \times G}$, where $H_k = h_{:, \mathcal{I}_k} \in \mathbb{R}^{B_t \times F}$ is a feature tile gathered once into shared memory, and $W_k \in \mathbb{R}^{G \times F}$ contains group weights. This is a **dense** GEMM that utilizes Tensor Core MMA primitives. The backward pass for weights $\nabla W_k = (\nabla Z_k)^\top H_k$ is similarly a dense GEMM. For gradients w.r.t. features, a Split-$K$ strategy is used across labels to handle index overlaps.
    - **Design Motivation**: Unlike Spartex, where each label performs a thin vector dot product, HASTE enables arithmetic intensity equivalent to dense GEMMs by reusing the gathered $H_k$ across $G$ labels.

3. **Head–Tail Split vs. Auxiliary Supervision**:
    - **Function**: Splits the label set by frequency $\mathcal{Y} = \mathcal{H} \cup \mathcal{T}$.
    - **Mechanism**: The dense head provides stable, dense gradient feedback to the encoder since high-frequency labels are activated in almost every batch. The sparse tail then fine-tunes the encoder for specific long-tail features.
    - **Design Motivation**: This replaces the need for auxiliary losses with a data-driven architectural prior. It avoids hyperparameter sensitivity (e.g., loss weights, temperature) and directly leverages the long-tail distribution of the data.

### Loss & Training
End-to-end BCE with BF16 precision. The encoder uses Adam, while the output layer uses SGD with momentum. Dynamic sparse training follows the RigL approach, periodically updating group-level $\mathcal{I}_k$ while maintaining $|\mathcal{I}_k| = F$.

## Key Experimental Results

### Main Results
Evaluated on four XMC datasets with up to 8.6M labels.

| Dataset | Metric | Dense | Spartex (Prev. SOTA) | Block Sparse | HASTE | VRAM (GiB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Amazon-670K | P@1 | 50.6 | 47.1 | 45.0 | **48.1** | 2.1 (vs Spartex 3.7) |
| AmazonTitles-670K | P@1 | 43.7 | 42.6 | 39.4 | **43.0** | 3.2 (vs Spartex 5.0) |
| Amazon-3M | P@1 | 52.6 | 50.2 | 27.9 | **52.5** | 5.67 (vs Spartex 13.5) |
| LF-Paper2Keywords-8.6M | P@1 | 43.6 | 40.7 | 22.8 | **47.5** | 12.5 (vs Spartex 18.4) |

Ours **consistently outperforms Spartex** while using 1.5–2.5$\times$ less memory. Epoch time on Amazon-3M is reduced from 86:38 to 21:39. On the 8.6M label dataset, P@1 even **exceeds the dense baseline** by 3.9 points.

### Ablation Study
| Configuration | P@1 (Amazon-670K) | Note |
| :--- | :--- | :--- |
| HASTE (Full) | 48.1 | Semantic grouping + HT split |
| Random grouping | 46.3 | Semantic grouping Gain: +1.8 |
| Frequency grouping | 46.7 | Inferior to semantic grouping |
| No Head–Tail Split | 46.8 | HT split Gain: +1.3 |
| Group size $G=16$ | 48.1 | Best expressivity |
| Group size $G=64$ | 47.5 | Fastest kernel, lowest accuracy |

### Key Findings
- **Kernel-level micro-benchmarks are a major highlight**: The forward pass is up to $4.4\times$ faster and the backward pass up to $25\times$ faster than standard fixed fan-in. Sparse FLOPs are finally converted into wall-clock acceleration.
- **Semantic grouping outperforms frequency grouping** (+1.4 P@1), validating the inductive bias of sharing fan-ins based on task structure.
- **PSP@k (Propensity-Scored Precision) increases**: On Amazon-3M, PSP@1 rises from 14.3 (Spartex) to 15.9, showing that the head–tail split benefits long-tail labels by improving the shared encoder.
- **Group size $G$ is a classic trade-off**: Larger $G$ facilitates better hardware utilization (reuse, Split-$K$ parallelism) but slightly reduces precision; $G=16 \sim 32$ is identified as the "sweet spot."

## Highlights & Insights
- **Alignment of Inductive Bias and Hardware**: Usually, being "GPU-friendly" conflicts with "task-accuracy," but here, semantically similar labels sharing features aligns corporate logic with the Tensor Core's need for regular data tiling.
- **Data Structure over Auxiliary Loss**: Instead of adding a second loss to stabilize training, the model architectural split (Head-Tail) uses the dataset's intrinsic frequency distribution to provide stable gradients, making it more robust across different labels.
- **Honest Metrics for Sparsity**: The paper reports wall-clock time and VRAM instead of just FLOPs, and compares against FLOPs-matched dense models, setting a higher standard for sparsity research.

## Limitations & Future Work
- Kernels were only evaluated on a single A100 GPU; the interaction between group-shared fan-in and multi-GPU NCCL all-reduce remains unexplored.
- Grouping requires an initial encoder representation to calculate $e_\ell$; the authors use a pre-trained BERT, but bootstrapping from a from-scratch encoder is not explicitly detailed.
- $G$ is a manual hyperparameter; automated tuning or integration with N:M sparsity is a potential next step.

## Related Work & Insights
- **vs. Spartex**: HASTE removes the "random gather" bottleneck and stabilizes the encoder via Head-Tail splits instead of auxiliary supervised tasks.
- **vs. BLOCK-SPARSE (2025)**: While block sparsity is fast, it is too restrictive. HASTE finds a middle ground with flexible "group-shared but arbitrary indices."
- **vs. ELMO (FP8 Quantization)**: These methods are orthogonal. HASTE reduces connection density whereas ELMO reduces numerical precision. On 8.6M labels, HASTE outperforms ELMO, suggesting "cutting connections" is more effective than "cutting precision" at extreme scales.

## Rating
- Novelty: ⭐⭐⭐⭐ Group-shared fan-in is a natural interpolation between fixed fan-in and block sparsity, executed with synergistic hardware and architectural designs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Massive scales (8.6M labels), detailed micro-benchmarks, and comparisons across dense, sparse, and quantized baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear formulas and hardware context; though the choice of the head-tail split point could use more ablation.
- Value: ⭐⭐⭐⭐⭐ Effectively solves the "FLOPs $\neq$ Speed" issue in sparse training for industrial XMC applications like search and recommendation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AMDP: Asynchronous Multi-Directional Pipeline Parallelism for Large-Scale Models Training](amdp_asynchronous_multi-directional_pipeline_parallelism_for_large-scale_models_.md)
- [\[CVPR 2026\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](../../CVPR2026/others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)
- [\[ICML 2026\] Decision Tree Learning on Product Spaces](decision_tree_learning_on_product_spaces.md)
- [\[ICML 2026\] Torus Graphs for Large-Scale Neural Phase Analysis](torus_graphs_for_large_scale_neural_phase_analysis.md)
- [\[NeurIPS 2025\] Sign-In to the Lottery: Reparameterized Sparse Training from Scratch](../../NeurIPS2025/others/sign-in_to_the_lottery_reparameterizing_sparse_training_from_scratch.md)

</div>

<!-- RELATED:END -->
