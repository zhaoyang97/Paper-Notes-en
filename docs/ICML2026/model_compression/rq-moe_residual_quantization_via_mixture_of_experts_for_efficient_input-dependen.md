---
title: >-
  [Paper Note] RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression
description: >-
  [ICML 2026][Model Compression][Residual Quantization] RQ-MoE utilizes a "two-level MoE + dual-stream quantization" design to enable dynamically generated codebooks for Residual Quantization (RQ). By decoupling the instru…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Residual Quantization"
  - "MoE"
  - "Input-Adaptive Codebook"
  - "Parallel Decoding"
  - "Normalized Residual Loss"
date: 2026-05-08
content_hash: aea50c0088026f84
---

# RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression

**Conference**: ICML 2026  
**arXiv**: [2605.14359](https://arxiv.org/abs/2605.14359)  
**Code**: [KDEGroup/RQ-MoE](https://github.com/KDEGroup/RQ-MoE)  
**Area**: Model Compression / Vector Quantization  
**Keywords**: Residual Quantization, MoE, Input-Adaptive Codebook, Parallel Decoding, Normalized Residual Loss

## TL;DR
RQ-MoE utilizes a "two-level MoE + dual-stream quantization" design to enable dynamically generated codebooks for Residual Quantization (RQ). By decoupling the instruction stream from the reconstruction flow, it achieves $6-14\times$ decoding speedups while maintaining or exceeding QINCo's MSE/Recall performance across four retrieval benchmarks.

## Background & Motivation
**Background**: Vector Quantization (VQ) achieves compression by mapping high-dimensional vectors to "codebook centers." Multi-Codebook Quantization (MCQ) further reduces errors using multiple small codebooks, with the "stepwise approximation" strategy of Residual Quantization (RQ) widely applied in recommendation systems, speech codecs, and generative RecSys tokenization. Recently, QINCo upgraded RQ to a "dynamic codebook" approach, where an MLP dynamically generates the next-step codebook based on the partially reconstructed result, significantly improving reconstruction quality.

**Limitations of Prior Work**: (i) Traditional RQ uses static codebooks, applying a "one-size-fits-all" approach to the local manifold geometry of different regions, which limits expressiveness; (ii) QINCo introduces strict sequential dependencies—the step $m$ codebook requires the reconstruction from steps $1 \ldots m-1$, preventing parallel decoding and increasing deployment latency; (iii) "Explicit gating" when applying MoE wastes bit budgets (e.g., 4 experts require 2 extra bits, a 25% overhead for 256-entry codebooks).

**Key Challenge**: There is a natural conflict between dynamic codebooks (for quality) and parallel decoding (for speed). If the codebook depends on prior reconstruction, it must be sequential; if parallelized, it loses input-adaptive capabilities.

**Goal**: To fully parallelize decoding without adding extra bits or losing input adaptability, while maintaining or surpassing the reconstruction and retrieval accuracy of QINCo.

**Key Insight**: The authors re-examine RQ and observe that it can be viewed as a degenerate MoE (nearest neighbor search = top-1 implicit routing). By binding the "expert information" of each codeword with its "quantization component" under the same index, routing becomes "free." Simultaneously, by stripping "instruction propagation" from the "reconstruction path," parallelism is achieved.

**Core Idea**: A high-dimensional codebook $\mathbf{w}_k^m=[\mathbf{c}_k^m;\mathbf{e}_k^m]$ is used to bind quantization components and expert components to the same index (level-1 MoE implicit routing). The instruction accumulation stream is decoupled from the codebook generation stream (level-2 MoE deforms a base codebook according to accumulated instructions), supporting fully parallel decoding.

## Method

### Overall Architecture
RQ-MoE follows the "stepwise residual refinement" backbone of RQ: $M$ quantization steps, selecting an index $i^m$ at each step. However, it maintains two streams in parallel:

- **Instruction Stream**: Stores accumulated expert information $\mathbf{I}^m \in \mathbb{R}^{D_e}$ with a minimal update rule: $\mathbf{I}^m = \mathbf{I}^{m-1} + \mathbf{E}_{i^{m-1}}^{m-1}$, with initial $\mathbf{I}^1 = \mathbf{0}$.
- **Quantization Stream**: At step $m$, a static base codebook $\mathcal{C}^m$ is deformed into a dynamic codebook $\tilde{\mathcal{C}}^m = \{\tilde{\mathbf{c}}_k^m\}$ via a level-2 MoE function $f_t$ based on $\mathbf{I}^m$. Nearest neighbor search is then performed on the dynamic codebook: $i^m = \arg\min_k \|\mathbf{r}^m - \tilde{\mathbf{c}}_k^m\|_2^2$.

The final reconstruction $\hat{\mathbf{x}} = \sum_{m=1}^M \tilde{\mathbf{c}}_{i^m}^m$ is consistent with the standard RQ summation. This means that once the index sequence is obtained during decoding, all $\mathbf{I}^m$ can be derived via "index lookup + addition" in parallel, and all $\tilde{\mathcal{C}}^m$ can be generated concurrently, completely removing sequential dependencies from the decoding path.

### Key Designs

1.  **Level-1 MoE with Implicit Routing + Index Reuse (High-Dimensional Codebook)**:
    - **Function**: Enables the nearest neighbor index to simultaneously perform "codeword selection" and "expert routing" without adding extra bits.
    - **Mechanism**: Each codeword is defined as a $(D+D_e)$-dimensional vector $\mathbf{w}_k^m=[\mathbf{c}_k^m;\mathbf{e}_k^m]$. The first $D$ dimensions $\mathbf{c}_k^m$ constitute the "base codebook" for residual matching, while the latter $D_e$ dimensions $\mathbf{e}_k^m$ represent the expert component encoding local manifold features. Nearest neighbor search calculates distance only on the first $D$ dimensions, but the selected index $i^m$ simultaneously identifies the $D_e$-dimensional expert signal $\mathbf{e}_{i^m}^m$, which is added to $\mathbf{I}^{m+1}$.
    - **Design Motivation**: Naive MoE must store $\log_2 N$ bits per expert for gating. RQ-MoE embeds gating into the "index that must be stored anyway," making routing "zero-cost" while elegantly preserving the simple storage format of RQ.

2.  **Dual-Stream Quantization + Level-2 MoE Codebook Deformation**:
    - **Function**: Uses accumulated instructions $\mathbf{I}^m$ to shift static base codewords $\mathbf{c}_k^m$ near the current input manifold, yielding an "input-adaptive local codebook."
    - **Mechanism**: For each candidate $k$, $\mathbf{z}_k^m = \text{Linear}([\mathbf{c}_k^m;\mathbf{I}^m])$ is first computed to inject instructions into the base codeword. Then, $N$ expert MLPs (each being an $L$-layer bottleneck residual) concurrently compute $\mathcal{E}_n(\mathbf{z}_k^m)$. A gating mechanism $\boldsymbol{\alpha}_k^m = \text{softmax}(\text{Linear}(\mathbf{z}_k^m))$ performs a weighted sum to obtain the offset $\Delta\mathbf{c}_k^m = \sum_n \boldsymbol{\alpha}_{k,n}^m \mathcal{E}_n(\mathbf{z}_k^m)$. Finally, $\tilde{\mathbf{c}}_k^m = \mathbf{c}_k^m + \Delta\mathbf{c}_k^m$. The first step $\tilde{\mathcal{C}}^1 = \mathcal{C}^1$ starts with the base codebook.
    - **Design Motivation**: Decoupling "conditional information" from the "reconstruction path" is key to parallel decoding. Since $\mathbf{I}^m$ only depends on previous indices and expert components (lookup + add) rather than previous reconstruction vectors, $\{\mathbf{I}^1, \ldots, \mathbf{I}^M\}$ can be calculated at once, followed by parallel generation of $\{\tilde{\mathcal{C}}^m\}_{m=1}^M$. This provides theoretical $M\times$ decoding speedup, which can be further multiplied by $N\times$ through inter-expert parallelism.

3.  **Normalized Residual Loss (NRL)**:
    - **Function**: Replaces final-step MSE or stepwise MSE to ensure each quantization step receives balanced gradients according to the "remaining difficulty."
    - **Mechanism**: Defines the relative residual ratio $\rho^m = \|\mathbf{r}^{m+1}\|_2^2 / (\text{sg}(\|\mathbf{r}^m\|_2^2) + \epsilon)$ (where denominator denotes stop-gradient). The loss is $\mathcal{L}_{\text{NRL}} = \sum_{m=1}^M \log(1 + \rho^m)$. Its gradient $\nabla_{\mathbf{r}^{m+1}} \mathcal{L}_{\text{NRL}} = 2\|\mathbf{r}^{m+1}\|_2 / (\|\mathbf{r}^{m+1}\|_2^2 + C)$ increases with $\|\mathbf{r}^{m+1}\|_2$ for moderate residuals and approaches zero for extreme residuals, acting as a redescending influence function.
    - **Design Motivation**: The gradient of pure MSE, $2\|\mathbf{r}^{m+1}\|_2$, scales linearly with residual magnitude. Early residuals are large and late residuals are small, causing gradients for deep experts to be drowned out. NRL automatically normalizes based on "progress relative to the previous step," enabling effective training for deep experts and remaining robust to outliers.

### Loss & Training
All base/expert codebooks, MoE gates, and expert MLPs are optimized end-to-end using only the NRL loss. No auxiliary load-balance loss is introduced, as implicit routing naturally possesses balancing properties via nearest neighbor search.

## Key Experimental Results

### Main Results
Evaluated on Deep1M, BigANN1M, FB-ssnpp1M, and Contriever1M benchmarks with 10M training samples and 8/16 byte budgets. RQ-MoE uses $N=1, L=16$ (Contriever uses $L=12$ to align with QINCo).

| Dataset (8 bytes) | Metric | RQ-MoE | QINCo | OPQ |
|------------------|------|--------|-------|-----|
| Deep1M (D=96) | MSE / R@1 | Comparable or Better | -- | 0.25 / 15.2 |
| BigANN1M (D=128) | MSE (×$10^4$) / R@1 | Comparable or Better | -- | 2.97 / 21.4 |
| FB-ssnpp1M (D=256) | MSE / R@1 | Comparable or Better | -- | 9.51 / 2.5 |
| Contriever1M (D=768) | MSE / R@100 | Comparable or Better | -- | 1.87 / 50.6 |

**Decoding Speedup**: Achieves **$6\times$–$14\times$** speedup compared to QINCo / QINCo2 PAD (varying by dataset and $M$).

**Complexity** (FLOPS per vector, fixed $N \cdot L$ budget):

| Method | Encoding | Decoding |
|------|------|------|
| UNQ | $H'(D+H+Mb+MK)$ | $H'(b+H'+D+M)$ |
| QINCo | $2MKD(D+LH)$ | $2MD(D+LH)$ |
| **RQ-MoE** | $2MKD(D+NLH+N)$ | $2MD(D+NLH+N)$ |

Theoretical decoding speedup: step-level $M\times$ + expert-level $N\times = (M \cdot N)\times$.

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|------|------|
| Full RQ-MoE | SOTA / 6–14× Speedup | Main result |
| Replacing NRL with MSE-final | Insufficient training of late experts | NRL solves deep underfitting |
| Replacing NRL with per-step MSE | Early steps dominate optimization | Excessive initial gradients |
| Disabling level-2 MoE (Fixed base codebook) | Degenerates to RQ, reconstruction error rises | Input adaptability is necessary |
| Coupling instruction and reconstruction (QINCo-style) | Sequential dependency returns, speed drops | Dual-stream decoupling is key to parallelism |
| Explicit gating (extra bits) | Accuracy drops given fixed bit budget | Implicit routing + index reuse is superior |

### Key Findings
- **Theoretical Proof**: RQ-MoE degenerates to standard RQ when $D_e=0$ and $\Delta\mathbf{c}_k^m=0$; it degenerates to QINCo when $f_t$ is a residual-MLP and $D_e=D$. Both are "restricted special cases" of RQ-MoE, making it a unified framework.
- **Expert Dimension $D_e$ Guidance**: Setting $D_e=D$ generally yields stable performance across most benchmarks.
- **Speedup Sources**: Beyond "step-level parallelism," experts within the level-2 MoE can also be parallelized, leading to an $M \cdot N$ advantage in end-to-end latency over QINCo.

## Highlights & Insights
- "Hiding routing information inside existing quantization indices" is a clever design—achieving MoE routing with zero bit overhead while being naturally load-balanced.
- Dual-stream decoupling allows dynamic codebooks and parallel decoding to coexist, overcoming their perceived mutual exclusivity.
- NRL is equivalent to a redescending M-estimator in robust statistics, providing a statistical explanation for why deep experts train well. This loss design is transferable to other "stepwise refinement" tasks (diffusion, autoregressive tokens, etc.).
- RQ-MoE offers a generalized framework: using hyper-dimensional codebooks to bind "main task output + auxiliary routing signals" to a single discrete index is a lightweight way to integrate MoE.

## Limitations & Future Work
- Encoding remains sequential (calculating residuals step-by-step to query dynamic codebooks), although $N$ experts could theoretically accelerate it; parallel encoding is not fully resolved.
- Experiments focus on retrieval/reconstruction metrics; downstream effects on RecSys (e.g., generative recommendation tokenization) or speech codecs are not directly evaluated.
- Training stability is not extensively discussed—while standard MoE requires load balancing/gating noise, this work relies on implicit routing. Scalability robustness remains to be verified.
- $D_e=D$ doubles the codebook storage, which may pose overhead in extremely compact scenarios (e.g., IoT edge devices).

## Related Work & Insights
- **vs RQ / PQ / OPQ**: Classic MCQ methods use static codebooks. RQ-MoE introduces input-conditioned dynamic codebooks while maintaining RQ's "index sequence as encoding" simplicity.
- **vs QINCo / QINCo2**: QINCo pioneered dynamic codebooks but is strictly sequential; QINCo2 uses PAD/beam search but does not fully eliminate sequential dependencies. RQ-MoE eliminates them via dual-stream decoupling.
- **vs UNQ**: UNQ uses deep networks instead of Euclidean distance for lookups but remains a static codebook method; RQ-MoE places network capacity into "codebook generation," better utilizing the sparse activation of MoE.
- **Insight**: RQ-MoE can directly replace RVQ in retrieval-augmented LLMs or generative recommenders, providing faster decoding at the same accuracy level.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Solves contradictory goals of dynamic codebooks and parallel decoding via implicit routing + dual-stream decoupling.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid results on 4 benchmarks + complexity analysis + ablation, though downstream RecSys/Codec tasks are missing.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework diagrams, algorithms, and theoretical summaries.
- **Value**: ⭐⭐⭐⭐ Directly applicable for replacing RVQ in generative recommendation or speech codecs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts](dag-moe_from_simple_mixture_to_structural_aggregation_in_mixture-of-experts.md)
- [\[ICML 2026\] ReSpinQuant: Efficient Layer-Wise LLM Quantization via Subspace Residual Rotation Approximation](respinquant_efficient_layer-wise_llm_quantization_via_subspace_residual_rotation.md)
- [\[ICML 2026\] RaBiT: Residual-Aware Binarization Training for Accurate and Efficient LLMs](rabit_residual-aware_binarization_training_for_accurate_and_efficient_llms.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](../../CVPR2026/model_compression/quant_experts_token_aware_vlm_quantization.md)
- [\[ICML 2026\] Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts](scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts.md)

</div>

<!-- RELATED:END -->
