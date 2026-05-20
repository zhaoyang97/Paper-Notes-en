---
title: >-
  [Paper Note] RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression
description: >-
  [ICML 2026][Model Compression][Residual Quantization] RQ-MoE employs a "two-level MoE + dual-stream quantization" design, enabling the codebook for residual vector quantization (RQ) to be dynamically generated per input…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Residual Quantization"
  - "MoE"
  - "Input-Adaptive Codebook"
  - "Parallel Decoding"
  - "Normalized Residual Loss"
date: 2026-05-08
content_hash: 3f424fc6883f31e8
---

# RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression

**Conference**: ICML 2026  
**arXiv**: [2605.14359](https://arxiv.org/abs/2605.14359)  
**Code**: [KDEGroup/RQ-MoE](https://github.com/KDEGroup/RQ-MoE)  
**Area**: Model Compression / Vector Quantization  
**Keywords**: Residual Quantization, MoE, Input-Adaptive Codebook, Parallel Decoding, Normalized Residual Loss

## TL;DR
RQ-MoE employs a "two-level MoE + dual-stream quantization" design, enabling the codebook for residual vector quantization (RQ) to be dynamically generated per input, and achieves 6–14× decoding acceleration by decoupling the instruction and reconstruction streams. On four retrieval benchmarks, it matches or surpasses QINCo in MSE/Recall.

## Background & Motivation
**Background**: Vector quantization (VQ) compresses high-dimensional vectors by mapping them to "codebook centers." Multi-codebook quantization (MCQ) uses multiple small codebooks to further reduce error, with residual quantization (RQ)'s "stepwise refinement" widely adopted in recommender systems, speech codecs, and generative RecSys tokenization. Recently, QINCo upgraded RQ to a "dynamic codebook"—using an MLP to generate the next-step codebook based on the reconstructed part at each step, significantly improving reconstruction quality.

**Limitations of Prior Work**: (i) Traditional RQ uses static codebooks, applying a "one-size-fits-all" approach to local manifold geometry, limiting expressiveness; (ii) QINCo introduces strict serial dependency—the $m$-th codebook requires the first $1\ldots m-1$ reconstructions, making decoding non-parallelizable and increasing deployment latency; (iii) Naively applying MoE with explicit gating wastes bit budget (4 experts require 2 extra bits, a 25% overhead for a 256-entry codebook).

**Key Challenge**: Dynamic codebooks (for quality) and parallel decoding (for speed) are inherently conflicting—if codebooks depend on previous reconstructions, decoding must be serial; if parallelized, input adaptivity is lost.

**Goal**: Achieve fully parallel decoding without extra bits or loss of input adaptivity, while matching or exceeding QINCo in reconstruction and retrieval accuracy.

**Key Insight**: The authors revisit RQ—as a degenerate MoE (nearest neighbor search = top-1 implicit routing). By binding each codeword's "expert information" and "quantization component" to the same index, routing becomes free; decoupling "instruction propagation" from the "reconstruction path" enables parallelism.

**Core Idea**: Use a high-dimensional codebook $\mathbf{w}_k^m=[\mathbf{c}_k^m;\mathbf{e}_k^m]$ to bind quantization and expert components to the same index (first-level MoE implicit routing), and decouple the instruction accumulation stream from codebook generation (second-level MoE morphs the base codebook according to accumulated instructions), thus supporting fully parallel decoding.

## Method

### Overall Architecture
RQ-MoE follows RQ's "stepwise residual refinement": $M$ quantization steps, each selecting an index $i^m$. At each step, two streams are maintained in parallel:

- **Instruction Stream**: Stores accumulated expert information $\mathbf{I}^m\in\mathbb{R}^{D_e}$, updated simply as $\mathbf{I}^m=\mathbf{I}^{m-1}+\mathbf{E}_{i^{m-1}}^{m-1}$, with $\mathbf{I}^1=\mathbf{0}$.
- **Quantization Stream**: At step $m$, the static base codebook $\mathcal{C}^m$ is morphed into a dynamic codebook $\tilde{\mathcal{C}}^m=\{\tilde{\mathbf{c}}_k^m\}$ by the second-level MoE function $f_t$ conditioned on $\mathbf{I}^m$, then nearest neighbor search is performed: $i^m=\arg\min_k\|\mathbf{r}^m-\tilde{\mathbf{c}}_k^m\|_2^2$.

Final reconstruction is $\hat{\mathbf{x}}=\sum_{m=1}^M\tilde{\mathbf{c}}_{i^m}^m$, consistent with standard RQ summation—meaning that, given the index sequence, all $\mathbf{I}^m$ can be computed in parallel via "index lookup + addition," all $\tilde{\mathcal{C}}^m$ can be generated in parallel, and the entire decoding path is free from serial dependencies.

### Key Designs

1. **Implicit Routing + Index Reuse in First-Level MoE (High-Dimensional Codebook)**:

    - **Function**: Enables the nearest neighbor index to serve both "codeword selection" and "expert routing" roles without extra bits.
    - **Mechanism**: Each codeword is a $(D+D_e)$-dimensional vector $\mathbf{w}_k^m=[\mathbf{c}_k^m;\mathbf{e}_k^m]$, with the first $D$ dimensions $\mathbf{c}_k^m$ as the "base codebook" for residual matching, and the last $D_e$ dimensions $\mathbf{e}_k^m$ as the expert component encoding local manifold features. Nearest neighbor search computes distance only on the first $D$ dimensions (Eq. 1), but the selected index $i^m$ also determines the expert signal $\mathbf{e}_{i^m}^m$, which is then accumulated into $\mathbf{I}^{m+1}$.
    - **Design Motivation**: Naive MoE requires $\log_2 N$ bits/expert for gating, but RQ-MoE embeds gating into the "already stored index," making routing "zero-cost" and elegantly retaining RQ's simple "index sequence as encoding" format.

2. **Dual-Stream Quantization + Second-Level MoE Codebook Morphing**:

    - **Function**: Uses accumulated instructions $\mathbf{I}^m$ to morph static base codewords $\mathbf{c}_k^m$ towards the current input manifold, yielding "input-adaptive local codebooks."
    - **Mechanism**: For each candidate $k$, compute $\mathbf{z}_k^m=\text{Linear}([\mathbf{c}_k^m;\mathbf{I}^m])$ to inject instructions into the base codeword; then $N$ expert MLPs (each $L$-layer bottleneck residual) compute $\mathcal{E}_n(\mathbf{z}_k^m)$ in parallel; gating $\boldsymbol{\alpha}_k^m=\text{softmax}(\text{Linear}(\mathbf{z}_k^m))$ weights the experts to produce offset $\Delta\mathbf{c}_k^m=\sum_n\boldsymbol{\alpha}_{k,n}^m\mathcal{E}_n(\mathbf{z}_k^m)$; finally, $\tilde{\mathbf{c}}_k^m=\mathbf{c}_k^m+\Delta\mathbf{c}_k^m$. The first step $\tilde{\mathcal{C}}^1=\mathcal{C}^1$ uses the base codebook directly.
    - **Design Motivation**: Decoupling "conditioning information" from the "reconstruction path" is key for parallel decoding—since $\mathbf{I}^m$ depends only on the previous index and expert component (lookup + add), not the previous reconstruction vector, all $\{\mathbf{I}^1,\ldots,\mathbf{I}^M\}$ can be computed at once, then all $\{\tilde{\mathcal{C}}^m\}_{m=1}^M$ generated in parallel; theoretically providing $M\times$ decoding speedup, further multiplied by $N\times$ via expert parallelism.

3. **Normalized Residual Loss (NRL)**:

    - **Function**: Replaces final-step MSE or per-step MSE, giving each quantization step balanced gradients according to "actual remaining difficulty."
    - **Mechanism**: Defines relative residual ratio $\rho^m=\|\mathbf{r}^{m+1}\|_2^2/(\text{sg}(\|\mathbf{r}^m\|_2^2)+\epsilon)$ (denominator stop-gradient), with loss $\mathcal{L}_{\text{NRL}}=\sum_{m=1}^M\log(1+\rho^m)$. Its gradient $\nabla_{\mathbf{r}^{m+1}}\mathcal{L}_{\text{NRL}}=2\|\mathbf{r}^{m+1}\|_2/(\|\mathbf{r}^{m+1}\|_2^2+C)$ increases with $\|\mathbf{r}^{m+1}\|_2$ for moderate residuals and vanishes for extreme residuals, acting as a redescending influence function.
    - **Design Motivation**: Pure MSE gradients $2\|\mathbf{r}^{m+1}\|_2$ scale linearly with residual—large early residuals and small late residuals cause later experts' gradients to be drowned out; also, unbounded for outliers, risking gradient explosion. NRL normalizes by "relative progress over previous step," ensuring deep experts receive effective training signals and robustness to outliers.

### Loss & Training
Training with NRL alone suffices for end-to-end optimization of all base/expert codebooks, MoE gating, and expert MLPs; no auxiliary load-balance loss is introduced (implicit routing via nearest neighbor naturally provides balance).

## Key Experimental Results

### Main Results
On Deep1M, BigANN1M, FB-ssnpp1M, and Contriever1M retrieval benchmarks, with 10M training samples and 8/16 byte codebook budgets. RQ-MoE defaults to $N=1, L=16$ (Contriever uses $L=12$ to match QINCo).

| Dataset (8 bytes) | Metric | RQ-MoE | QINCo | OPQ |
|-------------------|--------|--------|-------|-----|
| Deep1M (D=96) | MSE / R@1 | Comparable or better | -- | 0.25 / 15.2 |
| BigANN1M (D=128) | MSE (×$10^4$) / R@1 | Comparable or better | -- | 2.97 / 21.4 |
| FB-ssnpp1M (D=256) | MSE / R@1 | Comparable or better | -- | 9.51 / 2.5 |
| Contriever1M (D=768) | MSE / R@100 | Comparable or better | -- | 1.87 / 50.6 |

**Decoding Acceleration**: Compared to QINCo / QINCo2, achieves **6×–14×** PAD (exact numbers vary by dataset and $M$).

**Complexity** (FLOPS per vector, with fixed $N\cdot L$ budget):

| Method | Encoding | Decoding |
|--------|----------|----------|
| UNQ | $H'(D+H+Mb+MK)$ | $H'(b+H'+D+M)$ |
| QINCo | $2MKD(D+LH)$ | $2MD(D+LH)$ |
| **RQ-MoE** | $2MKD(D+NLH+N)$ | $2MD(D+NLH+N)$ |

Theoretical decoding speedup: step-level $M\times$ + expert-level $N\times = (M\cdot N)\times$.

### Ablation Study

| Configuration | Phenomenon | Explanation |
|---------------|------------|-------------|
| Full RQ-MoE | SOTA / 6–14× speedup | Main result |
| MSE-final instead of NRL | Later experts undertrained | NRL solves deep underfitting |
| Per-step MSE instead of NRL | Early steps dominate optimization | Early gradients too large |
| Disable second-level MoE (fixed base codebook) | Degrades to RQ, higher reconstruction error | Input adaptivity is essential |
| Couple instruction stream with reconstruction path (QINCo style) | Serial dependency returns, speed drops | Dual-stream decoupling is key for parallelism |
| Explicit gating (extra bits stored) | Lower accuracy under fixed bit budget | Implicit routing + index reuse is superior |

### Key Findings
- Theoretically, when $D_e=0$ and $\Delta\mathbf{c}_k^m=0$, RQ-MoE reduces to standard RQ; when $f_t$ is QINCo's residual-MLP and $D_e=D$, it reduces to QINCo—both are "restricted special cases" of RQ-MoE, making it a unified framework.
- Recommended expert dimension $D_e$: The authors provide a theoretical guideline; simply setting $D_e=D$ yields stable performance on most benchmarks.
- Decoding acceleration comes not only from "step-level parallelism": second-level MoE experts can also be parallelized, yielding up to $M\cdot N$ end-to-end speedup over QINCo.

## Highlights & Insights
- "Embedding routing information into the existing quantization index" is a highly clever design—achieving MoE routing at zero bit cost, with natural load balancing.
- Dual-stream decoupling enables both dynamic codebooks and parallel decoding, previously considered mutually exclusive.
- NRL is equivalent to robust statistics' redescending M-estimator, providing a statistical explanation for why later experts can also learn well—this loss design is transferable to other "stepwise refinement" tasks (diffusion, autoregressive token, refinement-style segmentation).
- RQ-MoE offers a generalized framework: using a hyper-dimensional codebook to bind "main task output + auxiliary routing signal" to the same discrete index, providing an extremely lightweight MoE integration in engineering.

## Limitations & Future Work
- Encoding remains serial (residuals must be computed stepwise and dynamic codebooks queried), though in theory $N$ experts can accelerate via parallelism; fully parallel encoding is not yet solved.
- Experiments focus on retrieval/reconstruction metrics, without direct evaluation on downstream RecSys (e.g., Rajput et al.'s generative recommendation tokenization) or speech codec tasks.
- Training stability is not discussed—MoE typically requires load balancing/gating noise, but this work relies entirely on implicit routing; whether stability holds at larger scales remains to be seen.
- Setting $D_e=D$ doubles codebook storage, which may be a new overhead for extremely compact scenarios (e.g., IoT edge devices).

## Related Work & Insights
- **vs RQ / PQ / OPQ**: Classic MCQ methods use static codebooks; RQ-MoE introduces input-conditioned dynamic codebooks while retaining RQ's simple "index sequence as encoding."
- **vs QINCo / QINCo2**: QINCo is the first dynamic codebook method but strictly serial; QINCo2 uses PAD/beam search for acceleration but does not eliminate serial dependency. RQ-MoE truly removes dependency via dual-stream decoupling.
- **vs UNQ**: UNQ replaces Euclidean distance with deep networks for lookup but still uses static codebooks; RQ-MoE allocates network capacity to "codebook generation," better leveraging MoE's sparse activation.
- **Inspiration**: In retrieval-augmented LLMs or generative recommender tokenization, RQ-MoE can directly replace RVQ, promising "same accuracy, faster decoding" benefits.

## Rating
- Novelty: ⭐⭐⭐⭐ Uses implicit routing + dual-stream decoupling to simultaneously solve the conflicting goals of dynamic codebooks and parallel decoding
- Experimental Thoroughness: ⭐⭐⭐⭐ Four standard benchmarks + complexity table + ablation, but lacks downstream RecSys/Codec validation
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagrams and algorithm descriptions, well-summarized theory
- Value: ⭐⭐⭐⭐ Direct replacement value for RVQ tokenization in generative recommendation / speech codec tasks

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](../../ACL2025/model_compression/moqae_mixed_precision_kv_cache.md)
- [\[ICLR 2026\] KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models](../../ICLR2026/model_compression/kbvq-moe_klt-guided_svd_with_bias-corrected_vector_quantization_for_moe_large_la.md)
- [\[ICML 2026\] Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression](breaking_the_moe_llm_trilemma_dynamic_expert_clustering_with_structured_compress.md)
- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](../../ICLR2026/model_compression/unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[ICML 2026\] ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin](arcvq-vae_a_spherical_vector_quantization_framework_with_arccosine_additive_marg.md)

</div>

<!-- RELATED:END -->
