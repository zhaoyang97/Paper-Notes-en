---
title: >-
  [Paper Note] Structured Generalized Linear Token Mixing: Shifting Gears Between Complexity and Expressivity with SND + Kronecker
description: >-
  [ICML 2026][LLM (Other)][token mixing] The paper proposes a unified "direct input mixing $\mathbf{A}$ + output recursive mixing $\mathbf{B}$" framework $Y = (I - B)^{-1} A X$ that encompasses attention, SSMs, linear recurrence, and high-order recurrence. It proves that the sparsity pattern of $A$ and $B$ directly controls the complexity gradient from $\mathcal{O}(n \log n)$ to $\mathcal{O}(n^2)$. Two translation-invariant modes, $f(k) = 2^k$ and $f(k) = k^2+1$…
tags:
  - "ICML 2026"
  - "LLM (Other)"
  - "token mixing"
  - "attention"
  - "SSM"
  - "high-order recurrence"
  - "time complexity"
  - "cache size"
date: 2026-05-08
content_hash: 63c8c237c97761c4
---

# Structured Generalized Linear Token Mixing: Shifting Gears Between Complexity and Expressivity with SND + Kronecker

**Conference**: ICML 2026  
**arXiv**: [2605.31367](https://arxiv.org/abs/2605.31367)  
**Code**: Not listed  
**Area**: Attention / State Space Models / Efficient Sequence Modeling  
**Keywords**: token mixing, attention, SSM, high-order recurrence, time complexity, cache size

## TL;DR
The paper proposes a unified "direct input mixing $\mathbf{A}$ + output recursive mixing $\mathbf{B}$" framework $Y = (I - B)^{-1} A X$ that encompasses attention, SSMs, linear recurrence, and high-order recurrence. It proves that the sparsity pattern of $A$ and $B$ directly controls the complexity gradient from $\mathcal{O}(n \log n)$ to $\mathcal{O}(n^2)$. Two translation-invariant modes, $f(k) = 2^k$ and $f(k) = k^2+1$, are introduced as new choices for $\mathcal{O}(n \log n)$ and $\mathcal{O}(n \sqrt{n})$ complexity, with cache sizes reducible to $\mathcal{O}(\log n)$ or $\mathcal{O}(\sqrt{n})$.

## Background & Motivation

**Background**: Token mixing is the core of sequence models, determining how information is exchanged across tokens. Transformers use self-attention for global one-hop interaction at $\mathcal{O}(n^2)$; linear attention uses kernels to reduce this to $\mathcal{O}(n)$; state space models (S4/Mamba) utilize linear recurrence at $\mathcal{O}(n)$; and hybrid models (Griffin/Nemotron-H) combine multiple mixers.

**Limitations of Prior Work**: (1) The relationships between different mixers are typically analyzed case-by-case (e.g., equivalence between Mamba-2 and linear attention, or Chimera and SSM-on-graph), lacking a unified framework. (2) Most mixers utilize first-order recurrence (tracking only the previous state), but Chen et al. (2025) proved that memory must grow as $L^\beta$ to scale effectively; first-order models are inherently limited on long sequences. (3) Higher-order recurrences (log-linear attention, ChaCAL) have appeared sporadically without a systematic theory explaining which sparsity patterns yield specific complexity and expressivity.

**Key Challenge**: Global connectivity for long-range dependencies requires $\mathcal{O}(n^2)$ attention, while speed necessitates local or recurrent designs. The intermediate design space (e.g., $\mathcal{O}(n \log n)$ or $\mathcal{O}(n \sqrt{n})$) has remained largely unexplored.

**Goal**: (1) Provide a unified form for all causal linear token mixers; (2) Systematically characterize the correspondence between sparsity patterns and complexity, cache size, and expressivity; (3) Design new mixers spanning the spectrum from $\mathcal{O}(n)$ to $\mathcal{O}(n^2)$.

**Key Insight**: It is observed that any causal linear token mixer can be decomposed into "direct input influence" and "recursive output propagation." By introducing matrix $A$ (lower triangular, input influence) and $B$ (strict lower triangular, output recurrence), the system is represented as $Y = AX + BY$, which simplifies to $Y = (I-B)^{-1} A X$. Attention corresponds to $B=0$, while recurrence involves diagonal $A$ and sub-diagonal $B$. Hybrid models are achieved by mixing sparsity patterns.

**Core Idea**: The sparsity patterns of $A$ and $B$ are used as design knobs. Different patterns correspond to varying complexities (time per token, cache size) and expressivity (shortest path between tokens, graph congestion). New mixers are designed using translation-invariant patterns like $f(k) = 2^k$ (exponential) or $f(k) = k^2+1$ (quadratic), providing intermediate complexity tiers at $\mathcal{O}(n \log n)$ and $\mathcal{O}(n \sqrt{n})$.

## Method

### Overall Architecture

The paper decomposes any causal linear token mixer into "direct input influence" and "recursive output propagation," formulated as a Generalized Linear Recurrence Layer: $y_i = \sum_{j=1}^i \alpha_{i,j} x_j + \sum_{j=1}^{i-1} \beta_{i,j} y_j$. In matrix form, this is $Y = (I-B)^{-1} A X$, where $A$ is lower triangular (input influence) and $B$ is strict lower triangular (output recurrence, ensuring $I-B$ is invertible). The core observation is that the sparsity patterns of $A$ and $B$ simultaneously determine complexity (sparser patterns lead to faster matrix-vector multiplication) and expressivity (though $(I-B)^{-1}$ expands to a dense lower-triangular matrix, distant tokens communicate via multi-hop recurrence). Thus, designing a sequence mixer is reduced to selecting a sparsity pattern.

### Key Designs

**1. Unified Framework: Unifying Causal Linear Mixers via $(A, B)$ Sparsity Patterns**

Previously, comparing attention and SSMs required case-by-case analysis. This work reveals they are instances of $Y=(I-B)^{-1}AX$ under different sparsity modes. Standard attention uses $B=0$ and $A=\mathrm{softmax}(QK^\top/\sqrt{d_k})$, resulting in $Y=AX$ with $\mathcal{O}(n^2)$ complexity. Local attention restricts $A$ to a banded matrix ($\mathcal{O}(nk)$). Gated linear recurrence uses diagonal $A$ and sub-diagonal $B$, expanding to $y_t=\alpha_{t,t}x_t+\beta_{t,t}y_{t-1}$. Diagonal SSMs are specific parameterizations of recurrence with state expansion. Mamba-2 is equivalent to a 1-semiseparable transformation matrix, bridging SSMs and masked linear attention. This algebraic language translates "why SSMs are faster" into sparsity and "why attention is more expressive" into the density of $A$.

**2. Translation-invariant Pattern: Tuning the $\mathcal{O}(f^{-1}(n))$ Complexity Spectrum via a Single Function $f$**

Sparse attention models like Longformer or BigBird used heuristic connectivity (sliding windows + global tokens), lacking a principled complexity ladder and clear expressivity bounds. Ours generates patterns using a strictly increasing function $f:\mathbb{N}_{\ge 0}\to\mathbb{N}_{>0}$ such that $\alpha_{i,j}\ne 0 \iff \exists k:\ j=i-f(k)$. For $f(k)=2^k$, token $i$ only attends to $i-1, i-2, i-4, \dots, i-2^{\lfloor\log_2 i\rfloor}$, requiring $\mathcal{O}(\log n)$ operations per token. Proposition 4.2 generalizes this: complexity is $\mathcal{O}(f^{-1}(n))$. Linear $f$ yields $\mathcal{O}(n)$, quadratic $f$ yields $\mathcal{O}(\sqrt n)$, and exponential $f$ yields $\mathcal{O}(\log n)$, turning the intermediate $\mathcal{O}(n\log n)$ and $\mathcal{O}(n\sqrt n)$ ranges into adjustable engineering knobs.

**3. Shortest Path and Congestion: Quantifying Expressivity with Graph Metrics**

Long-range dependency capability is quantified using the communication graph $\mathcal{G}$ (edge $i\to j$ exists if $i-j=f(k)$) through two metrics. First, the shortest path $d(i,j)=\min\{d:\exists a\in\mathbb{N}^d,\ \sum_k f(a_k)=i-j\}$ measures communication latency: for $f(k)=2^k$, $d(i,j)\le\log_2(i-j)$; for $f(k)=k^2+1$, Lagrange's four-square theorem ensures $d(i,j)\le 4$. Second, congestion $C(\mathcal{G})=\min_{\mathcal{P}}\max_i \#\{p\in\mathcal{P}:i\in p\}$ measures information bottlenecks: standard first-order recurrence compresses everything into one state ($C=n$), while higher-order patterns can achieve $\log n$ or even 4. This explains why $f(k)=k^2+1$ achieves a superior trade-off with $\mathcal{O}(\sqrt n)$ complexity and a path/congestion bound of 4.

**4. Cache-efficient Pattern: Compressing KV Cache from $\mathcal{O}(n)$ to $\mathcal{O}(f^{-1}(n))$**

While translation-invariant patterns reduce computation, the cache usually remains $\mathcal{O}(n)$. Definition 4.10 introduces a constraint: decoding token $i$ only allows attending to positions in $S_{i-1}\cup\{i\}$, reducing cache size to $\mathcal{O}(f^{-1}(n))$. Proposition 4.12 provides a closed-form for these positions: $S_i=\{a_k\lceil(i-f(k))/a_k\rceil:k\in\mathbb{N},\ f(k)<i\}$, where $a_{k+1}=a_k\lceil(f(k+1)-f(k))/a_k\rceil$. These positions fall on lattices with step size $a_k$, creating a periodic structure that is hardware-friendly.

## Key Experimental Results

### Complexity + Expressivity Comparison Table

| Structure | Time/token | Cache | Shortest path | Congestion | Copy% | Assoc recall% | Multi-hop% |
|---|---|---|---|---|---|---|---|
| Attention | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ | 1 | 1 | 100.00 | 100.00 | 39.21 |
| Local attention | $\mathcal{O}(k)$ | $\mathcal{O}(k)$ | $\infty$ | 1 | 23.75 | 26.20 | 23.59 |
| Diagonal SSM | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ | $n$ | $n$ | 42.98 | 32.53 | 27.17 |
| k-th order recurrence | $\mathcal{O}(k)$ | $\mathcal{O}(k)$ | $n/k$ | $n/k$ | 74.66 | 41.12 | 39.08 |
| Dense recurrence | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ | 1 | 1 | 100.00 | 99.99 | 99.80 |
| $f(k) = 2^k$ | $\mathcal{O}(\log_2 n)$ | $\mathcal{O}(n)$ | $\le \log_2 n$ | $\le \log_2 n$ | 92.63 | 49.03 | 34.85 |
| $f(k) = 2^k$ + cache-eff | $\mathcal{O}(\log_2 n)$ | $\mathcal{O}(\log_2 n)$ | — | — | 75.47 | 52.59 | 38.63 |
| $f(k) = k^2+1$ | $\mathcal{O}(\sqrt n)$ | $\mathcal{O}(n)$ | $\le 4$ | $\le 4$ | 99.66 | 53.61 | 35.68 |
| $f(k) = k^2+1$ + cache-eff | $\mathcal{O}(\sqrt n)$ | $\mathcal{O}(\sqrt n)$ | — | — | 91.59 | 54.56 | 38.02 |

### Key Findings

- **Clear Ladder from Theoretical Complexity to Empirical Expressivity**: From Diagonal SSM (weakest) to Attention (strongest), the sparse pattern $f(k) = k^2+1$ achieves 99.66% copy accuracy at $\mathcal{O}(\sqrt n)$ complexity, demonstrating the existence of "nearly free" intermediate tiers.
- **Dense Recurrence (Infinite Order) Matches Attention**: It achieves 100% copy and near-perfect associative/multi-hop recall, indicating that recurrence efficiency limits are a matter of order, not an inherent weakness. This aligns with the "memory must grow with $L^\beta$" theory.
- **Minimal Sacrifice for Cache-efficiency**: The cache-efficient version of $f(k) = 2^k$ reduces copy accuracy from 92.63% to 75.47%, but reduces cache from $\mathcal{O}(n)$ to $\mathcal{O}(\log n)$, showing significant value for long-sequence deployment.
- **Congestion as a Hard Bottleneck**: Diagonal SSM has congestion $n$ and poor copy performance (42.98%), while $f(k) = k^2+1$ has congestion 4 and high performance (99.66%). Congestion metrics directly predict copy capability.
- **Tight Shortest Path Bounds**: The use of Lagrange's four-square theorem to guarantee a path length $\le 4$ for $f(k) = k^2+1$ is validated by the strong correlation between short paths and long-range dependency performance.

## Highlights & Insights

- **$Y = (I-B)^{-1} A X$ as an Elegant Unified Framework**: It places various mixers into a single algebraic structure, serving as an actionable design tool rather than just an academic categorization.
- **Design Space of Translation-invariant Patterns**: Choosing $f$ allows architects to navigate the complexity spectrum systematically rather than through trial-and-error manual tuning.
- **Application of Lagrange's Four-Square Theorem**: Utilizing classical number theory to guarantee shortest path length is both rare and mathematically elegant.
- **Congestion as an Expressivity Metric**: Analyzing information bottlenecks from a graph routing perspective complements existing research on the limitations of recurrent model copying.
- **Closed-form Lattice for Cache Efficiency**: Proposition 4.12's periodic structure provides a foundation for developing hardware-friendly GPU kernels similar to FlashAttention.
- **Theory-Driven Empirical Validation**: Table 1 represents an empirical ladder predicted by theory, with each trade-off point supported by clear mathematical reasoning.

## Limitations & Future Work

- **Restricted to Linear Token Mixers**: Non-linear mixers (e.g., MoE, attention with non-linear maps) are not currently covered and require expansion.
- **Focus on Synthetic Tasks**: Reliability on controlled tasks like copy or recall is shown, but full-scale language model pre-training perplexity is not extensively detailed in the main results.
- **Limited Function Choices**: Only linear, $2^k$, and $k^2+1$ were explored; more nuanced functions (e.g., $f(k) = k \log k$) remain untested.
- **Pending Hardware Implementation**: The closed-form lattices for cache efficiency are theoretical; optimized GPU kernels for these structures have yet to be developed.
- **Training Stability of Higher-order Recurrence**: Data regarding the stability of infinite-order dense recurrence, especially for models larger than 8B parameters, is missing.
- **Throughput Comparisons**: The study focuses on task accuracy rather than wall-clock speed, which is a critical reference for deployment.

## Related Work & Insights

- **vs S4/Mamba/Mamba-2**: These are first-order linear recurrences; Ours proves that first-order models are sub-optimal for long sequences (congestion $n$), making higher-order designs necessary.
- **vs Log-linear Attention (Guo et al. 2026)**: They introduced a specific logarithmic-order recurrence instance; Ours provides a systematic framework treating it as a special case of $f(k) = 2^k$.
- **vs ChaCAL / Block-Chacal**: They proposed infinite-order recurrence instances; the unified framework in Ours includes these while providing finer-grained complexity-expressivity trade-offs.
- **vs FlashAttention**: While FlashAttention optimizes kernels for standard attention, Ours reduces complexity at the algorithmic level. Both are orthogonal and potentially combinable.
- **vs Chimera (Lahoti et al. 2025)**: They generalized SSMs to graphs; Ours uses communication graphs as an analytical tool for expressivity.
- **Insights**: (1) Sequence model design can be explored systematically via $(A, B)$ sparsity; (2) Pareto frontiers for complexity and expressivity can be parameterized by $f$; (3) Congestion should be a key metric in future architecture searches.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The $(I-B)^{-1} A$ framework and the application of number theory to design patterns are highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage of synthetic tasks and LM validation, though missing wall-clock comparisons and large-scale scaling data.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematically rigorous with clear visualizations and a comprehensive trade-off table.
- **Value**: ⭐⭐⭐⭐⭐ Provides a principled framework for designing next-generation hybrid models and hardware-aware kernels.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Why Are Linear RNNs More Parallelizable?](why_are_linear_rnns_more_parallelizable.md)
- [\[ICML 2026\] Token-Efficient Change Detection in LLM APIs](token-efficient_change_detection_in_llm_apis.md)
- [\[ACL 2026\] Characterizing the Expressivity of Local Attention in Transformers](../../ACL2026/llm_nlp/characterizing_the_expressivity_of_local_attention_in_transformers.md)
- [\[ICML 2025\] Generalized Interpolating Discrete Diffusion](../../ICML2025/llm_nlp/generalized_interpolating_discrete_diffusion.md)
- [\[NeurIPS 2025\] Composing Linear Layers from Irreducibles](../../NeurIPS2025/llm_nlp/composing_linear_layers_from_irreducibles.md)

</div>

<!-- RELATED:END -->
