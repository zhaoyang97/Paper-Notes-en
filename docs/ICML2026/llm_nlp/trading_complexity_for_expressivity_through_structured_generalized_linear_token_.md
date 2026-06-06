---
title: >-
  [Paper Note] Structured Generalized Linear Token Mixing: Shifting Gears Between Complexity and Expressivity with SND + Kronecker
description: >-
  [ICML 2026][LLM/NLP][token mixing] The paper proposes a unified "direct input mixing $\mathbf{A}$ + output recurrent mixing $\mathbf{B}$" framework $Y = (I - B)^{-1} A X$ that encompasses attention, SSM…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "token mixing"
  - "attention"
  - "SSM"
  - "higher-order recurrence"
  - "time complexity"
  - "cache size"
date: 2026-05-08
content_hash: 6b6da16fee5f0ab5
---

# Structured Generalized Linear Token Mixing: Shifting Gears Between Complexity and Expressivity with SND + Kronecker

**Conference**: ICML 2026  
**arXiv**: [2605.31367](https://arxiv.org/abs/2605.31367)  
**Code**: Not listed  
**Area**: Attention / State Space Models / Efficient Sequence Modeling  
**Keywords**: token mixing, attention, SSM, higher-order recurrence, time complexity, cache size

## TL;DR
The paper proposes a unified "direct input mixing $\mathbf{A}$ + output recurrent mixing $\mathbf{B}$" framework $Y = (I - B)^{-1} A X$ that encompasses attention, SSM, linear recurrence, and higher-order recurrence. It proves that the sparsity pattern of $A, B$ directly controls the complexity gradient from $\mathcal{O}(n^{\log n})$ to $\mathcal{O}(n^2)$. Two translation-invariant patterns, $f(k) = 2^k$ and $f(k) = k^2+1$, are introduced to provide new options with $\mathcal{O}(n \log n)$ and $\mathcal{O}(n \sqrt{n})$ complexity, while the cache can be reduced to $\mathcal{O}(\log n)$ or $\mathcal{O}(\sqrt{n})$.

## Background & Motivation

**Background**: Token mixing is the core of sequence models—it determines how models exchange information across tokens. Transformers use self-attention for global one-hop mixing at $\mathcal{O}(n^2)$; linear attention uses kernels to reduce this to $\mathcal{O}(n)$; State Space Models (S4/Mamba) utilize linear recurrence at $\mathcal{O}(n)$; and Hybrids (Griffin/Nemotron-H) mix various mixers.

**Limitations of Prior Work**: (1) Relationships between different mixers have been analyzed on a case-by-case basis (e.g., the equivalence of Mamba-2 and linear attention, or the connection between Chimera and SSM-on-graph), lacking a unified framework; (2) Most mixers utilize first-order recurrence (looking only at the previous state), but Chen et al. 2025 proved that memory must grow with $L^\beta$ to scale effectively, meaning first-order models are inherently limited on long sequences; (3) Higher-order recurrences (log-linear attention, ChaCAL) have appeared sporadically but without a systematic theory explaining "which sparsity pattern yields what complexity and expressivity."

**Key Challenge**: To express long-range dependencies, global connectivity is required ($\mathcal{O}(n^2)$ attention); for speed, models must be local or recurrent. The design space in between (e.g., $\mathcal{O}(n \log n)$ or $\mathcal{O}(n \sqrt{n})$) has seen almost no systematic exploration.

**Goal**: (1) Provide a unified form for all causal linear token mixers; (2) Systematically characterize the correspondence between sparsity pattern → complexity/cache/expressivity; (3) Design new mixers that span the full spectrum from $\mathcal{O}(n)$ to $\mathcal{O}(n^2)$.

**Key Insight**: Any causal linear token mixer can be decomposed into "direct input influence + output recurrent propagation." By introducing matrices $A$ (lower triangular, input influence) and $B$ (strict lower triangular, output recurrence), the relationship $Y = AX + BY$ or $Y = (I-B)^{-1} A X$ holds. Attention corresponds to $B=0$; recurrence corresponds to diagonal $A$ + sub-diagonal $B$; hybrids are achieved by mixing sparsity patterns.

**Core Idea**: Sparsity patterns of $A$ and $B$ are used as design knobs—different patterns correspond to different complexities (time per token, cache size) and expressivity (shortest path between tokens, graph congestion). New mixers using translation-invariant patterns like $f(k) = 2^k$ (exponential) or $f(k) = k^2+1$ (quadratic) provide intermediate tiers at $\mathcal{O}(n \log n)$ and $\mathcal{O}(n \sqrt{n})$.

## Method

### Overall Architecture

The Generalized Linear Recurrence Layer is defined as: $y_i = \sum_{j=1}^i \alpha_{i,j} x_j + \sum_{j=1}^{i-1} \beta_{i,j} y_j$, where $\alpha_{i,j}, \beta_{i,j}$ are input-dependent coefficients (attention weights, SSM gating, etc.). In matrix form, $Y = (I-B)^{-1} A X$, where $A$ is lower triangular and $B$ is strict lower triangular (ensuring $I - B$ is invertible). The sparsity pattern determines everything—sparsity leads to lower complexity, while $(I-B)^{-1}$ remains a dense lower-triangular matrix that preserves expressivity.

### Key Designs

1.  **Unified Framework: $A, B$ Sparsity Patterns Covering All Causal Linear Mixers**:
    - **Function**: Views attention, SSM, linear attention, and recurrence as special cases of $(A, B)$ selection, providing a unified language for system design.
    - **Mechanism**: (a) Standard attention: $B=0, A=\mathrm{softmax}(QK^\top/\sqrt{d_k})$ $\rightarrow$ $Y = AX$, complexity $\mathcal{O}(n^2)$; (b) Local attention: $A$ is banded $\rightarrow$ $\mathcal{O}(nk)$; (c) Gated linear recurrence: $A$ is diagonal, $B$ is sub-diagonal $\rightarrow$ $y_t = \alpha_{t,t} x_t + \beta_{t,t} y_{t-1}$; (d) Diagonal SSM: A specific parameterization of linear recurrence + state expansion; (e) Mamba-2: Equivalent to a 1-semiseparable transformation matrix, connecting SSM and masked linear attention. All these are instances of the same $(I-B)^{-1} A$ framework under different sparsity patterns.
    - **Design Motivation**: Previous comparisons between attention and SSM were case-by-case; this paper provides a unified matrix algebra language. Discussing "why SSM is faster than attention" is equivalent to discussing "sparser patterns $\rightarrow$ faster matrix-vector products"; discussing "why attention is more expressive" is equivalent to discussing "$A$ dense $\rightarrow$ any pairwise relationship is expressible." This unification is the foundation for subsequent system design.

2.  **Translation-invariant Patterns + $\mathcal{O}(f^{-1}(n))$ Complexity Family**:
    - **Function**: Uses a single function $f$ to control the sparsity pattern, allowing complexity, shortest path, and congestion to be derived directly.
    - **Mechanism**: Defines a pattern induced by a strictly increasing $f: \mathbb{N}_{\ge 0} \to \mathbb{N}_{> 0}$ such that $\alpha_{i,j} \ne 0 \iff \exists k: j = i - f(k)$, meaning token $i$ can only attend to past tokens at distance $f(k)$. For example, $f(k) = 2^k$ restricts token $i$ to look at $i-1, i-2, i-4, i-8, \dots, i-2^{\lfloor \log_2 i \rfloor}$, resulting in $\mathcal{O}(\log n)$ complexity per token. Proposition 4.2 generalizes this: complexity is $\mathcal{O}(f^{-1}(n))$—linear $f$ $\rightarrow$ $\mathcal{O}(n)$, quadratic $f$ $\rightarrow$ $\mathcal{O}(\sqrt{n})$, and exponential $f$ $\rightarrow$ $\mathcal{O}(\log n)$.
    - **Design Motivation**: Previous sparse attention (e.g., Longformer, BigBird) used handcrafted sparsity patterns (sliding window + global tokens) without a principled complexity hierarchy. The abstraction of translation-invariance + $f$ makes "choosing $f$ = choosing a complexity tier" a simple engineering decision, covering the full spectrum from $\mathcal{O}(\log n)$ and $\mathcal{O}(\sqrt n)$ to $\mathcal{O}(n)$.

3.  **Shortest Path + Congestion: Two New Expressivity Metrics**:
    - **Function**: Quantifies "how fast and smoothly information travels under different patterns" from a graph theory perspective.
    - **Mechanism**: Constructs a communication graph $\mathcal{G}$, where token $i \to$ token $j$ has an edge iff $i - j = f(k)$ for some $k$. The shortest path is $d(i, j) = \min\{d : \exists a \in \mathbb{N}^d, \sum_k f(a_k) = i - j\}$. When $f(k) = 2^k$, $d(i, j) \le \log_2 (i-j)$ (number of 1s in binary decomposition); when $f(k) = k^2+1$, $d(i, j) \le 4$ (Lagrange's four-square theorem). Congestion $C(\mathcal{G}) = \min_\mathcal{P} \max_i \#\{p \in \mathcal{P}: i \in p\}$ quantifies "how many paths crowd through a single node for copy tasks"; standard recurrent model congestion = $n$ (all information through one state), whereas higher-order recurrence can reduce this to $\log n$ or even 4. Propositions 4.7-4.8 provide tight upper/lower bounds for congestion.
    - **Design Motivation**: Previously, "long-range dependency capability" could only be described qualitatively. This paper provides two quantitative indicators. Shortest path measures "steps for information from $j$ to $i$," and congestion measures the "degree of information bottleneck." Together, they explain why $f(k) = k^2+1$ achieves an elegant trade-off of $\mathcal{O}(\sqrt n)$ complexity with a 4-step path and 4 congestion.

### Cache-efficient pattern

Translation-invariant pattern decoding is $\mathcal{O}(f^{-1}(n))$, but the cache remains $\mathcal{O}(n)$ because any token $j$ might be attended to by an arbitrarily distant future token. Definition 4.10 introduces a constraint: when decoding token $i$, it can only attend to positions in $S_{i-1} \cup \{i\}$, allowing the cache to evolve to $\mathcal{O}(f^{-1}(n))$. Proposition 4.12 provides a closed-form $S_i = \{a_k \lceil (i - f(k))/a_k \rceil : k \in \mathbb{N}, f(k) < i\}$, where $a_{k+1} = a_k \lceil (f(k+1) - f(k))/a_k \rceil$. Cache positions are quantized on a lattice of step $a_k$, creating a periodic structure that hardware implementations can exploit.

## Key Experimental Results

### Complexity vs. Expressivity Comparison

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

- **Clear Ladder from Theoretical Complexity to Empirical Expressivity**: Moving from Diagonal SSM (weakest) to Attention (strongest), the sparse pattern $f(k) = k^2+1$ at $\mathcal{O}(\sqrt n)$ complexity achieves 99.66% Copy accuracy—nearly matching Attention's 100%. This identifies an "almost free intermediate tier" in the design space.
- **Dense recurrence (infinite order) is as strong as attention**: Achieving 100% Copy, 99.99% Assoc Recall, and 99.80% Multi-hop demonstrates that recurrence is not inherently weaker than attention; the key factor is the "order." This supports the theory from Chen et al. 2025 that memory must scale with $L^\beta$.
- **Cache-efficient version involves minimal sacrifice**: $f(k) = 2^k$ + cache-eff drops from 92.63 to 75.47 in Copy accuracy, but the cache is reduced from $\mathcal{O}(n)$ to $\mathcal{O}(\log n)$, which is highly valuable for long-sequence deployment.
- **Congestion acts as a hard bottleneck**: Diagonal SSM (congestion = $n$) results in only 42.98% Copy accuracy, while $f(k) = k^2+1$ (congestion = 4) reaches 99.66%. Congestion directly predicts copy performance.
- **Shortest path bounds are tight**: The use of Lagrange's four-square theorem to guarantee a shortest path $\le 4$ for $f(k) = k^2+1$ is an elegant mathematical result, and experiments confirm that shorter paths correlate with better long-range dependency capture.

## Highlights & Insights

- **$Y = (I-B)^{-1} A X$ as an Elegant Unified Framework**: Incorporates attention, SSM, linear recurrence, higher-order recurrence, and Mamba-2 into a single matrix algebra. This unification is an actionable design tool rather than just a retroactive summary.
- **Design Space of Translation-invariant + $f$**: Choosing $f$ is equivalent to choosing complexity. This single-knob design allows architects to systematically slide along the complexity ladder rather than making case-by-case adjustments.
- **Ingenious Use of Number Theory**: Utilizing the four-square theorem to guarantee expressivity limits is a rare and elegant application of "borrowed" mathematical concepts.
- **Congestion as an Expressivity Metric**: Quantifying "information bottlenecks" from a graph routing perspective creates a perfect theoretical loop with researchers like Jelassi et al. 2024 regarding the copy dilemma in recurrent models.
- **Closed-form Lattice for Cache Efficiency**: Proposition 4.12 reveals that cache positions follow a periodic step-$a_k$ lattice, providing direct inspiration for hardware-friendly implementations like FlashAttention-style GPU kernels.
- **Theory Guiding Empirical Validation**: The table represents an empirical ladder predicted by theory; every trade-off point has a clear mathematical explanation without cherry-picking.

## Limitations & Future Work

- **Restricted to Linear Token Mixers**: Non-linear mixers (e.g., Mixture-of-Experts, attention with non-linear feature maps) are currently outside this framework and require extension.
- **Focus on Synthetic Tasks**: Tasks like Copy, Associative Recall, and Multi-hop are controlled; the perplexity gap in real-world language model pre-training is not as exhaustively demonstrated in the main results.
- **Limited Selection of $f$**: Experiments only explored linear, $2^k$, and $k^2+1$. Finer-grained functions like $f(k) = k \log k$ remain unexplored.
- **Lack of Hardware implementation for Cache-efficiency**: The closed-form lattice quantization is a theoretical result; actual GPU kernel optimization has not yet been demonstrated.
- **Training Stability of Higher-order Recurrence**: While infinite-order Dense recurrence is highly expressive, there is no data on whether it remains stable during training on models larger than 8B parameters.
- **Missing Throughput Comparisons**: The empirical study focuses on task accuracy rather than wall-clock speed, which is a critical reference for practitioners compared to Mamba-2 or Linear Attention.

## Related Work & Insights

- **vs. S4/Mamba/Mamba-2**: These are first-order linear recurrences. This paper proves first-order models are inherently sub-optimal for long sequences (congestion = $n$), and higher-order represents a necessary enhancement.
- **vs. Log-linear attention (Guo et al. 2026)**: They proposed an instance of logarithmic-order recurrence; this paper provides a systematic framework where that is a special case of $f(k) = 2^k$.
- **vs. ChaCAL (Fagnou et al. 2024) / Block-Chacal**: They proposed instances of infinite-order recurrence. Ours includes these while offering finer complexity-expressivity trade-offs.
- **vs. FlashAttention**: FlashAttention optimizes kernels for standard attention; this paper reduces complexity at the algorithmic level. Both are orthogonal and combinable.
- **vs. Chimera (Lahoti et al. 2025)**: They generalized SSMs to graphs; this paper uses communication graphs as an analytical tool for expressivity, introducing graph theory from a different angle.
- **Insights**: (1) Any sequence model design problem can be explored via $(A, B)$ sparsity patterns; (2) The Pareto frontier of complexity-expressivity can be parameterized by a single function $f$; (3) Congestion is an undervalued dimension of expressivity that should be included in future architecture searches; (4) The closed-form lattice for cache efficiency provides a building block for hardware-aware design.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The unified framework $(I-B)^{-1} A$ is a fresh perspective, and the combination of translation-invariant patterns with number theory is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic tasks are fully covered with LM validation, though wall-clock speeds and large-scale model validation are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically rigorous with proofs in the appendix; the sparsity visualizations are intuitive, and the main table clearly illustrates the trade-offs.
- Value: ⭐⭐⭐⭐⭐ Provides a principled framework for sequence model architecture design, directly guiding the design of future hybrid models or Mamba-3; offers closed-form tools for hardware-aware kernel development.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Token-Efficient Change Detection in LLM APIs](token-efficient_change_detection_in_llm_apis.md)
- [\[ICML 2026\] Why Are Linear RNNs More Parallelizable?](why_are_linear_rnns_more_parallelizable.md)
- [\[ACL 2026\] Characterizing the Expressivity of Local Attention in Transformers](../../ACL2026/llm_nlp/characterizing_the_expressivity_of_local_attention_in_transformers.md)
- [\[ICML 2026\] Express Your Doubts: Probabilistic World Modeling Should Not Be Based on Token logprobs](express_your_doubts_--_probabilistic_world_modeling_should_not_be_based_on_token.md)
- [\[ACL 2026\] Repeated Sequences Reveal Gaps between Large Language Models and Natural Language](../../ACL2026/llm_nlp/repeated_sequences_reveal_gaps_between_large_language_models_and_natural_languag.md)

</div>

<!-- RELATED:END -->
