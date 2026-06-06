---
title: >-
  [Paper Note] A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers
description: >-
  [NeurIPS 2025][Video Understanding][log-depth transformer] This paper proves that increasing Transformer depth from a constant to $\Theta(\log n)$ unlocks the ability to recognize regular languages and solve graph connec…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "log-depth transformer"
  - "expressive power"
  - "regular language recognition"
  - "graph connectivity"
  - "computational complexity"
date: 2026-05-08
content_hash: a4f9a485e50a0354
---

# A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2503.03961](https://arxiv.org/abs/2503.03961)  
**Code**: Available (provided in the camera-ready version)  
**Area**: Video Understanding
**Keywords**: log-depth transformer, expressive power, regular language recognition, graph connectivity, computational complexity

## TL;DR

This paper proves that increasing Transformer depth from a constant to $\Theta(\log n)$ unlocks the ability to recognize regular languages and solve graph connectivity — two problems provably beyond the reach of fixed-depth Transformers — and that depth scaling is strictly more efficient than width scaling (which requires super-polynomial growth) or Chain-of-Thought (CoT) steps (which requires super-logarithmic growth).

## Background & Motivation

A series of recent theoretical works has shown that fixed-depth Transformers are computationally bounded within the complexity class TC⁰, and are incapable of exactly solving fundamental problems requiring sequential reasoning, such as regular language recognition and graph connectivity. However, these impossibility results all treat depth as a constant — leaving open a critical question: if depth is allowed to grow by even a minimal amount as a function of input length $n$, how does the expressive power of Transformers change? Meanwhile, prior work on log-depth Transformers for regular language recognition (e.g., Liu et al. 2023) relies on non-uniform parameters and non-standard positional encodings, preventing a single fixed-parameter model from handling inputs of arbitrary length. This paper aims to close this theoretical gap.

## Core Problem

1. **Bounded-context problem**: How long an input can a fixed-depth-$d$ Transformer process? Can the relationship between depth and effective context length be quantified?
2. **Dynamic depth problem**: Does growing depth at the minimal rate — logarithmically in input length — suffice for Transformers to solve regular languages and graph connectivity? How does this compare in efficiency to scaling width or CoT steps?

The significance of these questions lies in their direct bearing on the fundamental capability boundaries of LLMs in long-context sequential reasoning (state tracking, multi-step logical inference), as well as their practical implications for choosing model depth.

## Method

### Overall Architecture

The paper analyzes **(s, r, t)-universal transformers** (also called looped transformers): $s$ initial layers + $r$ recurrent layers (looped $d(n) = \lceil \log n \rceil$ times) + $t$ termination layers. This is a fully uniform model whose parameters do not vary with $n$. Within this framework, the paper proves that log-depth suffices to solve two key problems, while also establishing the efficiency disadvantage of width and CoT scaling.

Key assumptions include: (1) averaging-hard attention (saturated attention); (2) masked pre-norm (a learned mask vector added to standard pre-norm to select specific dimensions of the residual stream); (3) $O(\log n)$-bit precision.

### Key Designs

1. **Lemma 1 (Integer Division)**: The core technical contribution. It proves that a fixed-depth Transformer of only 7 layers can compute the quotient and remainder of $a_i \div m$ for integers $a_i$ not exceeding the current position. Specifically: Layer 1 identifies multiples of $m$ and stores quotients; Layer 2 retrieves the quotient at the nearest multiple position via attention to obtain $\lfloor i/m \rfloor$; Layers 3–5 cooperatively compute $i \bmod m$; Layer 6 attends to position $a_i$ to obtain $\lfloor a_i/m \rfloor$ and $a_i \bmod m$; Layer 7 cleans up intermediate values. This construction requires no special positional encoding and is implemented directly using masked pre-norm, resolving the dependence on non-standard positional embeddings found in prior work.

2. **Theorem 1 (Regular Language Recognition)**: A $(0, 8, 9)$-universal transformer, looped $\lceil \log_2 |w| \rceil$ times, can recognize any regular language. The core mechanism is binary-tree prefix products — each "level" uses 8 layers to perform one doubling merge (using Lemma 1 to compute which tokens are "active" at the current level), for a total of $O(\log n)$ levels. Model dimension is $O(|Q|^2)$ (NFA) or $O(|Q| \log |Q|)$ (DFA), with feedforward width $O(2^{|Q|^4})$ or $O(2^{|Q|^2 \log^2 |Q|})$.

3. **Theorem 2 (Graph Connectivity)**: A $(17, 2, 1)$-universal transformer, looped $\lceil \log_2 n \rceil$ times, solves $s$-$t$ connectivity for $n$-vertex graphs. The approach is classical doubling reachability: a predicate $B_\ell(i,j)$ (existence of a path of length $\leq 2^\ell$) is maintained over $n^2$ adjacency matrix tokens, and an intermediate predicate $C_\ell(i,k,j)$ is computed over $n^3$ padding tokens. Each loop alternately updates $C$ and $B$; after $\lceil \log n \rceil$ rounds, full reachability information is obtained.

4. **Theorems 3 & 4 (Disadvantages of Width and CoT Scaling)**:
    - Width scaling: Fixed-depth Transformers remain within TC⁰ even with polynomial width growth; escaping TC⁰ requires super-polynomial width expansion — which is infeasible.
    - CoT scaling: $O(\log n)$ CoT steps also cannot escape TC⁰; thus logarithmic CoT is insufficient to solve regular languages and graph connectivity, making depth scaling strictly more powerful.

### Loss & Training

The theoretical contributions involve no training. Empirically, a standard Transformer is trained on the $A_5$ state-tracking task using **curriculum learning**: starting from sequences of length 2 and doubling up to 1024. Multiple seeds are trained for each depth/width configuration, and the best $n^*$ (the largest token position achieving 95% validation accuracy) is reported.

## Key Experimental Results

| Setup | Independent Variable | Dependent Variable | Fitted Relation | $r^2$ |
|---|---|---|---|---|
| Fixed width 512, varying depth | Depth $d \in \{6,9,12,15,18,21,24\}$ | $\log_2(n^*)$ | $d = 4.8 \log_2 n^* - 15.8$ | 0.93 |
| Fixed depth 6, varying width | Width $w \in \{128,256,512,1024\}$ | $n^*$ | $n^* = 7.2 \log_2 w - 41.7$ | 0.98 |

Key findings:
- Depth and effective context length share a **logarithmic relationship** (validating Theorem 1).
- Width and effective context length share an **exponential relationship** (validating Theorem 3).
- The empirical slope of 4.8 layers/log-token lies between the theoretical uniform construction (8) and the non-uniform construction (4).

### Ablation Study

- Experiments compare varying depth (6→24) and varying width (128→1024); every ~4.8 additional layers doubles the effective context length.
- A depth-32 model (e.g., LLaMA 3.1 7B) can theoretically recognize regular language strings of length up to $10^7$.
- A depth-80 model (e.g., LLaMA 3.1 70B) can theoretically handle strings of 440K tokens and graphs with 2.1 billion vertices.

## Highlights & Insights

- **Lemma 1 (Integer Division)** is the most elegant technical contribution: 7 layers implement modular arithmetic in a fully uniform manner without special positional encodings, serving as the critical bridge from fixed-depth computation to log-depth expressive power.
- This is the first proof that a **single Transformer with fixed parameters** (independent of $n$) can recognize any regular language through log-depth looping — resolving two core weaknesses of prior work (non-uniform parameters and non-standard architectures).
- The three-way comparison of depth vs. width vs. CoT is notably clean: $O(\log n)$ depth > super-logarithmic CoT > polynomial width, providing theoretical grounding for choosing inference-time compute strategies.
- Theoretical predictions align closely with experiments ($r^2 > 0.93$), and the results yield actionable quantitative relationships.

## Limitations & Future Work

- **Averaging-hard attention assumption**: Practical softmax attention is soft; the theoretical construction relies on exact maximum matching, which may require large temperature to approximate in practice.
- **$O(\log n)$-bit precision assumption**: While more reasonable than finite precision, practical float16/bfloat16 may not satisfy this requirement; the paper acknowledges that mixed-precision models are insufficient.
- **Experiments only on non-looped Transformers**: The core theory concerns universal (looped) transformers, but experiments use standard non-weight-sharing models, leaving the effect of weight sharing on learning unvalidated.
- **Learning dynamics and inductive bias not considered**: Expressive power $\neq$ learnability; the paper does not analyze whether SGD can efficiently find the theoretical constructions.
- **Only regular language recognition is validated experimentally**: Graph connectivity has no experimental verification.
- **P-complete problems remain intractable**: Even polylog-depth cannot solve linear systems or context-free language recognition (unless NC = P).

## Related Work & Insights

| Work | Key Distinction |
|---|---|
| **Liu et al. 2023** | Also uses log-depth + prefix sums for regular language recognition, but relies on non-uniform parameters and a non-standard architecture (no residual connections, no layer norm, special positional encodings). This paper achieves the same with a fully uniform standard architecture, using a single model for all input lengths. |
| **Merrill & Sabharwal 2023** | Proves fixed-depth Transformer $\subseteq$ TC⁰. This paper builds on that result to show log-depth can surpass the expressive limitations of TC⁰. |
| **Sanford et al. 2024** | Proves non-uniform log-depth + arbitrarily powerful FFNs can solve a variant of graph connectivity. This paper uses a fully uniform finite-width FFN. |
| **Geiping et al. 2025** | Empirically explores looped transformers for reasoning. This paper provides the theoretical foundation. |

The paper's analysis of depth vs. CoT suggests an interesting direction: since increasing loop iterations is theoretically more efficient than adding CoT steps, **adaptive depth** strategies in practical LLMs (dynamically adjusting loop count based on input difficulty) may hold more potential than pure CoT. The integer division construction of Lemma 1 may also inspire the design of **interpretable attention patterns**, illustrating how attention can implement non-trivial arithmetic operations. The quantitative relationship established (depth $d$ can process inputs of length $2^{O(d)}$) provides theoretical guidance for practical model depth selection.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First proof using a fully uniform standard Transformer architecture that log-depth breaks the TC⁰ expressive barrier; the integer division construction in Lemma 1 is remarkably elegant.
- Experimental Thoroughness: ⭐⭐⭐ — Experiments are limited to the $A_5$ state-tracking task; graph connectivity lacks experimental validation; looped transformers are not tested.
- Writing Quality: ⭐⭐⭐⭐⭐ — An exemplary theory paper: problem motivation is clear, theorem statements are precise, proof sketches precede full proofs, and the three-way comparison in Figure 1 is highly intuitive.
- Value: ⭐⭐⭐⭐⭐ — Provides a precise characterization of the fundamental limits of Transformer reasoning, with direct implications for model architecture design and inference-time compute strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases](../../ICCV2025/video_understanding/flowseek_optical_flow_made_easier_with_depth_foundation_models_and_motion_bases.md)
- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](../../ICCV2025/video_understanding/vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)
- [\[ICLR 2026\] The Expressive Limits of Diagonal SSMs for State-Tracking](../../ICLR2026/video_understanding/the_expressive_limits_of_diagonal_ssms_for_state-tracking.md)
- [\[NeurIPS 2025\] VideoLucy: Deep Memory Backtracking for Long Video Understanding](videolucy_deep_memory_backtracking_for_long_video_understanding.md)
- [\[ICLR 2026\] Log Probability Tracking of LLM APIs](../../ICLR2026/video_understanding/log_probability_tracking_of_llm_apis.md)

</div>

<!-- RELATED:END -->
