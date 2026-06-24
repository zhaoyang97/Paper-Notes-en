---
title: >-
  [Paper Note] Subquadratic Algorithms and Hardness for Attention with Any Temperature
description: >-
  [ICLR 2026][Learning Theory][Attention Mechanism] This paper answers the fundamental question of whether attention can be computed efficiently at any temperature. For head dimension $d=O(1)$, it provides the first true subquadratic algorithm $\tilde{O}(n^{2-1/d}\cdot\mathrm{polylog}(B/\varepsilon))$ with only logarithmic dependence on the entry bound $B$. Furthermore, it uses Max-IP / OV reductions to prove that the standard algorithm is essentially optimal in the ranges $d=2…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Attention Complexity"
  - "Attention Mechanism"
  - "Subquadratic Algorithms"
  - "Fine-grained Complexity"
  - "Polynomial Method"
  - "SETH"
date: 2026-05-08
content_hash: a4bd386d217491e2
---

# Subquadratic Algorithms and Hardness for Attention with Any Temperature

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=PSaJZktut7](https://openreview.net/forum?id=PSaJZktut7)  
**Code**: None (Theoretical paper)  
**Area**: Learning Theory / Attention Complexity  
**Keywords**: Attention Mechanism, Subquadratic Algorithms, Fine-grained Complexity, Polynomial Method, SETH

## TL;DR
This paper answers the fundamental question of whether attention can be computed efficiently at any temperature. For head dimension $d=O(1)$, it provides the first true subquadratic algorithm $\tilde{O}(n^{2-1/d}\cdot\mathrm{polylog}(B/\varepsilon))$ with only logarithmic dependence on the entry bound $B$. Furthermore, it uses Max-IP / OV reductions to prove that the standard algorithm is essentially optimal in the ranges $d=2^{\Omega(\log^* n)}$ and $d=\mathrm{poly}(n)$, nearly completing the complexity landscape of attention computation.

## Background & Motivation

**Background**: The attention mechanism in Transformers computes $\mathrm{softmax}(QK^\top)V$ for $Q,K,V\in\mathbb{R}^{n\times d}$. The standard approach explicitly constructs the $n\times n$ attention matrix $A=\exp(QK^\top)$, requiring $O(n^2 d)$ time, which grows quadratically with long context $n$. Given that the input and output sizes are only $O(nd)$, it is natural to ask: can attention be computed in subquadratic or even near-linear time without explicitly forming $A$?

**Limitations of Prior Work**: Prior work by Alman & Song (2024) characterized that for $d=\Theta(\log n)$, attention can be computed in $n^{1+o(1)}$ time **if and only if** the input entry bound $B=o(\sqrt{\log n})$; otherwise, $n^{2-o(1)}$ time is required under the SETH assumption. However, their fast algorithm has an **exponential dependence** $2^{O(B^2)}$ on $B$, failing to reach even polynomial time once $B$ is slightly large.

**Key Challenge**: A small $B$ is equivalent to the softmax having a **high temperature** $T$ ($A=\exp(QK^\top/T)$), where high temperatures make the distribution tend toward uniform. This means existing fast algorithms only hold in degenerate "high-temperature, nearly non-selective" cases. Yet, temperature is a critical hyperparameter in machine learning affecting accuracy and expressivity (low/learnable temperatures are proven useful in NLP, vision, and contrastive learning, while high-temperature Transformers are proven to have weaker expressivity). No subquadratic algorithm existed for low-entropy (low-temperature) scenarios. Worse, having a runtime that explodes with the magnitude of input values is a poor property—analogous to Knapsack or APSP, one desires algorithms that grow polynomially with the **length** of the values (i.e., $\log B$).

**Goal**: ① Does a "true subquadratic" algorithm exist with only $\mathrm{polylog}$ dependence on $B$ (as opposed to "pseudo-subquadratic" growing polynomially with values)? ② Is the standard $O(n^2 d)$ algorithm already optimal when $d$ is super-constant or $d=\mathrm{poly}(n)$?

**Key Insight**: The authors observe that the softmax distribution is highly concentrated—for each query, only a few keys whose inner product $Q_i\cdot K_j$ is close to the maximum contribute non-negligible weight, while "irrelevant keys" can be safely discarded. After discarding, the inner products of relevant keys are compressed into a narrow interval of length $O(t)$, and $\exp$ can be accurately approximated by **low-degree polynomials** on bounded intervals. This provides an entry point for the "polynomial method" in fine-grained complexity.

**Core Idea**: First, use halfspace geometry to filter out relevant keys for each query. Then, approximate $\exp$ on this narrow interval using a low-degree polynomial and **decouple** the polynomial into factors depending only on $q$ and only on $k$. Leveraging simplex range searching data structures, the summation is amortized to $\tilde{O}(n^{1-1/d})$ per query—replacing the exponential dependence on $B$ with logarithmic dependence.

## Method

### Overall Architecture

The core computation the paper addresses is: for each $i$, compute $O_{i,t}=\sum_j p_{i,j}V_{j,t}$, where $p_{i,j}\propto\exp(Q_i\cdot K_j)$ are softmax probabilities. The work follows two lines: the **algorithmic side** provides a true subquadratic algorithm for $d=O(1)$, and the **hardness side** proves this upper bound is nearly unimprovable.

The algorithmic pipeline consists of three serial steps: given $Q,K,V\in[-B,B]^{n\times d}$, the **first step** uses halfspaces for each query $Q_i$ to filter out "irrelevant keys," leaving only relevant keys with inner products close to the maximum (the error from discarding is proven to be controlled). The **second step** replaces the exponential with a low-degree polynomial $P(x)\approx\exp(x)$ on the compressed interval of length $O(t)$ and decouples $P(Q_i\cdot K_j)$ into "terms with only $q_i \times$ terms with only $k_j$." The **third step**, since summation only occurs over relevant keys (points falling within a halfspace), uses Matoušek’s simplex range searching data structure for preprocessing, allowing each query summation to be completed in $\tilde{O}(n^{1-1/d})$ time, totaling $\tilde{O}(n^{2-1/d})$ for all $n$ queries. Low-rank cases reduce the effective dimension from $d$ to $r=\min(\mathrm{rank}(Q),\mathrm{rank}(K))$ via decomposition.

The hardness side independently reduces (bi-chromatic) Max-IP and OV problems to attention computation, providing hardness matching the standard algorithm in the intervals $d=2^{\Omega(\log^* n)}$ and $d=\mathrm{poly}(n)$.

This is a pure theory paper; the method is a sequence of mathematical constructions and reductions without a visual pipeline.

### Key Designs

**1. Discarding Irrelevant Keys: Shrinking Softmax to a Halfspace**

Addressing the pain point of "must iterate over all $n$ keys leading to quadratic time," the authors prove only a few "relevant keys" are needed for each $Q_i$. Define $s^{(i)}_{\max}$ as the largest integer such that the halfspace $\{x:Q_i\cdot x\ge s\log(1+\varepsilon)\}$ contains at least one $K_j$; $j$ is **irrelevant** if $Q_i\cdot K_j < s^{(i)}_{\max}\log(1+\varepsilon)-\log(n/\varepsilon)$, otherwise **relevant**. The intuition is: softmax probability $\propto\exp(Q_i\cdot K_j)$ decays exponentially with the inner product; keys trailing the maximum by more than a threshold $t=\Theta(\log(n/\varepsilon))$ have negligible normalized probabilities. Re-normalizing over relevant keys yields $p^{(r)}_{i,j}$ and output $O^{(r)}_{i,t}$; the paper proves $\big|O^{(r)}_{i,t}-O_{i,t}\big|\le 3\varepsilon B$ (Lemma 3.2), showing that discarding irrelevant keys introduces only controllable additive error. Critically, the "relevant key set = $K_j$ falling within the halfspace $\{x:Q_i\cdot x\ge\max_j Q_i\cdot K_j - t\}$"—transforming an $n^2$ operation into a geometric query solvable via range searching.

**2. Polynomial Approximation + Decoupling: Replacing Exp with Separable Low-degree Polynomials**

After filtering, the inner products $Q_i\cdot K_j$ all fall within a narrow interval of length $O(t)$, which is the prerequisite for the polynomial method: $\exp$ can only be well-approximated by a degree-$p$ polynomial when $|x|\le p$. Citing Aggarwal–Alman, there exists a polynomial $P$ of degree $g=\Theta\!\big(\max\{\tfrac{\log(1/\varepsilon)}{\log(\log(1/\varepsilon)/B)},\,B\}\big)$ such that $|P(x)-\exp(x)|<\varepsilon\exp(x)$ for all $x$ in the interval. The authors set $c_i=\max_j Q_i\cdot K_j-O(t)$, write $\exp(Q_i\cdot K_j)$ as proportional to $\exp(Q_i\cdot K_j-c_i)$ (shifted near 0), and approximate with $P$ to get $\hat p_{i,j}\propto P(Q_i\cdot K_j-c_i)$, resulting in $\hat o_i=\sum_j\hat p_{i,j}v_j$.

Why this step is key: Unlike $\exp(Q_i\cdot K_j)$, the polynomial $P(Q_i\cdot K_j-c_i)$ can be **decoupled** into a product of factors depending only on $q_i$ and factors depending only on $k_j$—thus the part regarding $k_j$ in $\sum_j P(Q_i\cdot K_j-c_i)v_j$ can be preprocessed in $\tilde{O}(n)$ time, with each query requiring only $\tilde{O}(1)$ lookups. This differs from Alman & Song’s route of direct $2^{O(B^2)}$ rank approximation of $\exp(QK^\top)$; by approximating only on relevant indices, this work achieves $\mathrm{polylog}$ dependence on $B$ rather than exponential.

**3. Simplex Range Searching for High Dimensions and Low Rank: Origin of $\tilde{O}(n^{2-1/d})$**

In 1D, relevant keys are simply "large enough $k_j$," handled by interval structures; however, for $d>1$, different $Q_i$ correspond to different "large" criteria. The key observation is that the relevant key set is exactly the points within the halfspace $\{x:Q_i\cdot x\ge\max_j Q_i\cdot K_j-t\}$, which can be handled by Matoušek’s (1992) **simplex range searching data structure**. Initialized with $\{K_j\}$ ($O(n\log n)$ preprocessing), for each $Q_i$, a halfspace query returns the sum of weights in the region in $\tilde{O}(n^{1-1/d})$ time. Since the number of monomials in $P$ grows exponentially with $d$, $2^{\Omega(d)}$ Matoušek structures must be instantiated, which remains a sub-polynomial factor for constant $d=O(1)$. Summing over all $i$ yields the main Theorem 1.1: $\tilde{O}(n^{2-1/d}\cdot\mathrm{polylog}(B/\varepsilon))$. The low-rank extension (Theorem 1.2) factorizes $Q=U_Q V_Q^\top,\,K=U_K V_K^\top$ and applies the algorithm to $Q'=U_Q$ and $K'^\top=V_Q^\top U_K V_K^\top$, yielding $\tilde{O}(nd+n^{2-1/r}\cdot\mathrm{polylog}(B/\varepsilon))$ where $r=\min(\mathrm{rank}(Q),\mathrm{rank}(K))$. Additionally, gradient computation is reduced to $O(d)$ attention calls plus $O(nd^2)$ overhead (Theorem 1.3), making the entire LLM training process true subquadratic for $d=O(1)$.

**4. Hardness Results: Provable Optimality via Max-IP / OV**

To address whether the upper bound can be faster, the authors provide hardness in two ranges. First (Theorem 1.4): Reduction from (bi-chromatic) Max-IP to attention—any fast $\mathrm{AttC}(n,d,B,\varepsilon)$ algorithm can compute $\max_{a\in A,b\in B}a\cdot b$. Chen (2018) proved Max-IP requires $n^{2-o(1)}$ for $d=2^{\Theta(\log^* n)}$ under SETH, thus attention also requires $n^{2-o(1)}$ for $d=2^{\Omega(\log^* n)}$ and $B=\mathrm{poly}(n)$. Second (Theorem 1.5): For $d=\mathrm{poly}(n)$, the standard algorithm takes $O(T_{\mathrm{MUL}}(n,d,n))$ using fast matrix multiplication. Under the "Generalized High-dimensional OV Hypothesis," the authors prove $\mathrm{AttC}$ requires $T_{\mathrm{MUL}}(n,d,n)^{1-o(1)}$, making the standard algorithm **conditionally optimal** in this range and closing the gap between $O(n^2 d)$ and the old $n^{2-o(1)}$ lower bound. Together, these bounds characterize the complexity except for the narrow gap $1\ll d\ll 2^{\Theta(\log^* n)}$.

### An Example: Intuition of the 1D Rounding Algorithm

Consider $d=1, q_i=1$. Given keys $k_1,\dots,k_8$ and values $v_1,\dots,v_8$, first discard keys with small inner products via threshold $q_i k_{\max}-t$ (e.g., $k_1$). For relevant keys, round each $k_j$ to $\bar k_j$ such that $q_i k_j\le q_i\bar k_j\le q_i k_j+\log(1+\varepsilon)$, making $e^{q_i\bar k_j}$ a $(1+\varepsilon)$ multiplicative approximation of $e^{q_i k_j}$. Keys falling into the same width $\log(1+\varepsilon)$ interval (e.g., $\{k_2,k_3\}$) are merged. Since relevant keys span length $t$ with width $\log(1+\varepsilon)$, only $\tilde{O}(1/\varepsilon)$ intervals exist. Using a prefix sum structure with $\tilde{O}(n)$ preprocessing and $\tilde{O}(1)$ query time per query for "sum of values in interval," the total time is $\tilde{O}(nB/\varepsilon)$. While this rounding method is subquadratic for $B=o(n)$, it still has polynomial dependence on $B$; upgrading "rounding" to "polynomial approximation + decoupling" yields the true subquadratic algorithm with logarithmic dependence on $B$.

## Key Experimental Results

This is a pure theory paper with no experiments. Its core conclusion is a summary table characterizing attention computation complexity (under $B=\mathrm{poly}(n), \varepsilon=1/\mathrm{poly}(n)$, omitting sub-polynomial factors):

| Head Dimension $d$ | Upper Bound (Prior) | Upper Bound (Ours) | Lower Bound (Prior) | Lower Bound (Ours) |
|---|---|---|---|---|
| $O(1)$ | $n^2$ | $n^{2-1/d}$ (Thm 1.1) | $n$ | — |
| $2^{\Theta(\log^* n)}$ | $n^2$ | — | $n$ | $n^{2-o(1)}$ (Thm 1.4) |
| $\Theta(\log n)$ | $n^2$ | $n^{2-o(1)}$* | — | $n^{2-o(1)}$ (Thm C.7) |
| $\mathrm{poly}(n)$ | $T_{\mathrm{MUL}}(n,d,n)$ | — | $n^{2-o(1)}$* | $T_{\mathrm{MUL}}(n,d,n)^{1-o(1)}$ (Thm 1.5) |

(* Denotes results from Alman & Song (2024); their $d=\Theta(\log n)$ lower bound required $B=\Omega(\sqrt{\log n})$, whereas this paper holds for $B\ge\log 2$.)

Key Complexity Conclusions:
- **Main Algorithm**: $\tilde{O}(n^{2-1/d}\cdot\mathrm{polylog}(B/\varepsilon))$ for $d=O(1)$, the first logarithmic dependence on $B$ (prior best was exponential $2^{O(B^2)}$).
- **Low Rank**: $\tilde{O}(nd+n^{2-1/r}\cdot\mathrm{polylog}(B/\varepsilon))$ for $r=\min(\mathrm{rank}(Q),\mathrm{rank}(K))$.
- **Gradient/Training**: General reduction to $O(d)$ attention calls + $O(nd^2)$ overhead.
- **Hardness**: $n^{2-o(1)}$ required for $d=2^{\Omega(\log^* n)}$ under SETH; $T_{\mathrm{MUL}}(n,d,n)^{1-o(1)}$ for $d=\mathrm{poly}(n)$, proving standard algorithms are conditionally optimal.

## Highlights & Insights

- **Transforming "Temperature/Entropy Bound" into an Analyzable Geometric Quantity**: Previous fast attention algorithms implicitly assumed high temperatures (small $B$). This paper explicitly handles any temperature using "discarding irrelevant keys + halfspace range search," providing the first true subquadratic algorithm with $\mathrm{polylog}$ dependence on $B$, moving beyond "pseudo-subquadratic."
- **Polynomial Method + Decoupling as Core Technique**: Replacing $\exp$ with a separable low-degree polynomial on a narrow interval allows amortizing "per query summation" from $O(n)$ to $\tilde{O}(n^{1-1/d})$.
- **Near-matching Upper and Lower Bounds**: Except for the $1\ll d\ll 2^{\Theta(\log^* n)}$ gap, the complexity is fully characterized. The paper equates improvements in this gap to breakthroughs in Max-IP algorithms.
- **Subquadratic Training Throughout**: By reducing gradient computation to attention calls, the subquadratic result extends to the full LLM training process ($d=O(1)$), not just inference.

## Limitations & Future Work

- **Exponential Dependence on $d$**: The algorithm requires $2^{\Omega(d)}$ Matoušek structures, which is only sub-polynomial for constant $d$. For $d=\omega(1)$, it degrades to $n^{2-o(1)}$, making theoretical acceleration hard to implement directly for high-dimensional practical use.
- **Element-wise vs. Norm Approximation**: This work provides $\ell_\infty$ additive guarantees, unlike Performer/Reformer which provide operator norm approximations in linear time; the hardness results here show linear time is impossible for such strong element-wise guarantees.
- **Purely Theoretical**: Constants and sub-polynomial factors may be large; whether it can outperform engineering implementations like FlashAttention on real sequences remains unverified.
- **Unclosed Gap & Reliance on Conjectures**: The range $1\ll d\ll 2^{\Theta(\log^* n)}$ remains open, and the high-dimensional hardness relies on the relatively new "Generalized High-dimensional OV Hypothesis."

## Related Work & Insights

- **Attention in Fine-grained Complexity**: Extends the hardness routes of Alman & Song (2024), Keles et al. (2023), and Alman & Yu (2025) by applying SETH-based OV/Max-IP reductions to attention.
- **Approximate Attention Engineering**: Methods like Performer, Reformer, and BigBird pursue linear time but only guarantee matrix norm approximations; this work theoretically bounds why they cannot achieve element-wise strong approximations.
- **Polynomial Method and Range Search**: Adopts tools from Williams, Abboud, and Matoušek, serving as a paradigm of tool migration from geometry and theoretical CS to attention computation.
- **Importance of Temperature**: Echoes research on the role of temperature in contrastive learning and Transformers (Chen 2020, Wang & Liu 2021), explaining from a computational complexity perspective "why low temperatures are harder to compute."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Poly-attention: a general scheme for higher-order self-attention](poly-attention_a_general_scheme_for_higher-order_self-attention.md)
- [\[ICLR 2026\] Quantum Machine Learning Advantages Beyond Hardness of Evaluation](quantum_machine_learning_advantages_beyond_hardness_of_evaluation.md)
- [\[ICLR 2026\] Parameterized Hardness of Zonotope Containment and Neural Network Verification](parameterized_hardness_of_zonotope_containment_and_neural_network_verification.md)
- [\[ICLR 2026\] Understanding In-Context Learning on Structured Manifolds: Bridging Attention to Kernel Methods](understanding_in-context_learning_on_structured_manifolds_bridging_attention_to_.md)
- [\[ICLR 2026\] Critical Attention Scaling in Long-Context Transformers](critical_attention_scaling_in_long-context_transformers.md)

</div>

<!-- RELATED:END -->
