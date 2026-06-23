---
title: >-
  [Paper Note] Proving the Limited Scalability of Centralized Distributed Optimization via a New Lower Bound Construction
description: >-
  [ICLR 2026][Optimization & Theory][Paper Note] This paper proves that in classical federated learning (centralized distributed optimization), once the "server $\to$ worker" communication cost $\tau_s$ is factored in, even unbiased sparsification compressors cannot simultaneously improve the "communication term $\tau_s d L\Delta/\varepsilon$" and the "variance term
tags:
  - ICLR 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 4811df8be176600f
---
# Proving the Limited Scalability of Centralized Distributed Optimization via a New Lower Bound Construction

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=0KXI6lDM9C](https://openreview.net/forum?id=0KXI6lDM9C)  
**Code**: To be confirmed (Theoretical paper, no code)  
**Area**: Distributed Optimization Theory / Lower Bounds  
**Keywords**: Federated Learning, Communication Compression, Non-convex Optimization Lower Bounds, Worst-case Construction, Scalability

## TL;DR
This paper proves that in classical federated learning (centralized distributed optimization), once the "server $\to$ worker" communication cost $\tau_s$ is factored in, even unbiased sparsification compressors cannot simultaneously improve the "communication term $\tau_s d L\Delta/\varepsilon$" and the "variance term $h\sigma^2 L\Delta/\varepsilon^2$" with the number of workers $n$. Even in the homogeneous (i.i.d.) setting where all workers share the same function, the improvement magnitude is at most a polynomial-logarithmic factor $\mathrm{poly}\log n$. To this end, the authors construct a novel chain-based worst-case function and develop a proof framework that reduces the lower bound to "random sum concentration inequalities."

## Background & Motivation

**Background**: Distributed optimization connects multiple workers (CPUs/GPUs/mobile devices) to a central server. Each worker computes stochastic gradients, transmits them to the server for aggregation, and the server broadcasts the result back. The goal is to find an $\varepsilon$-stationary point ($\mathbb{E}[\|\nabla f(\bar x)\|^2]\le\varepsilon$) of an $L$-smooth non-convex function $f$. The fundamental motivation for using many workers is "scalability": as the number of workers $n$ increases, the execution should be faster. Theoretically, this holds for cases like Synchronous SGD, where the time complexity is $O\!\big(h(\tfrac{L\Delta}{\varepsilon}+\tfrac{\sigma^2 L\Delta}{n\varepsilon^2})\big)$, and the "statistical term" $h\sigma^2 L\Delta/(n\varepsilon^2)$ improves linearly with $n$. With unbiased sparsification compressors (e.g., RandK), even the "communication term" can improve with $n$, potentially yielding a coupling term $\sqrt{d\tau_w h\sigma^2/(n\varepsilon)}\cdot L\Delta/\varepsilon$ that improves with $\sqrt n$.

**Limitations of Prior Work**: These optimistic conclusions about "more workers being faster" are almost entirely built on the unrealistic assumption that server-to-worker broadcasting is free, i.e., $\tau_s=0$. In real-world internet or 4G/5G networks, downlink broadcast is equally expensive. Once $\tau_s>0$ is acknowledged, the complexity of Batch QSGD includes an additional term $\tau_s d L\Delta/\varepsilon$ (see Eq. (4)), which **does not improve with $n$ at all**: the server must send the $d$-dimensional $x^k$ to all workers every round, and adding more workers does not alleviate this downlink bandwidth requirement.

**Key Challenge**: In the heterogeneous (non-i.i.d.) case, Gruntkowska et al. (2024) proved that iteration complexity does not improve with $n$. However, the homogeneous (i.i.d., where all workers have the same $f$/distribution) setting is intuitively "easier." Since everyone has the same function, can workers cooperatively explore coordinates to amortize the downlink cost $\tau_s d$ with $n$? This is the contradiction: intuition suggests it might be possible, but no one has proved whether it is feasible.

**Goal**: In the simplest homogeneous setting, the paper asks a clear diagnostic question: Can a method using unbiased sparsification compressors be designed such that the downlink communication term $\tau_s d L\Delta/\varepsilon$ and the variance term $h\sigma^2 L\Delta/\varepsilon^2$ **simultaneously** improve (linearly or by $\sqrt n$) with $n$?

**Key Insight**: Lower bounds. Proving "it cannot be done" is more definitive than proving "it can." By constructing a worst-case function and a worst-case compressor such that any "zero-respecting" algorithm cannot accelerate on it, the entire class of methods is effectively ruled out. The difficulty lies in the fact that old constructions for the heterogeneous case (assigning different blocks of Eq. (6) to different workers) fail in the homogeneous setting. In the homogeneous case, all workers can advance toward the next coordinate simultaneously, so the $\min_{i\in[n]}$ term that could save a factor of $n$ in old arguments actually helps the algorithm.

**Core Idea**: The worst-case function is modified from "discovering the next coordinate requires only the previous coordinate to be non-zero" to "a sequence of $K$ coordinates must be non-zero to take one step forward." This $K$-chain forces even the "luckiest worker" to satisfy $K$ rare events simultaneously, thereby weakening the $1/n$ decay brought by $\min_{i\in[n]}$ in the lower bound to $1/n^{1/K}$. By setting $K=\Theta(\log n)$ such that $n^{1/K}=\Theta(1)$, the benefit of $n$ is completely neutralized, leaving only polynomial-logarithmic terms.

## Method

### Overall Architecture

This is a purely theoretical lower bound paper without algorithms or experimental figures. The "Method" refers to **how the lack of scalability with $n$ is proven**. The proof chain is: 
1. Fix a sufficiently general family of distributed methods (Protocol 1 + zero-respecting algorithms $\mathcal{A}_{zr}$), covering almost all methods that "act based on the span of observed coordinates" like GD, Adam, MARINA-P.
2. Provide all workers with the same newly constructed worst-case function $F_{T,K,a}$ and use a worst-case randomized sparsification compressor (RandK family) for downlink communication.
3. Prove that under this construction, the minimum stochastic time required for the algorithm to "discover the $T$-th coordinate (the prerequisite for reaching an $\varepsilon$-stationary point)" equals a specific random sum $t_B$ (Eq. (13)).
4. Prove a concentration inequality for $t_B$ to obtain a high-probability lower bound.
5. Substitute the values of $T, p_\sigma, p_K$ and select $K=\Theta(\log n)$ and $a=1+1/K$ to finalize the Main Theorem.

Recall the classical construction being replaced: the chain function of Carmon et al. (2020)
$$F_T(x)=\sum_{i=1}^{T}\big(\Psi(-x_{i-1})\Phi(-x_i)-\Psi(x_{i-1})\Phi(x_i)\big),\quad x_0\equiv1,$$
The key property is "the gradient is only potentially non-zero at $\mathrm{prog}(x)+1$." Starting from $x_i=0$, one gradient query lights up at most the next coordinate, requiring $T=\Theta(L\Delta/\varepsilon)$ queries. Heterogeneous lower bounds (Gruntkowska et al. 2024) assign different blocks to different workers so only one worker can advance at a time. However, in the homogeneous case, if everyone has the same $F_T$, all workers can advance at once, leading to $\tau_s\sum_j\min_{i\in[n]}\eta_{ji}\gtrsim\tau_s\tfrac{d}{n}\tfrac{L\Delta}{\varepsilon}$ (Eq. (8)), which improves with $n$. This necessitates a new construction.

### Key Designs

**1. K-chain worst-case function $F_{T,K,a}$: Changing "one step" from requiring 1 non-zero coordinate to requiring K**

In the old construction, a worker could discover the $i$-th coordinate if the previous $x_{i-1} \neq 0$. This low threshold allowed all workers to easily synchronize and advance. The authors replace the negative block $-\Psi(x_{i-1})\Phi(x_i)$ with a product of $K$ terms of $\Psi_a$:
$$F_{T,K,a}(x)=-\sum_{i=1}^{T}\Psi_a(x_{i-K})\cdots\Psi_a(x_{i-2})\Psi_a(x_{i-1})\Phi(x_i)+\sum_{i=1}^{T}\Gamma(x_i),$$
where $x_0=\cdots=x_{-K+1}\equiv1$, and $\Gamma(x)=-x e^{1/x+1}$ ($x<0$) or $0$ ($x\ge0$). The new progress measure is
$$\mathrm{prog}_K(x):=\max\{i\ge 0:\ x_i\ne 0, x_{i-1}\ne 0, \dots, x_{i-K+1}\ne 0\},$$
meaning " $K$ consecutive coordinates must be non-zero to progress." Lemma 3.1 ensures $\mathrm{supp}(\nabla F_{T,K,a}(x))\subseteq\{1,\dots,\mathrm{prog}_K(x)+1\}\cup\mathrm{supp}(x)$, and $\|\nabla F_{T,K,a}(x)\|>1$ as long as $\mathrm{prog}_K(x)<T$. The $\Gamma$ term prevents the algorithm from "climbing" to a negative stationary point to take a shortcut.

The cost of multiplying $K$ terms of $\Psi$ is that function geometry (value range, smoothness constant, $\ell_\infty$ gradient bound) might explode exponentially with $K$. The authors introduce a parameters $a\in(1,e]$ to modify $\Psi$ with $\log a$ to suppress this. Lemma 3.2 yields controllable constants: $\Delta_0(K,a)=\sqrt{2\pi e}\,a^K$, $\ell_1(K,a)=\tfrac{12\sqrt{2\pi}e^{5/2}K^2 a^K}{\log a}$, and $\gamma_\infty(K,a)=6\sqrt{2\pi}e^{3/2}\tfrac{K a^K}{\sqrt{\log a}}$. Choosing $1<a\ll e$ cancels the explosion caused by $K$, which is the prerequisite for selecting $K=\Theta(\log n)$.

**2. Reduction to a concentration inequality for random sum $t_B$: Neutralizing $\min_{i\in[n]}$ via "double counting + summation"**

With the $K$-chain, the minimum stochastic time for an algorithm to discover the $T$-th coordinate is characterized as a random sum. Workers can light up new coordinates in two ways: computing a "lucky" local stochastic gradient (probability $p_\sigma=\Theta(\varepsilon\gamma_\infty^2(K,a)/\sigma^2)$) or receiving a coordinate within $[K]$ from the server (probability $p_K:=2K/T$). Let $\eta_{b,i,k}$ be the number of gradients computed before the $k$-th "lucky" gradient and $\mu_{b,i,k}$ be the number of coordinates received before the $k$-th $[K]$-coordinate. These follow geometric distributions. To complete a segment of $K$ coordinates, one path must contribute at least $K/2$ coordinates. Thus, the total time is no less than:
$$t_B:=\sum_{b=1}^{B}\min_{i\in[n]}\Big\{\min\Big\{\,h\!\sum_{k=1}^{K/2}\eta_{b,i,k},\ \ \tau_s\!\sum_{k=1}^{K/2}\mu_{b,i,k}\Big\}\Big\},\qquad B=\lfloor T/K\rfloor.$$
Like Eq. (8), the $\min_{i\in[n]}$ exists (we only wait for the "luckiest worker"). **The essence of the new construction**: every worker now has an internal sum $\sum_{k=1}^{K/2}$. A lucky worker is no longer one who "gets lucky once," but one who "gets lucky $K/2$ times in a row," which significantly offsets the decay of $\min_{i\in[n]}$. A concentration inequality for $t_B$ is proven:
$$t_B\ \gtrsim\ \frac{BK}{n^{1/K}}\,\min\{h/p_\sigma,\ \tau_s/p_K\}\quad(\text{with high probability}),$$
where the denominator is $n^{1/K}$ instead of $n$—this is yielded by the $K$-chain.

**3. Selecting $K=\Theta(\log n)$ and $a=1+1/K$: Crushing the benefit of $n$ into a poly-logarithmic barrier**

The previous steps leave a tension: larger $K$ suppresses $n^{1/K}$, but the constants in Lemma 3.2 grow with $K$. Substituting $T, p_\sigma, p_K$ back:
$$t_B\ \gtrsim\ \frac{L\Delta}{n^{1/K}\Delta_0(K,a)\,\ell_1(K,a)\,\varepsilon}\,\min\Big\{\max\Big\{h,\ \tfrac{h\sigma^2}{\varepsilon\gamma_\infty^2(K,a)}\Big\},\ \tfrac{\tau_s d}{K}\Big\}.$$
The final stroke: select $K=\Theta(\log n)$, so $n^{1/K} = n^{1/\Theta(\log n)} = \Theta(1)$, removing $n$ from the main term. Setting $a=1+1/K$ keeps $\Delta_0, \ell_1, \gamma_\infty$ at $\mathrm{poly}\log n$ levels. This leads to Theorem 4.2 (informal Eq. 11):
$$\widetilde\Omega\Big(\min\Big\{\tau_s d\,\tfrac{L\Delta}{\varepsilon},\ \ h\tfrac{L\Delta}{\varepsilon}+h\tfrac{\sigma^2 L\Delta}{\varepsilon^2}\Big\}\Big).$$
Implication: One can improve dependence on $d$ OR dependence on $\sigma^2/\varepsilon$, but **not both simultaneously**. The improvement scale of $d$ and $\sigma^2/\varepsilon$ is capped at $\log^4(n+1)$ and $\log^6(n+1)$ respectively, far worse than linear $n$ or $\sqrt n$.

**4. Incorporating uplink communication $\tau_w>0$ for the complete lower bound**

Theorem 4.2 only accounts for downlink $\tau_s$. The authors additionally prove Theorem F.1, extending and improving results from Tyurin et al. (2024), to incorporate uplink cost $\tau_w$, yielding the full lower bound in Eq. (5). When $\tau_s \simeq \tau_w$, it simplifies to:
$$\widetilde\Omega\Big(\min\Big\{h\big(\tfrac{\sigma^2}{n\varepsilon}+1\big)\tfrac{L\Delta}{\varepsilon}+\tau_s d\,\tfrac{L\Delta}{\varepsilon},\ \ h\tfrac{L\Delta}{\varepsilon}+h\tfrac{\sigma^2 L\Delta}{\varepsilon^2}\Big\}\Big),$$
which is matched by **uncompressed** Batch Synchronized SGD (Eq. 2) or local plain SGD. This implies that when $\tau_s \simeq \tau_w$, unbiased sparsification compression is **useless**.

## Key Experimental Results

### Main Results: Complexity Scalability with n across Three Communication Scenarios

| Scenario | Representative Methods | Time Complexity (Key Terms) | Scales with $n$? |
|------|---------|------|------|
| Free communication $\tau_s=\tau_w=0$ | Synchronous SGD | $h\big(\tfrac{L\Delta}{\varepsilon}+\tfrac{\sigma^2 L\Delta}{n\varepsilon^2}\big)$ | Statistical term improves linearly ✅ |
| Uplink only $\tau_w>0,\ \tau_s=0$ | Batch QSGD | $h(1+\tfrac{\sigma^2}{n\varepsilon})\tfrac{L\Delta}{\varepsilon}+\tau_w(\tfrac{d}{n}+1)\tfrac{L\Delta}{\varepsilon}+\sqrt{\tfrac{d\tau_w h\sigma^2}{n\varepsilon}}\tfrac{L\Delta}{\varepsilon}$ | Communication improved by $n$, coupling by $\sqrt n$ ✅ |
| Bidirectional $\tau_s,\tau_w>0$ | Batch QSGD | Extra term $\tau_s d\,\tfrac{L\Delta}{\varepsilon}$ (Eq. 4) | Downlink term **does not improve** ❌ |

### Lower Bound and Matching Algorithms

| Metric | Conclusion | Description |
|------|------|------|
| Lower Bound (11) (only $\tau_s$) | $\widetilde\Omega\big(\min\{\tau_s d\tfrac{L\Delta}{\varepsilon},\ h\tfrac{L\Delta}{\varepsilon}+h\tfrac{\sigma^2 L\Delta}{\varepsilon^2}\}\big)$ | $d$ and $\sigma^2/\varepsilon$ cannot simultaneously improve with $n$ |
| Improvement capacity for $d$ | At most $\log^4(n+1)$ | Far worse than linear $n$ |
| Improvement capacity for $\sigma^2/\varepsilon$ | At most $\log^6(n+1)$ | Far worse than $\sqrt n$ |
| Matching algorithms ($\tau_s\simeq\tau_w$) | Batch Synchronized SGD (uncompressed) / Plain SGD | Compression provides no gain |

### Key Findings
- **Homogeneous is not "easier" than heterogeneous**: Intuition suggests shared functions allow cooperative downlink amortization, but with $\tau_s \simeq \tau_w$, the lower bound matches the heterogeneous case; compression does not save downlink bandwidth.
- **The logarithmic barrier is steep**: For $n=10,000$, keeping $\tau_s d/\log^4(n+1)$ constant while increasing $d$ by 10x requires increasing $n$ by $10^3x$, making scalability almost "illusory."
- **$K=\Theta(\log n)$ is the sweet spot**: Too small $K$ fails to suppress $n^{1/K}$; too large $K$ leads to constant explosion in Lemma 3.2.
- **Lower bound applies to biased compressors**: Since unbiased compressor families include biased counterparts, worst-case compressors exist for TopK/RankK that prevent acceleration.

## Highlights & Insights
- **"Raising the threshold" to neutralize $\min$ gains**: Changing the progress requirement from 1 coordinate to $K$ coordinates forces the luckiest worker to "win $K/2$ times" rather than once, softening $1/n$ to $1/n^{1/K}$.
- **Logarithmic trick for $n^{1/K}\to\Theta(1)$**: Setting $K=\Theta(\log n)$ turns exponential decay into a constant, a clean shift from construction to poly-log barriers.
- **Parameter $a$ decouples expressivity from geometry**: Multiplying $K$ terms of $\Psi$ causes smoothness constants to explode; $a=1+1/K$ specifically suppresses this back to poly-log levels.
- **Reduction to concentration of random sums**: Eq. (13) translates abstract "discovery time" into a sum of geometric random variables, providing a reusable proof framework.

## Limitations & Future Work
- The lower bound is precise only up to logarithmic factors; improving the power of $\log$ is difficult and may require new constructions.
- The worst-case compressor used was randomized sparsification (RandK); the authors **conjecture** (based on Safaryan et al. 2022) that this applies to the entire unbiased family, but a formal proof is missing.
- These are pessimistic worst-case constructions: In practice, TopK/RankK may perform better and EF/EF21-P on the server side might alleviate the downlink bottleneck.
- Additional assumptions (convexity, second-order smoothness) might break this pessimistic lower bound.

## Related Work & Insights
- **vs Carmon et al. (2020)**: Ours extends their chain function $F_T$ by upgrading coordinate thresholds to $K$-chains and using $\Gamma$ and $a$ to control geometry.
- **vs Gruntkowska et al. (2024)**: They proved non-scalability in **heterogeneous** settings; this paper tackles the harder **homogeneous** setting where simple block partitioning fails.
- **vs Tyurin et al. (2024)**: Their methods (Shadowheart SGD / Batch QSGD) are matching algorithms for the lower bound, and their lower bound results are extended here via Theorem F.1.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The $K$-chain construction and $n^{1/K}$ log-trick are original and reusable tools.
- Experimental Thoroughness: ⭐⭐⭐⭐ While theoretical, the comparison between lower bounds and matching algorithms across three scenarios is clear.
- Writing Quality: ⭐⭐⭐⭐⭐ Progresses logically from the failure of old constructions to new technical breakthroughs.
- Value: ⭐⭐⭐⭐⭐ Sets a hard lower bound for the optimistic narrative of "infinite acceleration via compression," reshaping understanding of scalability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bilevel Optimization with Lower-Level Uniform Convexity: Theory and Algorithm](bilevel_optimization_with_lower-level_uniform_convexity_theory_and_algorithm.md)
- [\[ICLR 2026\] LEGACY: A Lightweight Dynamic Gradient Compression Strategy for Distributed Deep Learning](legacy_a_lightweight_dynamic_gradient_compression_strategy_for_distributed_deep_.md)
- [\[ICLR 2026\] Conformal Robustness Control: A New Strategy for Robust Decision](conformal_robustness_control_a_new_strategy_for_robust_decision.md)
- [\[ICLR 2026\] Towards Better Branching Policies: Leveraging the Sequential Nature of Branch-and-Bound Tree](towards_better_branching_policies_leveraging_the_sequential_nature_of_branch-and.md)
- [\[ICLR 2026\] MT-DAO: Multi-Timescale Distributed Adaptive Optimizers with Local Updates](mt-dao_multi-timescale_distributed_adaptive_optimizers_with_local_updates.md)

</div>

<!-- RELATED:END -->
