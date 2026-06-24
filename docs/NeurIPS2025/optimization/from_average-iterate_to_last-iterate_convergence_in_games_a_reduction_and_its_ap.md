---
title: >-
  [Paper Note] From Average-Iterate to Last-Iterate Convergence in Games: A Reduction and Its Applications
description: >-
  [NeurIPS 2025][Optimization][Game learning] Proposes the A2L (Average to Last-iterate) black-box reduction. For games where the utility function is linear with respect to both a player's own strategy and the opponents' joint strategy, A2L converts the average-iterate of any uncoupled learning dynamics into the last-iterate of a new dynamics. Consequently, it achieves state-of-the-art (SOTA) last-iterate convergence rates of $O(\log d / T)$ under gradient feedback and $\tilde{…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Game learning"
  - "last-iterate convergence"
  - "OMWU"
  - "zero-sum games"
  - "black-box reduction"
date: 2026-05-08
content_hash: bf1cb2d15937bf0a
---

# From Average-Iterate to Last-Iterate Convergence in Games: A Reduction and Its Applications

**Conference**: NeurIPS 2025  
**arXiv**: [2506.03464](https://arxiv.org/abs/2506.03464)  
**Code**: None  
**Area**: Optimization  
**Keywords**: Game learning, last-iterate convergence, OMWU, zero-sum games, black-box reduction

## TL;DR
Proposes the A2L (Average to Last-iterate) black-box reduction. For games where the utility function is linear with respect to both a player's own strategy and the opponents' joint strategy, A2L converts the average-iterate of any uncoupled learning dynamics into the last-iterate of a new dynamics. Consequently, it achieves state-of-the-art (SOTA) last-iterate convergence rates of $O(\log d / T)$ under gradient feedback and $\tilde{O}(d^{1/5}T^{-1/5})$ under bandit feedback in multi-player zero-sum polymatrix games.

## Background & Motivation

**Background**: The convergence of online learning algorithms in games is a foundational problem in game theory and machine learning. Classical no-regret algorithms (e.g., MWU) guarantee that the average-iterate converges to a Nash equilibrium, but the actual/last-iterate strategies may diverge, cycle, or exhibit chaotic behavior.

**Limitations of Prior Work**:
   - Last-iterate convergence is significantly more challenging than average-iterate convergence; algorithm designs require optimism/regularization, and their analyses rely on customized Lyapunov functions.
   - Existing optimal last-iterate rates are $O(\text{poly}(d)/T)$ (e.g., the AOG algorithm), where the dependence on dimension is polynomial.
   - Under bandit feedback, the problem is even harder, with the prior best rates being $\tilde{O}(T^{-1/8})$ with high probability or $\tilde{O}(T^{-1/6})$ in expectation.

**Key Challenge**: Average-iterate convergence only requires the no-regret property (which is simple and straightforward), whereas last-iterate convergence requires dedicated designs—can this gap be bridged?

**Key Insight**: The linearity of the utility function implies that $u_i(\cdot, \bar{x}_{-i}) = \frac{1}{t}\sum_{k=1}^t u_i(\cdot, x_{-i}^k)$. Therefore, the feedback of the original strategies can be reconstructed from the observed feedback of the mixed strategies.

**Core Idea**: Design a general reduction, A2L, that lets each player output the running average of their internal strategies as the actual deployed strategy. Through linearity, the utility feedback of the internal strategies is reconstructed, thereby converting the last-iterate problem into an average-iterate problem.

## Method

### Overall Architecture
A2L is a black-box reduction: it takes an arbitrary online learning algorithm $\mathcal{R}$ as input and outputs a new algorithm $\text{A2L}(\mathcal{R})$. In each round $t$, it obtains strategy $x^t$ from $\mathcal{R}$ (which is not actually executed internally), and deploys the running average $\bar{x}^t = \frac{1}{t}\sum_{k=1}^t x^k$. Upon receiving the feedback $\bar{u}^t$, it reconstructs $u^t = t \cdot \bar{u}^t - \sum_{k=1}^{t-1} u^k$ and passes it back to $\mathcal{R}$.

### Key Designs

1. **A2L Black-Box Reduction (Algorithm 1)**:

    - **Function**: Converts the average-iterate of any online learning algorithm into the last-iterate of a new algorithm.
    - **Mechanism**: Player $i$ executes $\bar{x}_i^t = \frac{1}{t}\sum_{k=1}^t x_i^k$ and observes $\bar{u}_i^t = u_i(\cdot, \bar{x}_{-i}^t)$. By utility linearity, $\bar{u}_i^t = \frac{1}{t}\sum_{k=1}^t u_i(\cdot, x_{-i}^k)$, so $u_i^t = t \cdot \bar{u}_i^t - (t-1)\bar{u}_i^{t-1}$. The reconstructed $u^t$ is fed back into sub-procedure $\mathcal{R}$.
    - **Design Motivation**: The reduction preserves uncoupledness (each player only observes their own utility) and does not require shared randomness or communication.

2. **A2L-OMWU (Gradient Feedback)**:

    - **Function**: Applies A2L to the OMWU algorithm to achieve optimal last-iterate convergence.
    - **Mechanism**: The RVU property of OMWU guarantees that $\sum_i \text{Reg}_i(T) \leq \frac{\sum_i \log d_i}{\eta}$ (since the variation term is counteracted by the stability term when $\eta \leq \frac{1}{2(n-1)}$). By Lemma 1, $\text{TGap}(\bar{x}^T) = \frac{1}{T}\sum_i \text{Reg}_i(T)$. By Theorem 2, $\bar{x}^T$ is precisely the last-iterate of A2L-OMWU.
    - **Design Motivation**: OMWU already possesses an $O(\log d/T)$ average-iterate convergence rate; applying A2L directly yields last-iterate guarantees of the same rate.

3. **A2L-OMWU-Bandit (Bandit Feedback, Algorithm 2)**:

    - **Function**: Achieves last-iterate convergence when only observing the payoff of a single action (rather than the full utility vector).
    - **Mechanism**: In each round $t$, it mixes in $\epsilon_t$-exploration: $\bar{x}_{i,\epsilon}^t = (1-\epsilon_t)\bar{x}_i^t + \epsilon_t \text{Uniform}(d_i)$. It performs $B_t$ rounds of sampling to estimate the utility vector $\hat{U}_{i,\epsilon}^t$, reconstructs $\hat{u}_{i,\epsilon}^t = t\hat{U}_{i,\epsilon}^t - (t-1)\hat{U}_{i,\epsilon}^{t-1}$, and passes it to OMWU.
    - **Design Motivation**: Since bandit feedback does not directly provide the utility vector, estimation via replicated sampling is needed. A key technique in analyzing the estimation error is utilizing the negative terms of the RVU inequality to absorb second-order error terms.

### Loss & Training
- Gradient feedback: $\eta \leq \frac{1}{2(n-1)}$, without needing to know the time horizon $T$ (anytime guarantee).
- Bandit feedback: $\epsilon_t = t^{-1}$, $B_t = d \cdot t^4$, $\eta \leq \frac{1}{6n}$.
- Adversarial robustness: Monitors cumulative regret, switching to a standard no-regret algorithm if it exceeds a threshold.

## Key Experimental Results

### Main Theoretical Results

| Feedback Type | Algorithm | Last-Iterate Convergence Rate | Dimensional Dependence | Prev. SOTA |
|---------|------|-------------------|---------|----------|
| Gradient | A2L-OMWU | $O(\frac{\log d}{T})$ | $\log d$ | $O(\frac{\text{poly}(d)}{T})$ (AOG) |
| Bandit | A2L-OMWU-Bandit | $\tilde{O}(d^{1/5}T^{-1/5})$ | $d^{1/5}$ | $\tilde{O}(\sqrt{d}T^{-1/8})$ (high probability) |

### Corollary: Dynamic Regret

| Algorithm | Dynamic Regret | Prev. SOTA |
|------|---------|----------|
| A2L-OMWU | $O(\log d \cdot \log T)$ | $O(\text{poly}(d) \cdot \log T)$ |

### Key Findings
- Under gradient feedback, the dimensional dependence is exponentially improved from $\text{poly}(d)$ to $\log d$.
- Under bandit feedback, the rate is improved from $\tilde{O}(T^{-1/8})$ to $\tilde{O}(T^{-1/5})$. The key lies in the second-order error analysis technique (using the negative term of the RVU inequality $-\frac{1}{8\eta}\|x_i^t - x_i^{t-1}\|_1^2$ to absorb $\|\delta_i^t\|_\infty^2$).
- A2L is applicable to multi-player zero-sum polymatrix games (not just two-player games) and can be extended to proportional response dynamics (PRD) in Fisher markets.
- No regularization is needed (unlike the method of Cen et al.), and knowing $T$ is not required (offering a true anytime guarantee).
- The dynamic regret is only $O(\log d \cdot \log T)$, closely matching the $O(\log T)$ static regret of OMWU.

## Highlights & Insights
- **Conceptual Elegance**: The A2L reduction is incredibly concise (Algorithm 1 is only a few lines) yet yields powerful theoretical results. The core observation—that utility linearity allows reconstructing pure feedback from mixed feedback—is simple in hindsight but had not been uncovered until now.
- **Anytime Guarantee**: Unlike regularization methods that require knowing $T$ in advance to tune parameters, A2L-OMWU guarantees an $O(\log d/T)$-approximate Nash equilibrium for every $T \geq 1$.
- **Technical Innovation in Bandit Analysis**: Utilizing the negative term of the RVU inequality to absorb the second-order estimation error improves the rate from a naive $\tilde{O}(T^{-1/7})$ to $\tilde{O}(T^{-1/5})$.
- **Extension to Non-linear Utilities**: It remains applicable to PRD in Fisher markets, as its feedback exhibits a similar reconstructible structure.

## Limitations & Future Work
- It relies on the utility linearity assumption (Assumption 1), making it inapplicable to non-linear utility settings such as convex-concave min-max optimization—where EG/OG still maintain an advantage.
- It requires computing the running average and storing the historical utility vectors $\sum_{k=1}^{t-1} u^k$, incurring an $O(d \cdot T)$ space overhead, unlike average-free algorithms like EG/OG.
- Whether the $\tilde{O}(T^{-1/5})$ rate under bandit feedback is optimal remains an open question. Additionally, the total sample complexity for $B_t = d \cdot t^4$ is $\sum_{t=1}^T dt^4 \approx dT^5/5$, demanding huge resources in practice.
- Extending the framework to Markov games and time-varying games is an important future direction.
- The weighted version of A2L can utilize non-uniform weights $\alpha_t$ (e.g., linear weights $\alpha_t = t$), which could show empirical value for algorithms like regret matching, though theoretical analysis has not yet been conducted.
- The switching mechanism for adversarial robustness (switching to standard no-regret after detecting that regret exceeds a threshold) is not discussed in detail regarding its detection efficiency and latency impacts in practice.

## Related Work & Insights
- **vs AOG (Cai et al. 2023)**: AOG employs accelerated optimistic gradient to achieve an $O(\text{poly}(d)/T)$ last-iterate rate, whereas A2L-OMWU exponentially improves the dimensional dependence to $\log d$. Under the bandit setting, AOG only obtains $T^{-1/8}$ since its potential function is sensitive to errors, whereas A2L circumvents this difficulty by turning the problem into a regret analysis.
- **vs Regularization Methods (Cen et al. 2021/2024)**: These methods require knowing $T$ in advance to tune regularization parameters, only guarantee convergence for the very last step (not anytime), and their rate of $\tilde{O}(\log T / T)$ suffers from an extra $\log$ factor. In contrast, A2L-OMWU requires no regularization, does not need to know $T$, and eliminates the $\log T$ factor.
- **vs Original OMWU**: Cai & Zheng (2024) proved that the last-iterate of OMWU can be arbitrarily slow ($\Omega(1)$ lower bound); the A2L reduction elegantly bypasses this "impossibility" result.
- **vs Decay Regularization (Park et al. 2023)**: Decay regularization gives an $\tilde{O}(T^{-1/4})$ anytime rate, which is far slower than the $O(\log d/T)$ of A2L-OMWU.
- **vs Cutkosky (2019) AO2B reduction**: AO2B targets static optimization problems and does not guarantee the equivalence of original and new algorithm iterates; A2L targets dynamic games and strictly ensures that last-iterate = average-iterate.
- **Fisher Market PRD Application**: Demonstrates that A2L is not limited to Assumption 1, as long as the feedback satisfies a reconstructible structure (such as the proportional structure of PRD), obtaining an $O(1/T)$ last-iterate rate and improving upon the average-iterate-only result of Cheung (2025).

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The black-box reduction idea is extremely elegant, systematically reducing the last-iterate problem to average-iterate for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐ Purely theoretical work with no experiments, but the theoretical results are already very comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The paper is clearly structured, with the core idea apparent at a glance and highly informative proof sketches.
- **Value**: ⭐⭐⭐⭐⭐ Resolves core open problems in game learning, with results simultaneously improving both gradient and bandit settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Improved Last-Iterate Convergence of Shuffling Gradient Methods for Nonsmooth Convex Optimization](../../ICML2025/optimization/improved_last-iterate_convergence_of_shuffling_gradient_methods_for_nonsmooth_co.md)
- [\[ICLR 2026\] High-Probability Bounds for the Last Iterate of Clipped SGD](../../ICLR2026/optimization/high-probability_bounds_for_the_last_iterate_of_clipped_sgd.md)
- [\[NeurIPS 2025\] Sharper Convergence Rates for Nonconvex Optimisation via Reduction Mappings](sharper_convergence_rates_for_nonconvex_optimisation_via_reduction_mappings.md)
- [\[NeurIPS 2025\] Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner](purifying_shampoo_investigating_shampoos_heuristics_by_decomposing_its_precondit.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)

</div>

<!-- RELATED:END -->
