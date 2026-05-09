---
title: >-
  [Paper Note] Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD
description: >-
  [ICLR 2026][AI Safety][differential privacy] This paper proposes the Banded Inverse Square Root (BISR) matrix factorization method, which imposes a banded structure on the inverse correlation matrix (rather than on the correlation matrix itself). This approach achieves, for the first time, an asymptotically optimal factorization error bound for multi-epoch differentially private SGD, and is accompanied by a memory-efficient variant, BandInvMF.
tags:
  - ICLR 2026
  - AI Safety
  - differential privacy
  - matrix factorization
  - DP-SGD
  - multi-epoch participation
  - banded factorization
  - optimal error bounds
date: 2026-05-08
content_hash: 3d3b2c5e811e59ca
---

# Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD

**Conference**: ICLR 2026
**arXiv**: [2505.12128](https://arxiv.org/abs/2505.12128)
**Code**: None (uses the jax-privacy library for baseline comparisons)
**Area**: AI Safety / Differential Privacy
**Keywords**: differential privacy, matrix factorization, DP-SGD, multi-epoch participation, banded factorization, optimal error bounds

## TL;DR
This paper proposes the Banded Inverse Square Root (BISR) matrix factorization method, which imposes a banded structure on the inverse correlation matrix (rather than on the correlation matrix itself). This approach achieves, for the first time, an asymptotically optimal factorization error bound for multi-epoch differentially private SGD, and is accompanied by a memory-efficient variant, BandInvMF.

## Background & Motivation
**Background**: The Matrix Factorization Mechanism is an important technique in differentially private training that improves model utility by injecting correlated noise. It has been deployed by Google in production-scale on-device language model training.

**Limitations of Prior Work**: In multi-epoch training, each data point is used multiple times, necessitating a characterization of the factorization error as a function of the number of participation rounds. However, a significant gap exists between existing upper and lower bounds — the dependence on bandwidth $p$ in the error bound of the Banded Square Root (BSR) method is implicit, making it impossible to assess optimality.

**Key Challenge**: It is theoretically unclear what the optimal growth rate of the factorization error is under multi-epoch participation, and in practice there is no decomposition method that is both computationally efficient and admits an explicit error characterization.

**Goal**: To establish a tight bound on the matrix factorization error under multi-epoch participation and to provide a computationally efficient, theoretically optimal factorization method with an explicit closed-form characterization.

**Key Insight**: Rather than banding the correlation matrix $C$ as in BSR, the proposed approach bands $C^{-1}$ — a perspective shift that simultaneously enables explicit error characterization and efficient implementation.

**Core Idea**: Imposing a banded structure on the inverse correlation matrix allows noise injection to be implemented efficiently via convolution, while yielding an explicit and optimal error bound as a function of bandwidth.

## Method

### Overall Architecture
In differentially private SGD training, a public coefficient matrix is factorized as $A = BC$, and the private estimate is $\widehat{AX} = B(CX + Z)$. The central objective is to minimize the factorization error $\mathcal{E}(B,C)$, which is jointly determined by $\|B\|_F$ and the sensitivity of $C$.

### Key Designs

1. **BISR Factorization (Definition 1)**:

    - Compute the positive-definite diagonal matrix square root $C$ of the workload matrix $A$ (i.e., $C^2 = A$).
    - Compute $C^{-1}$ and truncate it to a $p$-banded matrix.
    - Invert the result to obtain $C^p$, yielding the factorization $A = B^p C^p$.
    - Key advantage: a banded $C^{-1}$ means noise injection requires only a convolution with $p$ coefficients, which can be accelerated via FFT.

2. **New Lower Bound (Theorem 3)**:

    - For any factorization, when $\alpha = 1$, the error is $\Omega(\sqrt{k}\log n + k)$.
    - When $\alpha < 1$ (with weight decay), the error is $\Omega_\alpha(\sqrt{k})$.
    - Proof technique: probabilistic method combined with norm bounds on participation vectors.

3. **BISR Upper Bound (Theorem 4)**:

    - The error explicitly depends on the bandwidth $p$, participation count $k$, matrix size $n$, and separation parameter $b$.
    - Selecting the optimal $p^* = O(b \log b)$ makes the upper bound match the lower bound, proving that BISR is asymptotically optimal.

4. **BandInvMF (Memory-Efficient Variant)**:

    - Retains the banded Toeplitz structure of the inverse matrix, but determines coefficients via numerical optimization rather than a closed-form solution.
    - Even with unit bandwidth, achieves $O(n^{1/4})$ error (compared to $O(\sqrt{n})$ for standard factorization), representing a significant improvement.
    - Initialized with BISR coefficients and converges in approximately 20 steps.

5. **Algorithm Implementation (Algorithm 1)**:

    - Only a buffer of $p$ noise vectors needs to be stored at each step.
    - Noise injection: $\hat{x}_i = x_i + \zeta \sum_{t=0}^{\min(p,i)-1} c_t Z_{i-t}$
    - Compatible with momentum and weight decay.

## Key Experimental Results

### Table 1: CIFAR-10 Test Accuracy ((9, 10⁻⁵)-DP, 10 epochs)

| Method | Epoch 1 | Epoch 5 | Epoch 10 |
|--------|---------|---------|----------|
| DP-SGD (Amp.) | 12.7±2.2 | 39.8±1.2 | 44.6±0.7 |
| BSR (Amp.) | 28.3±0.7 | 48.0±2.0 | 49.8±0.3 |
| **BISR (Amp.)** | **32.3±0.7** | **52.8±2.0** | **61.8±0.3** |
| Band-MF (Amp.) | 27.7±2.0 | 46.8±0.8 | 50.0±0.4 |
| Band-Inv-MF (Amp.) | 23.6±2.8 | 48.6±1.0 | 57.4±1.2 |
| DP-SGD (Non-Amp.) | 19.5±3.0 | 37.7±1.2 | 39.0±0.7 |
| **BISR (Non-Amp.)** | **31.8±1.5** | **51.1±1.0** | **56.2±0.2** |

### Table 2: RMSE Comparison (Matrix Factorization Error, n=16384)

| Method | k=4, α=1, β=0 | k=16, α=1, β=0 | k=16, α=1, β=0.9 |
|--------|----------------|-----------------|-------------------|
| BSR | Comparable to BISR | Notably worse than BISR | Worse than BISR |
| BLT | Comparable to BISR | Comparable to BISR | Supports prefix-sum only |
| BandMF | Slightly better (small matrices) | Slightly better but not scalable | Prohibitively expensive |
| **BISR** | **Optimal or near-optimal** | **Substantially better than BSR** | **Consistently best** |

> BISR's advantage is especially pronounced at high participation counts (k=16); while BandMF achieves slightly lower RMSE, it does not scale beyond n>4096.

## Highlights & Insights
- **The power of perspective shift**: Transitioning from "banding $C$" to "banding $C^{-1}$" yields an explicit error characterization — a seemingly minor structural change that enables a theoretical breakthrough.
- **Closing the theory-practice loop**: BISR simultaneously achieves theoretical optimality (matching upper and lower bounds) and practical competitiveness (accuracy comparable to BLT/BandMF), with an extremely simple implementation (convolution operations).
- **Insight in the memory-constrained regime**: Lower RMSE does not imply higher model accuracy — Band-Inv-MF achieves lower RMSE than BISR, yet both yield similar training accuracy, suggesting that the relationship between factorization error and model utility is not simply monotone.
- **Practical appeal**: Requiring only a convolution with $p$ coefficients, BISR incurs far lower storage and computational costs than methods such as BandMF that require solving an optimization problem.

## Limitations & Future Work
1. **Asymptotic optimality ≠ finite-size optimality**: BISR is optimal in the asymptotic sense, but numerical optimization methods such as BandMF may still achieve marginally better performance at finite matrix sizes.
2. **Constant factors not optimized**: The upper and lower bounds match only in terms of order; the gap in constant factors has not been fully eliminated.
3. **Disconnect between RMSE and accuracy**: Lower factorization RMSE does not necessarily translate into higher model accuracy, particularly when amplification by subsampling is used.
4. **Limited BLT comparison**: BLT is only implemented for prefix-sum matrices, precluding comparison under momentum and weight decay settings.

## Supplementary Experiments: IMDB Sentiment Analysis (BERT-base)
- Fine-tuning BERT-base under (9, 10⁻⁵)-DP, BISR outperforms BSR and Band-MF in both amplified and non-amplified settings.
- After 10 epochs, BISR (Amplified) substantially outperforms DP-SGD, demonstrating the advantage of the matrix factorization mechanism.
- In the memory-constrained regime, Band-Inv-MF achieves accuracy close to BISR but requires solving an optimization problem, whereas BISR does not.

## Related Work & Insights
- **Matrix Factorization Mechanism**: Choquette-Choo et al. (2023a) formalize the optimal factorization problem under multi-epoch participation; BLT (Dvijotham et al., 2024) provides a buffer-based approach; BandMF (McKenna, 2025) solves for the optimal banded factorization via numerical optimization.
- **Square Root Factorization**: Henzinger et al. (2024) introduce the approach; Kalinin & Lampert (2024) extend it to BSR and establish the first upper and lower bounds, though the bandwidth dependence remains implicit.
- **Privacy Accounting**: This work uses the MCMC accountant (Choquette-Choo et al., 2024b) and bins-and-balls subsampling (Chua et al., 2025) for privacy analysis.
- **Matrix Factorization in Federated Learning**: Zhang et al. (2025) and Bienstock et al. (2025) extend matrix factorization mechanisms to federated learning settings.

## Rating
- **Novelty**: ★★★★☆ — The perspective shift of banding the inverse matrix is elegant and effective; the accompanying tight theoretical bounds represent a significant contribution.
- **Practicality**: ★★★★☆ — Implementation is simple and efficient; convolution operations are parallelizable and a JAX implementation is available.
- **Theoretical Depth**: ★★★★★ — Closes the theoretical gap in factorization error under multi-epoch participation, with asymptotically matching upper and lower bounds.
- **Experimental Thoroughness**: ★★★★☆ — Dual evaluation via RMSE and training accuracy covers diverse optimizer settings and datasets, though large-scale LLM experiments are absent.
- **Writing Quality**: ★★★★☆ — Mathematical derivations are rigorous, algorithmic descriptions are clear, and the visualization in Figure 1 is intuitive.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization](unified_privacy_guarantees_for_decentralized_learning_via_matrix_factorization.md)
- [\[ICLR 2026\] Skirting Additive Error Barriers for Private Turnstile Streams](skirting_additive_error_barriers_for_private_turnstile_streams.md)
- [\[AAAI 2026\] An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses](../../AAAI2026/ai_safety/an_improved_privacy_and_utility_analysis_of_differentially_p.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](../../NeurIPS2025/ai_safety/differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[NeurIPS 2025\] Mitigating Disparate Impact of Differentially Private Learning through Bounded Adaptive Clipping](../../NeurIPS2025/ai_safety/mitigating_disparate_impact_of_differentially_private_learning_through_bounded_a.md)

<!-- RELATED:END -->
