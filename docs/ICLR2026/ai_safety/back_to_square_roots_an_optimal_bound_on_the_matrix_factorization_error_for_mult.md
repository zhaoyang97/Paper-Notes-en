---
title: >-
  [Paper Note] Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD
description: >-
  [ICLR 2026][AI Safety][differential privacy] This paper proposes the Banded Inverse Square Root (BISR) matrix factorization method. By imposing a banded structure on the inverse correlation matrix (rather than the correlation matrix itself), it achieves the first asymptotically optimal factorization error bound for multi-epoch Differentially Private SGD, accompan
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
content_hash: f1b723e5957ae8ce
---
# Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD

**Conference**: ICLR 2026  
**arXiv**: [2505.12128](https://arxiv.org/abs/2505.12128)  
**Code**: None (baseline comparisons performed using the jax-privacy library)  
**Area**: AI Safety / Differential Privacy  
**Keywords**: differential privacy, matrix factorization, DP-SGD, multi-epoch participation, banded factorization, optimal error bounds

## TL;DR
This paper proposes the Banded Inverse Square Root (BISR) matrix factorization method. By imposing a banded structure on the inverse correlation matrix (rather than the correlation matrix itself), it achieves the first asymptotically optimal factorization error bound for multi-epoch Differentially Private SGD, accompanied by a low-storage optimized variant, BandInvMF.

## Background & Motivation
**Background**: The Matrix Factorization Mechanism is a critical method for enhancing model utility in differentially private training by injecting correlated noise. It has been deployed by Google for production-level on-device language model training.

**Limitations of Prior Work**: In multi-epoch training, the same data point is utilized multiple times, requiring a characterization of the relationship between factorization error and the number of participations. However, a significant gap exists between existing upper and lower bounds—specifically, the dependence on the bandwidth $p$ in the error bound of Banded Square Root (BSR) is implicit, making it impossible to determine its optimality.

**Key Challenge**: Theoretically, the optimal growth rate of factorization error under multi-epoch participation remains unclear. Practically, there is a lack of factorization methods that are both computationally efficient and possess explicit error characterizations.

**Goal**: To establish a tight bound for the matrix factorization error under multi-epoch participation and provide a computationally efficient, theoretically optimal, and explicit factorization method.

**Key Insight**: Instead of making the correlation matrix $C$ banded like BSR, this work makes $C^{-1}$ banded. This perspective shift provides the dual advantages of explicit error characterization and efficient implementation.

**Core Idea**: Imposing a banded structure on the inverse correlation matrix enables noise injection to be efficiently implemented via convolution while obtaining an explicitly optimal error bound relative to the bandwidth.

## Method

### Overall Architecture
The matrix factorization mechanism decomposes the workload matrix $A$ to be published into $A = BC$. After injecting correlated noise on the private side, an unbiased estimate $\widehat{AX} = B(CX + Z)$ is obtained. The efficacy of the method depends entirely on the factorization error $\mathcal{E}(B,C)$, which is determined by $\|B\|_F$ and the column sensitivity of $C$. Existing BSR methods directly truncate the correlation matrix $C$ into a banded form; however, the dependence of the error on the bandwidth is buried within closed-form solutions, obscuring its optimality. This paper conversely bands $C^{-1}$: starting from the workload $A$, its positive definite square root $C$ (satisfying $C^2 = A$) is taken, then its inverse $C^{-1}$ is truncated into a $p$-banded matrix. Finally, re-inverting yields $C^p$, resulting in the factorization $A = B^p C^p$, where $B^p = A (C^p)^{-1}$ (Definition 1). This step provides two primary benefits: **explicit analyzability**, as the norms and sensitivities of the factors can be expressed in closed form once $C^{-1}$ is banded; and **inexpensive implementation**, as a banded $C^{-1}$ means $(C^p)^{-1}Z$ is equivalent to a convolution of the noise sequence with $p$ fixed coefficients, accelerable via FFT. In training (Algorithm 1), this involves maintaining a noise buffer of length $p$, adding weighted noise samples to gradients.

### Key Designs

**1. BISR Decomposition: Banding the Inverse Correlation Matrix for Convolutional Noise Injection**

The limitation of BSR is that it truncates the correlation matrix $C$ directly, making the dependence of the error on bandwidth $p$ implicit and requiring online optimization during training. BISR shifts the focus: starting from the workload $A$, it identifies the square root $C$, then truncates $C^{-1}$ to be $p$-banded, yielding the decomposition $A = B^p C^p$. This allows the error to be precisely characterized relative to bandwidth. For implementation, since $C^{-1}$ is banded, the transformation $(C^p)^{-1}Z$ is equivalent to a convolution, which only requires storing $p$ lines of noise in streaming scenarios. As it only affects linear transformations of gradients, it is naturally compatible with momentum and weight decay.

**2. Alignment of Bounds: Establishing the Optimal Growth Rate and Matching with BISR**

To establish "optimality," any decomposition's inherent lower error bound must be identified. This paper provides a tight lower bound (Theorem 3) using probabilistic methods to define participation vectors, yielding $\Omega(\sqrt{k}\log n + k)$ without weight decay ($\alpha = 1$) and $\Omega_\alpha(\sqrt{k})$ with weight decay ($\alpha < 1$). This defines the "square root" scaling of error relative to the number of participations $k$, hence the title "Back to Square Roots." The upper bound (Theorem 4 + Corollary 1) derived from BISR's explicit factor norms matches this lower bound in terms of order when the bandwidth is optimal. This alignment closes the gap left by BSR.

**3. BandInvMF: A Low-Storage Variant via Numerical Optimization**

BISR coefficients originate from a closed-form square root construction, which may not be literally optimal for finite scales. BandInvMF retains the "banded inverse matrix + Toeplitz" structure but uses numerical optimization to fit coefficients directly, initialized with BISR coefficients. This further reduces error in low-storage regimes while remaining simple to implement.

## Key Experimental Results

### Main Results: Table 1: CIFAR-10 Test Accuracy ((9, 10⁻⁵)-DP, 10 epochs)

| Method | Epoch 1 | Epoch 5 | Epoch 10 |
|------|---------|---------|----------|
| DP-SGD (Amp.) | 12.7±2.2 | 39.8±1.2 | 44.6±0.7 |
| BSR (Amp.) | 28.3±0.7 | 48.0±2.0 | 49.8±0.3 |
| **BISR (Amp.)** | **32.3±0.7** | **52.8±2.0** | **61.8±0.3** |
| Band-MF (Amp.) | 27.7±2.0 | 46.8±0.8 | 50.0±0.4 |
| Band-Inv-MF (Amp.) | 23.6±2.8 | 48.6±1.0 | 57.4±1.2 |
| DP-SGD (Non-Amp.) | 19.5±3.0 | 37.7±1.2 | 39.0±0.7 |
| **BISR (Non-Amp.)** | **31.8±1.5** | **51.1±1.0** | **56.2±0.2** |

### Table 2: RMSE Comparison (Factorization Error, n=16384)

| Method | k=4, α=1, β=0 | k=16, α=1, β=0 | k=16, α=1, β=0.9 |
|------|---------------|----------------|------------------|
| BSR | Comparable to BISR | Significantly worse than BISR | Worse than BISR |
| BLT | Comparable to BISR | Comparable to BISR | Prefix-sum only |
| BandMF | Slightly better (small n) | Slightly better but not scalable | High computational cost |
| **BISR** | **Optimal or near-optimal** | **Significantly better than BSR** | **Best consistency** |

## Highlights & Insights
- **Power of Perspective Shift**: Moving from banding $C$ to banding $C^{-1}$ enables explicit error characterization, a structural change that leads to a theoretical breakthrough.
- **Closed Loop of Theory and Practice**: BISR achieves both theoretical optimality (matching bounds) and practical competitiveness, with an extremely simple implementation (convolution).
- **Insight into Low-Storage Regimes**: Lower RMSE does not always equate to higher model accuracy. BandInvMF achieves better RMSE than BISR, but training accuracies are similar, suggesting the relationship between factorization error and model utility is not strictly monotonic.
- **Practicality**: Requiring only a convolution of $p$ coefficients, storage and computational costs are significantly lower than BandMF.

## Limitations & Future Work
1. **Asymptotic vs. Finite Scale Optimality**: BISR is asymptotically optimal, but numerical methods like BandMF may still perform slightly better for small matrix sizes.
2. **Unoptimized Constant Factors**: While bounds match in terms of order, gaps in constant terms remain.
3. **RMSE-Accuracy Disconnect**: Lower decomposition RMSE does not always translate to higher accuracy, particularly when using amplification by subsampling.
4. **BLT Comparison Restrictions**: BLT is only implemented for prefix-sum matrices, limiting comparison in momentum/weight decay settings.

## Supplementary Experiment: IMDB Sentiment Analysis (BERT-base)
- Fine-tuning BERT-base under (9, 10⁻⁵)-DP shows BISR outperforms BSR and Band-MF in both amplified and non-amplified settings.
- BISR (Amplified) significantly leads DP-SGD after 10 epochs.
- In low-storage regimes, BandInvMF and BISR show similar accuracy, but BISR avoids optimization solvers.

## Related Work
- **Matrix Factorization Mechanisms**: Choquette-Choo et al. (2023a) defined optimal factorization under multi-epoch participation; BLT (Dvijotham et al., 2024) provided buffer-based methods; BandMF (McKenna, 2025) solved optimal banded factorization via numerical optimization.
- **Square Root Factorization**: Proposed by Henzinger et al. (2024), extended to BSR by Kalinin & Lampert (2024).
- **Privacy Accounting**: This work utilizes MCMC accountant and bins-and-balls subsampling for analysis.

## Rating
- **Novelty**: ★★★★☆ — The inverse banding perspective is elegant and effective.
- **Practicality**: ★★★★☆ — Simple implementation with low overhead.
- **Theoretical Depth**: ★★★★★ — Closes the theoretical gap for multi-epoch factorization error.
- **Experimental Thoroughness**: ★★★★☆ — Dual evaluation of RMSE and training accuracy across multiple settings.
- **Writing Quality**: ★★★★☆ — Rigorous derivations and clear algorithmic descriptions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PE-SGD: Differentially Private Deep Learning via Evolution of Gradient Subspace for Text](pe-sgd_differentially_private_deep_learning_via_evolution_of_gradient_subspace_f.md)
- [\[ICLR 2026\] Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization](unified_privacy_guarantees_for_decentralized_learning_via_matrix_factorization.md)
- [\[ICLR 2026\] On Optimal Hyperparameters for Differentially Private Deep Transfer Learning](on_optimal_hyperparameters_for_differentially_private_deep_transfer_learning.md)
- [\[ICLR 2026\] Skirting Additive Error Barriers for Private Turnstile Streams](skirting_additive_error_barriers_for_private_turnstile_streams.md)
- [\[AAAI 2026\] An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses](../../AAAI2026/ai_safety/an_improved_privacy_and_utility_analysis_of_differentially_p.md)

</div>

<!-- RELATED:END -->
