---
title: >-
  [Paper Note] Optimal Online Change Detection via Random Fourier Features
description: >-
  [NeurIPS 2025][LLM Pretraining][online change detection] This paper proposes the Online RFF-MMD algorithm, which approximates the MMD statistic via random Fourier features and embeds it into a sequential testing framework over a binary grid. The method achieves online nonparametric change detection without requiring training data or window size parameters, with $\mathcal{O}(r \log n)$ time and space complexity, and establishes minimax optimality of the detection delay.
tags:
  - NeurIPS 2025
  - LLM Pretraining
  - online change detection
  - random Fourier features
  - MMD
  - minimax optimality
  - nonparametric detection
date: 2026-05-08
content_hash: 850ffc183369507c
---

# Optimal Online Change Detection via Random Fourier Features

**Conference**: NeurIPS 2025
**arXiv**: [2505.17789](https://arxiv.org/abs/2505.17789)
**Code**: [GitHub](https://github.com/FlopsKa/rff-change-detection)
**Area**: LLM Pretraining
**Keywords**: online change detection, random Fourier features, MMD, minimax optimality, nonparametric detection

## TL;DR

This paper proposes the Online RFF-MMD algorithm, which approximates the MMD statistic via random Fourier features and embeds it into a sequential testing framework over a binary grid. The method achieves online nonparametric change detection without requiring training data or window size parameters, with $\mathcal{O}(r \log n)$ time and space complexity, and establishes minimax optimality of the detection delay.

## Background & Motivation

Online change detection is a classical statistical problem: when the distribution of a data stream shifts, the change should be detected as quickly as possible while controlling the false alarm rate. Modern applications (audio streams, video, traffic data, ECG time series) face two key challenges: data are high-dimensional with unknown distributions (ruling out parametric methods), and data volumes are large with real-time processing requirements.

Two major limitations of existing kernel methods:

**Not truly online**: Most methods assume knowledge of the pre-change distribution or access to historical data drawn from it (e.g., Scan B-statistics, Online kernel CUSUM).

**Require window parameters**: Nearly all methods require specifying a window size for computing local test statistics (including NEWMA). Poor window selection reduces power or increases delay, and the window size limits the minimum detectable change magnitude.

MMD has a core advantage as a detection statistic — under a characteristic kernel, MMD metrizes the space of probability distributions and can detect arbitrary distributional changes. However, standard MMD estimators require $\mathcal{O}(n^2)$ time, making them unsuitable for online settings.

The authors' key insight is to approximate MMD with RFF for online updatability, and eliminate the window parameter via binary grid scanning.

## Method

### Overall Architecture

The algorithm constructs a stopping time $N$: for each $n \geq 2$, RFF-MMD statistics are computed at $\log_2 n$ binary candidate change-point locations $n - 2^j$, and a change is declared when any statistic exceeds the threshold.

$$N = \inf\left\{n \geq 2 \mid \bigcup_{j=0}^{\lfloor\log_2(n)\rfloor - 1} \sqrt{\frac{2^j(n-2^j)}{n}} \text{MMD}_{\hat{K}}[X_{1:(n-2^j)}, X_{(n-2^j+1):n}] > \lambda_n \right\}$$

### Key Designs

1. **RFF Approximation of MMD**: For a translation-invariant kernel $K(\mathbf{x}, \mathbf{y}) = \psi(\mathbf{x} - \mathbf{y})$, Bochner's theorem implies $K$ can be expressed as an expectation under a spectral measure $\Lambda$. Sampling $\omega_1, \ldots, \omega_r \sim \Lambda$, the feature map is constructed as:

    $\hat{z}_K(\mathbf{x}) = \frac{1}{\sqrt{r}}((\sin(\omega_j^\top\mathbf{x}), \cos(\omega_j^\top\mathbf{x})))_{j=1}^r \in \mathbb{R}^{2r}$

   Then $\text{MMD}_{\hat{K}}[X_{1:n}, Y_{1:m}] = \|n^{-1}\sum_i \hat{z}_K(X_i) - m^{-1}\sum_j \hat{z}_K(Y_j)\|_2$. The key advantage is that mean embeddings become Euclidean vectors, computable in linear time with constant-time updates.

2. **Binary Grid Sequential Testing**: Rather than enumerating all $\mathcal{O}(n)$ candidate change-point locations, the algorithm tests only at positions $n - 2^j$, yielding $\log_2 n$ tests. This is implemented by maintaining a window list $\mathcal{W}$:

    - Each new observation creates a new window of size 1
    - Windows of equal size are automatically merged (z-values and counts are summed)
    - The merging process naturally maintains a binary structure

   Through memoization, the change detection component requires only a single pass over $\mathcal{W}$, with runtime $\mathcal{O}(|\mathcal{W}|r) = \mathcal{O}(r\log n)$.

3. **Two Threshold Modes**:

    - **ARL control** (Theorem 1): $\lambda_n \geq \sqrt{2} + \sqrt{2\log(4\gamma\log_2(2\gamma))}$ guarantees average run length $\geq \gamma$
    - **Uniform false alarm control** (Theorem 2): $\lambda_n \geq \sqrt{2} + \sqrt{2(\log(n/\alpha) + 2\log(\log_2 n) + \log(\log_2(2n)))}$ guarantees $\mathbb{P}_\infty(N < \infty) \leq \alpha$

   Both guarantees are independent of the number of RFF components and the pre-change distribution.

### Theoretical Guarantees

**Detection delay upper bound** (Theorem 3): Under compact support assumptions and signal strength condition $\eta \geq C_1\log(2\eta/\alpha)/\text{MMD}_K^2[\mathbb{P}, \mathbb{Q}]$, with probability $\geq 1-\alpha$:

$$(N - \eta)^+ \leq 1 \vee C_3 \frac{\log(2\eta/\alpha)}{(\text{MMD}_K[\mathbb{P}, \mathbb{Q}])^2}$$

**Minimax optimality** (Theorem 4): There exist absolute constants $\alpha_0, \beta_0$ such that for all $\alpha \leq \alpha_0$:

$$\inf_{N: \mathbb{P}_\infty(N<\infty)\leq\alpha} \sup_{\eta, \mathbb{P}, \mathbb{Q}} \mathbb{P}_\eta\left(N \geq \eta + C_K \frac{\log(1/\alpha)}{(\text{MMD}_K[\mathbb{P},\mathbb{Q}])^2}\right) \geq \beta_0$$

Both the upper and lower bounds are $\Theta(\log(1/\alpha)/\text{MMD}^2)$, establishing optimality up to logarithmic factors.

## Key Experimental Results

### Runtime Verification

| No. of RFF $r$ | Per-insertion time after 250K observations | Space complexity |
|---|---|---|
| 10 | ~0.01 ms | $\mathcal{O}(10\log n)$ |
| 100 | ~0.1 ms | $\mathcal{O}(100\log n)$ |
| 1000 | ~1 ms | $\mathcal{O}(1000\log n)$ |

The theoretical $\mathcal{O}(r\log n)$ runtime is confirmed empirically.

### Synthetic Data: Detection Delay Comparison

| Change Type | Target ARL | RFF-MMD EDD | OKCUSUM EDD | ScanB EDD | NEWMA EDD |
|---|---|---|---|---|---|
| $\mathcal{N}(0,I_{20}) \to$ Mixture Gaussian | 5000 | **~30** | ~60 | ~70 | ~100 |
| $\mathcal{N}(0,I_{20}) \to$ Laplace | 5000 | **~40** | ~80 | ~90 | ~120 |
| $\mathcal{N}(0,I_{20}) \to$ Uniform | 5000 | **~15** | ~35 | ~40 | ~50 |

### MNIST Experiment ($d=784$)

| Change Type | EDD at Target ARL=1000 | Comparison |
|---|---|---|
| Digit 0→1 | RFF-MMD **best** | Best across all ARL levels |
| Digit 0→2 | RFF-MMD **best** | |
| Digit 0→3 | RFF-MMD **best** | |

### Key Findings

- RFF-MMD outperforms all competitors across all tested post-change distributions and target ARL values.
- Increasing the number of RFF components does not adversely affect detection delay, unlike window-based methods.
- The constraint of requiring no training data does not noticeably degrade detection performance.

## Highlights & Insights

- **A genuine "free lunch"**: The method simultaneously addresses two practical challenges (no training data required, no window parameter required) without sacrificing theoretical guarantees.
- **The minimax optimality proof** is the first such result for kernel-based online change detection; analogous results previously existed only for offline settings and parametric online settings.
- **Algorithmic elegance**: The binary structure of window merging perfectly aligns with the additivity of MMD, enabling the maintenance of $\log n$ test statistics at near-zero overhead.
- The theoretical threshold formula is directly applicable without any distributional knowledge, making the method genuinely plug-and-play.

## Limitations & Future Work

- The choice of kernel and kernel parameters still influences detection power, even though the theoretical guarantees hold for all kernels satisfying Assumption 1.
- The i.i.d. assumption is required; the method is not directly applicable to strongly mixing or functionally dependent time series.
- The required number of RFF components in Theorem 3 depends on the unknown quantity $\text{MMD}_K[\mathbb{P}, \mathbb{Q}]$; in practice, using as many as computationally feasible is recommended.
- The paper focuses on single change-point detection; multi-change-point extensions are discussed in the appendix but with weaker theoretical guarantees.
- Expected risk analysis for worst-case average detection delay appears intractable in the fully nonparametric setting.

## Related Work & Insights

- NEWMA (Keriven et al., 2020) also uses RFF but implicitly retains a "window" (exponential smoothing is equivalent to two windows of different sizes).
- Lai (1995) first employed an exponential grid in change detection; the binary grid in this work refines that idea.
- The combination of RFF approximation and sequential testing may extend to online computation of other kernel statistics.
- This work is complementary to the exponential-window MMD approach of Kalinke et al. (2025).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of RFF and binary grid builds on known components but integrates them innovatively; the minimax optimality proof is a theoretical breakthrough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comparisons on synthetic data and MNIST are thorough, though more real-world time series experiments would strengthen the evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The structure is clear; Algorithm 1 and Example 1 are highly intuitive; theoretical statements are precise.
- **Value**: ⭐⭐⭐⭐⭐ The method simultaneously addresses two core practical problems with optimality guarantees, representing a significant contribution to online detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Law of Data Reconstruction for Random Features (and Beyond)](../../ICLR2026/llm_pretraining/a_law_of_data_reconstruction_for_random_features_and_beyond.md)
- [\[NeurIPS 2025\] Mouse-Guided Gaze: Semi-Supervised Learning of Intention-Aware Representations for Reading Detection](mouse-guided_gaze_semi-supervised_learning_of_intention-aware_representations_fo.md)
- [\[ICLR 2026\] Imagine How To Change: Explicit Procedure Modeling for Change Captioning](../../ICLR2026/llm_pretraining/imagine_how_to_change_explicit_procedure_modeling_for_change_captioning.md)
- [\[CVPR 2026\] Watch and Learn: Learning to Use Computers from Online Videos](../../CVPR2026/llm_pretraining/watch_and_learn_learning_to_use_computers_from_online_videos.md)
- [\[NeurIPS 2025\] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)

</div>

<!-- RELATED:END -->
