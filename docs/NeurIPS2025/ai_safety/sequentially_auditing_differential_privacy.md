---
title: >-
  [Paper Note] Sequentially Auditing Differential Privacy
description: >-
  [NeurIPS 2025][AI Safety][differential privacy] This paper proposes a differential privacy auditing framework based on sequential hypothesis testing and kernel MMD statistics, enabling valid detection of privacy violations at any point during streaming mechanism outputs. The approach reduces the required sample count from 50K (as needed by existing methods) to just a few hundred, and can identify DP-SGD privacy violations within less than one full training run.
tags:
  - NeurIPS 2025
  - AI Safety
  - differential privacy
  - Sequential Testing
  - MMD
  - E-values
  - DP-SGD Auditing
date: 2026-05-08
content_hash: c0156d29ab7ce3eb
---

# Sequentially Auditing Differential Privacy

**Conference**: NeurIPS 2025
**arXiv**: [2509.07055](https://arxiv.org/abs/2509.07055)
**Code**: [Available](https://github.com/google-research/google-research/tree/master/dp_sequential_test)
**Area**: AI Safety
**Keywords**: differential privacy, Sequential Testing, MMD, E-values, DP-SGD Auditing

## TL;DR

This paper proposes a differential privacy auditing framework based on sequential hypothesis testing and kernel MMD statistics, enabling valid detection of privacy violations at any point during streaming mechanism outputs. The approach reduces the required sample count from 50K (as needed by existing methods) to just a few hundred, and can identify DP-SGD privacy violations within less than one full training run.

## Background & Motivation

Differential privacy (DP) has been widely adopted by industry and government agencies, yet the realization of its theoretical guarantees depends on correct algorithmic design and implementation. Subtle code bugs—such as missing constants, incorrect sampling, or fixed random seeds—can entirely undermine privacy protections. Moreover, theoretical privacy parameters $\varepsilon$ and $\delta$ are typically worst-case upper bounds, making empirical auditing essential for assessing actual privacy leakage.

Existing privacy auditing methods suffer from two major limitations:

**Parametric tests**: High statistical power but rely on strong assumptions (e.g., knowledge of mechanism internals and output distributions).

**Black-box tests**: Fewer assumptions but require large numbers of samples (typically 50K+), making them infeasible for computationally expensive algorithms such as DP-SGD—where each sample may require training a full model.

The root cause is that existing methods are **batch-based**: the sample count $n$ must be fixed in advance; too large wastes computation, too small yields inconclusive results, and samples generally cannot be reused.

The paper's key contribution is the introduction of **sequential testing**: samples are processed one at a time, the test statistic accumulates continuously, and the procedure stops as soon as statistical significance is reached—automatically adapting to the unknown complexity of the problem without requiring a predetermined sample size.

## Method

### Overall Architecture

Auditing DP consists of two steps: (1) identifying the pair of adjacent datasets that maximizes the divergence between $\mathcal{A}(S)$ and $\mathcal{A}(S')$; and (2) testing whether the privacy guarantee holds for that pair. This paper focuses on step (2).

The core mechanism constructs a stochastic process $\{\mathcal{K}_t\}$ that is a supermartingale (bounded in expectation) under the null hypothesis (DP is satisfied) and grows exponentially under the alternative hypothesis (DP is violated), enabling efficient detection of violations.

### Key Designs

1. **Novel connection between MMD and Hockey-Stick divergence (Theorem 3.1)**:

    - Approximate DP is defined via the Hockey-Stick divergence $D_{e^\varepsilon}(P\|Q) \leq \delta$.
    - When the likelihood ratio is unknown, direct testing of Hockey-Stick divergence is difficult.
    - The paper establishes a tighter MMD upper bound: if $\mathcal{A}$ is $(\varepsilon, \delta)$-DP, then $\text{MMD}(\mathcal{A}(S), \mathcal{A}(S')) \leq \sqrt{2}\left(1 - \frac{2(1-\delta)}{1+e^\varepsilon}\right)$.
    - This bound is strictly tighter than prior work and does not become vacuous as $\varepsilon \to \infty$ (it approaches $\sqrt{2}$ rather than $\infty$).
    - This justifies replacing the Hockey-Stick test with an MMD-based test.

2. **Sequential test statistic construction (Theorem 3.2)**:

    - Define the threshold $\tau(\varepsilon, \delta) = \sqrt{2}(1 - \frac{2(1-\delta)}{1+e^\varepsilon})$.
    - Hypothesis setting: $H_0: \text{MMD} \leq \tau$ vs. $H_1: \text{MMD} > \tau$.
    - Construct the product process $\mathcal{K}_t^*(\lambda) = \prod_{i=1}^t (1 + \lambda[f^*(X_i) - f^*(Y_i) - \tau])$.
    - Under $H_0$, this is a non-negative supermartingale for appropriate $\lambda$; Ville's inequality controls the Type I error.
    - Under $H_1$, $\log \mathcal{K}_t^*$ grows at rate $\Omega(\Delta^2)$, where $\Delta = \text{MMD} - \tau$.

3. **Practical algorithm (Algorithm 1)**:

    - Challenge: the optimal $\lambda^*$ and witness function $f^*$ are unknown and must be learned.
    - **Online Newton Step (ONS)** is used to adaptively learn $\lambda_t$.
    - **Online Gradient Ascent (OGA)** is used to learn $f_t$ from samples.
    - Theoretical guarantees (Theorem 3.3): under $H_0$, $\mathbb{P}(\mathcal{T} < \infty) \leq \alpha$; under $H_1$, the expected stopping time satisfies $\mathbb{E}[\mathcal{T}] = O\!\left(\frac{\log(1/(\alpha\Delta^2))}{\Delta^2}\right)$.

### Loss & Training

- ONS loss: $\ell_t(\lambda) = -\log(1 + \lambda[\langle f_t, K(X_t, \cdot) - K(Y_t, \cdot)\rangle_\mathcal{H} - \tau])$
- OGA loss: $h_t(f) = \langle f, K(X_t, \cdot) - K(Y_t, \cdot)\rangle_\mathcal{H}$, projected onto $\|f\|_\mathcal{H} \leq 1$
- Kernel bandwidth is set using the median of pairwise distances from the first 20 samples, which are excluded from the formal test.

## Key Experimental Results

### Main Results: Additive Noise Mean Estimation Mechanisms

| Mechanism | Rejection Rate ($\varepsilon=0.01$) | Avg. Samples | Rejection Rate ($\varepsilon=0.1$) | Avg. Samples |
|-----------|-----------------------------------|--------------|------------------------------------|--------------|
| DPGaussian (✓DP) | 0.0 ± 0.0 | — | 0.0 ± 0.0 | — |
| NonDPGaussian1 | **1.0 ± 0.0** | 264 ± 9 | **1.0 ± 0.0** | 562 ± 29 |
| NonDPGaussian2 | 0.85 ± 0.08 | 1139 ± 126 | 0.05 ± 0.05 | 4776 |
| DPLaplace (✓DP) | 0.0 ± 0.0 | — | 0.0 ± 0.0 | — |
| NonDPLaplace1 | **1.0 ± 0.0** | 331 ± 14 | **1.0 ± 0.0** | 920 ± 62 |
| NonDPLaplace2 | **1.0 ± 0.0** | 192 ± 18 | 0.95 ± 0.05 | 770 ± 262 |

(The prior method DPAuditorium fails to detect NonDPLaplace2 and NonDPGaussian2 even with 500K samples.)

### DP-SGD Auditing Experiments

| Setting | Result | Samples |
|---------|--------|---------|
| Correct DP-SGD ($\varepsilon_{\text{canary}}=0.01$) | No rejection across 5 runs | 500 |
| Non-private SGD, $H_0: \varepsilon \leq 0.01$ | Rejection at avg. **60** samples | 60 |
| Non-private SGD, $H_0: \varepsilon \leq 0.1$ | Rejection at avg. **75** samples | 75 |
| Non-private SGD, privacy lower bound after 250 steps | $\varepsilon = 0.43$ | 250 |

### Key Findings

- The sequential test achieves **two to three orders of magnitude** better sample efficiency than batch-based methods.
- Privacy violations in DP-SGD can be detected within **less than one full training run**.
- Correctly implemented DP mechanisms are never falsely rejected (Type I error is well controlled).
- Limitation: when the true $\varepsilon$ is large (>0.6), both the MMD and the threshold $\tau$ approach their theoretical upper bounds, reducing detection efficiency.

## Highlights & Insights

- **The central contribution is introducing sequential testing into DP auditing**—a natural yet non-trivial innovation, as adapting one-sided tests to a non-zero threshold requires substantial theoretical development.
- The tighter MMD–Hockey-Stick divergence bound directly improves test power and remains non-vacuous as $\varepsilon$ grows.
- Running tests in parallel across multiple $\varepsilon$ levels allows estimation of privacy lower bounds from a single observation stream, which is highly practical.
- The white-box DP-SGD auditing setup (projecting gradients onto the canary direction) produces one sample per training step, perfectly matching the sequential framework.

## Limitations & Future Work

- Auditing capability is limited for large $\varepsilon$ regimes, where $\Delta^2$ is too small and sample complexity becomes prohibitive.
- The method relies on a fixed pair of adjacent datasets; ideally, the worst-case pair should be identified adaptively.
- The choice of kernel function and bandwidth may affect test power, and optimal selection strategies remain underexplored.
- Extension to other privacy definitions such as Rényi DP or $f$-DP is a natural direction for future work.

## Related Work & Insights

- Sequential tests based on e-values and test supermartingales have recently been successfully applied in auditing fairness, financial risk, and elections; this paper is the first to apply them to general black-box DP auditing.
- The comparison with Richter et al. (language model behavioral drift detection) is particularly instructive: the parametric process $\mathcal{K}_t^*(\lambda)$ proposed here grows faster under $H_1$.
- Privacy auditing for LLMs (e.g., user data leakage in RLHF) is a promising direction for future application.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First application of modern sequential testing to DP auditing, with substantial theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers mean estimation and DP-SGD with thorough comparison against prior methods.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear and motivation is compellingly presented.
- Value: ⭐⭐⭐⭐⭐ — High practical value; substantially reduces the computational cost of DP auditing.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Multi-Class Support Vector Machine with Differential Privacy](multi-class_support_vector_machine_with_differential_privacy.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)
- [\[NeurIPS 2025\] Differential Privacy for Euclidean Jordan Algebra with Applications to Private Symmetric Cone Programming](differential_privacy_for_euclidean_jordan_algebra_with_applications_to_private_s.md)
- [\[NeurIPS 2025\] Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy](mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](../../AAAI2026/ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)

<!-- RELATED:END -->
