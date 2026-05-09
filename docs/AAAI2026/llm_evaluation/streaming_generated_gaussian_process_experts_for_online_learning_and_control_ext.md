---
title: >-
  [Paper Note] Streaming Generated Gaussian Process Experts for Online Learning and Control: Extended Version
description: >-
  [AAAI 2026][LLM Evaluation][Gaussian Process] This paper proposes SkyGP (Streaming Kernel-induced Progressively Generated Expert GP), which handles streaming data via **kernel-distance-driven progressive expert generation** and **time-aware configurable aggregation**, inheriting the learning guarantees of exact GP while maintaining bounded computational complexity. SkyGP comprehensively outperforms state-of-the-art methods on both benchmark regression tasks and real-time control experiments.
tags:
  - AAAI 2026
  - LLM Evaluation
  - Gaussian Process
  - Online Learning
  - Streaming Data
  - Mixture of Experts
  - Robot Control
date: 2026-05-08
content_hash: 5852ef7b2f84c3a4
---

# Streaming Generated Gaussian Process Experts for Online Learning and Control: Extended Version

**Conference**: AAAI 2026
**arXiv**: [2508.03679](https://arxiv.org/abs/2508.03679)
**Code**: [https://github.com/Zewen-Yang/SkyGP](https://github.com/Zewen-Yang/SkyGP)
**Area**: Machine Learning / Online Learning
**Keywords**: Gaussian Process, Online Learning, Streaming Data, Mixture of Experts, Robot Control

## TL;DR

This paper proposes SkyGP (Streaming Kernel-induced Progressively Generated Expert GP), which handles streaming data via **kernel-distance-driven progressive expert generation** and **time-aware configurable aggregation**, inheriting the learning guarantees of exact GP while maintaining bounded computational complexity. SkyGP comprehensively outperforms state-of-the-art methods on both benchmark regression tasks and real-time control experiments.

## Background & Motivation

1. **Background**: Gaussian Processes (GPs), as nonparametric methods, offer flexible modeling and calibrated uncertainty quantification, supporting online updates in polynomial time, making them well-suited for safety-critical systems.
2. **Limitations of Prior Work**: Exact GP inference over streaming data incurs $O(N^3)$ time and $O(N^2)$ memory costs, which do not scale with growing data. Existing solutions include sparse GPs (requiring expensive optimization and losing error guarantees) and distributed GPs (e.g., LoG-GP splits data along only a single dimension and does not handle non-stationarity).
3. **Key Challenge**: Online learning demands rapid adaptation and bounded complexity, yet the predictive performance guarantees of exact GPs require the full data matrix. Existing distributed methods either neglect online learning requirements or employ partitioning strategies that fail to exploit spatial/temporal correlations.
4. **Goal**: Design a streaming GP framework that dynamically manages a bounded collection of experts while inheriting the prediction error bounds of exact GPs.
5. **Key Insight**: Use kernel-function distance to decide whether incoming data should be assigned to an existing expert or used to initialize a new one, combined with a temporal decay factor to manage expert staleness.
6. **Core Idea**: Kernel-distance-driven adaptive expert assignment + time-aware aggregation = online learning with bounded complexity that preserves exact GP performance guarantees.

## Method

### Overall Architecture

When a new data point $(x^k, y^k)$ arrives: (1) the nearest expert is retrieved within an adaptive window via kernel distance; (2) if the nearest expert is not full, the data point is added (SkyGP-Fast: rank-1 Cholesky update); if full, the variant determines whether to replace data or create a new expert (SkyGP-Dense: data replacement); (3) at prediction time, the $\bar{N}$ nearest experts are selected and aggregated via MoE/PoE/BCM.

### Key Designs

1. **Kernel-Distance-Driven Expert Localization and Generation**

    - **Function**: Adaptively assigns streaming data to the most suitable expert, or creates a new expert when necessary.
    - **Mechanism**: Each expert $\mathcal{GP}_i$ maintains a center $c_i$ (updated incrementally as $c_i^k = (k-1)c_i^{k-1}/k + x^k/k$) and a kernel distance $d_i^k = 1/\kappa(c_i^k, x^k)$. The nearest expert is searched within an adaptive window $W = \min(\bar{W}, \lfloor\exp(d_{temp}/\varrho)\rfloor)$. If the nearest expert is full and the incoming data better matches the distribution of previously discarded data ($\Delta < 0$), a data replacement is performed (SkyGP-Dense); otherwise, a new expert is created. Experts are maintained in a sorted list by center position, and new experts are inserted at the neighbor position of the nearest expert.
    - **Design Motivation**: Conventional methods naively assign data to the first available expert without considering distributional properties. Kernel distance enforces expert locality — data from similar contexts is handled by the same expert.

2. **Time-Aware Aggregation Framework**

    - **Function**: Manages expert staleness and ensures prediction quality in non-stationary environments.
    - **Mechanism**: Each expert maintains a time-aware factor $\vartheta \in (0, 1]$, which is reset to 1 upon being queried and decays at rate $\rho$ otherwise. During aggregation, only active experts with $\vartheta > \bar{\vartheta}$ are selected. Three aggregation strategies are supported: MoE ($\omega_i = w_i$), PoE ($\omega_i = w_i \sigma_i^2 / \varpi_i$), and BCM (incorporating prior variance $\sigma_*$).
    - **Design Motivation**: Stale experts may represent outdated data distributions and are harmful to prediction in non-stationary settings. Temporal decay naturally retires obsolete experts.

3. **Two Variants: SkyGP-Dense vs. SkyGP-Fast**

    - **SkyGP-Dense**: Full experts maintain local representativeness via data replacement, requiring a full Cholesky recomputation ($O(\bar{N}^3)$ per expert); suited for memory-constrained scenarios.
    - **SkyGP-Fast**: No replacement is performed; new experts are created directly, with rank-1 Cholesky updates ($O(\bar{N}^2)$); suited for low-latency scenarios.

### Loss & Training

No conventional training is performed. GPs are used for direct inference via Cholesky decomposition. For control tasks, a learning-based control strategy is derived that leverages GP uncertainty to design a safe feedback controller. Theoretical guarantees are provided for bounded prediction error: $|f(x) - \tilde{\mu}| \leq \beta\sigma(x) + \gamma(x)$.

## Key Experimental Results

### Main Results

Results on online regression benchmarks (RMSE↓) and real-time control experiments:

| Method | Online Regression Accuracy | Inference Time | Memory | Control Error |
|---|---|---|---|---|
| Exact GP | Best but not scalable | $O(N^3)$ | $O(N^2)$ | - |
| SSGP | Moderate | High | Moderate | - |
| LoG-GP | Moderate | Moderate | Bounded | Moderate |
| **SkyGP-Dense** | **Close to Exact GP** | **Bounded** | **Bounded** | **Best** |
| **SkyGP-Fast** | Moderately superior | **Fastest** | Bounded | Superior |

### Ablation Study

| Configuration | Effect | Notes |
|---|---|---|
| MoE vs. PoE vs. BCM aggregation | BCM generally best | Prior variance calibrates uncertainty |
| Time-aware vs. without time-awareness | Time-aware superior | Stale experts are harmful on non-stationary data |
| Event-triggered vs. always-replace | Event-triggered superior | Reduces unnecessary Cholesky recomputation |
| Window size $W$ | Adaptive is optimal | Fixed window fails to adapt to varying data density |

### Key Findings

- SkyGP-Dense achieves prediction accuracy close to exact GP with bounded complexity — validating the claim of "scalability without sacrificing performance."
- SkyGP-Fast achieves the lowest inference latency, making it well-suited for real-time control.
- The time-aware factor is critical for non-stationary data — removing it leads to significant performance degradation.
- In robot control experiments, SkyGP's uncertainty estimates directly drive safe control policies, enabling online-adaptive and safe trajectory tracking.
- The theoretical guarantee (Lemma 1) is empirically validated — prediction errors consistently fall within the $\beta\sigma(x) + \gamma(x)$ bound.

## Highlights & Insights

- **Kernel-Distance-Driven Intelligent Data Assignment**: Rather than simple first-come-first-served or random partitioning, data is adaptively assigned based on kernel-space similarity, ensuring local consistency within each expert.
- **Inherited Theoretical Guarantees**: Error bounds are rigorously derived from exact GP guarantees to the aggregated SkyGP setting, providing theoretical assurances rarely seen in distributed GP literature.
- **Dense/Fast Dual-Mode Design**: A single framework supports two deployment modes, flexibly accommodating different computation-memory trade-off requirements.

## Limitations & Future Work

- Kernel hyperparameters (e.g., lengthscale) are assumed to be pre-determined or globally optimized; online hyperparameter adaptation is not addressed.
- The number of experts has no hard upper bound — highly non-stationary environments may lead to excessive expert creation.
- Only single-output GPs are validated; extensions to multi-output GPs (e.g., vector-valued functions) are not discussed.
- Real-time control experiments are conducted exclusively on Euler–Lagrange systems.

## Related Work & Insights

- **vs. LoG-GP**: Partitions data along only a single dimension, offering limited flexibility in high-dimensional spaces; SkyGP adaptively assigns data in multi-dimensional space via kernel distance.
- **vs. SSGP**: Requires expensive optimization at each step and provides no error guarantees; SkyGP maintains low latency via rank-1 updates with theoretical guarantees.
- **vs. Sparse GPs (FITC/VFE)**: Require global training and non-trivial inducing point selection; SkyGP is fully online and requires no global retraining.
- The framework is generalizable to multi-robot collaborative learning — each robot maintains local GP experts and shares information via a communication network.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of kernel-distance-driven progressive expert generation and time-aware aggregation is an innovative contribution
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of benchmark regression, real-time control, and theoretical validation
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations and clear algorithmic descriptions
- Value: ⭐⭐⭐⭐ Practical deployment value for online learning in safety-critical systems

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[NeurIPS 2025\] Turbocharging Gaussian Process Inference with Approximate Sketch-and-Project](../../NeurIPS2025/llm_evaluation/turbocharging_gaussian_process_inference_with_approximate_sketch-and-project.md)
- [\[AAAI 2026\] TRACE: A Generalizable Drift Detector for Streaming Data-Driven Optimization](trace_a_generalizable_drift_detector_for_streaming_data-driven_optimization.md)
- [\[NeurIPS 2025\] Conformal Online Learning of Deep Koopman Linear Embeddings](../../NeurIPS2025/llm_evaluation/conformal_online_learning_of_deep_koopman_linear_embeddings.md)
- [\[CVPR 2026\] Enhancing Out-of-Distribution Detection with Extended Logit Normalization](../../CVPR2026/llm_evaluation/enhancing_out-of-distribution_detection_with_extended_logit_normalization.md)

</div>

<!-- RELATED:END -->
