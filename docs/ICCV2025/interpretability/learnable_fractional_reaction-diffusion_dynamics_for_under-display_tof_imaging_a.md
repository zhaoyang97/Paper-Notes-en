---
title: >-
  [Paper Note] Learnable Fractional Reaction-Diffusion Dynamics for Under-Display ToF Imaging and Beyond
description: >-
  [ICCV 2025][Interpretability][Under-display ToF imaging] LFRD² proposes a hybrid framework that combines learnable time-fractional reaction-diffusion equations with neural networks for under-display ToF (UD-ToF) depth ma…
tags:
  - "ICCV 2025"
  - "Interpretability"
  - "Under-display ToF imaging"
  - "fractional reaction-diffusion"
  - "continuous convolution"
  - "depth restoration"
  - "physics-driven"
date: 2026-05-08
content_hash: 408a2669f09cb3cc
---

# Learnable Fractional Reaction-Diffusion Dynamics for Under-Display ToF Imaging and Beyond

**Conference**: ICCV 2025
**arXiv**: [2511.01704](https://arxiv.org/abs/2511.01704)
**Code**: [https://github.com/wudiqx106/LFRD2](https://github.com/wudiqx106/LFRD2)
**Area**: Depth Restoration / Computational Imaging
**Keywords**: Under-display ToF imaging, fractional reaction-diffusion, continuous convolution, depth restoration, physics-driven

## TL;DR

LFRD² proposes a hybrid framework that combines learnable time-fractional reaction-diffusion equations with neural networks for under-display ToF (UD-ToF) depth map restoration. The approach captures long-range memory dependencies across iterations via fractional calculus and introduces an efficient continuous convolution operator to replace discrete convolution, achieving state-of-the-art performance on UD-ToF depth restoration, ToF denoising, and depth super-resolution tasks.

## Background & Motivation

- **The full-screen trend** has driven the development of under-display sensors: under-display RGB cameras are already commercialized, with under-display ToF (UD-ToF) depth cameras as the next frontier.
- **Transparent OLED (TOLED) panels** severely degrade ToF camera signals through signal attenuation, multi-path interference (MPI), temporal noise, and other artifacts, substantially reducing depth quality.
- **Traditional diffusion methods** (e.g., P-M diffusion) exploit physical priors for depth refinement and offer robust adaptability and generalization, but suffer from complex parameter modeling and high computational cost.
- **Deep learning methods** excel at high-level image understanding and contextual reasoning, yet rely heavily on architectural design and data quality, lacking physical interpretability.
- **Integer-order algorithm unrolling** maps iterative algorithms to deep network layers, but relies on **integer-order differential equations (IDEs)** in which the predicted state depends only on the current state, ignoring historical information.
- **Fractional-order differential equations (FDEs)** possess **memory properties** — the current state depends on all historical states — which better reflects real physical processes.
- **Core Motivation**: Can neural networks be used to learn solutions to fractional reaction-diffusion equations, improving depth restoration quality while maintaining physical interpretability?

## Method

### Overall Architecture

LFRD² consists of two stages:
1. **Deep Initial State Builder (DISB)**: Uses an existing network (e.g., UD-ToFnet) to generate an initial depth map $u_0$.
2. **Deep Fractional Reaction-Diffusion Module**: Performs iterative depth refinement based on the Caputo fractional derivative.

### Key Designs

1. **Fractional Reaction-Diffusion Dynamics**:

    - Employs the Caputo fractional derivative of order $0 < \alpha < 1$:
    ${}^C_0 D_t^\alpha u(t) = \frac{1}{\Gamma(1-\alpha)} \int_0^t (t-\tau)^{-\alpha} u'(\tau) d\tau$
    - Discretized via the L1 approximation, yielding the iterative scheme:
    $u_{n+1} = u_n + S[\text{div}(g|\nabla u_n|\nabla u_n) + \lambda(u_0 - u_n)] - \sum_{k=1}^n a_k^{(\alpha)}(u_{n+1-k} - u_{n-k})$
    - where $S = \Gamma(2-\alpha)/a_0^\alpha$ and $a_k^{(\alpha)} = (k+1)^{1-\alpha} - k^{1-\alpha}$.
    - **Memory property**: The current state $u_{n+1}$ depends on all historical states $u_0, \ldots, u_n$, rather than $u_n$ alone.
    - **The fractional order $\alpha$ is dynamically predicted by the neural network**, rather than being fixed a priori.
    - In the diffusion term $\text{div}(g|\nabla u|\nabla u)$, the function $g(\cdot)$ is learned by the neural network rather than defined by a traditional conductance function.
    - The reaction term $\lambda(u_0 - u_n)$ drives the depth evolution toward the target state.

2. **Efficient Continuous Convolution Operator**:

    - Conventional discrete convolution disregards the continuity of natural scenes.
    - Existing continuous convolution implementations (MLP-based Neural Fields) incur high computational cost and complex hyperparameter tuning.
    - This paper exploits the repeated differentiation/integration property: $u * \mathcal{K} = u^{(-n)} * \mathcal{K}^{(n)}$.
    - At $n=2$, the estimated kernel $\hat{\mathcal{K}}^{(2)}$ degenerates into a sparse Dirac delta.
    - **Novelty**: Rather than predefining Gaussian kernels and control points, the DISB directly generates the Dirac delta.
    - The antiderivative of the signal is efficiently approximated as $u^{(-2)} \approx A \cdot u(x_0, y_0) + B$, where coefficients $A$ and $B$ are predicted by a three-layer convolution network.
    - Compared to NFC (Neural Field Convolution), this design **reduces FLOPs by 62%** (7.69G vs. 20.5G) with faster inference speed (22.75ms vs. 28.42ms).

3. **Physical Interpretability**:

    - The entire iterative process encodes a time-fractional reaction-diffusion equation.
    - The non-local nature of fractional calculus provides an appropriate framework for describing dynamic processes with memory effects.
    - The neural network serves as an estimator of the fractional order, which can be viewed as a form of physics-informed neural networks (PINNs).

### Loss & Training

- Adam optimizer with initial learning rate $1 \times 10^{-4}$ and batch size 16.
- SUD-ToF trained for 250 epochs; RUD-ToF trained for 1000 epochs.
- Original $180 \times 240$ images are cropped to $176 \times 240$ patches.
- DISB is based on UD-ToFnet with its original settings preserved.
- Reaction term coefficient $\lambda = 0.01$.
- Experiments conducted on an NVIDIA RTX 3090.

## Key Experimental Results

### Main Results (Tables)

**SUD-ToF / RUD-ToF Datasets**:

| Method | SUD-ToF MAE↓ | SUD-ToF RMSE↓ | RUD-ToF MAE↓ | RUD-ToF RMSE↓ |
|--------|-------------|--------------|-------------|--------------|
| PE-ToF | 9.77 | 15.92 | 21.22 | 48.76 |
| NAFNet | 11.08 | 18.24 | 20.41 | 33.83 |
| Restormer | 9.75 | 14.76 | 18.94 | 31.78 |
| UD-ToFnet | 8.88 | 11.50 | 17.29 | 31.11 |
| **LFRD² (Ours)** | **8.41** | **10.99** | **16.73** | **30.94** |

**FLAT Dataset (ToF Denoising)**:

| Method | MAE↓ | RMSE↓ |
|--------|------|-------|
| SHARPnet | 4.62 | 10.26 |
| UD-ToFnet | 4.41 | 8.23 |
| **LFRD²** | **4.13** | **7.35** |

**NYUv2 Dataset (Depth Super-Resolution, MSE/MAE)**:

| Method | 4× | 8× | 16× |
|--------|----|----|-----|
| DSR-EI | 2.94/0.49 | 13.3/1.19 | 57.0/2.70 |
| **LFRD²** | **2.85/0.47** | **12.8/1.16** | **52.3/2.58** |

### Ablation Study (Tables)

**Core Component Ablation (RUD-ToF)**:

| Configuration | Params/M | FLOPs/G | Speed/ms | MAE↓ | RMSE↓ |
|---------------|----------|---------|----------|------|-------|
| Baseline (UD-ToFnet) | 2.17 | 8.65 | 15.20 | 17.29 | 31.11 |
| + GRU | +0.18 | +7.62 | 19.89 | 17.02 | 31.09 |
| + LSTM | +0.24 | +10.4 | 22.15 | 16.96 | 31.22 |
| **+ LFRD²** | **+0.18** | **+7.69** | **22.75** | **16.73** | **30.94** |
| w/o FC (integer-order) | +0.01 | +0.41 | 20.67 | 17.00 | 30.99 |
| w/o CC (no continuous conv.) | +0.17 | +7.28 | 22.11 | 16.88 | 31.03 |
| NFC | +0.13 | +20.5 | 28.42 | 16.97 | 31.00 |

**Fractional Order Ablation**:

| Order $\alpha$ | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | Learnable (Ours) |
|----------------|------|------|------|------|------|-----------------|
| MAE/mm | 18.12 | 18.38 | 18.86 | 19.01 | 18.29 | **17.62** |
| $\rho_{1.02}$/% | 66.79 | 66.80 | 66.16 | 65.73 | 66.04 | **67.43** |

### Key Findings

- **Fractional vs. integer order**: Removing fractional calculus (w/o FC) increases MAE from 16.73 to 17.00, confirming the importance of memory properties.
- **Continuous vs. discrete convolution**: Removing continuous convolution (w/o CC) increases MAE from 16.73 to 16.88.
- **vs. RNN variants**: LFRD² outperforms both GRU and LSTM on MAE and RMSE, while maintaining parameter count comparable to GRU.
- **vs. NFC**: Achieves comparable accuracy while reducing FLOPs by 62% and improving speed by 25%.
- **Learnable order outperforms all fixed orders**, dynamically adapting to the optimal order for each sample.
- **Cross-task generalization**: The same framework achieves state-of-the-art results across three distinct tasks — UD-ToF restoration, ToF denoising, and depth super-resolution.
- **Plug-and-play**: The DISB can be replaced by various baselines (PE-ToF, NAFNet, Restormer, etc.), consistently yielding improvements.

## Highlights & Insights

- **Hybrid physics-driven and data-driven design**: Embedding fractional PDEs into neural network iterations provides both physical interpretability and learning flexibility.
- **Memory property of fractional calculus** is the key innovation: leveraging a weighted combination of all historical iteration states makes the approach more robust than integer-order methods that rely solely on the current state.
- **Dynamic fractional order** learned by the network addresses the difficulty of manual parameter selection in traditional fractional-order methods.
- The efficient continuous convolution implementation (coefficient prediction + repeated differentiation) is more practical than MLP-based Neural Fields.
- **Strong cross-task applicability**: The same framework, without modification, applies to different depth restoration tasks.

## Limitations & Future Work

- Training requires careful configuration to avoid numerical instability (NaN values); robustness needs improvement.
- The current explicit numerical scheme may benefit from implicit formulations for greater stability and efficiency.
- The number of iteration steps is fixed; adaptive step-size control could further improve efficiency.
- The choice of DISB affects final performance, but selection criteria for the optimal initializer are not thoroughly investigated.
- Although more efficient than NFC, continuous convolution still increases inference time by approximately 50%.

## Related Work & Insights

- **Perona-Malik Diffusion**: A classical image enhancement diffusion model serving as the integer-order baseline for this work.
- **TNRD (Chen & Pock, 2016)**: Trainable Nonlinear Reaction Diffusion, learning time-varying filter parameters from data.
- **UD-ToFnet (Qiao et al.)**: A pioneering work on UD-ToF depth restoration, serving as the DISB baseline in this paper.
- **NFC (Nsampi et al.)**: Continuous convolution implementation based on repeated differentiation, the reference method for the continuous convolution design.
- **Algorithm Unrolling**: Unrolls iterative algorithms into network layers, forming the methodological foundation of LFRD².
- Insight: The potential of fractional calculus in image processing remains largely underexplored.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Combining fractional reaction-diffusion with deep learning and the learnable-order design are both novel contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four datasets, three tasks, and detailed ablations covering core components, continuous convolution inputs, and order selection.
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous mathematical derivations, clear illustrations, and thorough physical explanations.
- **Value**: ⭐⭐⭐⭐ — Methodological contribution to physics-driven depth restoration with strong cross-task generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](../../NeurIPS2025/interpretability/beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)
- [\[CVPR 2026\] Pixel2Phys: Distilling Governing Laws from Visual Dynamics](../../CVPR2026/interpretability/pixel2phys_distilling_governing_laws_from_visual_dynamics.md)
- [\[NeurIPS 2025\] Dynamic Algorithm for Explainable k-medians Clustering under lp Norm](../../NeurIPS2025/interpretability/dynamic_algorithm_for_explainable_k-medians_clustering_under_lp_norm.md)
- [\[NeurIPS 2025\] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](../../NeurIPS2025/interpretability/beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)
- [\[NeurIPS 2025\] Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT](../../NeurIPS2025/interpretability/beyond_token_probes_hallucination_detection_via_activation_tensors_with_act-vit.md)

</div>

<!-- RELATED:END -->
