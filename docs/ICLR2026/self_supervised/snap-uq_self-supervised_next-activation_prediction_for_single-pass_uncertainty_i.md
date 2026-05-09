---
title: >-
  [Paper Note] SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty
description: >-
  [ICLR 2026][Self-Supervised Learning][Uncertainty Estimation] SNAP-UQ proposes a single-forward-pass uncertainty estimation method tailored for TinyML scenarios. Lightweight int8 prediction heads are attached at selected tap layers of a backbone network; these heads predict the activation statistics of the next layer in a self-supervised manner. The deviation ("surprisal") between predicted and actual activations is aggregated into an uncertainty score. The method requires no additional forward passes, temporal buffers, or ensembles, and adds only tens of kilobytes of flash memory, enabling reliable distribution-shift detection and failure detection on microcontrollers.
tags:
  - ICLR 2026
  - Self-Supervised Learning
  - Uncertainty Estimation
  - TinyML
  - Single-Pass Inference
  - Microcontroller Deployment
  - OOD Detection
date: 2026-05-08
content_hash: df4e598ab02bf282
---

# SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty

**Conference**: ICLR 2026
**arXiv**: [2508.12907](https://arxiv.org/abs/2508.12907)
**Code**: None
**Area**: Self-Supervised Learning
**Keywords**: Uncertainty Estimation, TinyML, Single-Pass Inference, Self-Supervised Learning, Microcontroller Deployment, OOD Detection

## TL;DR

SNAP-UQ proposes a single-forward-pass uncertainty estimation method tailored for TinyML scenarios. Lightweight int8 prediction heads are attached at selected tap layers of a backbone network; these heads predict the activation statistics of the next layer in a self-supervised manner. The deviation ("surprisal") between predicted and actual activations is aggregated into an uncertainty score. The method requires no additional forward passes, temporal buffers, or ensembles, and adds only tens of kilobytes of flash memory, enabling reliable distribution-shift detection and failure detection on microcontrollers.

## Background & Motivation

TinyML models are increasingly deployed on battery-powered microcontrollers (MCUs) for privacy-preserving, low-latency visual and audio perception. However, the input distribution evolves continuously after deployment—sensor drift, lighting and acoustic environment changes, and alternating occurrences of in-distribution corruption (CID) and out-of-distribution (OOD) samples are common. Modern neural networks frequently exhibit overconfidence under such shifts, even when well-calibrated on held-out sets.

Addressing uncertainty estimation on MCUs faces severe constraints:

- **MC Dropout** and **Deep Ensembles** require multiple forward passes, multiplying latency and flash costs.
- **Early-exit ensembles** still require additional classification heads and memory bandwidth at inference time, and rely on softmax signals that are fragile under CID.
- **Post-hoc calibration** (Temperature Scaling) typically fails under distribution shift.
- **Classical OOD detectors** (ODIN/G-ODIN) transfer poorly to ultra-compact models.

The key insight is that inter-layer dynamics reflect distribution shift earlier than softmax confidence—features become atypical with respect to the network's own transformations before the class posterior flattens.

## Method

### Overall Architecture

SNAP-UQ attaches lightweight prediction heads at 2–3 selected tap layers $\mathcal{S}$ of the backbone. Each head predicts the conditional Gaussian parameters (mean $\mu_\ell$ and diagonal variance $\sigma_\ell^2$) of the next layer's activations from a low-rank projection of the current layer's activations. The normalized prediction error ("surprisal") is computed, aggregated with learned weights, and mapped through a lightweight function to produce the final uncertainty estimate.

### Key Designs

1. **Depth-wise next-activation prediction model**:

   - At each tap layer $\ell \in \mathcal{S}$, the preceding layer's activations are compressed via projector $P_\ell$: $z_\ell = P_\ell a_{\ell-1} \in \mathbb{R}^{r_\ell}$ ($r_\ell \ll d_{\ell-1}$).
   - The prediction head $g_\ell$ outputs diagonal Gaussian parameters: $(\mu_\ell, \log \sigma_\ell^2) = g_\ell(z_\ell)$.
   - An optional low-rank plus diagonal covariance structure is supported: $\Sigma_\ell = \text{diag}(\sigma_\ell^2) + B_\ell B_\ell^\top$, computed efficiently via the Woodbury identity.
   - **Design Motivation**: The method models the conditional inter-layer relationship $a_{\ell-1} \mapsto a_\ell$, rather than relying on conventional unconditional class-level statistics.

2. **Self-supervised training objective**:

   - The auxiliary loss is based on diagonal Gaussian NLL: $\mathcal{L}_{SS} = \frac{1}{|\mathcal{B}|}\sum_{x \in \mathcal{B}} \sum_{\ell \in \mathcal{S}} \frac{1}{2}[\|(a_\ell - \mu_\ell) \odot \sigma_\ell^{-1}\|^2 + \mathbf{1}^\top \log \sigma_\ell^2]$
   - Total loss: $\mathcal{L} = \mathcal{L}_{clf} + \lambda_{SS}\mathcal{L}_{SS} + \lambda_{reg}\mathcal{R}$, with $\lambda_{SS}$ small ($10^{-3} \sim 10^{-2}$).
   - Regularization: a variance lower bound (softplus + $\epsilon^2$) prevents collapse; scale control $\mathcal{R}_{var} = \sum_\ell \|\log \sigma_\ell^2\|_1$ prevents over-dispersion.
   - An optional detach mode applies stop-gradient to $a_\ell$ for small backbones to avoid gradient conflicts.

3. **Single-pass surprisal aggregation and mapping**:

   - Normalized error: $\bar{e}_\ell(x) = \frac{1}{d_\ell}\|(a_\ell - \mu_\ell) \odot \sigma_\ell^{-1}\|^2$
   - SNAP score: $S(x) = \sum_{\ell \in \mathcal{S}} w_\ell \bar{e}_\ell(x)$
   - Final uncertainty via logistic mapping fused with an optional confidence proxy: $U(x) = \sigma(\beta_0 + \beta_1 S(x) + \beta_2 m(x))$
   - Mapping parameters are fitted offline once; no online labels are required.

4. **MCU-friendly integer implementation**:

   - $P_\ell$, $W_\mu$, and $W_\sigma$ are quantized to int8.
   - Exponentiation for $\exp(-\frac{1}{2}\log\sigma^2)$ is replaced by a 256-entry lookup table (LUT).
   - With 2 tap points and $r_\ell \in [32, 128]$, the additional computation is less than 2% of backbone cost, and flash overhead is only tens of KB.

### Loss & Training

- Dimension normalization is applied to prevent large-dimensional layers from dominating the loss.
- Layer weights are either uniform or inverse-variance-weighted ($w_\ell \propto 1/\hat{\text{Var}}[\bar{e}_\ell]$).
- Quantization-aware training (QAT) inserts fake quantization during the final 20% of training epochs.
- Student-$t$ and Huberized variants are provided as robust alternatives.

## Key Experimental Results

### Main Results — MCU Deployment Performance

| Platform/Task | Method | Flash (KB) | Peak RAM (KB) | Latency (ms) | Energy (mJ) |
|---|---|---|---|---|---|
| Big-MCU/SpeechCmd | BASE | 220 | 84 | 60 | 2.1 |
| | EE-ens | 360 | 132 | 85 | 3.0 |
| | DEEP | 290 | 108 | 70 | 2.5 |
| | **SNAP-UQ** | **182** | **70** | **52** | **1.7** |
| Small-MCU/CIFAR-10 | BASE | 180 | 92 | 260 | 9.5 |
| | EE-ens | OOM | — | — | — |
| | DEEP | OOM | — | — | — |
| | **SNAP-UQ** | **158** | **85** | **178** | **6.4** |

### Failure Detection

| Method | MNIST ID✓–ID× | SpeechCmd ID✓–ID× | CIFAR-10 ID✓–OOD | SpeechCmd ID✓–OOD |
|---|---|---|---|---|
| BASE | 0.75 | 0.90 | 0.90 | 0.88 |
| EE-ens | 0.85 | 0.90 | 0.90 | 0.90 |
| DEEP | 0.85 | 0.91 | 0.92 | 0.92 |
| **SNAP-UQ** | **0.90** | **0.94** | **0.92** | **0.94** |

### CID Stream Monitoring

| Method | MNIST-C AUPRC | Latency (frames) | SpeechCmd-C AUPRC | Latency (frames) |
|---|---|---|---|---|
| BASE | 0.54 | 42 | 0.52 | 67 |
| EE-ens | 0.63 | 31 | 0.59 | 55 |
| SNAP-UQ | **0.66** | **24** | **0.65** | **41** |

### Ablation Study

| Configuration | AUPRC (CIFAR-10-C) | Latency (ms) |
|---|---|---|
| P only, r=32 | 0.62 | 88 |
| M+P, r=64 | 0.70 | 83 |
| M+P, r=128 | 0.72 | 86 |
| M+P+early, r=64 | 0.71 | 90 |

### Key Findings

- **SNAP-UQ is the only viable UQ solution on Small-MCU**: Both EE-ens and DEEP run out of memory on CIFAR-10/Small-MCU, whereas SNAP-UQ deploys normally.
- **Depth-wise surprisal detects shifts earlier than softmax signals**: In stream monitoring experiments, SNAP-UQ's detection latency is approximately 25–30% shorter than that of EE-ens.
- **INT8 quantization incurs negligible degradation**: The INT8 head reduces AUPRC by only 0.01 compared to FP32, while reducing flash usage by 1.6–2.1×.
- **Two tap points are optimal**: The mid + penultimate combination consistently achieves the best accuracy–latency trade-off; adding an early tap degrades performance due to increased noise.

## Highlights & Insights

- The core innovation lies in using "deviation from inter-layer dynamics" as an uncertainty signal—unlike conventional softmax-, energy-, or Mahalanobis-based methods, SNAP-UQ captures conditional, depth-wise signals.
- The theoretical analysis is clean and elegant: Proposition 2.1 shows that the SNAP score is an affine transformation of the depth-wise negative log-likelihood; Proposition 2.2 establishes its equivalence to the Mahalanobis distance to the conditional mean; Proposition 2.3 proves invariance to batch normalization scale transformations.
- The entire design is highly engineering-oriented: int8 quantization, LUT-based exponentiation, CMSIS-NN compatibility, and no temporal buffering.
- Head-to-head comparisons with activation-shaping methods such as ASH and ReAct (Appendix O) show that SNAP-UQ consistently outperforms them on risk-at-coverage and AURC metrics.

## Limitations & Future Work

- Some firmware fuses or omits intermediate activations; exposing tap points may require runtime modifications.
- The diagonal covariance fails to capture full cross-channel structure, potentially under- or over-estimating surprisal under extreme distributional distortions.
- Performance is sensitive to the choice of tap layer positions and projector rank.
- The optional confidence blending and mapping still require a small labeled development set.
- Evaluation is limited to four benchmarks and two MCU tiers; broader modalities and tiny transformer architectures remain unexplored.

## Related Work & Insights

- MC Dropout (Gal & Ghahramani, 2016) and Deep Ensembles (Lakshminarayanan et al., 2017) are classical UQ methods but impose prohibitive resource costs.
- Energy score (Liu et al., 2020) and Mahalanobis detection (Lee et al., 2018) are single-pass methods but rely on unconditional statistics.
- QUTE (Ghanathe & Wilton, 2024) is a recent TinyML UQ approach but still depends on early-exit architectures.
- **Insight**: The paradigm of using a network's own depth-wise dynamics as an anomaly signal is generalizable to online monitoring of larger models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](soft_equivariance_regularization_for_invariant_self-supervised_learning.md)
- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](../../NeurIPS2025/self_supervised/uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [\[ICLR 2026\] Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models](regularized_latent_dynamics_prediction_is_a_strong_baseline_for_behavioral_found.md)
- [\[ICLR 2026\] Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning](why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic.md)
- [\[AAAI 2026\] Self-Supervised Inductive Logic Programming](../../AAAI2026/self_supervised/self-supervised_inductive_logic_programming.md)

</div>

<!-- RELATED:END -->
