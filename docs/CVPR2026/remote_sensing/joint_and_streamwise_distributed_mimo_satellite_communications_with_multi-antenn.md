---
title: >-
  [Paper Note] Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users
description: >-
  [CVPR 2026][Remote Sensing][Distributed MIMO] Two downlink transmission schemes (joint transmission & streamwise transmission) are proposed for distributed LEO satellite systems serving multi-antenna ground users. Throug…
tags:
  - "CVPR 2026"
  - "Remote Sensing"
  - "Distributed MIMO"
  - "LEO satellite communications"
  - "multi-antenna users"
  - "non-coherent joint transmission"
  - "fronthaul overhead optimization"
date: 2026-05-08
content_hash: 2d788b9352a79e41
---

# Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users

**Conference**: CVPR 2026
**arXiv**: [2603.12914](https://arxiv.org/abs/2603.12914)  
**Code**: None  
**Area**: Remote Sensing
**Keywords**: Distributed MIMO, LEO satellite communications, multi-antenna users, non-coherent joint transmission, fronthaul overhead optimization

## TL;DR

Two downlink transmission schemes (joint transmission & streamwise transmission) are proposed for distributed LEO satellite systems serving multi-antenna ground users. Through WMMSE precoding design based on statistical CSI and a stream-satellite association strategy based on the Hungarian algorithm, the proposed framework achieves a flexible trade-off between high spectral efficiency and low fronthaul overhead without requiring inter-satellite phase synchronization.

## Background & Motivation

### 1. State of the Field

One of the core objectives of 6G networks is to achieve seamless global connectivity. LEO satellite systems, with their advantages of low latency, low propagation loss, and low deployment cost, are becoming critical infrastructure for bridging the digital divide in remote areas. Introducing MIMO technology into satellite communications has attracted widespread attention—multi-antenna beamforming can focus energy, suppress interference, and improve coverage quality. Furthermore, the cell-free massive MIMO concept has been extended from terrestrial networks to satellite scenarios, where multiple satellites form a virtual antenna array to jointly serve ground users.

### 2. Limitations of Prior Work

- **Single-antenna user assumption**: Most existing multi-satellite cooperative works (e.g., [9][23][31]) assume that ground users have only a single antenna, limiting the degrees of freedom for spatial multiplexing and stream-level processing.
- **Difficulty of coherent joint transmission**: Most prior schemes rely on coherent joint transmission across multiple satellites; however, the large inter-satellite distances, propagation delay differences, and severe phase jitter in LEO systems make tight inter-satellite synchronization extremely challenging.
- **Excessive fronthaul overhead**: Joint transmission requires distributing all data streams of each user to all satellites, imposing heavy burdens on inter-satellite links, feeder links, and onboard processing in terms of bandwidth, latency, and complexity.

### 3. Root Cause

In distributed satellite MIMO, fully exploiting the spatial resources of multiple satellites and multiple antennas to maximize spectral efficiency requires jointly transmitting all streams—yet this fundamentally conflicts with limited fronthaul capacity and the inability to achieve inter-satellite phase synchronization.

### 4. Paper Goals

To design distributed transmission schemes for multi-antenna ground users that do not rely on inter-satellite phase synchronization, while providing flexible design choices between spectral efficiency and fronthaul overhead.

### 5. Starting Point

By leveraging the property that LoS-dominant channel models require only statistical CSI (angle information and large-scale fading) rather than instantaneous CSI, the exact ergodic SE is approximated as a tractable deterministic expression, making precoding design naturally independent of instantaneous phase—thereby eliminating the need for inter-satellite synchronization.

### 6. Core Idea

A two-tier scheme is proposed: (1) **Joint non-coherent transmission**—all satellites transmit all streams, with precoding solved under general convex power constraints via the WMMSE framework combined with the ellipsoid method; (2) **Streamwise transmission**—each stream is transmitted by only one satellite, where the eigenmodes of the aggregated channel are analyzed via SVD, and participation factors combined with the Hungarian algorithm perform one-to-one stream-satellite assignment, substantially reducing fronthaul requirements.

## Method

### Overall Architecture

The system consists of $L$ LEO satellites (each with $N$ antennas) serving $K$ ground users (each with $M$ antennas), where each user receives $S \leq M$ spatial data streams. The channel is modeled as LoS-dominant Rician fading, depending only on angle information and large-scale fading variance $\beta_{l,k}$ (statistical CSI). Building on this model, the paper proposes two transmission modes and designs complete transceiver optimization algorithms for each.

### Key Designs

#### Design 1: Problem Tractabilization via SE Approximation

- **Function**: Approximates the exact ergodic spectral efficiency (involving the expectation of matrix logarithmic determinants) as a deterministic SE expression.
- **Mechanism**: Employs the approximation $\mathbb{E}\{\log_2|\mathbf{I}+\mathbf{X}\mathbf{Y}^{-1}|\} \approx \log_2|\mathbf{I}+\mathbb{E}\{\mathbf{X}\}\mathbb{E}\{\mathbf{Y}\}^{-1}|$, computing the expectations of signal and interference terms separately to eliminate the effect of the random phase $\psi_{l,k}$.
- **Design Motivation**: The exact SE has no closed form and is difficult to optimize. After approximation, the SE depends only on deterministic quantities such as $\beta_{l,k}$, angles of arrival/departure, etc., making precoding design feasible and naturally free from inter-satellite phase synchronization.

#### Design 2: WMMSE Joint Precoding (Joint Transmission Mode)

- **Function**: Maximizes the sum spectral efficiency of all users under general convex power constraints.
- **Mechanism**: Exploiting the classical equivalence between sum-SE maximization and weighted sum-MSE minimization, the non-convex problem is transformed into a block coordinate descent iteration over the receive combining matrix $\mathbf{U}_k$, the MSE weight matrix $\mathbf{C}_k$, and the transmit precoding matrix $\mathbf{W}_{l,k}$:
    - Fix precoding, solve for the MMSE-optimal receiver $\mathbf{U}_k^\star$;
    - Fix receiver and precoding, solve for the optimal MSE weight $\mathbf{C}_k^\star = \frac{1}{\ln 2}\mathbf{E}_k^{-1}$;
    - Fix receiver and weights, solve for the precoding in closed form with Lagrangian multipliers updated via the **ellipsoid method**.
- **Design Motivation**: The general convex power constraint framework (encompassing per-satellite total power, per-antenna power, etc.) makes the scheme adaptable to various practical scenarios; the ellipsoid method efficiently handles multi-dimensional multiplier updates.

#### Design 3: Eigenmode-Based Stream-Satellite Association (Streamwise Transmission Mode)

- **Function**: Assigns each data stream to a single satellite, reducing fronthaul data exchange.
- **Mechanism**:
  1. Construct the aggregated channel $\tilde{\mathbf{H}}_k = [\tilde{\mathbf{H}}_{1,k}, \ldots, \tilde{\mathbf{H}}_{L,k}]$ and perform SVD;
  2. Compute the **participation factor** $\eta_{l,k,m} = \|\mathbf{v}_{k,m}^{(l)}\|^2$ for each satellite and each eigenmode (the energy proportion of the right singular vector corresponding to the satellite);
  3. Using participation factors as weights, formulate a maximum-weight bipartite matching problem and solve the one-to-one stream-satellite assignment via the **Hungarian algorithm**.
- **Design Motivation**: Participation factors directly characterize each satellite's contribution to the user's spatial eigenmodes; the one-to-one matching ensures that each stream is transmitted by the most suitable satellite while each satellite handles at most one stream, avoiding resource waste.

### Loss & Training

This work is an optimization-theoretic method with no neural network training. The core optimization objective is:

$$\max_{\{\mathbf{W}_{l,k}\}} \sum_{k=1}^{K} \bar{R}_k, \quad \text{s.t. } \sum_{k=1}^{K} \text{Tr}(\mathbf{W}_{l,k}^H \mathbf{A}_{l,x} \mathbf{W}_{l,k}) \leq \rho_{l,x}$$

- Joint transmission: Algorithm 1 (WMMSE iteration + Algorithm 2 ellipsoid method), converging to a stationary point.
- Streamwise transmission: Algorithm 3 (Hungarian matching first, then WMMSE iteration + bisection method for multiplier updates).
- Initialization: MMSE precoder with power allocated according to large-scale fading.
- Convergence criteria: $I_{\max}=40$, $\epsilon=10^{-4}$.

## Key Experimental Results

### Main Results

Simulation parameters: orbital altitude 560 km, carrier frequency 20 GHz, bandwidth 400 MHz, default $L=4$ satellites, $N=64$ antennas/satellite, $K=2$ users, $M=4$ antennas/user, $S=2$ streams/user, Rician factor $\kappa=12$ dB.

**Table 1: Joint vs. Streamwise Transmission — SE Comparison under Orthogonal and Non-Orthogonal Channels**

| Channel Condition | Transmission Mode | Low-Power SE (bps/Hz) | High-Power SE (bps/Hz) | SE Gap |
|---|---|---|---|---|
| UE-side orthogonal | Joint transmission | ~6 | ~22 | Baseline |
| UE-side orthogonal | Streamwise transmission | ~6 | ~21.5 | Negligible (<3%) |
| UE-side non-orthogonal | Joint transmission | ~5 | ~20 | Baseline |
| UE-side non-orthogonal | Streamwise transmission | ~5 | ~15 | Significant (~25%) |

**Table 2: Joint Transmission vs. Baseline Methods — SE Comparison (Non-Orthogonal Channel, L=4/8, High-Power Regime)**

| Method | L=4 SE (bps/Hz) | L=8 SE (bps/Hz) | Relative Gain |
|---|---|---|---|
| Proposed joint transmission | ~20 | ~28 | Baseline |
| MMSE baseline | ~16 | ~22 | −20%~−21% |
| ZF baseline | ~15 | ~20 | −25%~−29% |
| Orthogonal MRT (K=4) | — | ~8 | −71% |

### Ablation Study

- **Approximation accuracy validation (Fig. 4)**: The SE approximation almost coincides with exact Monte Carlo results in the low-power regime; deviations increase at high power and large $L$, but trends remain consistent, validating its use as a proxy objective for precoding design.
- **Effect of stream count (Fig. 7)**: Joint transmission gains significantly from $S=1 \to S=2$, while $S=3$ leads to performance degradation due to interference dominance; streamwise transmission improves monotonically from $S=1 \to 2 \to 3$; both modes converge in performance at $S=3$.
- **Stream-satellite association strategy (Fig. 9)**: The proposed Hungarian matching versus random assignment shows increasing gains with higher power and more antennas, validating the effectiveness of eigenmode-based matching.

### Key Findings

1. When UE-side channels are orthogonal, streamwise transmission incurs negligible loss (<3% SE degradation) while significantly reducing fronthaul overhead.
2. Joint transmission holds a clear advantage (~25% gain) under non-orthogonal channels, as it can shape interference across satellites.
3. Stream count selection requires care: too many streams under joint transmission can backfire due to interference degradation.
4. The number of simultaneously served users also involves a trade-off: $K=4$ outperforms both $K=2$ (insufficient spatial multiplexing gain) and $K=6$ (excessive interference).

## Highlights & Insights

- The **non-coherent design requiring no inter-satellite phase synchronization** is the central highlight—the deterministic approximation based on statistical CSI naturally circumvents the synchronization challenge.
- The concept of **participation factors** is intuitive and elegant: the block-wise energy proportion of SVD right singular vectors clearly characterizes "which satellite matters most for which spatial eigenmode."
- The **general convex power constraint framework** unifies various power limitations including per-satellite and per-antenna constraints, offering strong engineering adaptability.
- The **orthogonality condition analysis** for joint versus streamwise transmission provides clear guidelines for mode selection.

## Limitations & Future Work

- The SE approximation degrades in accuracy at high power or with many satellites; tighter approximations (e.g., Jensen gap compensation) could be explored.
- The channel model only considers LoS and Rician fading, without addressing blockage or rich multipath scenarios.
- The stream-satellite association is a static, one-shot decision that does not account for dynamic reassignment under time-varying channels.
- Only the downlink is studied; distributed reception design for the uplink warrants further investigation.
- User scheduling (selecting which users to serve simultaneously) is not incorporated into the joint optimization.

## Related Work & Insights

- **Cell-free massive MIMO** (Ngo et al., Demir et al.): The key distinction when extending from terrestrial to satellite scenarios is the infeasibility of coherent joint transmission.
- **WMMSE framework** (Shi et al.): Cleverly adapted to the satellite setting with general convex constraints; using the ellipsoid method for multi-dimensional multiplier updates represents a technical novelty.
- **Hungarian algorithm** applied to stream-satellite matching inspires the integration of combinatorial optimization into satellite resource allocation.
- The participation factor concept is generalizable to resource allocation decisions in other distributed systems (e.g., RIS-assisted communications, UAV formations).

## Rating

⭐⭐⭐⭐ A theoretically rigorous and structurally complete work on distributed satellite MIMO. The complementary design of two transmission modes is elegant, and the participation factor concept is intuitive and generalizable. However, experiments are purely simulation-based without real-world measurement validation, and the channel model is relatively simplified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration?](pretrained_image_matchers_for_sar_optical_satellite_registration.md)
- [\[AAAI 2026\] M3SR: Multi-Scale Multi-Perceptual Mamba for Efficient Spectral Reconstruction](../../AAAI2026/remote_sensing/m3sr_multi-scale_multi-perceptual_mamba_for_efficient_spectral_reconstruction.md)
- [\[ICCV 2025\] AstroLoc: Robust Space to Ground Image Localizer](../../ICCV2025/remote_sensing/astroloc_robust_space_to_ground_image_localizer.md)
- [\[AAAI 2026\] Debiasing Machine Learning Predictions for Causal Inference Without Additional Ground Truth Data](../../AAAI2026/remote_sensing/debiasing_machine_learning_predictions_for_causal_inference_without_additional_g.md)
- [\[ICCV 2025\] WildSAT: Learning Satellite Image Representations from Wildlife Observations](../../ICCV2025/remote_sensing/wildsat_learning_satellite_image_representations_from_wildlife_observations.md)

</div>

<!-- RELATED:END -->
