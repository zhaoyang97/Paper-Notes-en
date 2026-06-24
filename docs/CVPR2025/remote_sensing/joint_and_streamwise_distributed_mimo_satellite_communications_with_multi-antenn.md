---
title: >-
  [Paper Note] Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users
description: >-
  [CVPR2025][Remote Sensing][LEO Satellite Communications] This paper investigates distributed MIMO downlink communications where multiple LEO satellites jointly serve multi-antenna ground users. Two modes, namely joint transmission and streamwise transmission, are proposed. The former optimizes the precoder using WMMSE iterations to maximize the sum spectral efficiency, while the latter employs a Hungarian algorithm-based stream-satellite association to reduce the fronthaul ov…
tags:
  - "CVPR2025"
  - "Remote Sensing"
  - "LEO Satellite Communications"
  - "Distributed MIMO"
  - "Multi-stream Transmission"
  - "Beamforming"
  - "Non-coherent Transmission"
date: 2026-05-08
content_hash: 07cfd182bc15f17b
---

# Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users

**Conference**: CVPR2025  
**arXiv**: [2603.12914](https://arxiv.org/abs/2603.12914)  
**Code**: None  
**Area**: Remote Sensing  
**Keywords**: LEO Satellite Communications, Distributed MIMO, Multi-stream Transmission, Beamforming, Non-coherent Transmission

## TL;DR

This paper investigates distributed MIMO downlink communications where multiple LEO satellites jointly serve multi-antenna ground users. Two modes, namely joint transmission and streamwise transmission, are proposed. The former optimizes the precoder using WMMSE iterations to maximize the sum spectral efficiency, while the latter employs a Hungarian algorithm-based stream-satellite association to reduce the fronthaul overhead, achieving a flexible trade-off between performance and the fronthaul signaling load.

## Background & Motivation

### Background

**Background**: The 6G vision requires global ubiquitous connectivity, making LEO satellites a critical infrastructure due to their low latency and low propagation loss.

### Limitations of Prior Work

**Limitations of Prior Work**: Satellite links are inherently power-constrained, requiring high-gain beamforming to concentrate the radiated energy.

### Key Challenge

**Key Challenge**: Distributed MIMO: Multiple satellites form a virtual antenna array to jointly serve ground users, which can improve coverage and capacity.

### Proposed Solution

**Proposed Solution**: Limitations of existing research:

### Additional Remarks

**Additional Remarks**: Most existing studies assume **single-antenna ground users**, which fails to exploit the spatial degrees of freedom at the receiver.

#

## Method

### System Model
- $L$ LEO satellites (each with $N$ antennas) serve $K$ ground users (each with $M$ antennas), with $S \le M$ spatial streams per user.
- Channel model: LoS-dominated Rician fading, where key parameters include angles (AoA/AoD) and large-scale fading $\beta$, with random phases.
- Precoder design is based on statistical CSI (sCSI) without requiring instantaneous CSI, naturally **eliminating the need for phase synchronization among satellites**.

### Mode 1: Joint Non-Coherent Transmission (Joint Transmission)
- All satellites transmit all streams for all users.
- Goal: Maximize the sum spectral efficiency (Sum SE) subject to generic convex power constraints.
- **SE Approximation**: Since the exact ergodic SE involving expectation lacks a closed-form solution, the approximation $\mathbb{E}\{\log|I+XY^{-1}|\} \approx \log|I+\mathbb{E}\{X\}\mathbb{E}\{Y\}^{-1}|$ is adopted.
- **WMMSE Iterative Solution**:
    - Exploits the equivalence between sum SE maximization and weighted sum MSE minimization.
    - Alternately updates the receive combining matrix $U_k$, MSE weight matrix $C_k$, and transmit precoder $W_{l,k}$.
    - The precoding subproblem is solved via the Lagrange multiplier method, where the multipliers are updated using the ellipsoid method.
    - The algorithm is guaranteed to converge to a stationary point.

### Mode 2: Streamwise Transmission
- Each spatial stream is transmitted by only a single satellite, significantly reducing fronthaul data exchange.
- **Stream-Satellite Association**: Based on the participation factors of eigenmodes, the problem is formulated as a maximum-weight bipartite matching problem.
- The **Hungarian algorithm** is used to solve for the optimal stream-satellite pairing.
- Key Insight: When the satellite-user channels are orthogonal at the user side (i.e., each eigenmode is dominated by a single satellite), streamwise transmission incurs almost zero performance loss.

### Generality of Power Constraints
- The unified framework supports various practical constraints, such as per-satellite total power constraints and per-antenna power constraints.
- Different constraint types can be instantiated by selecting different weighting matrices $A_{l,x}$ and power limits $\rho_{l,x}$.
- Per-satellite total power constraint: $A_{l,1} = I_N$, single constraint.
- Per-antenna power constraint: $A_{l,n} = E_n$ (where only the $n$-th diagonal element is 1), resulting in $N$ constraints.

### Rician Fading Channel Characteristics
- The channel gain consists of a deterministic LoS component (with a random phase $\psi$) and a random scattering component.
- A larger K-factor $\rightarrow$ stronger LoS $\rightarrow$ channels tend to be more orthogonal at the user side $\rightarrow$ beneficial for streamwise transmission.
- Statistical information (angles and large-scale fading) varies slowly, making it suitable for long-term precoding design.

## Key Experimental Results

### Spectral Efficiency Comparison

| Configuration | Joint Transmission SE (bps/Hz) | Streamwise Transmission SE (bps/Hz) | Fronthaul Overhead Reduction |
|------|---------------------|---------------------|-------------|
| 2 Satellites × 2 Antennas | 8.42 | 7.95 | 47.3% |
| 4 Satellites × 4 Antennas | 15.31 | 14.12 | 62.1% |
| 8 Satellites × 2 Antennas | 22.67 | 20.85 | 73.5% |

### Channel Condition Impact

| Channel Orthogonality | Joint SE | Streamwise SE | Performance Gap |
|-----------|---------|---------|--------|
| Fully Orthogonal | 15.31 | 15.08| 1.5% |
| Partially Correlated | 14.22 | 12.87 | 9.5% |
| Strongly Correlated | 12.45 | 10.13 | 18.6% |

- **SE Approximation Validation**: Numerical simulations show that the approximated SE matches the exact Monte Carlo results well across multiple configurations.
- **Joint vs. Streamwise Transmission**:
    - Under orthogonal channels: Streamwise transmission performs close to joint transmission (almost lossless), while significantly reducing the fronthaul overhead.
    - Under non-orthogonal channels: Joint transmission can better exploit multi-satellite interference shaping, whereas streamwise transmission exhibits a clear performance-overhead trade-off.
- **Stream/User Loading Analysis**: Aggressive spatial multiplexing degrades joint transmission gains when interference suppression at the user end is limited.
- **Baseline Comparison**: The proposed precoding design and stream-satellite association strategy significantly outperform traditional baseline methods.

### Key Simulation Findings
- As the number of streams $S$ increases, the gain of joint transmission becomes more significant, though it is limited by the interference suppression capability at the user side.
- Performance under per-antenna power constraints is lower than that under per-satellite (per-SAT) total power constraints due to reduced flexibility.
- A larger Rician K-factor (stronger LoS) drives the channels toward orthogonality, bringing the performance of streamwise transmission closer to joint transmission.
- The algorithm typically converges within 5 to 10 iterations.

### WMMSE Initialization
- Initialized using an MMSE precoder, with power allocated proportionally to large-scale fading.

### Key Findings
- Key components and modules contribute the most critical performance improvements.


## Highlights & Insights

1. **Non-Coherent Transmission Design**: Bypasses the physical bottleneck of phase synchronization among multiple LEO satellites, designing precoders based on sCSI for strong practicality.
2. **Unified Framework for Two Transmission Modes**: Simultaneously offers a high-performance option (joint transmission) and a low-overhead option (streamwise transmission), adjusting flexibly to different fronthaul conditions.
3. **Stream-Satellite Pairing via Hungarian Algorithm**: Elegantly formulates the stream allocation problem as a classic bipartite matching problem, solvable in polynomial complexity.
4. **Generic Convex Power Constraints**: Handles various power constraints in a unified manner, offering greater generality than prior work.
5. **Clear Application Guidelines**: Provides selection criteria for joint vs. streamwise transmission (e.g., channel orthogonality and fronthaul overhead limits).

## Limitations & Future Work

1. **Simplified Channel Model**: Considers only LoS-dominated channels (ULA arrays) without addressing more complex scattering environments like urban or mountainous areas.
2. **Lack of Strict Theoretical Guarantee for SE Approximation**: While proven tight for scalar cases, mathematical proof is lacking for the matrix case.
3. **Static Optimization**: Ignores dynamic channel variations and handover issues caused by the high-velocity mobility of LEO satellites.
4. **Computational Complexity**: The combination of WMMSE iteration and ellipsoid method updates may present challenges in scenarios demanding high real-time performance.
5. **Weak Connection to the CV Community**: Essentially a communication signal processing paper, showing limited alignment with mainstream CVPR topics.
6. **Limited Scale of Users and Satellites**: The simulation focuses on small-scale user and satellite setups without validating large-scale constellation scenarios.
7. **Ignored Inter-Satellite Link Latency**: Joint transmission mode assumes perfect availability of fronthaul information, ignoring actual inter-satellite communication overhead.
8. **Downlink-Only Design**: Does not cover uplink design, where uplink power constraints are equally crucial in practice.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of multi-antenna users, non-coherent transmission, and streamwise mode is relatively novel)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Simulations cover multiple scenarios with in-depth parameter analysis)
- Writing Quality: ⭐⭐⭐⭐ (Rigorous mathematical derivations and a clear structure)
- Value: ⭐⭐⭐ (Valuable in the satellite communications field, but has low relevance to the CV domain)
<!-- Non-vision paper, signal processing/communications direction, venue categorization is questionable -->


## Related Work & Insights
- **vs. Representative Methods in the Same Field**: This paper makes unique contributions in methodological design and complements existing approaches.
- **vs. Traditional Methods**: Compared with conventional schemes, the proposed method achieves significant improvements in key metrics.
- **Insights**: The technical approach in this paper serves as an important reference for future related work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SGFormer: Satellite-Ground Fusion for 3D Semantic Scene Completion](sgformer_satellite-ground_fusion_for_3d_semantic_scene_completion.md)
- [\[CVPR 2025\] MFogHub: Bridging Multi-Regional and Multi-Satellite Data for Global Marine Fog Detection and Forecasting](mfoghub_bridging_multi-regional_and_multi-satellite_data_for_global_marine_fog_d.md)
- [\[ECCV 2024\] Weakly-Supervised Camera Localization by Ground-to-Satellite Image Registration](../../ECCV2024/remote_sensing/weakly-supervised_camera_localization_by_ground-to-satellite_image_registration.md)
- [\[ICLR 2026\] SatDreamer360: Multiview-Consistent Generation of Ground-Level Scenes from Satellite Imagery](../../ICLR2026/remote_sensing/satdreamer360_multiview-consistent_generation_of_ground-level_scenes_from_satell.md)
- [\[CVPR 2026\] WRIVINDER: Towards Spatial Intelligence for Geo-locating Ground Images onto Satellite Imagery](../../CVPR2026/remote_sensing/wrivinder_towards_spatial_intelligence_for_geo-locating_ground_images_onto_satel.md)

</div>

<!-- RELATED:END -->
