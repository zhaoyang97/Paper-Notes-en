---
title: >-
  [Paper Note] Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users
description: >-
  [CVPR 2026][Remote Sensing][Distributed MIMO] This paper studies downlink transmission from multiple LEO satellites jointly serving multi-antenna ground users. Two non-coherent transmission modes are proposed—joint transmission and streamwise transmission—with precoders designed under the WMMSE framework and stream-to-satellite association solved via the Hungarian algorithm, achieving near-optimal spectral efficiency while substantially reducing fronthaul overhead.
tags:
  - CVPR 2026
  - Remote Sensing
  - Distributed MIMO
  - LEO satellite communications
  - multi-stream transmission
  - beamforming
  - fronthaul optimization
date: 2026-05-08
content_hash: 23d8666ada9a49b1
---

# Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users

**Conference**: CVPR 2026
**arXiv**: [2603.12914](https://arxiv.org/abs/2603.12914)
**Code**: None
**Area**: Remote Sensing / Satellite Communications
**Keywords**: Distributed MIMO, LEO satellite communications, multi-stream transmission, beamforming, fronthaul optimization

## TL;DR

This paper studies downlink transmission from multiple LEO satellites jointly serving multi-antenna ground users. Two non-coherent transmission modes are proposed—joint transmission and streamwise transmission—with precoders designed under the WMMSE framework and stream-to-satellite association solved via the Hungarian algorithm, achieving near-optimal spectral efficiency while substantially reducing fronthaul overhead.

## Background & Motivation

1. **Background**: LEO satellite communications have emerged as a key technology for 6G global coverage due to low latency and low propagation loss. Existing research has primarily focused on cooperative transmission for single-antenna ground users, while multi-satellite joint MIMO is becoming an active research direction.
2. **Limitations of Prior Work**: (a) Most existing multi-satellite cooperative transmission studies assume single-antenna users, which limits spatial multiplexing gains; (b) many methods rely on phase-synchronized coherent joint transmission across satellites, which is practically infeasible for LEO constellations due to large inter-satellite separations and significant propagation delay differences; (c) joint transmission requires distributing all data streams to all satellites, incurring prohibitively large fronthaul overhead.
3. **Key Challenge**: Joint transmission offers superior performance but demands high fronthaul capacity, whereas reducing fronthaul requirements may degrade performance—striking a balance between the two is the central challenge.
4. **Goal**: Two non-coherent transmission modes are proposed—joint transmission (all satellites transmit all streams) and streamwise transmission (each stream is transmitted by a single satellite)—with corresponding precoder optimization and stream-to-satellite assignment algorithms.
5. **Core Idea**: Statistical CSI (angular information and large-scale fading) is exploited to eliminate the need for inter-satellite phase synchronization, while eigenmode analysis enables efficient stream-to-satellite association.

## Method

### Overall Architecture

Consider $L$ LEO satellites each equipped with $N$ antennas, serving $K$ multi-antenna ground users (each with $M$ antennas) receiving $S$ spatially multiplexed streams per user. The system operates under a LoS-dominant Rician channel model, and precoders are designed using only statistical CSI (AoA, AoD, and large-scale fading coefficients $\beta_{l,k}$).

### Key Designs

1. **Non-Coherent Joint Transmission Precoder Design**:
   - **Function**: Maximize the sum spectral efficiency across all users.
   - **Mechanism**: Since the exact ergodic SE involves an expectation over random phases that admits no closed-form expression, the approximation $\mathbb{E}\{\log_2|I+XY^{-1}|\} \approx \log_2|I+\mathbb{E}\{X\}\mathbb{E}\{Y\}^{-1}|$ is adopted to render the problem tractable. Via WMMSE equivalence, sum SE maximization is reformulated as weighted sum MSE minimization, solved by block coordinate descent that alternately updates the receive combiner, MSE weight matrices, and transmit precoders. The precoder subproblem updates Lagrange multipliers via the ellipsoid method to handle general convex power constraints.
   - **Design Motivation**: The approximated SE depends only on the deterministic effective channel matrix $\tilde{H}_{l,k}=\sqrt{\beta_{l,k}} b_{l,k} a_{l,k}^T$, requiring no instantaneous phase information, which inherently enables non-coherent transmission.

2. **Streamwise Transmission and Stream-to-Satellite Association**:
   - **Function**: Each data stream is transmitted by a single satellite, significantly reducing fronthaul overhead.
   - **Mechanism**: For each user $k$, the aggregated channel $\tilde{H}_k = [\tilde{H}_{1,k}, ..., \tilde{H}_{L,k}]$ is formed and decomposed via SVD. A participation factor $\alpha_{l,k,m} = \|v_{k,m}^{(l)}\|^2$ is defined to quantify each satellite's contribution to each eigenmode. The problem is then formulated as a maximum-weight bipartite matching problem and solved optimally via the Hungarian algorithm.
   - **Design Motivation**: When the channel directions from different satellites are sufficiently orthogonal, each eigenmode is naturally dominated by a single satellite, so streamwise transmission incurs negligible performance loss.

3. **General Convex Power Constraint Framework**:
   - **Function**: Uniformly handle multiple types of power constraints (e.g., per-satellite total power, per-antenna power).
   - **Mechanism**: Constraints are parameterized via weight matrices $A_{l,x}$ as $\sum_k \text{Tr}(W_{l,k}^H A_{l,x} W_{l,k}) \leq \rho_{l,x}$, where different choices of $A_{l,x}$ correspond to different constraint types. Lagrange multiplier updates employ the ellipsoid method: a feasible upper bound is first located via geometric expansion, followed by iterative convergence through central cutting planes.
   - **Design Motivation**: Practical satellite systems face diverse power limitations; a unified framework avoids deriving separate solutions for each constraint type.

### Loss & Training

- **Objective**: $\max \sum_{k=1}^K \bar{R}_k$ (approximated sum spectral efficiency)
- **WMMSE Equivalent Objective**: $\min \sum_{k=1}^K \text{Tr}(C_k E_k) - \log_2|C_k|$
- **Initialization**: MMSE precoders are used as initialization, with power allocated proportionally to large-scale fading coefficients.
- **Convergence**: Each subproblem is solved optimally per iteration, ensuring monotonic decrease of the objective; boundedness guarantees convergence to a stationary point.

## Key Experimental Results

### Main Results

Performance is evaluated via numerical simulation, with spectral efficiency (SE, bit/s/Hz) as the primary metric.

| Scenario | Joint Transmission SE | Streamwise Transmission SE | Performance Ratio |
|---|---|---|---|
| Orthogonal channels | High | ≈ Joint transmission | ~100% |
| Non-orthogonal channels | High | Below joint | Noticeable gap |
| High user load | Degraded | More degraded | Gap widens |

### Ablation Study

| Configuration | Observation |
|---|---|
| Joint vs. streamwise transmission | Negligible gap under orthogonal channels; joint transmission clearly superior under non-orthogonal channels |
| Effect of stream count / number of users | Excessive spatial multiplexing streams degrade joint transmission gains when receiver interference suppression is limited |
| SE approximation accuracy | The approximated SE provides reasonable accuracy in most tested scenarios |

### Key Findings

- When satellite-to-user channels are sufficiently orthogonal at the receiver, streamwise transmission incurs minimal performance loss, as each channel eigenmode is naturally dominated by a single satellite.
- Under non-orthogonal channels, joint transmission better exploits multiple satellites for interference shaping, whereas streamwise transmission introduces a clear performance–overhead tradeoff.
- The choice of stream count and number of users requires careful consideration—aggressive spatial multiplexing can reduce gains when receiver interference suppression capability is limited.
- The proposed precoder design and stream-to-satellite association strategy yield substantial improvements over conventional baselines such as MF/ZF precoders.

## Highlights & Insights

- **Non-Coherent Transmission Design**: Statistical CSI is elegantly exploited to eliminate the requirement for multi-satellite phase synchronization, which is highly practical for LEO constellations with wide inter-satellite separations. This approach is transferable to other distributed non-coherent cooperative scenarios.
- **Eigenmode-Based Stream Assignment**: By analyzing each satellite's contribution to eigenmodes via SVD right singular vectors, the continuous optimization problem is discretized into a matching problem and solved efficiently with the classical Hungarian algorithm.
- **Unified Power Constraint Handling**: Multiple practical constraint types are subsumed under a single framework, avoiding the need for case-by-case derivations.

## Limitations & Future Work

- The channel model assumes LoS-dominant Rician fading; applicability to rich-scattering urban environments remains to be verified.
- Only the downlink is considered; joint reception design for the uplink is not addressed.
- Theoretical tightness guarantees for the SE approximation in the matrix case are absent; the authors acknowledge that approximation accuracy may vary across scenarios.
- The practical effects of inter-satellite link (ISL) latency and synchronization errors are not considered.

## Related Work & Insights

- **vs. Cell-Free Massive MIMO**: Terrestrial cell-free systems typically assume coherent joint transmission; this paper explicitly designs a non-coherent scheme that is more consistent with the practical constraints of satellite systems.
- **vs. Multi-Sensor Fusion (e.g., BEVFusion)**: Though from a different domain, the notion of "selective fusion to reduce overhead" shares conceptual similarities with streamwise transmission.
- **vs. Prior Multi-Satellite Works [9, 26, 31]**: This paper is the first to systematically address the scenario of multi-antenna users with multi-stream transmission.

## Rating

- **Novelty**: ⭐⭐⭐ The problem formulation (multi-antenna users with multi-stream non-coherent transmission) is novel, though the methodological framework (WMMSE) is well-established.
- **Experimental Thoroughness**: ⭐⭐⭐ Validation is purely simulation-based, with no real-world measurement data.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous and problem formulation is clearly presented.
- **Value**: ⭐⭐⭐ Engineering value for satellite communications, though connections to the vision/remote sensing community are limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration?](pretrained_image_matchers_for_sar_optical_satellite_registration.md)
- [\[CVPR 2026\] GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing](geommbench_and_geommagent_toward_expert_level_multimodal_intelligence_in_geoscience_and_remote_sensing.md)
- [\[CVPR 2026\] AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network](avion_aerial_visionlanguage_instruction_from_offli.md)
- [\[CVPR 2026\] GeoFlow: Real-Time Fine-Grained Cross-View Geolocalization via Iterative Flow Prediction](geoflow_real-time_fine-grained_cross-view_geolocalization.md)
- [\[CVPR 2026\] MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging](metaspectra_a_compact_broadband_metasurface_camera_for_snapshot_hyperspectral_im.md)

</div>

<!-- RELATED:END -->
