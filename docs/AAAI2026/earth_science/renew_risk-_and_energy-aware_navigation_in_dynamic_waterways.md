---
title: >-
  [Paper Note] RENEW: Risk- and Energy-Aware Navigation in Dynamic Waterways
description: >-
  [AAAI 2026 (Oral)][Earth Science][ASV] This paper proposes RENEW, a global path planner for autonomous surface vessels (ASVs) operating in dynamic water current (ocean current) environments. It introduces a unified risk- and energy-aware strategy via adaptive no-go zone identification, best-effort contingency planning, and a hierarchical architecture based on Constrained Delaunay Triangulation (CDT), achieving zero collisions in emergency maneuver tests.
tags:
  - "AAAI 2026 (Oral)"
  - "Earth Science"
  - "ASV"
  - "path planning"
  - "adaptive safety constraints"
  - "homotopy classes"
  - "energy-awareness"
  - "No Go Zone"
date: 2026-05-08
content_hash: 3fb79c4e6aeb5951
---

# RENEW: Risk- and Energy-Aware Navigation in Dynamic Waterways

**Conference**: AAAI 2026 (Oral)  
**arXiv**: [2601.16424](https://arxiv.org/abs/2601.16424)  
**Code**: [https://github.com/dartmouthrobotics/RENEW.git](https://github.com/dartmouthrobotics/RENEW.git)  
**Area**: Robot Navigation / Path Planning / Autonomous Surface Vessel Navigation  
**Keywords**: ASV, path planning, adaptive safety constraints, homotopy classes, energy-awareness, No Go Zone

## TL;DR

This paper proposes RENEW, a global path planner for autonomous surface vessels (ASVs) operating in dynamic water current (ocean current) environments. It introduces a unified risk- and energy-aware strategy via adaptive no-go zone identification, best-effort contingency planning, and a hierarchical architecture based on Constrained Delaunay Triangulation (CDT), achieving zero collisions in emergency maneuver tests.

## Background & Motivation

ASVs navigating busy waterways such as the Strait of Malacca and the Singapore Strait face the core challenge that external disturbances such as ocean currents dynamically alter navigable regions and optimal paths. Existing planners exhibit the following limitations:

- **Sampling-based methods** (RRT/PRM/A*): minimize distance rather than energy consumption and ignore the effect of disturbances on turning behavior.
- **Fixed safety margin methods**: use fixed padding instead of dynamic adaptation, resulting in overly conservative or insufficient clearances.
- **Existing energy-aware methods**: do not simultaneously handle dynamic no-go zones and multi-topology path selection.
- **Trajectory optimization methods**: mostly confined to local planning, lacking global topological guarantees.

The core problem is: given a known 2D vector flow field (ocean current forecast), how can a globally safe (avoiding Inevitable Collision States, ICS) and energy-efficient path be found for a nonholonomic ASV, while maintaining topological diversity to support route replanning?

## Method

### Overall Architecture

A five-stage hierarchical architecture:
1. **Constrained Delaunay Triangulation (CDT)** → constructs a navigation mesh, separating Hard No Go Zones ($\mathcal{L}$) from navigable regions ($\mathcal{L}^c$).
2. **Homotopic channel search** → DFS over the CDT dual graph to identify $k$ topologically distinct channels.
3. **Adaptive padding** → sampling-based worst-case hard-over turn analysis to determine safety margins for each channel.
4. **High-level planner** → fuel efficiency evaluation to select the optimal homotopy class.
5. **Low-level planner** → optimizes the final path within the selected homotopic channel.

### Key Designs

1. **Adaptive No Go Zone**:

    - Without disturbances, ASV turning traces a regular circle ($r = v_\text{thrust}/\omega$); under ocean currents, the turning shape becomes irregular.
    - For each constrained edge, hard-over turns ($\tau \in \{-\omega_\text{max}, +\omega_\text{max}\}$) are sampled, considering heading $\phi$, lateral offset $\Delta\phi$, and mean disturbance.
    - Under disturbance noise, resampling is performed to compute the probability distribution of the minimum distance to the constrained edge.
    - A percentile $\sigma$ (e.g., 95%) defines the padding offset, guaranteeing collision avoidance with probability $\sigma$, with residual risk $1-\sigma$.
    - **Key insight**: the same constrained edge may have different padding values for different homotopic traversal directions, as the relationship between current, vessel, and obstacle changes with topology.

2. **Homotopic Channel**:

    - Exploits the dual structure of CDT: triangles → nodes, shared edges → connections.
    - DFS searches for topologically distinct channel sequences, with parameter $k \leq 2^n$ ($n$ = number of obstacles).
    - Concave obstacles are handled via the crossing-sequence method for homotopy class identification.
    - Retaining multiple homotopic options enables ASVs to switch or replan routes at any time, enhancing robustness.

3. **Fuel Cost Model**:

    - $F = \int \alpha \cdot v_\text{thrust}^k \cdot ds\,/\,(v_\text{thrust} + \vec{c} \cdot \hat{T}(s))$, with $k=2$ (quadratic drag), accounting for favorable/adverse current directions.
    - **Harmonic mean** is used to evaluate homotopy classes (penalizing high-fuel outliers): $h^* = \arg\min_h (\text{harmonic mean } F_{h,i})$.
    - An effective turning radius $r' = v'_\text{world}/\omega$ replaces the fixed $r$, ensuring kinematic feasibility.

### Loss & Training

No learning components are involved; RENEW is a purely sampling-based planner. Key parameters:
- ASV model: length 2 m, maximum linear velocity 1.0 m/s, maximum angular velocity 35°/s.
- Safety probability threshold $\sigma = 0.95$.
- $N = 500$ paths sampled per homotopy class.
- Computation time: in a 7-obstacle environment, homotopy class identification takes ~10 s; full path optimization takes ~300 s.

## Key Experimental Results

### Main Results: Method Comparison Across Environments

| Environment | Metric | RENEW | Grid A* | RRT | RRT-D | PRM | PRM-D |
|------|------|:-:|:-:|:-:|:-:|:-:|:-:|
| 4-Gyre ($k$=16) | Fuel↓ | **202.25** | 224.84 | 310.57 | 236.27 | 288.07 | 235.09 |
| 4-Gyre | Safety↑ | **16.40** | 4.20 | 1.77 | 1.18 | 1.77 | 1.05 |
| 4-Gyre | F/D↓ | **0.680** | 0.860 | 0.99 | 0.92 | 1.01 | 0.93 |
| Hansando (real currents) | Fuel↓ | **2639** | 2845 | 3567 | 3958 | 3565 | 3959 |

### Ablation Study

| Ablation | Fuel | Safety | Note |
|--------|:-:|:-:|------|
| $k$=1 (single homotopy) | 325 | 7.7 | Too few homotopy classes; trapped in suboptimal paths |
| $k$=16 (multi-homotopy) | **202** | **16.4** | More options reveal better paths |
| No adaptive padding | — | Collision | Unsafe |
| Fixed padding | Higher | Overconservative | Navigable space compressed |
| Emergency test: RENEW | — | **0 collisions** | Meets maritime safety standards |
| Emergency test: Grid A* | — | 62–128 collisions | Unsafe |
| Emergency test: RRT family | — | 29–213 collisions | Unsafe |
| Emergency test: PRM family | — | 45–185 collisions | Unsafe |

### Key Findings

- **Largest advantage under adverse currents (270°)**: RENEW generates zigzag paths to avoid opposing currents; though longer in distance, fuel consumption is lower.
- **Emergency maneuver test**: simulating a sudden obstacle every meter along the path, RENEW achieves zero collisions while all baselines fail.
- **Palawan Passage case study**: under different seasonal current patterns (summer vs. winter), RENEW automatically selects entirely different optimal routes for the same region.
- **Computational efficiency**: the main bottleneck lies in path sampling and optimization (~300 s / 500 samples), while homotopy search is fast (~10 s).

## Highlights & Insights

- **First global path planner to jointly address adaptive no-go zones and topological diversity**, filling a gap in maritime ASV navigation research.
- **Zero collisions in emergency maneuver tests** — directly satisfying the concept of abort position in maritime safety regulations.
- Core insight of adaptive padding: the required safety margin for the same obstacle boundary differs by traversal direction, as the effect of ocean currents on turning behavior depends on relative orientation.
- The best-effort contingency strategy, inspired by maritime emergency planning, elegantly resolves the safety guarantee problem.
- Selecting homotopy classes via harmonic mean is more robust than arithmetic mean — it penalizes paths with anomalously high energy consumption.

## Limitations & Future Work

- Assumes ASV thrust exceeds current speed ($\|V_\text{thrust}\| > \|V_c\|$); cannot handle extreme current scenarios.
- Assumes the flow field remains constant during planning (relies on short-term forecasts); long-range missions require online replanning.
- Path optimization is computationally expensive (~300 s), precluding real-time rapid replanning.
- Validated only in 2D planar environments; three-dimensional effects (underwater or surface undulation) are not considered.

## Related Work & Insights

- The approach is transferable to UAV path planning in wind fields, which shares the same structure of external disturbances and nonholonomic constraints.
- The adaptive padding concept can be applied to dynamic safety distance computation in urban autonomous driving.
- The homotopic channel framework provides a topology-level path allocation scheme for multi-robot coordinated planning.

## Rating

- Novelty: ⭐⭐⭐⭐ (novel combination of adaptive No Go Zones and homotopic channels)
- Technical Depth: ⭐⭐⭐⭐ (complete integration of probabilistic safety analysis, kinematic constraints, and energy optimization)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (real current data + ablations + emergency tests + multiple environments + multiple baselines)
- Practical Value: ⭐⭐⭐⭐ (directly targets maritime navigation; open-source code available)

## Limitations & Future Work
- Assumes fixed thrust velocity; variable-speed optimization is not considered.
- Relies on a priori known environment maps and flow field forecasts.
- Computation time is substantial (7-obstacle environment: ~10 s for homotopy search + ~300 s for path optimization).
- Online replanning in partially unknown environments is not addressed.

## Related Work & Insights
- vs. Grid A*: energy-aware vs. distance-only; homotopic diversity provides more route options.
- vs. RRT/PRM: safety guarantees significantly superior to sampling-based methods; fuel efficiency is also better.
- vs. Kularatne et al.: first to incorporate adaptive no-go zones and topological diversity.

## Insights & Connections
The concept of adaptive safety margins generalizes to dynamic safety distance computation in urban autonomous driving. The homotopy class search strategy for path diversity is a valuable reference for multi-path planning research.

## Rating ⭐⭐⭐⭐ (4/5)
An Oral paper with a clearly defined problem and rigorous methodology. The zero-collision result in emergency maneuver tests is impressive. However, computational cost and the fixed-thrust assumption limit practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MdaIF: Robust One-Stop Multi-Degradation-Aware Image Fusion with Language-Driven Semantics](mdaif_robust_one-stop_multi-degradation-aware_image_fusion_with_language-driven_.md)
- [\[ICLR 2026\] GeoFAR: Geography-Informed Frequency-Aware Super-Resolution for Climate Data](../../ICLR2026/earth_science/geofar_geography-informed_frequency-aware_super-resolution_for_climate_data.md)
- [\[ICLR 2026\] TianQuan-S2S: Constructing Subseasonal-to-Seasonal Global Weather Forecasting Models by Incorporating Climatology](../../ICLR2026/earth_science/tianquan-s2s_a_subseasonal-to-seasonal_global_weather_model_via_incorporate_clim.md)
- [\[ICML 2026\] Scaling Laws of Global Weather Models](../../ICML2026/earth_science/scaling_laws_of_global_weather_models.md)
- [\[ICLR 2026\] Uncovering the Mechanism of Continuous Representation Full Waveform Inversion: A Wave-based Neural Tangent Kernel Framework](../../ICLR2026/earth_science/unveiling_the_mechanism_of_continuous_representation_full-waveform_inversion_a_w.md)

</div>

<!-- RELATED:END -->
