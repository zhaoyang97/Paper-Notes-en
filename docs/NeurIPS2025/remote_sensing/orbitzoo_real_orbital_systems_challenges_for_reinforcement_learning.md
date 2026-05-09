---
title: >-
  [Paper Note] OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning
description: >-
  [NeurIPS 2025][Remote Sensing][Orbital dynamics] This paper presents OrbitZoo, a multi-agent RL environment built on the industrial-grade Orekit orbital mechanics library, supporting realistic orbital tasks such as collision avoidance, Hohmann transfers, and constellation coordination. It provides standardized MARL training through the PettingZoo interface, and achieves 24-meter RMSE (over a 16.6-hour propagation) for the low-error group in validation against real Starlink ephemeris data.
tags:
  - NeurIPS 2025
  - Remote Sensing
  - Orbital dynamics
  - multi-agent RL
  - satellite maneuvering
  - collision avoidance
  - simulation environment
date: 2026-05-08
content_hash: 1647eadb82da79a2
---

# OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2504.04160](https://arxiv.org/abs/2504.04160)
**Code**: Open source (specific link not provided)
**Area**: Remote Sensing / Reinforcement Learning
**Keywords**: Orbital dynamics, multi-agent RL, satellite maneuvering, collision avoidance, simulation environment

## TL;DR

This paper presents OrbitZoo, a multi-agent RL environment built on the industrial-grade Orekit orbital mechanics library, supporting realistic orbital tasks such as collision avoidance, Hohmann transfers, and constellation coordination. It provides standardized MARL training through the PettingZoo interface, and achieves 24-meter RMSE (over a 16.6-hour propagation) for the low-error group in validation against real Starlink ephemeris data.

## Background & Motivation

**Background**: Approximately 20,000 satellites (roughly 50% operational) and around 140 million debris objects currently occupy Earth orbit. The deployment of large-scale constellations such as Starlink has intensified congestion in low Earth orbit (LEO). Traditional satellite operations rely heavily on human decision-making, a paradigm that is becoming unsustainable given growing complexity. RL has shown promise for autonomous satellite maneuvering.

**Limitations of Prior Work**: Existing RL orbital environments suffer from three major shortcomings: (1) most employ simplified dynamics models (e.g., two/three-body problems, Newtonian gravity only), which fail to capture realistic orbital perturbations; (2) most are custom-built from scratch, lacking standardization and reproducibility—dynamics validation is labor-intensive and error-prone; (3) no existing environment simultaneously supports multi-agent scenarios, continuous control, realistic thrust modeling, and visualization (Table 1 surveys 14 existing environments, none of which satisfies all criteria).

**Key Challenge**: RL requires massive simulation interaction to train policies, demanding environments that are both high-fidelity (accurate dynamics) and computationally efficient (fast propagation). Simultaneously, the sim-to-real gap is the central obstacle to deploying RL policies on real satellites—simulations must be validated against real data.

**Goal**: (1) Provide a unified environment combining high-fidelity dynamics with a standard RL interface; (2) support cooperative, competitive, and mixed multi-agent scenarios; (3) validate simulation accuracy against real ephemeris data.

**Key Insight**: The paper leverages Orekit—an industrial-grade open-source orbital mechanics library—for high-fidelity dynamics, while wrapping it in a PettingZoo-compliant interface so that RL researchers need not master orbital mechanics.

**Core Idea**: Combine an industrial-grade orbit propagator with a standard MARL framework to construct a satellite RL benchmark environment that is simultaneously realistic and accessible.

## Method

### Overall Architecture

OrbitZoo adopts a three-layer architecture. The bottom layer is a high-fidelity orbit propagation engine based on Orekit, handling gravitational fields, atmospheric drag, solar radiation pressure (SRP), and third-body effects. The middle layer is a task definition layer specifying observation spaces, action spaces, reward functions, and termination conditions. The top layer is a PettingZoo-compliant MARL interface in which each satellite acts as an independent agent, supporting partially observable POMDP modeling.

### Key Designs

1. **High-Fidelity Data Generation Engine**

   - **Function**: Provides a simulation environment with high consistency to real orbital behavior.
   - **Mechanism**: Uses Orekit's numerical propagator, supporting Holmes-Featherstone harmonic gravitational fields, atmospheric drag based on historical space weather data, solar radiation pressure (SRP) accounting for solar/lunar occultation, and third-body gravitational perturbations from all solar system planets as well as the Sun, Moon, and Earth-Moon barycenter. Supports three state representations: Cartesian coordinates, Keplerian orbital elements, and equinoctial orbital elements. Employs a Dormand-Prince variable-step integrator for high-accuracy propagation, with parallel computation to accelerate multi-body propagation.
   - **Design Motivation**: Most existing RL environments use simplified models (e.g., J2-only or Newtonian gravity only), which preclude training transferable policies for real environments.

2. **Standardized MARL Interface**

   - **Function**: Enables RL researchers to focus on algorithm design rather than environment implementation.
   - **Mechanism**: Implements a POMDP structure via the PettingZoo framework, with each satellite agent possessing independent observation and action spaces. Thrust actions are parameterized in polar coordinates $(T, \theta, \phi)$—thrust magnitude and direction angles in the RSW frame. Supports cooperative, competitive, and mixed scenario configurations, and integrates directly with mainstream MARL libraries such as MARLlib and EPyMARL.
   - **Design Motivation**: Table 1 shows that most existing environments do not simultaneously support multi-agent interaction and industrial-grade simulators.

3. **Modular Reward Framework**

   - **Function**: Supports flexible, physically constrained reward design.
   - **Mechanism**: Provides two categories of signals—inter-body metrics (relative distance, collision probability PoC, line-of-sight conditions) and individual metrics (fuel consumption, mass change). Supports reward modes ranging from dense to sparse, as well as multi-objective trade-offs (performance vs. safety vs. efficiency). Collision probability is computed using the Akella method; $\text{PoC} > 10^{-6}$ is treated as high risk.
   - **Design Motivation**: Reward design in orbital control inherently faces delayed feedback and coupled dynamics; a modular framework facilitates experimentation with diverse reward strategies.

### Visualization Component

OrbitZoo provides a Python-based real-time 3D visualization tool, which the authors claim is the first Python implementation offering real-time orbital visualization natively integrated within an RL framework. It supports policy inspection, fault diagnosis, and behavior interpretation.

## Key Experimental Results

### Main Results 1: Single-Agent Hohmann Transfer

| Metric | Result |
|--------|--------|
| Task | 30 km altitude raise |
| Algorithm | PPO (continuous actions) |
| Result | Near-optimal orbital transfer; semi-major axis matches theoretical value |
| Finding | Agent adapts to perturbation forces not accounted for in the theoretical solution |

### Main Results 2: Single-Agent Collision Avoidance Maneuver (CAM)

| Algorithm | Training Dynamics | Evaluation Dynamics | PoC Reduction |
|-----------|------------------|--------------------|--------------------|
| DQN (discrete) | Newtonian + drag | Full perturbations | Effective but weaker generalization |
| PPO (continuous) | Newtonian + drag | Full perturbations | Superior, stronger generalization |

### Main Results 3: Validation Against Real Starlink Data

| Group | Satellites | Mean RMSE (m) | Propagation Duration |
|-------|-----------|--------------|----------------------|
| Low-RMSE group | — | 24.14 | 16.6 hours |
| Mid-RMSE group | — | 83.75 | 16.6 hours |
| High-RMSE group | — | 1924.90 | 16.6 hours |
| Total | 31 | — | — |

### Key Findings

- Continuous action spaces (PPO) generalize significantly better than discrete actions (DQN) under realistic perturbations, as orbital maneuvering is inherently continuous in nature.
- Policies trained under simplified dynamics can be evaluated in fully perturbed environments, but a generalization gap remains—precisely what OrbitZoo aims to help researchers close.
- A cooperative PPO agent controlling 4 GEO constellation satellites learns to maintain equal angular spacing while minimizing fuel consumption; the learned policy also generalizes to perturbations unseen during training (third-body forces, SRP, drag).
- The high-RMSE group (1924.90 m) exhibits larger deviations, likely due to missing precise physical parameters (drag coefficients, reflectivity coefficients, etc.); Bayesian optimization-based parameter tuning can mitigate this.

## Highlights & Insights

- Comprehensiveness is OrbitZoo's greatest strength: Table 1 compares 14 existing environments, and only OrbitZoo satisfies all seven capability criteria simultaneously (multi-agent support, industrial simulator, high-fidelity dynamics, continuous control, realistic thrust modeling, visualization, and open-source code). This "all-rounder" positioning fills a clear gap in the field.
- The engineering contribution of bridging Orekit (Java ecosystem) with PettingZoo (Python ecosystem) should not be understated; it makes industrial-grade orbital simulation accessible to the RL community.
- Validation against real Starlink data provides credible evidence of simulation fidelity—the low-RMSE group achieves only 24-meter error after 16.6 hours of propagation, which is sufficiently accurate for operational decision windows of under 2 hours.

## Limitations & Future Work

- The high-RMSE group deviation (1924 m) indicates that simulation accuracy for satellites with unknown physical properties remains improvable, requiring better parameter estimation methods.
- The RL experiments primarily demonstrate environment usability but do not include rigorous comparative experiments against other environments on identical tasks.
- Scalability to large constellations (thousands of satellites) is mentioned in terms of parallelization support, but no concrete computational performance data are provided.
- Validation is limited to LEO and GEO scenarios; more complex regimes such as deep space or lunar orbits are not tested.
- No baseline comparisons against traditional control methods such as Model Predictive Control are provided.

## Related Work & Insights

- **vs. ColAvGym (Kazemi 2024)**: Uses real conjunction data messages (CDMs) but is limited to single-agent CAM scenarios; OrbitZoo supports multi-agent settings and a broader range of tasks.
- **vs. Poliastro**: Python-native but lower fidelity than Orekit, lacking high-order gravitational fields and multi-perturbation support.
- **vs. STK**: High commercial fidelity but costly, closed-source, and not directly integrable into RL training loops.
- OrbitZoo's design philosophy—standardized environment + high-fidelity simulation + real data validation—offers a useful reference for constructing RL environments in other safety-critical domains.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Systematically integrates high-fidelity simulation with a MARL framework, filling the gap left by the absence of a comprehensive orbital RL environment.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers diverse task scenarios and real-data validation; RL experiment depth could be further strengthened.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with thorough background exposition, accessible to readers from both the RL and aerospace communities.
- **Value**: ⭐⭐⭐⭐ As open-source infrastructure, this environment holds significant value for advancing RL research on autonomous space operations.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] CityNav: A Large-Scale Dataset for Real-World Aerial Navigation](../../ICCV2025/remote_sensing/citynav_a_large-scale_dataset_for_real-world_aerial_navigation.md)
- [\[ACL 2026\] MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems](../../ACL2026/remote_sensing/moneta_multimodal_industry_classification_through_geographic_information_with_mu.md)
- [\[NeurIPS 2025\] Mass Conservation on Rails – Rethinking Physics-Informed Learning of Ice Flow Vector Fields](mass_conservation_on_rails_--_rethinking_physics-informed_learning_of_ice_flow_v.md)
- [\[NeurIPS 2025\] ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning](chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)
- [\[NeurIPS 2025\] Connecting the Dots: A Machine Learning Ready Dataset for Ionospheric Forecasting Models](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)

<!-- RELATED:END -->
