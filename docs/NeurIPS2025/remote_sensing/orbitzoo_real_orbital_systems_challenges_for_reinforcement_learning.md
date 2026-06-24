---
title: >-
  [Paper Note] OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning
description: >-
  [NeurIPS 2025][Remote Sensing][Multi-agent reinforcement learning] This paper presents OrbitZoo, a multi-agent RL environment built on the industrial-grade astrodynamics library Orekit. It integrates high-fidelity orbital dynamics (including atmospheric drag, solar radiation pressure, and third-body effects), a PettingZoo multi-agent interface, and real-time 3D visualization. Validation against real Starlink ephemerides yields a mean MAPE of only 0.16%.
tags:
  - "NeurIPS 2025"
  - "Remote Sensing"
  - "Multi-agent reinforcement learning"
  - "orbital dynamics"
  - "high-fidelity simulation"
  - "collision avoidance"
  - "PettingZoo"
date: 2026-05-08
content_hash: 06aae7b906d82331
---

# OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2504.04160](https://arxiv.org/abs/2504.04160)  
**Code**: Available (open-source)  
**Area**: Other
**Keywords**: Multi-agent reinforcement learning, orbital dynamics, high-fidelity simulation, collision avoidance, PettingZoo

## TL;DR

This paper presents OrbitZoo, a multi-agent RL environment built on the industrial-grade astrodynamics library Orekit. It integrates high-fidelity orbital dynamics (including atmospheric drag, solar radiation pressure, and third-body effects), a PettingZoo multi-agent interface, and real-time 3D visualization. Validation against real Starlink ephemerides yields a mean MAPE of only 0.16%.

## Background & Motivation

**Background**: The rapid growth in satellite and debris populations has made low Earth orbit (LEO) increasingly congested. Tasks such as collision avoidance, formation keeping, and orbital transfer demand autonomous decision-making systems. Reinforcement learning has shown promise in these domains by enabling adaptive policy learning in dynamic and uncertain environments.

**Limitations of Prior Work**: Most existing RL frameworks are built from scratch using simplified dynamical models (e.g., the circular restricted three-body problem, CR3BP), neglecting critical real-world perturbations such as atmospheric drag, solar radiation pressure (SRP), and multi-body gravitational effects. This results in large sim-to-real gaps. Furthermore, the majority of existing environments support only single-agent, fully observable, and impulsive-thrust settings, lacking standardization and reproducibility.

**Key Challenge**: There exists a fundamental tension between high-fidelity simulation and RL training efficiency. Accurate numerical integrators and full perturbation models incur high computational costs, yet policies trained on simplified models generalize poorly to real scenarios. At the same time, existing environments lack unified multi-agent support and validation standards.

**Goal**: (1) The absence of standardized multi-agent RL environments integrating industrial-grade dynamics libraries; (2) the difficulty of quantifying the sim-to-real gap; (3) insufficient scalability and reproducibility in existing environments.

**Key Insight**: The paper leverages the Python wrapper of Orekit—a mature Java astrodynamics library—as the dynamics engine, combined with the PettingZoo multi-agent RL framework, to construct a modular, high-fidelity, and open-source orbital RL platform.

**Core Idea**: Build a standardized MARL environment on top of an industrial-grade orbital simulation library, validate fidelity using real ephemeris data, and fill the gap in standardized platforms for orbital maneuvering RL research.

## Method

### Overall Architecture

OrbitZoo consists of three layers: (1) a low-level dynamics engine (Orekit) supporting full orbital perturbations and high-precision numerical integration; (2) an RL environment layer (PettingZoo interface) that models orbital maneuvering as a POMDP, with each satellite treated as an independent agent; and (3) an application layer providing customizable task scenarios (collision avoidance, orbital transfer, formation keeping, etc.) and real-time 3D visualization.

### Key Designs

1. **High-Fidelity Dynamics Integration**:

    - Function: Provides realistic orbital environment simulation.
    - Mechanism: Wraps Orekit's numerical propagator to support Holmes-Featherstone harmonic gravity, atmospheric drag (computed using historical space weather data), solar radiation pressure (accounting for lunar shadowing), and third-body effects (all solar system planets, the Sun, and the Moon). Supports Cartesian, Keplerian, and Equinoctial state representations, as well as variable-step high-accuracy integration methods such as Dormand-Prince.
    - Design Motivation: Using Orekit avoids the validation burden of a custom dynamics model, directly achieves industrial-grade precision, and minimizes the sim-to-real gap.

2. **PettingZoo Multi-Agent RL Interface**:

    - Function: Supports partially observable, decentralized multi-agent training.
    - Mechanism: Each satellite acts as an independent agent with its own observation space (orbital state plus neighbor information) and action space (polar-coordinate thrust parameters). The environment is modeled as a MA-POMDP supporting cooperative (formation keeping), competitive (pursuit-evasion), and mixed scenarios. Centralized training with decentralized execution (CTDE) and federated learning are both supported.
    - Design Motivation: Adopting the established PettingZoo framework lowers the development barrier, while Orekit's parallel propagation enables scalable computation over thousands of objects.

3. **Modular Reward and Visualization Framework**:

    - Function: Enables flexible task objective definition and agent behavior debugging.
    - Mechanism: The reward framework integrates inter-body quantities (relative distance, probability of collision PoC) and body-specific quantities (fuel consumption, mass change), supporting dense/sparse rewards and multi-objective optimization. A built-in Python real-time 3D visualization tool renders orbital trajectories and thrust actions directly during training and evaluation.
    - Design Motivation: Reward design for orbital tasks is non-trivial due to delayed effects and coupled dynamics; the modular design facilitates experimentation with different reward strategies, while visualization aids in understanding learned behaviors and diagnosing failure cases.

## Key Experimental Results

### Main Results

| Experiment | Algorithm | Key Results |
|------|------|---------|
| Hohmann Transfer (30 km) | PPO | Learns near-optimal orbital transfer; semi-major axis matches theoretical values |
| Collision Avoidance (CAM) | DQN, PPO | PPO performs better under full perturbations, effectively reducing PoC below $10^{-6}$ |
| GEO Formation (4 satellites) | PPO + GAE | Maintains equal angular spacing over 4 days; generalizes to unseen perturbations |
| Starlink Validation | — | 31 satellites, MAPE = 0.16%; RMSE as low as 24 m over 16.6 h propagation |

### Ablation Study (Validation Accuracy by Group)

| Satellite Group | Mean RMSE (m) | Note |
|---------|-------------|------|
| Low RMSE | 24.14 | Good match |
| Medium RMSE | 83.75 | Moderate deviation |
| High RMSE | 1924.90 | Insufficient physical parameter information |

### Key Findings

- Continuous action spaces (PPO) generalize better than discrete action spaces (DQN) under high-fidelity dynamics.
- In a sim-to-real setting where training uses simplified dynamics but evaluation employs full perturbations, PPO still effectively reduces collision probability.
- In formation keeping, learned policies generalize to third-body forces and SRP not seen during training, demonstrating that policies trained in high-fidelity environments are more robust.
- In the Starlink validation, most satellites exhibit very low RMSE; high-RMSE satellites are primarily attributable to insufficient physical parameter information (e.g., drag coefficients).

## Highlights & Insights

- **First standardized MARL orbital environment integrating an industrial-grade dynamics library**: The comparison table shows it is the only platform simultaneously satisfying all seven capability criteria (multi-agent, industrial simulator, high-fidelity dynamics, continuous control, realistic thrust modeling, interactive visualization, open-source). This provides a unified experimental infrastructure for space RL research.
- **Real-data validation methodology**: The approach of using Bayesian optimization to tune physical parameters (drag and reflectivity coefficients) for matching Starlink ephemerides offers a reusable framework for sim-to-real evaluation.
- **Uncertainty propagation modeling in collision avoidance**: The CAM task explicitly models state uncertainty and its temporal evolution, which is more operationally realistic than simply using Euclidean distance as a proxy.

## Limitations & Future Work

- The computational cost of high-fidelity propagation remains a bottleneck at the scale of thousands of objects; training efficiency for large-scale constellations (e.g., the full Starlink fleet of 4000+ satellites) has not been explicitly demonstrated.
- Quantification of the sim-to-real gap is limited to orbital prediction error; the transferability of trained policies to real spacecraft has not been validated.
- Reward function design still requires manual tuning; automated reward discovery mechanisms are absent.
- Formal safety guarantees (e.g., hard fuel constraints, exclusion zones) are not addressed.

## Related Work & Insights

- **vs. ColAvGym (Kazemi 2024)**: A single-agent CAM environment that also uses Orekit but does not support multi-agent interaction or interactive visualization; OrbitZoo offers substantially broader functionality.
- **vs. Dolan 2023 (GNN-based MARL)**: Uses simplified J2-only dynamics without integration of an industrial simulator; OrbitZoo achieves far superior dynamical fidelity.
- **vs. REDA (Holder 2024)**: Uses Poliastro for multi-agent task assignment without integrating real orbital data; OrbitZoo provides credibility guarantees through Starlink validation.

## Rating

- Novelty: ⭐⭐⭐ — Primarily an engineering integration contribution; methodological innovation is limited.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-task validation combined with real-data comparison.
- Writing Quality: ⭐⭐⭐⭐ — Comparison table is systematic and comprehensive; background is clearly presented.
- Value: ⭐⭐⭐⭐ — Fills the gap in standardized platforms for space RL; offers long-term value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](../../CVPR2025/remote_sensing/learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)
- [\[NeurIPS 2025\] Connecting the Dots: A Machine Learning Dataset for Ionospheric Prediction](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)
- [\[ICCV 2025\] CityNav: A Large-Scale Dataset for Real-World Aerial Navigation](../../ICCV2025/remote_sensing/citynav_a_large-scale_dataset_for_real-world_aerial_navigation.md)
- [\[NeurIPS 2025\] Mass Conservation on Rails – Rethinking Physics-Informed Learning of Ice Flow Vector Fields](mass_conservation_on_rails_--_rethinking_physics-informed_learning_of_ice_flow_v.md)
- [\[NeurIPS 2025\] ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning](chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)

</div>

<!-- RELATED:END -->
