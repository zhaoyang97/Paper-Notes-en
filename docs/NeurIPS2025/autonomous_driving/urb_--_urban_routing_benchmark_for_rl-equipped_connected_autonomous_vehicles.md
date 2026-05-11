---
title: >-
  [Paper Note] URB -- Urban Routing Benchmark for RL-Equipped Connected Autonomous Vehicles
description: >-
  [NeurIPS 2025][Autonomous Driving][Urban routing benchmark] This paper presents URB — the first large-scale MARL benchmark environment for urban mixed-traffic (human + CAV) routing…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "Urban routing benchmark"
  - "multi-agent reinforcement learning"
  - "traffic simulation"
  - "game theory"
date: 2026-05-08
content_hash: 1724ba5a804036f1
---

# URB -- Urban Routing Benchmark for RL-Equipped Connected Autonomous Vehicles

**Conference**: NeurIPS 2025
**arXiv**: [2505.17734](https://arxiv.org/abs/2505.17734)
**Code**: [GitHub](https://github.com/COeXISTENCE-PROJECT/URB)
**Area**: Autonomous Driving / Reinforcement Learning for Route Planning
**Keywords**: Urban routing benchmark, multi-agent reinforcement learning, autonomous driving, traffic simulation, game theory

## TL;DR

This paper presents URB — the first large-scale MARL benchmark environment for urban mixed-traffic (human + CAV) routing, integrating 29 real-world traffic networks, the microscopic traffic simulator SUMO, and empirical travel demand patterns. Experiments reveal that current state-of-the-art MARL algorithms rarely outperform human drivers, highlighting the urgent need for algorithmic advances in this domain.

## Background & Motivation

Connected autonomous vehicles (CAVs) hold promise for alleviating urban congestion through optimized routing decisions, yet the central question remains: which algorithms should govern collective routing? Reinforcement learning (RL) naturally frames routing as a sequential decision-making problem, but a standardized, realistic benchmark for evaluating and comparing algorithms is lacking.

Key limitations of prior work include: (1) the enormous action space of urban routing (the number of paths grows exponentially with network size), non-stationarity arising from agents competing for shared resources, and high simulation costs; (2) prior studies are confined to simplified scenarios without unified evaluation protocols; and (3) the impact of CAV deployment on human drivers is largely neglected.

The paper's starting point is to construct a comprehensive benchmark framework that enables RL researchers to test algorithms on realistic traffic scenarios while allowing transportation researchers to assess the societal implications of CAV deployment using state-of-the-art RL methods. The central finding is that current MARL algorithms are far from mature for large-scale urban routing and often fail to surpass even random policies.

## Method

### Overall Architecture

URB formalizes urban routing as an Agent-Environment Cycle (AEC) game: human drivers first learn and stabilize their routing strategies → a fraction of vehicles are converted to CAVs → CAVs are trained using MARL algorithms → strategies are evaluated. The environment is simulated using SUMO, human behavior is modeled via a Generic Learning Model (HLM), and CAVs select routes based on local observations.

### Key Designs

1. **Real-World Traffic Networks and Demand**:

    - 29 real-world traffic networks: 28 sub-regions of Île-de-France and 1 Ingolstadt network
    - Each network is paired with synthetic travel demand patterns derived from empirical data (AM peak period)
    - Network scales range from small (St. Arnoult) to large (Ingolstadt), covering varying levels of complexity

2. **Flexible Parameterization**:

    - CAV market penetration rate (0–100%) is configurable
    - CAV behavioral profiles: selfish (minimize own travel time), cooperative (minimize system-wide travel time), altruistic, or even adversarial
    - Observation space: CAVs observe the routing choices of agents that departed earlier
    - Action space: selection from $K_i$ pre-computed candidate paths

3. **Human Behavior Modeling**:

    - Based on the classical day-to-day routing learning model (Gawron 1998)
    - Human drivers update their expected travel times based on recent experience and select the subjectively optimal path each day
    - After sufficient learning iterations, behavior converges toward user equilibrium (a special case of Nash equilibrium)

4. **Comprehensive Evaluation Metrics**:

    - Core metric: travel time $t$ ($t^{pre}$, $t^{train}$, $t^{test}$, $t_{CAV}$, $t_{HDV}$)
    - Training cost $c_{all}$: measures the negative systemic impact incurred during training
    - Change analysis: mean speed change $\Delta_V$ and mileage change $\Delta_L$
    - Win rate $WR$: proportion of runs in which CAV post-training travel time is lower than the human baseline

### Loss & Training

The benchmark implements four MARL algorithms:
- **IQL (Independent Q-Learning)**: each agent trains an independent DQN
- **IPPO (Independent PPO)**: each agent independently applies PPO
- **MAPPO (Multi-Agent PPO)**: centralized critic with decentralized actors
- **QMIX**: factorizes the joint Q-function via a mixing network

Cooperative algorithms (MAPPO, QMIX) use a cooperative reward (minimizing system-wide travel time); independent algorithms (IQL, IPPO) use a selfish reward. Baselines include All-or-Nothing (AON), random, and human routing.

## Key Experimental Results

### Main Results

**Scenario 1: 40% CAV market penetration (average over 5 repetitions)**:

| Algorithm | St. Arnoult $t^{test}$ | Provins $t^{test}$ | Ingolstadt $t^{test}$ | St. Arnoult WR |
|-----------|----------------------|-------------------|-----------------------|---------------|
| Human baseline | 3.15 | 2.80 | 4.21 | 100% |
| AON | 3.15 | **2.67** | **4.11** | 100% |
| Random | 3.38 | 2.93 | 4.40 | 0% |
| IPPO | 3.28 | **2.90** | 4.37 | 0% |
| IQL | 3.36 | 2.91 | 4.41 | 0% |
| MAPPO | 3.35 | 2.91 | 4.38 | 0% |
| QMIX | **3.24** | 2.94 | 4.47 | 80% |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| QMIX 6000 episodes | 80% WR only on the smallest network, St. Arnoult |
| QMIX 20000 episodes | Extended training yields diminishing returns; performance remains poor on larger networks |
| Varying CAV penetration | Higher CAV penetration does not necessarily improve system-level performance |
| Fixed episode budget | MARL training converges extremely slowly; each episode requires a full SUMO simulation |

### Key Findings
- **MARL algorithms rarely surpass human routing**: Only QMIX achieves an 80% win rate on the smallest network (St. Arnoult); on larger networks it even underperforms the random policy
- **CAV deployment harms human drivers**: Across all scenarios $t_{HDV} > t^{pre}$, indicating that CAV presence increases travel time for human drivers
- **Training imposes substantial system costs**: System efficiency consistently degrades during training (reduced speeds, increased mileage)
- **The simple AON baseline achieves optimal performance in several scenarios**: exposing the immaturity of current MARL approaches

## Highlights & Insights

- **The value of negative results**: Reporting poor MARL performance candidly is more meaningful than presenting superficial successes, and points the community toward genuinely needed breakthroughs
- **Interdisciplinary design**: Transportation engineering (SUMO simulation), transport science (human routing behavior modeling), and machine learning (MARL) are organically integrated
- **Societal impact assessment**: Beyond evaluating CAV performance, the benchmark tracks the effects of CAV deployment on human drivers, reflecting responsible AI research practice
- **Open and extensible**: The modular design allows the community to freely add new algorithms, networks, and tasks

## Limitations & Future Work

- Simulation efficiency bottleneck: each episode requires a full SUMO simulation, making large-scale training extremely time-consuming
- Only pre-trip routing decisions are currently supported; en-route rerouting is not considered
- Human behavior is modeled with a standardized formulation that does not capture heterogeneity or irrational behavior
- Only discrete route selection is employed; continuous routing optimization remains unexplored
- Practical CAV deployment constraints such as communication latency and partial observability are not modeled

## Related Work & Insights

- **vs. FLOW / RESCO**: These traffic RL benchmarks focus on signal control rather than routing decisions; URB fills the gap for urban routing MARL benchmarks
- **vs. RouteRL**: URB extends RouteRL by incorporating 29 networks, system-level evaluation metrics, and benchmark algorithms
- **A cautionary message for MARL research**: The paper underscores the substantial gap between real-world large-scale non-stationary multi-agent problems and commonly used MARL benchmarks

## Rating

- Novelty: ⭐⭐⭐⭐ First large-scale urban routing MARL benchmark with an innovative problem formulation, though core methods rely on existing algorithms
- Experimental Thoroughness: ⭐⭐⭐⭐ Three networks, four algorithms, and multi-metric evaluation, though only a single scenario configuration is presented
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, interdisciplinary content is well organized, and background knowledge is sufficiently covered
- Value: ⭐⭐⭐⭐⭐ Provides important benchmark value for both MARL and autonomous vehicle routing; the negative results are highly instructive

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Extrapolated Urban View Synthesis Benchmark](../../ICCV2025/autonomous_driving/extrapolated_urban_view_synthesis_benchmark.md)
- [\[NeurIPS 2025\] CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation](cymbadiff_structured_spatial_diffusion_for_sketch-based_3d_semantic_urban_scene_.md)
- [\[NeurIPS 2025\] SimWorld-Robotics: Synthesizing Photorealistic and Dynamic Urban Environments for Multimodal Robot Navigation and Collaboration](simworld-robotics_synthesizing_photorealistic_and_dynamic_urban_environments_for.md)
- [\[AAAI 2026\] TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions](../../AAAI2026/autonomous_driving/tsbow_traffic_surveillance_benchmark_for_occluded_vehicles_under_various_weather.md)
- [\[ICCV 2025\] UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving](../../ICCV2025/autonomous_driving/uniocc_a_unified_benchmark_for_occupancy_forecasting_and_prediction_in_autonomou.md)

</div>

<!-- RELATED:END -->
