---
title: >-
  [Paper Note] DCcluster-Opt: Benchmarking Dynamic Multi-Objective Optimization for Geo-Distributed Data Center Workloads
description: >-
  [NeurIPS 2025][Reinforcement Learning][Data Center Optimization] This paper proposes DCcluster-Opt, an open-source high-fidelity simulation benchmark platform for geo-distributed data centers. It integrates real-world da…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Data Center Optimization"
  - "Multi-Objective Optimization"
  - "Carbon Emissions"
  - "Workload Scheduling"
date: 2026-05-08
content_hash: bcc50df54d329903
---

# DCcluster-Opt: Benchmarking Dynamic Multi-Objective Optimization for Geo-Distributed Data Center Workloads

**Conference**: NeurIPS 2025
**arXiv**: [2511.00117](https://arxiv.org/abs/2511.00117)  
**Code**: [GitHub (dc-rl)](https://github.com/HewlettPackard/dc-rl)  
**Area**: Reinforcement Learning / Sustainable Computing
**Keywords**: Data Center Optimization, Multi-Objective Optimization, Reinforcement Learning, Carbon Emissions, Workload Scheduling

## TL;DR

This paper proposes DCcluster-Opt, an open-source high-fidelity simulation benchmark platform for geo-distributed data centers. It integrates real-world datasets (carbon intensity, electricity prices, weather, etc.) and physics-based models to support reinforcement learning research on dynamic multi-objective workload scheduling.

## Background & Motivation

The rapid growth of large-scale AI has led to a dramatic increase in data center energy consumption and carbon emissions. Intelligent workload management across globally distributed data center clusters is of critical importance, yet research progress has been hindered by the lack of suitable benchmarks.

**Limitations of existing benchmarks:**

**Oversimplified environmental factors**: Existing benchmarks fail to realistically capture the interactions among time-varying grid carbon intensity, electricity prices, and weather conditions.

**Absence of data center physics models**: Detailed physical characteristics such as CPU, GPU, memory, and HVAC energy consumption are largely ignored.

**Lack of geo-distributed network dynamics**: Cross-data-center network dynamics such as latency and transmission costs are not modeled.

**Poor reproducibility**: Some prior work relies on proprietary data or non-reproducible experimental setups.

**Positioning of DCcluster-Opt:**

As an evolution of the predecessor SustainDC (NeurIPS 2024 Datasets and Benchmarks), DCcluster-Opt extends the scope from single data centers to geo-distributed clusters, adding a top-level coordinator agent for task assignment, cross-region network modeling, heat recovery, and other advanced components.

## Method

### Overall Architecture

DCcluster-Opt constructs a simulation environment composed of multiple geo-distributed data centers, centered on a hierarchical scheduling problem:

- **Top-level Coordinator Agent**: Receives global state (task queues, per-DC load, carbon intensity, electricity prices, etc.) and decides how to assign, defer, or redistribute incoming tasks across data centers.
- **Data Center-level Agents**: Manage intra-DC operations such as HVAC cooling optimization and battery charge/discharge strategies.
- **Task Characteristics**: Each task carries resource requirements (CPU/GPU/memory) and service-level agreement (SLA) constraints.

### Key Designs

**1. High-Fidelity Data Center Physics Models**

Each data center simulates the following components:
- **IT Systems**: CPU/GPU power models (utilization-to-power ratio), server rack thermal models.
- **HVAC Systems**: CRAC units, chillers (COP models), cooling towers, and pump energy consumption.
- **Battery Systems**: Charge/discharge cycles and state-of-charge (SoC) management.
- **Heat Recovery Systems**: Utilization of server waste heat for heating purposes.

**2. Real-World Dataset Integration**

Covering 20 global regions:
- **AI Workload Traces**: Derived from Alibaba and Google cluster data.
- **Grid Carbon Intensity**: Sourced from the EIA (U.S. Energy Information Administration), varying by region and time.
- **Electricity Market Prices**: Real-time electricity pricing data.
- **Weather Data**: Hourly weather profiles (temperature, humidity, etc.) in EnergyPlus .epw format.
- **Cloud Transfer Costs**: Cross-region data transfer pricing.
- **Network Latency**: Empirically measured cross-region latency parameters.

**3. Modular Reward System**

Supports flexible configuration of multi-objective weights:
- **Carbon Emissions**: Minimize total CO₂ emissions.
- **Energy Cost**: Minimize overall electricity expenditure.
- **SLA Violations**: Ensure tasks are completed before deadlines.
- **Water Consumption**: Minimize cooling water usage.

The reward function supports customizable weights, enabling research on Pareto trade-offs among different objectives.

**4. Gymnasium API Integration**

The environment implements the standard Gymnasium Env interface:
- **Observation Space**: Includes time encoding, carbon intensity forecasts, per-DC load status, and pending task queues.
- **Action Space**: Task assignment decisions (assign to a specific DC / defer / reject).
- Supports both single-agent and multi-agent modes.

### Loss & Training

As a benchmark environment, DCcluster-Opt provides multiple baseline controllers:

- **Rule-based**: Heuristic strategies such as greedy assignment based on carbon intensity and load balancing.
- **RL Methods**: PPO, IPPO, MAPPO, HAPPO, and other reinforcement learning algorithms.
- **Random**: Random assignment baseline.

## Key Experimental Results

### Main Results

**Table 1: Performance comparison of different scheduling strategies on a 5-DC cluster**

| Method | Carbon Emissions (kg CO₂) ↓ | Energy Cost ($) ↓ | SLA Violation Rate (%) ↓ | Water Usage (m³) ↓ |
|---|---|---|---|---|
| Random | 1250 | 8500 | 12.3 | 450 |
| Greedy-Carbon | 980 | 9200 | 8.5 | 380 |
| Load-Balance | 1150 | 7800 | 5.2 | 420 |
| PPO | 920 | 7600 | 6.1 | 360 |
| HAPPO | **870** | **7200** | 4.8 | **340** |
| MAPPO | 890 | 7400 | **4.5** | 355 |

Multi-agent RL methods (HAPPO, MAPPO) outperform rule-based strategies and single-agent RL on most objectives, though different strategies exhibit distinct trade-offs between carbon emissions and SLA compliance.

**Table 2: Scalability under different cluster size configurations**

| Cluster Scale | Training Time (h) | Carbon Reduction (%) vs. Random | SLA Improvement (%) |
|---|---|---|---|
| 3 DCs | 2.5 | 22.4 | 48.5 |
| 5 DCs | 5.8 | 30.4 | 56.2 |
| 10 DCs | 14.2 | 35.1 | 61.8 |
| 20 DCs | 38.6 | 38.7 | 65.3 |

As cluster size increases, RL agents can exploit greater cross-region carbon intensity variation to optimize scheduling, though training costs grow linearly.

### Ablation Study

**Analysis of individual environment component contributions**

| Configuration | HAPPO Carbon Reduction (%) |
|---|---|
| Full environment | 30.4 |
| Without weather variation | 26.1 |
| Without carbon intensity forecast | 22.8 |
| Without network latency modeling | 28.9 |
| Without heat recovery | 29.7 |

Carbon intensity forecast information has the greatest impact on RL agent performance (−7.6%), highlighting the importance of incorporating time-varying environmental signals.

### Key Findings

1. **Geographic diversity is critical**: Greater cross-region carbon intensity variation provides more scheduling opportunities for RL; the 20-DC configuration achieves approximately 16% more carbon reduction than the 3-DC configuration.
2. **Carbon vs. SLA trade-off**: Strategies that minimize carbon emissions may defer tasks to low-carbon time windows, leading to increased SLA violations.
3. **Rule-based strategies remain competitive**: In certain single-objective scenarios, well-designed greedy strategies approach RL-level performance.
4. **Forecast information is essential**: Look-ahead carbon intensity and electricity price forecasts significantly improve RL agent performance.

## Highlights & Insights

- **Realism and reproducibility**: Integrates multi-source real-world data with physics-based models while remaining fully open-source.
- **Extension from single DC to cluster**: Compared to the predecessor SustainDC, DCcluster-Opt adds a cross-DC scheduling dimension, making the problem substantially more challenging.
- **Modular design**: Reward functions, data center configurations, and the number of regions are all flexibly configurable.
- **Standard interface**: The Gymnasium API enables plug-and-play compatibility with a wide variety of RL algorithms.

## Limitations & Future Work

1. **Sim-to-real gap**: Despite the use of physics-based models and real-world data, the simulation inevitably deviates from actual data center operations.
2. **Simplified network model**: The current latency and transfer cost model is relatively simple and does not account for dynamic routing or congestion.
3. **Extensible task model**: The current task model assumes relatively simple resource requirements and does not deeply model complex workloads such as GPU cluster training jobs.
4. **Safety constraints**: Modeling of safety constraints in real data center operations (e.g., hard temperature limits, power redundancy) could be further strengthened.
5. **LLM workloads**: As LLM inference demand grows, workload patterns specific to LLM serving have yet to be incorporated.

## Related Work & Insights

- **SustainDC (Naug et al. 2024)**: The predecessor of this work; a single-DC multi-agent benchmark.
- **CarbonExplorer (Facebook)**: A carbon footprint analysis tool.
- **DCRL-Green**: An earlier version of green control for data centers.
- **EnergyPlus**: A building energy simulator providing the foundation for weather and thermal models.
- **Gymnasium/PettingZoo**: Standardized RL environment interfaces.

## Rating

| Dimension | Score (1–5) |
|---|---|
| Novelty | 3 — An incremental extension of SustainDC with moderate innovation. |
| Technical Quality | 4 — High-fidelity simulation with robust engineering implementation. |
| Experimental Thoroughness | 4 — Systematic evaluation across multiple strategies and scales. |
| Writing Quality | 4 — Clear structure typical of benchmark papers. |
| Impact | 4 — Provides an important benchmark for sustainable computing research. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[NeurIPS 2025\] Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach](multi-objective_reinforcement_learning_with_max-min_criterion_a_game-theoretic_a.md)
- [\[NeurIPS 2025\] PARCO: Parallel AutoRegressive Models for Multi-Agent Combinatorial Optimization](parco_parallel_autoregressive_models_for_multi-agent_combinatorial_optimization.md)
- [\[NeurIPS 2025\] MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)

</div>

<!-- RELATED:END -->
