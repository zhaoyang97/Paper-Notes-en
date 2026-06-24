---
title: >-
  [Paper Note] BEAVER: Building Environments with Assessable Variation for Evaluating Multi-Objective Reinforcement Learning
description: >-
  [ICML 2025][Reinforcement Learning][Multi-objective RL] Proposes the BEAVER benchmark, the first multi-objective contextual reinforcement learning evaluation framework for building energy management. By parameterizing thermal dynamics and climate zones, it constructs controllable environmental variations to systematically evaluate the cross-environment generalization capabilities of existing MORL algorithms.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Multi-objective RL"
  - "Building energy management"
  - "MOC-MDP"
  - "Generalization evaluation"
  - "Contextual MDP"
  - "Pareto frontier"
date: 2026-05-08
content_hash: 3f26888be22df803
---

# BEAVER: Building Environments with Assessable Variation for Evaluating Multi-Objective Reinforcement Learning

**Conference**: ICML 2025  
**arXiv**: [2507.07769](https://arxiv.org/abs/2507.07769)  
**Code**: [https://github.com/chennnnnyize/BEAVER](https://github.com/chennnnnyize/BEAVER)  
**Area**: Reinforcement Learning  
**Keywords**: Multi-objective RL, Building energy management, MOC-MDP, Generalization evaluation, Contextual MDP, Pareto frontier

## TL;DR

Proposes the BEAVER benchmark, the first multi-objective contextual reinforcement learning evaluation framework for building energy management. By parameterizing thermal dynamics and climate zones, it constructs controllable environmental variations to systematically evaluate the cross-environment generalization capabilities of existing MORL algorithms.

## Background & Motivation

**Background**: Reinforcement learning-based building HVAC control has achieved success in single-simulation environments, effectively optimizing thermal comfort and energy consumption. However, building operation and maintenance face a key real-world challenge: buildings vary significantly in materials, structures, and geographical locations, making control policies trained in one building difficult to transfer directly to another.

**Limitations of Prior Work**: (1) Existing MORL studies assume a single static environment and ignore variations in underlying dynamics; (2) Building RL benchmarks lack standardized support for evaluating multi-objective trade-offs (comfort vs. energy consumption vs. carbon emissions) and cross-environment generalization; (3) In practice, operators deploying controllers face different buildings and differing user preferences, yet existing benchmarks fail to evaluate robustness under such multi-dimensional variations.

**Key Challenge**: Building thermal dynamics parameters (thermal resistance and heat capacity) vary by materials and climates, but existing methods simplify or ignore these differences, leading to a "lab-to-field" performance gap.

**Goal**: Construct a standardized and automated benchmark framework to systematically evaluate the multi-objective trade-off capabilities and cross-environment generalization performance of MORL algorithms in building control scenarios.

**Key Insight**: Formulate the building RL problem as a Multi-Objective Contextual MDP (MOC-MDP), where the "context" wraps two types of controllable variables: thermal convection parameters (affecting state transitions) and climate zones (affecting external inputs). Based on physical principles (RC thermal network models + EnergyPlus simulation), automated variations of environment sets are constructed.

**Core Idea**: MOC-MDP = MOMDP + Contextual Parameterization, decoupling "environmental variation" and "objective preferences" into independent dimensions to achieve systematic evaluation.

## Method

### Overall Architecture

The BEAVER benchmark consists of three components: (1) physics-principled building thermal dynamics environments based on RC networks; (2) parameterized contextual variables ($U_{\text{wall}}$ thermodynamics + climate regions); and (3) multi-objective reward designs and standardized evaluation metrics. The framework supports automated environment construction, MORL algorithm integration, and quantitative/visual analysis.

### Key Designs

1. **MOC-MDP Formulation**: Introduces a context space $\mathcal{C}$ on top of the standard MOMDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}_{1:n}, \Omega, f, \gamma \rangle$, defining a mapping $\boldsymbol{M}(c)$ that maps context $c \in \mathcal{C}$ to a specific MOMDP. The context $c$ includes climate conditions and building thermal dynamics parameters, with different values of $c$ altering the state transition function $\mathcal{P}^c$. A key assumption is that the context is unobservable to the agent—if it were observable, the problem would reduce to an MOMDP with an augmented state space. Objective preferences $\omega \in \Omega$ are mapped to scalar utilities via linear scalarization $f_\omega(\mathbf{r}) = \omega^\top \mathbf{r}$.

2. **Parameterized Building Thermal Dynamics**: Employs a classic RC network model to describe zone temperature dynamics: $C_i \frac{dT_i}{dt} = \sum_{j \in \mathcal{N}(i)} \frac{T_j - T_i}{R_{i,j}} + Q_i^h + Q_i^a + Q_i^s$, where $C_i$ represents heat capacity and $R_{i,j}$ represents thermal resistance. By sampling the wall U-factor ($U_{\text{wall}}$, in W/m²·°C) from EnergyPlus reference buildings, these are converted into $R_i, C_i$ for each wall, enabling systematic variation of thermodynamic parameters. A lower U-factor indicates better thermal insulation.

3. **Multi-Objective Reward Design & Evaluation Metrics**: Defines two types of rewards—thermal comfort $\mathcal{R}_{\text{thermal}} = M - 0.05 \sum_i |T_i[t] - T_i^s[t]|$ (penalty for deviance from the set point) and energy cost $\mathcal{R}_{\text{cost}} = M - 0.05 \sum_i c[t] |P_i[t]|$ (penalty for electricity consumption)—with additional support for power ramp rate rewards. Evaluation employs three complementary metrics: Hypervolume (HV, quality of Pareto frontier approximation), Expected Utility (EU, mean preferred utility), and Sparsity (SP, density of the solution set).

### Loss & Training

- Supports two training modes: Static-Train (training in a single fixed environment) and Dynamic-Train ($U_{\text{wall}}$ is sampled from a distribution in each episode).
- Compatible with the MORL-Generalization framework, supporting constrained multi-objective optimization algorithms such as C-MORL.
- Results are averaged over 5 independent runs.

## Key Experimental Results

### Main Results: Generalization under Thermal Dynamics Variations

| Metric | Training Mode | Dynamics 1 | Dynamics 2 | Dynamics 3 | Dynamics 4 | Dynamics 5 |
|------|---------|-----------|-----------|-----------|-----------|-----------|
| HV($10^7$)↑ | Static-Train | 9.35±0.04 | 9.38±0.06 | **8.49±0.05** | 9.76±0.05 | 9.27±0.05 |
| HV($10^7$)↑ | Dynamic-Train | 9.38±0.06 | 9.37±0.07 | **8.59±0.09** | 9.67±0.14 | 9.32±0.09 |
| EU($10^3$)↑ | Static-Train | 9.34±0.01 | 9.33±0.02 | **9.14±0.02** | 9.47±0.02 | 9.32±0.01 |
| SP($10^5$)↓ | Static-Train | 0.36±0.18 | 0.41±0.14 | **1.27±1.26** | 0.31±0.07 | 0.65±0.42 |

### Generalization under Climate Variations (Training Environment: Warm Marine)

| Metric | Mixed Marine | Cool Marine | Warm Humid | Warm Dry | Hot Humid | Warm Marine* |
|------|-------------|-------------|-----------|---------|----------|-------------|
| HV($10^7$)↑ | 8.78±0.11 | 8.74±0.10 | 8.59±0.09 | 9.59±0.05 | 10.00±0.02 | **10.13±0.14** |
| EU($10^3$)↑ | 8.80±0.06 | 8.78±0.06 | 8.71±0.05 | 9.25±0.04 | 9.58±0.02 | **9.67±0.02** |
| SP($10^5$)↓ | 0.79±0.37 | 0.80±0.30 | **7.24±0.38** | 0.31±0.09 | 0.19±0.08 | 0.19±0.09 |

### Ablation Study

| Comparison Dimension | Dynamics Effect | Climate Effect |
|---------|---------|---------|
| Dynamics 3 Degradation | HV dropped to 8.49 (vs. 9.35 baseline, -9.2%) | — |
| Warm Humid Degradation | — | SP soared from 0.19 to 7.24 (38x) |
| Dynamic-Train vs. Static-Train | HV increased only marginally by 0.03–0.10 | — |

### Key Findings

- Dynamics 3 is the most challenging environmental variation, causing significant drops in both HV and EU, indicating that certain thermodynamic configurations pose severe challenges to existing MORL algorithms.
- Dynamic-Train yields only marginal improvements compared to Static-Train, suggesting that current uniform mixing training strategies are insufficient to enhance robustness.
- Cross-region generalization under climate variations is highly unstable, where only Hot Humid approaches the performance of the training environment, while all others degrade.
- Pareto frontier visualization shows that both training modes perform poorly on the user comfort objective.

## Highlights & Insights

- **Filling the Gap**: The first standardized benchmark applying MOC-MDP formulation to building HVAC control, systematically decoupling the two dimensions of "environmental variation" and "objective preference."
- **Physics-Driven**: Extracts realistic $U_{\text{wall}}$ values based on EnergyPlus reference buildings, improving the practical relevance of the evaluation.
- **Exposing Key Deficiencies**: Experiments clearly reveal that existing MORL methods lack cross-environment robustness, providing a clear direction for prospective algorithm design.

## Limitations & Future Work

- Currently, only single-zone buildings are supported, leaving multi-zone complex building layouts for future extensions.
- The MORL baselines are limited to C-MORL, without coverage of other algorithms like GPI-PD or PGMORL.
- The sampling strategy for Dynamic-Train is overly simplistic (uniform sampling); curriculum learning or adaptive sampling might be more effective.
- Additional contextual dimensions, such as initial state distributions and varying occupancy rates, can be incorporated in the future.

## Related Work & Insights

- Complementary to MORL-Generalization (Teoh et al. 2025): the latter provides a general-purpose MORL benchmark, whereas BEAVER focuses on practical scenarios in building control.
- C-MORL (Liu et al. 2025) performs well with many objectives, but cross-environment generalization remains a bottleneck.
- The RC network modeling in MPC methods (Ma et al. 2012) provides the physical foundation for BEAVER.

## Rating

⭐⭐⭐⭐ — Provides a rigorous MORL benchmark for the practical application domain of building control. The MOC-MDP formulation is elegant, and the experiments reveal valuable negative findings. The methodological innovation lies primarily in the benchmark design rather than algorithmic breakthroughs, and the coverage of baseline methods is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization](graph-supported_dynamic_algorithm_configuration_for_multi-objective_combinatoria.md)
- [\[ICLR 2026\] A Reward-Free Viewpoint on Multi-Objective Reinforcement Learning](../../ICLR2026/reinforcement_learning/a_reward-free_viewpoint_on_multi-objective_reinforcement_learning.md)
- [\[NeurIPS 2025\] Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach](../../NeurIPS2025/reinforcement_learning/multi-objective_reinforcement_learning_with_max-min_criterion_a_game-theoretic_a.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](../../NeurIPS2025/reinforcement_learning/thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](../../NeurIPS2025/reinforcement_learning/forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)

</div>

<!-- RELATED:END -->
