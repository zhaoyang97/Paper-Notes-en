---
title: >-
  [Paper Note] Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization
description: >-
  [ICML2025][Reinforcement Learning][Dynamic Algorithm Configuration] This paper proposes GS-MODAC, which leverages GNNs to map solutions in the objective space into a graph structure for learning state representations. Combined with PPO, it dynamically configures the parameters of Multi-Objective Evolutionary Algorithms (MOEAs). It outperforms static and existing DRL methods on scheduling and routing NP-hard combinatorial optimization problems, demonstrating generalization cap…
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Dynamic Algorithm Configuration"
  - "Graph Neural Networks"
  - "Deep Reinforcement Learning"
  - "Multi-Objective Evolutionary Algorithms"
  - "Pareto Optimization"
date: 2026-05-08
content_hash: 6a3fed7241c41b5f
---

# Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization

**Conference**: ICML2025  
**arXiv**: [2505.16471](https://arxiv.org/abs/2505.16471)  
**Code**: [GitHub](https://github.com/RobbertReijnen/GS-MODAC)  
**Area**: Algorithm Configuration / Multi-Objective Combinatorial Optimization  
**Keywords**: Dynamic Algorithm Configuration, Graph Neural Networks, Deep Reinforcement Learning, Multi-Objective Evolutionary Algorithms, Pareto Optimization

## TL;DR

This paper proposes GS-MODAC, which leverages GNNs to map solutions in the objective space into a graph structure for learning state representations. Combined with PPO, it dynamically configures the parameters of Multi-Objective Evolutionary Algorithms (MOEAs). It outperforms static and existing DRL methods on scheduling and routing NP-hard combinatorial optimization problems, demonstrating generalization capability across problem scales and numbers of objectives.

## Background & Motivation

**Core Problem:** Hyperparameters of Evolutionary Algorithms (EAs) (such as crossover rate, mutation rate, etc.) significantly impact solution quality, and the optimal parameters vary across stages in the search process. Existing Dynamic Algorithm Configuration (DAC) methods are primarily designed for continuous optimization problems, making them difficult to directly transfer to Multi-Objective Combinatorial Optimization (MOCO).

**Limitations of Prior Work:**

1. **Static Configuration Methods** (irace, SMAC3): These use fixed parameters throughout the search process, failing to adapt to changes across search stages.
2. **MADAC** (Xue et al., 2022): Although it is a DRL-based DAC, it relies on hand-crafted state features (e.g., number of elite solutions, spacing, hypervolume), which is time-consuming and potentially suboptimal; the discretized action space also limits parameter control precision.
3. Existing methods perform poorly on large-scale, multi-objective combinatorial optimization problems due to the non-smooth solution space of COPs and large discrepancies in objective value ranges.

**Design Motivation:** To automatically learn state representations from the graph structure of the objective space using GNNs, replacing hand-crafted feature engineering; and to design instance-independent reward functions that enable the model to generalize across different problem scales and types.

## Method

### Overall Architecture

GS-MODAC models dynamic algorithm configuration as a contextual Markov Decision Process (contextual MDP). The mechanism is: at each search iteration, the distribution of the current population in the objective space is transformed into a graph, GNNs are used to extract embeddings as states, and a DRL agent outputs the EA parameters for the next iteration.

### MDP Component Design

**State Space (States):** Mapping the objective space to a graph structure:

- Each node corresponds to a solution, with normalized objective values as node features.
- Normalization baseline: the best values encountered during the search + the worst values from the initial generation.
- Solutions are partitioned into different Pareto front layers via non-dominated sorting, with edges constructed between solutions in the same layer.
- Auxiliary feature vector: normalized generations used (reflecting the remaining search budget).

The key advantage of this design is that **the state dimension is independent of the number of objectives**, allowing natural scalability to an arbitrary number of objectives.

**Action Space (Actions):** Continuous values, normalized to $[-1, 1]$, and mapped to the range of EA parameters. Taking NSGA-II as an example:

- Crossover rate $\in [0.6, 1.0]$
- Mutation rate $\in [0.0, 0.1]$

**Reward Function (Rewards):** Progressive reward based on hypervolume improvement:

$$r_t = \begin{cases} \Delta_{\text{current}}^2 - \Delta_{\text{best}}^2 & \text{if } HV_{\text{current}} > HV_{\text{best}} \\ 0 & \text{otherwise} \end{cases}$$

Where the improvement ratio is defined as:

$$\Delta_{\text{current}} = \frac{HV_{\text{current}} - HV_{\text{initial}}}{HV_{\text{ideal}} - HV_{\text{initial}}} \times 100$$

Through the squared-difference design, **larger hypervolume improvements in the later stages receive higher rewards**, encouraging fine-grained evolution at the end of the search. The ideal hypervolume $HV_{\text{ideal}}$ is approximated through a single high-budget (e.g., double) run, making the rewards instance-independent.

### Policy Network Architecture

- A 2-layer GCN extracts graph node embeddings.
- Global mean pooling aggregates node embeddings into a single graph embedding.
- Concatenates the search budget feature.
- A linear layer outputs the mean of the action distribution.
- Trained using PPO.

### Supported MOEAs

- **NSGA-II**: Non-dominated sorting + crowding distance, used as the main experimental algorithm.
- **MOPSO**: Multi-Objective Particle Swarm Optimization, used to verify the generality of the method.

## Key Experimental Results

### Experimental Setup

- **Problems**: Flexible Job Shop Scheduling Problem (FJSP, 2/3/5 objectives) and Capacitated Vehicle Routing Problem (CVRP, 2 objectives).
- **Scales**: FJSP 5j5m / 10j5m / 25j5m; CVRP 100 / 200 / 500 customers.
- **Baselines**: NSGA-II default parameters, irace (static), SMAC3 (static), MADAC (DRL).
- **Metrics**: Mean, maximum (max), and standard deviation (std) of hypervolume (HV), across 100 instances × 10 runs.

### Main Results (Hypervolume, bold indicates the best)

| Problem | Scale | GS-MODAC Rank | Key Findings |
|------|------|--------------|----------|
| Bi-FJSP | 5j5m / 10j5m / 25j5m | **Best in all** | Best in both mean and max |
| Tri-FJSP | 5j5m | Second | Close to irace |
| Tri-FJSP | 10j5m / 25j5m | **Best** | Significantly outperforms all baselines |
| Penta-FJSP | All scales | **Best in all** | Edge is more pronounced with 5 objectives |
| Bi-CVRP | 100 / 200 / 500 | **Best or tied** | Comparable to MADAC |

### Generalization Experiments

- Models trained on small scales (e.g., 5j5m) can be directly applied to large-scale (25j5m) instances, still outperforming static methods trained directly on the target scale.
- Generalizes to more highly-constrained problem variants unseen during training.
- Shows strong generalization across numbers of objectives (from 2 to 3 to 5).

### MOPSO Adaptation

The effectiveness of GS-MODAC is also verified on MOPSO, proving that the method does not rely on a specific MOEA.

## Highlights & Insights

1. **Graph State Representation Replacing Hand-crafted Features**: GNNs automatically learn the structure of the objective space, eliminating the tedious process of manual state feature design, and naturally adapting to different numbers of objectives.
2. **Instance-Independent Reward Design**: Normalized hypervolume combined with progressive squared-difference rewards allows a single model to work across different problem scales and types.
3. **Continuous Action Space**: Provides finer parameter control compared to the discrete actions of MADAC.
4. **Greater Advantages with More Objectives**: GS-MODAC shows the largest performance gap on Penta-FJSP (5 objectives), demonstrating the advantages of graph representations in high-dimensional objective spaces.
5. **Strong Generalization Ability**: Demonstrates generalization capacity across three dimensions: scale, constraints, and number of objectives.

## Limitations & Future Work

1. **High Training Cost**: Training on large-scale problems takes days (up to 3 days for 25j5m FJSP), requiring extensive pre-training before actual deployment.
2. **Evaluation Limited to Two Problem Types**: Only FJSP and CVRP were evaluated; effectiveness on other MOCO problems (e.g., knapsack, graph coloring) remains to be verified.
3. **Ideal Hypervolume Requires Pre-computation**: $HV_{\text{ideal}}$ needs to be approximated with an extra high-budget run, which increases upfront computational overhead.
4. **Limited Configured Parameters**: The action space only covers two parameters (crossover and mutation rates), which may be insufficient for EAs with more parameters (such as CMA-ES with adaptation mechanisms).
5. **Simplistic GCN Pooling Method**: Utilizing global mean pooling might lose hierarchical structural information of the Pareto fronts.

## Related Work & Insights

- **MADAC** (Xue et al., 2022): The most direct predecessor, which uses multi-agent DRL + hand-crafted state features for DAC.
- **DAC-Bench** (Biedenkapp et al., 2020; Adriaensen et al., 2022): A standardized framework and benchmark for DAC.
- **irace / SMAC3**: Classical static algorithm configuration methods, serving as strong baselines.
- **GNN for CO**: Unlike directly solving CO problems using GNNs (e.g., Attention Models), this work uses GNNs for meta-level control.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of graph state representation and instance-independent rewards is novel, but the overall paradigm remains standard DAC + GNN.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Quite comprehensive, covering two types of problems, multiple scales, generalization experiments, MOPSO adaptation, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, complete description of methodology, and intuitive diagrams.
- Value: ⭐⭐⭐⭐ — Provides a practical and generalizable solution for the automatic configuration of MOEAs in combinatorial optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](../../NeurIPS2025/reinforcement_learning/sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[ICML 2025\] Preference Optimization for Combinatorial Optimization Problems](preference_optimization_for_combinatorial_optimization_problems.md)
- [\[NeurIPS 2025\] DCcluster-Opt: Benchmarking Dynamic Multi-Objective Optimization for Geo-Distributed Data Center Workloads](../../NeurIPS2025/reinforcement_learning/dccluster-opt_benchmarking_dynamic_multi-objective_optimization_for_geo-distribu.md)
- [\[ICML 2025\] Meta-Black-Box-Optimization through Offline Q-function Learning (Q-Mamba)](meta-black-box-optimization_through_offline_q-function_learning.md)
- [\[NeurIPS 2025\] PARCO: Parallel AutoRegressive Models for Multi-Agent Combinatorial Optimization](../../NeurIPS2025/reinforcement_learning/parco_parallel_autoregressive_models_for_multi-agent_combinatorial_optimization.md)

</div>

<!-- RELATED:END -->
