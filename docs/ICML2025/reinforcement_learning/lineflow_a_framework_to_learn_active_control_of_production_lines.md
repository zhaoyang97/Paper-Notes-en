---
title: >-
  [Paper Note] LineFlow: A Framework to Learn Active Control of Production Lines
description: >-
  [ICML2025][Reinforcement Learning][production line control] Proposes LineFlow, an extensible, open-source Python framework designed to simulate production lines of arbitrary complexity and train RL agents for active production line control (adaptive routing, worker reallocation, dispatching, etc.), while providing mathematical optimal solutions for several sub-problems as benchmarks.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "production line control"
  - "reinforcement-learning"
  - "discrete-event simulation"
  - "manufacturing optimization"
  - "open-source framework"
date: 2026-05-08
content_hash: e1c9ca02a10c37ac
---

# LineFlow: A Framework to Learn Active Control of Production Lines

**Conference**: ICML2025  
**arXiv**: [2505.06744](https://arxiv.org/abs/2505.06744)  
**Code**: [hs-kempten/lineflow](https://github.com/hs-kempten/lineflow)  
**Author**: Kai Müller, Martin Wenzel, Tobias Windisch
**Area**: Reinforcement Learning  
**Keywords**: production line control, reinforcement-learning, discrete-event simulation, manufacturing optimization, open-source framework

## TL;DR

Proposes LineFlow, an extensible, open-source Python framework designed to simulate production lines of arbitrary complexity and train RL agents for active production line control (adaptive routing, worker reallocation, dispatching, etc.), while providing mathematical optimal solutions for several sub-problems as benchmarks.

## Background & Motivation

### Challenges of Active Production Line Control

During operation, production lines encounter variations in processing conditions, stochastic fluctuations, and machine breakdowns, leading to buffer overflows, shifting bottlenecks, and line blockages. Traditional control relies on rule-based policies and mathematical models, which show limited effectiveness in highly uncertain, dynamic environments. RL has demonstrated the capability to learn complex state-action mappings in various domains and is a natural choice for production line control.

### Limitations of Prior Work

- Existing RL production line control studies utilize ad-hoc or domain-specific simulation environments, making reproduction and comparison difficult.
- There is a lack of a **standardized and general** framework to train and evaluate RL agents across diverse production line scenarios.
- Manufacturers are reluctant to disclose real production line layouts due to privacy concerns, leading to a scarcity of public datasets.

LineFlow is specifically designed to fill these gaps by providing a unified simulation environment, standardized action/state spaces, and mathematical optimal solutions for benchmarking.

## Method

### 1. Production Line Simulation Modeling

LineFlow abstracts production lines into the following core objects:

| Object | Function |
|------|------|
| **Source** | Emits parts (raw materials) |
| **Process** | Performs an operation on a single part |
| **Assembly** | Merges multiple components into one |
| **Sink** | Removes finished products from the line |
| **Buffer** | FIFO buffer between stations with limited capacity |
| **Switch** | Handles routing allocation of parts |
| **Carrier** | Transports parts between workstations |

The processing time is modeled as an exponential distribution:

$$\mathcal{T} = T + \mathrm{Exp}_S$$

where $T \geq 0$ represents the minimum processing time, and $\mathrm{Exp}_S$ is an exponential distribution with mean $S$. This modeling aligns with classical production line queuing theory literature.

### 2. Three Types of Interventions for Active Control

- **Waiting Time Regulation**: Controls the release rate from the Source to prevent downstream Assembly from producing scrap due to timeouts.
- **Part Distribution**: Changes the routing allocation ratio of parts among parallel workstations via Switches.
- **Worker Reallocation**: Reassigns workers from idle workstations to bottleneck workstations to reduce their processing times.

### 3. Performance Metrics — Generalized OEE

A cost model is introduced based on classical OEE (Overall Equipment Effectiveness). For a policy $\pi$, the production line value at time $t$ is formulated as:

$$C_\pi(t) = \frac{T_C}{t}\left(c \cdot n_{\text{ok}}^\pi(t) - \sum_{i=1}^{k} c_i \cdot n_{\text{nok}}^\pi(t, i)\right)$$

- $n_{\text{ok}}^\pi(t)$: Number of quality products up to time $t$
- $n_{\text{nok}}^\pi(t,i)$: Number of scrap components at workstation $P_i$
- $c_i$ : Unit cost at workstation $P_i$ (material/energy)
- $T_C$: Minimum theoretical cycle time to produce a single finished product

$C_\pi > 0$ indicates a profitable production line, while $C_\pi < 0$ denotes a loss. In a scrap-free scenario, maximizing $C_\pi$ is equivalent to maximizing throughput.

### 4. RL Formulation

Production line control is formulated as an **episodic Partially Observable Markov Decision Process (POMDP)**:

- **State Space**: Buffer fill levels (normalized to $[0,1]$), workstation processing times, production rates, workstation modes, and routing information at Switches.
- **Action Space**: Discrete control decisions — worker allocation (an independent dimension for each worker), routing adjustment, and workstation toggling (on/off).
- **Interaction Frequency**: The agent interacts with the production line at a fixed interval $T_{\text{step}}$, resulting in a total of $T = T_{\text{sim}} / T_{\text{step}}$ steps per episode.
- **Reward Design**: Decoupling $C_\pi$ into step-wise differences to support TD learning:

$$R(s_t, \pi(s_{t-1})) = C_\pi(T_{\text{step}} \cdot (t+1)) - C_\pi(T_{\text{step}} \cdot t)$$

thereby ensuring that $\sum_{t=0}^{T} R(t) = C_\pi(T_{\text{sim}})$.

### 5. Implementation Architecture

- The underlying discrete-event simulation is based on **SimPy**, with an object-oriented design facilitating custom workstation creation.
- The visualization module is built on **pygame**.
- The RL interface complies with the **Gymnasium API**, enabling direct integration with stable-baselines3 / skrl.
- Supports environment **vectorization and parallelization** to accelerate training.
- Although the agent interacts at discrete time steps, the underlying production line simulation remains continuous-time and event-driven.

## Experiments & Case Studies

### Benchmark Scenario Design

The authors designed three sub-problems that possess computable mathematical optimal solutions:

| Scenario | Control Variable | Key Challenge |
|------|----------|----------|
| **WT / WTJ** | Source waiting time | Balancing the release pace with Assembly timeout scraps; WTJ introduces sudden changes in processing time |
| **PD$_k$** | Allocation ratio of parts among $k$ parallel workstations | Optimal load balancing based on heterogeneous processing times of stations |
| **WA$_{k,n}$** | Allocation of $n$ workers among $k$ workstations | Dynamically assigning workers to bottleneck workstations |

Additionally, a **complex industrial scenario** is designed, which simultaneously involves waiting time, distribution, and worker dispatching, with no known mathematical optimal solution.

### WT/WTJ Scenario

- The Source $S_C$ needs to control its release interval to match the throughput rate of Assembly $A$.
- Under-waiting (waiting time too short) $\to$ components expire in the buffer, generating scrap and requiring cleanup time.
- Over-waiting (waiting time too long) $\to$ idle Assembly, leading to reduced throughput.
- The optimal waiting time is determined by the processing time difference between $A$ and $S_C$; analytical solutions are provided in the paper.
- In the WTJ variant, the processing time of $A$ undergoes a sudden change (multiplied by a factor $f$) at random moments, requiring the agent to perform online detection and dynamic adjustment.

### Training Setup

- SOTA RL algorithms such as **PPO** (Proximal Policy Optimization) and **DQN** are implemented.
- Training is conducted via stable-baselines3 / skrl.
- **Curriculum learning** strategies are adopted for complex scenarios, gradually scaling up difficulty.

### Main Results

1. **Simple Scenarios (WT, PD, WA)**: The performances of RL-learned policies are **close to the mathematical optimal solutions**, validating the rationality of LineFlow's modeling assumptions and reward design.
2. **Dynamic Scenario (WTJ)**: RL agents successfully learn to perform online detection of processing time mutations and adaptively adjust the waiting time. However, due to observation delay (the change can only be sensed after the first part is produced), they cannot fully reach the theoretical optimum.
3. **Complex Industrial Scenario**: RL faces **significant challenges** with suboptimal performance, exposing the following bottlenecks:
    - Sparse rewards and weak signals.
    - Combinatorial explosion of the action space.
    - Difficulty in learning compound side-effects from multiple types of interventions.

## Limitations & Future Work

1. **Restricted Processing Time Assumptions**: The uniform adoption of the exponential distribution ($T + \mathrm{Exp}_S$) does not fully reflect real-world production lines, where processing time distributions may be more complex (e.g., Weibull, log-normal).
2. **Insufficient Scalability of RL for Complex Lines**: In industrial-scale, multi-station scenarios, current RL algorithms struggle to converge, requiring superior reward shaping, hierarchical RL, and curriculum learning strategies.
3. **Simplified POMDP Modeling**: Although modeled as a POMDP, exploiting historical observation sequences (e.g., through RNN/Transformer policy networks) remains largely unexplored.
4. **Lack of Real-World Production Line Validation**: Despite modeling a real-world production line based on public data in the appendix, the framework has not been deployed or validated in actual manufacturing environments.
5. **Oversimplified Economic Cost Model**: Real-world factors such as dynamic energy fluctuations, product switchover costs, and maintenance scheduling are not considered.

## Reproducibility Points

- **Open-Source Code**: [github.com/hs-kempten/lineflow](https://github.com/hs-kempten/lineflow), containing the complete simulation engine and examples.
- **Standard API**: Adheres to the Gymnasium interface, allowing direct integration with stable-baselines3 and skrl.
- **Dependency Stack**: Python + SimPy (discrete-event simulation) + pygame (visualization) + pandas/numpy.
- **All Case Study Parameters** (processing times, buffer capacities, cost coefficients, etc.) are explicitly provided in the paper.
- **Mathematical Optimal Solutions** provide analytical benchmarks for each baseline scenario.

## Highlights & Insights

**Strengths**:

- Accurate positioning: RL in manufacturing lacks standardized environments rather than algorithms; LineFlow successfully fills this gap.
- Solid theoretical foundation: Mathematical optimal solutions are provided as baselines for each case, making RL performance evaluations rigorous.
- Sound engineering design: The use of the Gymnasium API, vectorized environments, and object-oriented production line modeling lowers the entry barrier.
- Open-source nature facilitates community reproduction and expansion.

**Weaknesses**:

- The paper leans more towards a framework introduction with limited innovation on the RL algorithm side—no tailor-made algorithms designed for production line control (such as GNN policy networks exploiting structural priors).
- Experimental results of complex scenarios are not integrated deeply; it merely states "RL faces challenges" without thoroughly analyzing the failure modes.
- Lack of comparison with OR (Operations Research) methods: Traditional MIP/CP solvers might still serve as stronger baselines in scheduling and allocation tasks.

**Insights**:

- Production line control is essentially a multi-agent coordination + combinatorial optimization problem. Pure end-to-end RL might be insufficient; integrating structured policies (e.g., decomposing bottlenecks before performing local optimization) could be necessary.
- Combining LineFlow with LLM-based planning/reasoning (e.g., using LLMs for high-level scheduling and RL for low-level control) could be an intriguing research direction.

## Rating
- Novelty: To be rated
- Experimental Thoroughness: To be rated
- Writing Quality: To be rated
- Value: To be rated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control](../../NeurIPS2025/reinforcement_learning/succeed_or_learn_slowly_sample_efficient_off-policy_reinforcement_learning_for_m.md)
- [\[ICML 2025\] Stochastic Encodings for Active Feature Acquisition](stochastic_encodings_for_active_feature_acquisition.md)
- [\[ICML 2025\] Learning Mean Field Control on Sparse Graphs](learning_mean_field_control_on_sparse_graphs.md)
- [\[NeurIPS 2025\] Shift Before You Learn: Enabling Low-Rank Representations in Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/shift_before_you_learn_enabling_low-rank_representations_in_reinforcement_learni.md)
- [\[NeurIPS 2025\] Open-World Drone Active Tracking with Goal-Centered Rewards](../../NeurIPS2025/reinforcement_learning/open-world_drone_active_tracking_with_goal-centered_rewards.md)

</div>

<!-- RELATED:END -->
