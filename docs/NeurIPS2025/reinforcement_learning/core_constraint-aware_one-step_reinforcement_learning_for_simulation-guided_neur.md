---
title: >-
  [Paper Note] CORE: Constraint-Aware One-Step Reinforcement Learning for Simulation-Guided Neural Network Accelerator Design
description: >-
  [NeurIPS 2025][Reinforcement Learning][one-step RL] This paper proposes CORE (Constraint-aware One-step REinforcement learning), a critic-free single-step RL framework that efficiently explores the joint hardware–mapping design space of DNN accelerators via structured distribution sampling, a scaling-graph decoder, and constraint-aware reward shaping, achieving at least 15× latency improvement across 7 DNN models.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - one-step RL
  - design space exploration
  - DNN accelerator
  - constraint-aware
  - scaling graph
date: 2026-05-08
content_hash: c0aec535105a6389
---

# CORE: Constraint-Aware One-Step Reinforcement Learning for Simulation-Guided Neural Network Accelerator Design

**Conference**: NeurIPS 2025
**arXiv**: [2506.03474](https://arxiv.org/abs/2506.03474)
**Code**: Not open-sourced
**Area**: Reinforcement Learning
**Keywords**: one-step RL, design space exploration, DNN accelerator, constraint-aware, scaling graph

## TL;DR
This paper proposes CORE (Constraint-aware One-step REinforcement learning), a critic-free single-step RL framework that efficiently explores the joint hardware–mapping design space of DNN accelerators via structured distribution sampling, a scaling-graph decoder, and constraint-aware reward shaping, achieving at least 15× latency improvement across 7 DNN models.

## Background & Motivation

**Background**: Simulation-based design space exploration (DSE) is critical for hardware–software co-design. Designing spatial DNN accelerators involves structured parameters such as PE count, buffer size, and mapping strategies (tiling, loop order, parallelism dimensions), resulting in an enormous search space with complex inter-parameter dependency constraints. For instance, tile sizes must satisfy hierarchical constraints $D_i \leq D_{i+1}$ across memory levels, and PE count bounds the degree of parallelism.

**Limitations of Prior Work**:
- Genetic algorithms (GA) and Bayesian optimization are inefficient in high-dimensional or sparse-feedback settings and lack effective constraint encoding mechanisms.
- Multi-step RL methods (e.g., ArchGym) model DSE as an MDP requiring long rollouts, suffering from sparse/delayed rewards, difficulty maintaining partial design states, and constraint violations caused by heuristic action masking.
- Two-stage methods such as HASCO first apply Bayesian optimization to search hardware configurations and then use heuristics or Q-learning for mapping optimization, ignoring the strong coupling between hardware and mapping.

**Key Challenge**: Joint optimization of hardware and mapping is necessary given their tight coupling, yet the joint space is prohibitively large and constraint-rich. Traditional methods either decouple the two objectives and sacrifice optimality, or perform exhaustive search with poor efficiency.

**Core Insight**: DSE can be modeled as a single-step MDP—the policy network generates a complete candidate configuration in one shot, eliminating the need to maintain intermediate states and rollouts, and naturally enabling batched parallel simulation. The key challenge is encoding inter-parameter dependency constraints within the single-step generation process.

**Core Idea**: A scaling graph encodes parameter dependencies to guarantee feasibility at sampling time, while intra-batch relative advantages replace a critic to enable efficient single-step policy optimization.

## Method

### Overall Architecture

The CORE pipeline operates in three cyclic stages: (1) the policy network outputs structured distribution parameters; (2) the scaling-graph decoder maps sampled actions to feasible design configurations; (3) a parallel simulator (MAESTRO) evaluates $E=32$ design points and returns metrics such as latency, area, and power; and (4) rewards and advantages are computed to update the policy. The input is a fixed context $s_0$ encoding DNN workload information, and the output is a complete configuration comprising hardware parameters and mapping strategies.

### Key Designs

1. **Single-Step MDP and Conditional Distribution Sampling**:

   - *Function*: Models design exploration as a single-step MDP $\mathcal{M}=(s_0, \mathcal{A}, R)$, where the policy generates a complete configuration in one forward pass.
   - *Mechanism*: The policy network $\pi_\theta$ outputs the joint distribution over all design parameters via conditional factorization $\pi_\theta(a_1, \ldots, a_N; s_0) = \prod_{i=1}^{N} f_{i,\theta}(a_i \mid a_{i+1} \ldots a_N; s_0)$. Discrete parameters (e.g., parallelism dimension, 6-way categorical) use a **Categorical distribution**; large discrete parameters (e.g., PE count with 512 choices) use a **Beta distribution** quantized to discrete values.
   - *Design Motivation*: This formulation eliminates reward propagation delays and intermediate state maintenance inherent to multi-step RL. The Beta distribution serves as a continuous relaxation, reducing the output dimensionality of the policy network.

2. **Scaling-Graph Decoder**:

   - *Function*: Encodes parameter dependencies via a directed graph and decodes sampled actions into feasible configurations following topological order.
   - *Mechanism*: Each node represents a design variable; directed edges encode constraint/scaling relationships. The decoded value of a source parameter dynamically constrains the range of its target parameters. For a sampled action $b \in [0,1]$ and source parameter values $\{A_i\}$, decoding proceeds as $B = B_{low} + \lfloor(\frac{\min_i\{A_i\} - B_{low}}{B_s} + 1) b \rfloor B_s$, replacing fixed upper bounds with $\min_i\{A_i\}$.
   - *Concrete Example*: The upper bound of parallelism $P_1$ is jointly constrained by PE count and the corresponding tile size: $P_1 = P_{low} + \lfloor(\min\{N_{pe}, X_2\} - P_{low} + 1) p_1 \rfloor$.
   - *Design Motivation*: Conventional methods either ignore dependencies (yielding many infeasible designs) or apply heuristic masking (non-differentiable and incomplete). The scaling graph guarantees feasibility at sampling time and is fully differentiable, supporting end-to-end training.
   - *Novelty vs. Prior Work*: ArchGym applies heuristic masking within a multi-step MDP to constrain the action space, which tends to miss complex transitive dependencies. The scaling graph is declarative; topological sorting automatically handles all transitive constraints.

3. **Constraint-Aware Reward Shaping**:

   - *Function*: A three-tier reward design that distinguishes normal designs, constraint violations, and anomalous (non-simulatable) designs.
   - *Mechanism*: Normal reward: $R(\xi_k) = \mathbf{w}^\top U(\xi_k)$; constraint violation: $R(\xi_k) = \mathbf{w}^\top U(\xi_k) - \alpha_c h(U(\xi_k))$, where $h$ quantifies the degree of violation; anomalous designs that cannot be simulated receive a penalty below the batch mean: $R_t(\xi') = \min(\mathbb{E}[R], \hat{R}_{t-1}) - \alpha_p \mathbb{E}[R]$.
   - *Design Motivation*: Assigning a fixed negative reward to invalid designs provides insufficient learning signal. Quantifying the degree of constraint violation enables the policy to learn *how much* a constraint is violated, rather than merely *whether* it is violated.

### Loss & Training

The **surrogate advantage function** uses intra-batch relative rewards (analogous to GRPO):

$$A_t(\xi_k) = R(\xi_k) - \hat{R}_t, \quad \hat{R}_t = \alpha_r \cdot \frac{1}{E}\sum_{k=1}^E R(\xi_k) + (1-\alpha_r)\hat{R}_{t-1}$$

The total objective $L(\theta_t) = L_{up} + L_r + L_e$ comprises:
- **Conditional update**: $L_{up} = \mathbb{E}[\frac{\pi_\theta}{\pi_{\theta_t}} A_t(\xi_k)]$ (importance-sampling ratio analogous to PPO)
- **KL regularization**: $L_r = -\beta_r \sum_i D_{KL}(\pi_{i,\theta} \| \pi_{i,\theta_t})$ (preventing excessively large updates)
- **Entropy regularization**: $L_e = \beta_e \sum_i \mathbb{H}(\pi_{i,\theta})$ ($\beta_e$ linearly decays from 1.0 to 0.02, encouraging early exploration and later exploitation)

A key property is the **absence of a critic**: no value function is learned; advantages are computed from intra-batch relative rewards. The policy network is a 4-layer MLP (512→4096→4096→4096→output), trained with the Adam optimizer at learning rate $10^{-5}$ for 2,000 episodes.

## Key Experimental Results

### Main Results (Latency, $\log_{10}$ cycles, lower is better)

| Model | GA | HASCO | CORE |
|---|---|---|---|
| ResNet-18 (Cloud) | 7.28 | 6.80 | **4.62** |
| ResNet-50 (Cloud) | 7.29 | 7.30 | **5.47** |
| MobileNetV2 (Cloud) | 7.01 | 6.79 | **4.33** |
| BERT (Cloud) | 7.31 | 6.85 | **5.60** |
| VGG-16 (Cloud) | 7.85 | 7.43 | **5.05** |

CORE substantially outperforms all baselines on all 7 DNN models, achieving an average latency reduction of **more than 15×** on the Cloud platform (a log-scale gap of 1.5–2.5, corresponding to 30×–300× in absolute terms). Performance on the Edge platform is similarly strong; e.g., MobileNetV2 latency decreases from 6.74 to 4.97.

### Ablation Study

| Configuration | ResNet-18 Latency | ResNet-50 Latency | Notes |
|---|---|---|---|
| w/o reward shaping | 6.68 | 7.31 | Constraint violation rate increases; multiple Edge-platform models yield no feasible solution ("—") |
| w/o scaling graph | 5.26 | 6.47 | Parameter dependencies ignored; sampling efficiency degrades |
| **CORE (full)** | **4.62** | **5.47** | Both components contribute synergistically |

### Key Findings
- The scaling graph contributes more: removing it causes a latency degradation of approximately 0.5–1.0 in log scale, underscoring the importance of modeling parameter dependencies for sampling quality.
- Reward shaping is especially critical on the Edge platform: under tighter area constraints (0.2 mm²), removing reward shaping renders the search entirely infeasible.
- Sampling efficiency: under a fixed budget of 40,000 evaluations, CORE converges within approximately 2,000 episodes (64,000 samples), whereas GA is far from convergence under the same budget.
- On small models (NCF, DLRM), HASCO remains competitive because the design space is smaller and two-stage optimization suffices.

## Highlights & Insights
- **Generality of the Scaling Graph**: The approach of encoding parameter dependencies via a directed graph and dynamically constraining ranges during decoding is broadly applicable beyond DNN accelerator design—e.g., compiler autotuning and chip floorplanning. The key insight is transforming constraint handling from "reject after generation" to "guarantee during generation," dramatically reducing invalid samples.
- **Advantages of Single-Step RL**: The formulation eliminates reward propagation delays, intermediate state maintenance, and Markov assumptions inherent to multi-step RL. For DSE problems—where evaluation is expensive but generating a configuration does not inherently require sequential decisions—the single-step formulation is a more natural fit.
- **Conceptual Alignment with GRPO/DeepSeek-R1**: Both CORE and GRPO independently arrive at the idea of replacing a critic with intra-batch relative advantages. This critic-free paradigm is particularly well-suited to simulation-expensive settings, where learning an accurate value function itself demands a large number of samples. The authors note that CORE and DeepSeek-R1 were developed independently yet converge on the same paradigm.

## Limitations & Future Work
- **Static Mapping Assumption**: The current formulation considers only compile-time fixed mapping strategies and does not support runtime-adaptive dataflow, which may be suboptimal under dynamic workload scenarios.
- **Dependence on Simulation Fidelity**: Performance relies on the accuracy of the MAESTRO cost model; post-silicon results may deviate, and no FPGA/ASIC physical validation is conducted.
- **Limitations of Single-Step Modeling**: For design problems requiring multi-stage decision-making (e.g., multi-chiplet system design, placement and routing), a single-step formulation may be insufficiently expressive.
- **Limited Advantage on Small Models**: HASCO remains competitive on NCF and DLRM, suggesting that CORE's structured modeling offers less benefit when the design space is small.
- **Directions for Improvement**: (1) Extend the scaling graph to a conditional graph supporting runtime dependencies; (2) incorporate a surrogate model to reduce simulator calls; (3) replace weighted-sum scalarization with multi-objective Pareto optimization.

## Related Work & Insights
- **vs. ArchGym**: ArchGym models DSE as standard multi-step RL requiring long rollouts and intermediate state maintenance, with constraints handled via heuristic masking. CORE generates complete configurations in a single step with constraint feasibility guaranteed by the scaling graph—simpler and more sample-efficient, at the cost of applicability to sequential decision problems.
- **vs. HASCO**: HASCO is a two-stage method (Bayesian optimization for hardware, then heuristic/Q-learning for mapping) that ignores hardware–mapping coupling. CORE optimizes both jointly, yielding pronounced advantages on large models (e.g., VGG-16 latency reduced by 2.4 orders of magnitude).
- **vs. DiGamma**: DiGamma performs joint optimization via heuristic search but does not scale well; CORE employs a learnable policy with stronger generalization.
- **Connection to GRPO in LLMs**: The critic-free, intra-batch baseline paradigm aligns with DeepSeek-R1's GRPO, applied here to hardware design rather than LLM training. This paradigm appears broadly effective for optimization problems characterized by expensive evaluation and large action spaces.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of single-step RL and a scaling graph for hardware DSE is novel, though individual components (Beta distribution sampling, critic-free optimization) are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 7 DNN models × 2 platforms × 2 metrics with complete ablations; lacks physical chip validation and comparison with a broader set of RL baselines.
- **Writing Quality**: ⭐⭐⭐⭐ — Architecture diagrams and algorithm pseudocode are clear; the scaling graph explanation is intuitive; the background section is somewhat lengthy.
- **Value**: ⭐⭐⭐⭐ — Achieves substantial improvements on an important hardware DSE problem; the scaling graph idea is transferable to other structured optimization domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scalable Neural Incentive Design with Parameterized Mean-Field Approximation](scalable_neural_incentive_design_with_parameterized_mean-field_approximation.md)
- [\[NeurIPS 2025\] Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning](counteractive_rl_rethinking_core_principles_for_efficient_and_scalable_deep_rein.md)
- [\[AAAI 2026\] One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow](../../AAAI2026/reinforcement_learning/one-step_generative_policies_with_q-learning_a_reformulation_of_meanflow.md)
- [\[NeurIPS 2025\] Reward-Aware Proto-Representations in Reinforcement Learning](reward-aware_proto-representations_in_reinforcement_learning.md)
- [\[NeurIPS 2025\] Adaptive Cooperative Transmission Design for URLLC via Deep RL](adaptive_cooperative_transmission_design_for_ultra-reliable_low-latency_communic.md)

</div>

<!-- RELATED:END -->
