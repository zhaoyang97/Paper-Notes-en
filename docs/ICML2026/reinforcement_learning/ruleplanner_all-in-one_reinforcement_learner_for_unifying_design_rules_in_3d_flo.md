---
title: >-
  [Paper Note] RulePlanner: All-in-One Reinforcement Learner for Unifying Design Rules in 3D Floorplanning
description: >-
  [ICML 2026][Reinforcement Learning][3D Floorplanning] This paper unifies seven industrial design rules for 3D chip floorplanning into an actor-critic RL framework. The core mechanism compiles each rule into a $W \times H…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "3D Floorplanning"
  - "Design Rules"
  - "Action Space Constraints"
  - "Masked RL"
  - "Hybrid PPO"
date: 2026-05-08
content_hash: 3ecdcab37784751c
---

# RulePlanner: All-in-One Reinforcement Learner for Unifying Design Rules in 3D Floorplanning

**Conference**: ICML 2026  
**arXiv**: [2601.22476](https://arxiv.org/abs/2601.22476)  
**Code**: https://github.com/Thinklab-SJTU/EDA-AI  
**Area**: Reinforcement Learning / Chip Design / Combinatorial Optimization  
**Keywords**: 3D Floorplanning, Design Rules, Action Space Constraints, Masked RL, Hybrid PPO  

## TL;DR
This paper unifies seven industrial design rules for 3D chip floorplanning into an actor-critic RL framework. The core mechanism compiles each rule into a $W \times H$ "adjacency matrix mask," which directly blocks illegal positions using a large negative value before the policy softmax. Combined with a hybrid action space (discrete position + continuous aspect ratio) and Transformer-encoded netlist features, this approach allows a single agent to satisfy seven rules simultaneously—including boundary, grouping, cross-layer alignment, and non-overlap—while demonstrating zero-shot transferability to unseen circuits.

## Background & Motivation

**Background**: In advanced process nodes (e.g., 5 nm), floorplanning is no longer just "placing blocks." It involves determining the coordinates $(x_i, y_i)$, layer $z_i$, and dimensions $(w_i, h_i)$ for each functional module while satisfying a long list of hardware design rules: boundary constraints (blocks must be near specific terminals), grouping constraints (blocks must form physical regions like voltage islands), cross-layer alignment, pre-placed blocks, non-overlap, contour, and shape constraints. Existing solutions fall into three categories: analytical methods (gradient descent like NTUplace/DREAMPlace), heuristics (B*-tree, Simulated Annealing), and RL-based methods (GraphPlace, MaskPlace, FlexPlanner).

**Limitations of Prior Work**: The authors compare representative methods in Table 1 and find that no single category "checks all boxes." Analytical methods rely on differentiable objectives, yet rule violation penalties are inherently non-differentiable. Heuristics rely on manually designed cost terms, which fail under complex rules. Most RL-based methods insert penalties into the reward function, treating hard constraints as soft ones, which requires labor-intensive manual post-processing by engineers to fix violations.

**Key Challenge**: Treating rules as rewards is equivalent to "post-hoc punishment" rather than "pre-emptive prohibition." In a discrete position space of magnitude $W \times H$, the vast majority of sampled actions are illegal, making it difficult for agents to converge using reward signals alone. Furthermore, rules often conflict (e.g., boundary vs. grouping, alignment vs. non-overlap), making it nearly impossible for a single-weighted reward to drive all violations to zero.

**Goal**: To enable the RL agent to perceive only legal positions at the moment of action sampling, ensuring the "filtering" mechanism is extensible to an arbitrary number of rules.

**Key Insight**: The authors observe that all spatial design rules can be rewritten as "outputting a scalar for every candidate position $(x, y)$." For instance, a boundary rule measures how far a block at $(x,y)$ is from a terminal, and a grouping rule measures the shared edge length with group members. This naturally forms a $W \times H$ matrix that can be computed in parallel, thresholded, and combined using logical AND operations.

**Core Idea**: Compile each rule into a matrix mask $\to$ compute a bitwise AND to obtain a global "availability mask" $\bm M$ $\to$ apply $(\bm M-1) \odot 10^8$ to the policy logits for a hard-masked softmax. This embeds hard constraints into the action space itself while optimizing continuous metrics (distance, alignment scores) as soft objectives via the reward function.

## Method

### Overall Architecture

RulePlanner models 3D floorplanning as an episodic MDP. At each step, a candidate block $b_t$ is selected. The state $s_t$ includes several $W \times H$ rule matrices, the canvas image, and the netlist graph $(\bm G, \bm E)$. The action $a_t = (x_t, y_t, \mathrm{AR}_{t+1})$ is hybrid, consisting of discrete 2D coordinates and a continuous aspect ratio for the **next block** (deciding the next shape early is necessary because $(w_{t+1}, h_{t+1})$ is required to recompute rule matrices for the next state). The workflow involves: rule matrices + netlist graph $\to$ CNN/Transformer encoding $\to$ policy network logits $\to$ filtering violations via the availability mask $\to$ sampling positions $\to$ sampling aspect ratio from a Gaussian and clipping to $[\mathrm{AR}_\min, \mathrm{AR}_\max]$ $\to$ reward calculation $\to$ Hybrid PPO update.

### Key Designs

1. **Matrix Representation of Design Rules (Adjacent Terminal / Block Mask)**:
    - **Function**: Uniformly encodes whether a block satisfies a rule at $(x, y)$ into a $W \times H$ scalar matrix, allowing modular combinations of rules.
    - **Mechanism**: Quantifiable metrics are defined for each rule. Boundary rules use block-terminal distance $d(b_i, t_j) = \min_{seg \in Segs(b_i)} d_m(seg, t_j)$ to form the adjacent terminal matrix $\bm T_{xy} = d(b, t)$. Grouping rules use block-to-block adjacency edge length $l(b_i, b_j)$ to form the adjacent block matrix $\bm B_{xy} = l(b_i, b_j)$. Cross-layer alignment utilizes the alignment score matrix $\bm A$ from FlexPlanner. Non-overlap and boundary violations are encoded as a position mask $\bm P \in \{0,1\}^{W \times H}$. Multiple constraints are merged: $\bm T^{(i)}_{xy} = \max_j \bm T^{(ij)}_{xy}$ for mandatory terminal alignment or $\min$ for optional ones; $\bm B^{(i)}_{xy} = \sum_j \bm B^{(ij)}_{xy}$ for voltage islands. All matrices are computed in parallel using GPU meshgrid operators to optimize $\mathcal{O}(WH)$ complexity.
    - **Design Motivation**: Previous methods either embedded rules in rewards or required specialized algorithmic modifications for every new rule. By unifying them into "matrices + merging operators," new rules only require a "position-to-scalar" mapping, making extensibility the primary advantage of this framework.

2. **Action Space Constraints via Availability Mask + Hard-masked Softmax**:
    - **Function**: Directly eliminates all violating discrete positions **before** policy sampling, converting hard constraints into zero probability rather than negative rewards.
    - **Mechanism**: Four matrices are binarized using thresholds $\bar t, \bar b, \bar a$: $\bar{\bm T}_{xy} = \mathbb{1}[\bm T_{xy} \le \bar t]$ (closeness), $\bar{\bm B}_{xy} = \mathbb{1}[\bm B_{xy} \ge \bar b]$ (contact length), $\bar{\bm A}_{xy} = \mathbb{1}[\bm A_{xy} \ge \bar a]$ (alignment), and $\bar{\bm P} = \bm P$ (basic legality). The total mask is $\bm M = \bar{\bm T} \odot \bar{\bm B} \odot \bar{\bm A} \odot \bar{\bm P}$. Logits for violated positions are pushed to $-\infty$ via $\bar p_\theta = \mathrm{softmax}(p_\theta + (\bm M - 1) \odot 10^8)$. For the continuous aspect ratio, the policy outputs $\mu_\theta \in [-1,1]$ (tanh) and $\sigma_\theta$. Sampling $z \sim \mathcal{N}(\mu_\theta, \sigma_\theta^2)$ is followed by an affine transformation $\bar z = \frac{z+1}{2}(\mathrm{AR}_\max - \mathrm{AR}_\min) + \mathrm{AR}_\min$ and clipping to ensure shape rules are perpetually satisfied.
    - **Design Motivation**: In a $W \times H$ space, legal positions often account for only 1%. Relying on RL to learn "where not to place" is inefficient and fails to satisfy all seven rules. By encoding rules into the action space, the agent's exploration budget is focused entirely on the legal subset, reducing training time and violation rates by orders of magnitude.

3. **Multi-modal State Encoding + Hybrid PPO Training**:
    - **Function**: Enables a single policy to process heterogeneous signals (spatial rule matrices and netlist graphs) and jointly optimize discrete position and continuous aspect ratio actions.
    - **Mechanism**: Rule matrices, canvas images, and wire masks are concatenated as channels for a CNN to extract spatial features $\bm y_v$. Netlist graphs $(\bm G, \bm E)$ are encoded via a Transformer. Node features $\bm g_j = [x_j, y_j, z_j, w_j, h_j, a_j, p_j]$ are injected with placement order $\bm o$ via FC + sinusoidal positional encoding. Self-attention masks are derived from $1 - \bm E$ to restrict message passing to topological neighbors. Features $\bm e_{local} = \mathrm{FC}(\bm H_2[\mathrm{idx}])$ and $\bm e_{global} = \mathrm{FC}(\mathrm{AvgPool}(\bm H_2))$ are concatenated with $\bm y_v$ for the policy/value heads. Training uses Hybrid PPO with separate clip objectives $L(\theta) = \sum_{k=1}^2 \lambda_k \hat{\mathbb{E}}_t[\min(r_t^{(k)}\hat A_t, \bar r_t^{(k)}\hat A_t)]$, where $\hat A_t$ is estimated using GAE. The reward function combines adjacency length, terminal distance, alignment, HPWL, and overlap using self-adaptive normalization.
    - **Design Motivation**: Topological information and spatial rule matrices are fundamentally different modalities; relying solely on CNNs or GNNs loses critical information. Hybrid PPO ensures gradients from continuous aspect ratios do not destabilize the discrete position policy.

### Loss & Training
The policy objective is a weighted sum of position and aspect ratio branches with PPO clipping. The critic optimizes $L(\phi) = \lambda_\phi \hat{\mathbb{E}}_t[(G_t - V_\phi(s_t))^2]$ using GAE. Rewards are scaled via adaptive normalization to ensure hard constraints (enforced by masks) and soft objectives (HPWL, overlap) are optimized in tandem.

## Key Experimental Results

### Main Results
On public 3D floorplanning benchmarks, RulePlanner is compared against analytical (NTUplace), heuristic (B*-3D-SA / WireMask-BBO), and RL baselines (GraphPlace, DeepPlace, MaskPlace, FlexPlanner). RulePlanner is the **only** method in Table 1 to satisfy all seven rules (a-g).

| Method | Type | (a) Bndry | (b) Group | (c) Align | (d) Pre-pl | (e) Overlap | (f) Contour | (g) Shape |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Analytical (Huang 2023) | Analytical | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ |
| B*-3D-SA | Heuristic | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| WireMask-BBO | Heuristic | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ |
| GraphPlace | RL | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ |
| MaskPlace | RL | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ |
| FlexPlanner | RL | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **RulePlanner (Ours)** | **RL** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

RulePlanner outperforms the strongest RL baseline, FlexPlanner, in HPWL, violation rates, and runtime, while exhibiting zero-shot transferability to unseen circuits.

### Ablation Study

| Configuration | Rule Satisfaction | HPWL / Overlap | Description |
| :--- | :--- | :--- | :--- |
| Full RulePlanner | All 7 satisfied | Lowest | Mask + Hybrid PPO + Transformer |
| w/o Adjacency Mask | Boundary/Group violations | Significantly worse | Soft constraints fail the rules |
| w/o Netlist Transformer | Rules satisfied | Higher HPWL | Poor routing due to missing topology |
| w/o AR Clipping | Shape violations | Unstable | Reward penalties insufficient for shapes |

### Key Findings
- **Masking > Reward Penalty**: Removing adjacency masks allows the policy to sample violating positions, necessitating post-processing. This confirms the necessity of hard-coding hard constraints.
- **Topology Impacts Routing**: Removing the Transformer does not affect rule satisfaction (guaranteed by masks) but significantly increases HPWL, indicating that netlist adjacency is crucial for learning module proximity.
- **Zero-Shot Transferability**: The model maintains hard constraint satisfaction on unseen circuits without retraining because the masks are universally applicable to any circuit geometry.

## Highlights & Insights
- Abstracting "spatial hard constraints" into "$W \times H$ matrices + thresholding + bitwise AND" provides a clean interface. New rules can be added by simply writing a "position-to-scalar" function without altering the training loop. This "Constraint as Matrix" approach is applicable to PCB routing and robot path planning.
- Utilizing "$-\infty$ in softmax" is an engineering-friendly trick. Unlike IPO or Lagrangian multipliers that require hyperparameter tuning, hard masking ensures zero violations with no extra parameters—a critical requirement for industrial EDA.
- Implementing the "aspect ratio for the next block" as part of the current action solves the circular dependency where the next state requires $(w_{t+1}, h_{t+1})$ to compute rule matrices. This "one-step ahead" design is highly effective for sequential decision tasks.

## Limitations & Future Work
- The hard-masking approach assumes all rules can be mapped to a "position-to-scalar" form. Non-spatial constraints (e.g., thermal, signal integrity) require specialized encoding.
- Thresholds $\bar t, \bar b, \bar a$ are manual hyperparameters. Their sensitivity across different process nodes or circuit scales was not systematically discussed.
- Sequential decision-making for large circuits (hundreds of blocks) leads to expanding episode lengths, increasing training costs compared to one-shot analytical optimization.
- For complex grouping structures, the selection of merging operators ($\sum$ vs $\max$) is manual; automating this is an open problem.

## Related Work & Insights
- **vs. MaskPlace (2022)**: MaskPlace introduced masks to block illegal positions but only handled non-overlap/contour. RulePlanner extends this to seven industrial rules and incorporates Transformer encoding.
- **vs. FlexPlanner (2024)**: FlexPlanner previously handled alignment and shapes via RL, but relied on rewards for boundaries and grouping. RulePlanner unifies all seven into the masking system.
- **vs. Analytical Methods**: Analytical methods are unable to handle non-differentiable rules. RulePlanner uses RL + masking to satisfy hard constraints precisely, though it lacks the mass-scale optimization speed of analytical solvers; they are complementary in industrial flows.

## Rating
- Novelty: ⭐⭐⭐⭐ Unifying seven rules into a masked framework with a clear extensible interface.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison with seven baselines and cross-circuit transfer tests.
- Writing Quality: ⭐⭐⭐⭐ Clear definitions of rules, derivations, and merging operators.
- Value: ⭐⭐⭐⭐ Direct industrial utility for EDA; the methodology is highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](../../ICLR2026/reinforcement_learning/one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)
- [\[NeurIPS 2025\] CORE: Constraint-Aware One-Step Reinforcement Learning for Simulation-Guided Neural Network Accelerator Design](../../NeurIPS2025/reinforcement_learning/core_constraint-aware_one-step_reinforcement_learning_for_simulation-guided_neur.md)
- [\[AAAI 2026\] Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework](../../AAAI2026/reinforcement_learning/distilling_deep_reinforcement_learning_into_interpretable_fuzzy_rules_an_explain.md)
- [\[ICML 2026\] One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models](one_bias_after_another_mechanistic_reward_shaping_and_persistent_biases_in_langu.md)
- [\[ICML 2026\] RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search](rl4rla_teaching_ml_to_discover_randomized_linear_algebra_algorithms_through_curr.md)

</div>

<!-- RELATED:END -->
