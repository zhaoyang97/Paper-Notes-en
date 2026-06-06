---
title: >-
  [Paper Note] Spatial-Aware Decision-Making with Ring Attractors in Reinforcement Learning Systems
description: >-
  [NeurIPS 2025][Reinforcement Learning][Ring Attractors] This paper integrates ring attractor models from neuroscience into action selection in deep reinforcement learning (DRL). By mapping actions to spatial positions on…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Ring Attractors"
  - "Biologically Inspired RL"
  - "Spatial Awareness"
  - "Action Space Encoding"
  - "Uncertainty Quantification"
date: 2026-05-08
content_hash: 2d2bd189a8be5b8e
---

# Spatial-Aware Decision-Making with Ring Attractors in Reinforcement Learning Systems

**Conference**: NeurIPS 2025
**arXiv**: [2410.03119](https://arxiv.org/abs/2410.03119)  
**Code**: [https://github.com/marcosaura/RA_RL](https://github.com/marcosaura/RA_RL)  
**Area**: Reinforcement Learning
**Keywords**: Ring Attractors, Biologically Inspired RL, Spatial Awareness, Action Space Encoding, Uncertainty Quantification

## TL;DR
This paper integrates ring attractor models from neuroscience into action selection in deep reinforcement learning (DRL). By mapping actions to spatial positions on a ring and injecting Gaussian signals encoding Q-values and uncertainty, the proposed approach achieves a 53% improvement over baseline on Atari 100K.

## Background & Motivation

**Background**: Fast and efficient action selection remains a core challenge in DRL, particularly in environments with spatial structure (e.g., coupled joint movements in robotic manipulation, adjacent directional actions in games). Existing methods treat actions as orthogonal, independent one-hot vectors, completely ignoring the topological relationships among actions.

**Limitations of Prior Work**:
   - Standard DQN represents actions as orthogonal vectors, failing to reflect the fact that "left" and "upper-left" are closer to each other than "left" and "right."
   - Existing spatially-aware methods (relational RL, cognitive maps, etc.) rely on complex architectures to implicitly learn spatial understanding from data, requiring large amounts of samples.
   - Uncertainty quantification methods (e.g., Bootstrapped DQN) treat uncertainty as an independent module, without integrating it with spatial structure.

**Key Challenge**: Action spaces possess inherent topological structure, yet this structural information is entirely discarded in standard DRL.

**Key Insight**: The ring attractor circuit found in the central complex of *Drosophila* is an experimentally validated neural circuit capable of stably encoding directional and spatial information.

**Core Idea**: The ring attractor serves as the "brain" for action selection — Q-values are transformed into Gaussian input signals on the ring (amplitude = Q-value, angle = action direction, width = uncertainty), and excitatory-inhibitory dynamics select the optimal action.

## Method

### Overall Architecture
Two implementations are proposed: (1) **Extrinsic model** — a ring attractor implemented via a continuous-time recurrent neural network (CTRNN), serving as a post-processing module at the DQN output layer to replace the standard argmax for action selection; (2) **Integrated model** — a reusable RNN-based deep learning module embedded within the DRL agent for end-to-end training.

### Key Designs

1. **Ring Attractor Architecture (Touretzky Model)**:

    - Function: $N$ excitatory neurons arranged in a ring plus one central inhibitory neuron, with distance-decaying synaptic connections implementing local excitation and global inhibition.
    - Core equation: Excitatory neuron dynamics $\frac{dv_n}{dt} = \frac{f(x_n + \epsilon_n + \eta_n)}{\tau} - v_n$, where $x_n$ is the external input, $\epsilon_n$ is excitatory feedback, and $\eta_n$ is inhibitory feedback.
    - Synaptic weights decay with distance: $w^{(E_m \rightarrow E_n)} = e^{-d^2_{(m,n)}}$
    - Design Motivation: The excitatory-inhibitory dynamics produce a winner-take-all effect — the activity peak stabilizes near the action with the highest Q-value, modulated by neighboring high-value actions, enabling smooth spatial reasoning.

2. **Mapping Q-Values to Ring Inputs**:

    - Function: Transforms DQN output Q-values into Gaussian input signals for the ring attractor.
    - Core equation: $x_n(Q(s,a)) = \sum_{a=1}^{A} \frac{Q(s,a)}{\sqrt{2\pi\sigma_a}} \exp(-\frac{(\alpha_n - \alpha_a(a))^2}{2\sigma_a^2})$
    - Three key parameters: $K_i = Q(s,a)$ (amplitude = action value), $\mu_i = \alpha_a(a)$ (angle = action position on the ring), $\sigma_i = \sigma_a$ (width = uncertainty of value estimate).
    - Design Motivation: Actions with higher Q-values generate stronger input signals on the ring; after excitatory-inhibitory dynamics, spatially adjacent high-value actions mutually reinforce each other — this is the core mechanism by which spatial information is exploited.

3. **Bayesian Uncertainty Injection**:

    - Function: Bayesian linear regression (BLR) is used as the DQN output layer, naturally providing the Q-value variance for each action.
    - Mechanism: $Q(s,a) = \Phi_\theta(s)^T w_a$, where $w_a$ is drawn from the BLR posterior. The Gaussian signal width $\sigma_a$ is directly set to the BLR posterior variance.
    - Design Motivation: Actions with high uncertainty yield more "diffuse" signals (wide Gaussians), while low-uncertainty actions yield sharper signals (narrow Gaussians) — automatically achieving exploration-exploitation balance.

4. **Integrated DL Module (Reusable RNN)**:

    - Function: Implements ring attractor dynamics using GRU/LSTM as a plug-and-play component within the DRL framework.
    - Mechanism: Recurrent layers simulate the temporal evolution of the ring attractor; inputs are Q-value sequences and outputs are action selections.
    - Design Motivation: CTRNN requires manual configuration of iteration steps (~50 steps) and is non-differentiable; the RNN implementation supports end-to-end training and is more efficient.

### Loss & Training
- Extrinsic model: The Q-network is trained with standard DQN loss; the ring attractor performs action selection without participating in gradient computation; the BLR output layer updates its posterior online.
- Integrated model: End-to-end training, with the ring attractor RNN module participating in backpropagation.

## Key Experimental Results

### Main Results — Atari 100K Benchmark (Extrinsic Model)

| Method | Median Human-Normalized Score (MHNS) | Mean MHNS | Superhuman Games |
|--------|--------------------------------------|-----------|-----------------|
| DQN baseline | ~50% | ~45% | 2/26 |
| DQN + Ring Attractor (w/o UQ) | ~72% | ~68% | 5/26 |
| DQN + Ring Attractor (**w/ UQ**) | **~80%** | **~75%** | **8/26** |
| Gain | - | **+53%** | - |

### Ablation Study — Contribution of Each Component

| Configuration | MHNS | Gain over Baseline | Notes |
|---------------|------|--------------------|-------|
| DQN baseline | ~50% | - | Standard argmax selection |
| +Ring Attractor (spatial structure only) | ~68% | +35% | Value of spatial topology |
| +Ring Attractor + Bayesian UQ | **~80%** | **+53%** | Uncertainty integration adds +18% |
| +Ring Attractor + varying ring size | ~65–78% | Varies | N=32 neurons is optimal |

### Integrated DL Model vs. Extrinsic Model

| Method | Training Speed | Final Performance | Scalability |
|--------|---------------|-------------------|-------------|
| Extrinsic CTRNN | Slow (~50 iterations required) | High | Limited |
| Integrated RNN | **Fast** | **Comparable/slightly higher** | **Good** |

### Key Findings
- The ring attractor yields the largest gains in games with **spatially structured action spaces** (directional movement games > discrete selection games). For example, improvements are substantial on Pong (continuous directionality) but limited on Montezuma's Revenge (exploration-intensive).
- The **additional 18% gain from uncertainty injection** is complementary to Thompson Sampling-style exploration: high-uncertainty actions produce wider Gaussian signals that "diffuse" into neighboring actions, equivalent to exploration.
- The **temporal filtering effect** of the ring attractor smooths the action selection sequence — reducing frequent oscillation between two similarly valued actions, which is particularly valuable in physical control settings.
- The number of neurons $N$ on the ring requires careful selection: too few leads to insufficient information; too many incurs excessive computational overhead. $N=32$ (approximately $4\times$ the action space size) is optimal across most games.

## Highlights & Insights
- **Introducing biological neural circuits as computational primitives in RL** represents a uniquely interdisciplinary perspective — ring attractors have been experimentally validated in the head-direction system of *Drosophila*, and leveraging this computational structure refined over millions of years of evolution reflects a deep methodological commitment to borrowing from biology.
- The elegance of the three-in-one design: Gaussian amplitude = Q-value (exploitation), angle = spatial position (structure), width = uncertainty (exploration) — a single formula simultaneously encodes three critical types of information.
- The method is **plug-and-play** — it only requires appending a ring attractor module at the DQN output, with no modification to the Q-network itself.
- The "winner-take-all + spatial diffusion" effect produced by excitatory-inhibitory dynamics is naturally suited to continuous or spatially structured action selection.

## Limitations & Future Work
- The mapping from actions to the ring assumes a **circular topology** among actions — not all action spaces satisfy this (e.g., hierarchical action spaces with tree-like structure).
- The CTRNN model requires approximately 50 iterations to converge to a steady state, introducing inference latency — this may be problematic in real-time control scenarios.
- Validation is currently limited to discrete action spaces — continuous action spaces would require discretizing the ring or extending the framework to higher-dimensional attractors.
- The BLR posterior update may not adapt quickly enough in non-stationary environments — forgetting mechanisms or online Bayesian methods should be considered.

## Related Work & Insights
- **vs. Bootstrapped DQN**: Bootstrapped DQN treats uncertainty as an independent module for Thompson Sampling-based exploration; the ring attractor unifies uncertainty and spatial structure within a single signal.
- **vs. Grid Cells (DeepMind)**: Grid cells encode positional information in state space; ring attractors encode structural information in action space — both are biologically inspired, but their encoding targets differ.
- **vs. Relational DRL**: Relational RL implicitly learns inter-entity relationships via attention mechanisms; the ring attractor provides spatial inductive bias through explicit ring topology — the latter is more sample-efficient.
- Future directions: Extending the ring attractor to multi-dimensional attractors (e.g., toroidal structures) for encoding high-dimensional continuous action spaces.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A unique and deep intersection of neuroscience and RL; using ring attractors for action space encoding has never been proposed before.
- Experimental Thoroughness: ⭐⭐⭐⭐ Standard Atari 100K benchmark, two implementations, and ablation analysis, though continuous control tasks are absent.
- Writing Quality: ⭐⭐⭐⭐ Detailed biological background and complete mathematical derivations, though the paper is lengthy.
- Value: ⭐⭐⭐⭐ Spatial action encoding is a promising direction, and the plug-and-play design is practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[NeurIPS 2025\] Modulation of Temporal Decision-Making in a Deep Reinforcement Learning Agent under the Dual-Task Paradigm](modulation_of_temporal_decision-making_in_a_deep_reinforcement_learning_agent_un.md)
- [\[NeurIPS 2025\] Reward-Aware Proto-Representations in Reinforcement Learning](reward-aware_proto-representations_in_reinforcement_learning.md)
- [\[NeurIPS 2025\] Blending Complementary Memory Systems in Hybrid Quadratic-Linear Transformers](blending_complementary_memory_systems_in_hybrid_quadratic-linear_transformers.md)
- [\[NeurIPS 2025\] Prompt Tuning Decision Transformers with Structured and Scalable Bandits](prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)

</div>

<!-- RELATED:END -->
