---
title: >-
  [Paper Note] Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models
description: >-
  [NeurIPS 2025][Multi-Agent][Multi-Agent Reinforcement Learning] This paper proposes an "intention communication" architecture based on lightweight world models, enabling multi-agent coordination by generating and sharing future trajectory plans. The approach comprehensively outperforms end-to-end emergent communication methods in both scalability and performance.
tags:
  - "NeurIPS 2025"
  - "Multi-Agent"
  - "Multi-Agent Reinforcement Learning"
  - "World Models"
  - "Communication Protocol"
  - "Intention Communication"
  - "Dec-POMDP"
date: 2026-05-08
content_hash: e894953a23164016
---

# Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models

**Conference**: NeurIPS 2025
**arXiv**: [2508.02912](https://arxiv.org/abs/2508.02912)  
**Code**: To be confirmed  
**Area**: Reinforcement Learning
**Keywords**: Multi-Agent Reinforcement Learning, World Models, Communication Protocol, Intention Communication, Dec-POMDP

## TL;DR

This paper proposes an "intention communication" architecture based on lightweight world models, enabling multi-agent coordination by generating and sharing future trajectory plans. The approach comprehensively outperforms end-to-end emergent communication methods in both scalability and performance.

## Background & Motivation

Multi-agent coordination is critical in scenarios such as autonomous driving and warehouse robotics. Multi-Agent Reinforcement Learning (MARL) faces a core challenge: since the actions of other agents alter state transition dynamics, the environment is non-stationary from the perspective of any individual agent, violating the Markov property assumed by single-agent algorithms.

Communication can alleviate this issue, but a fundamental debate exists over how to design effective communication protocols:
- **Emergent Communication**: Protocols are learned from scratch—general-purpose but sample-intensive, with fragile and uninterpretable conventions.
- **Engineered Communication**: Structural inductive biases are introduced under the assumption that structured reasoning is more efficient for complex coordination.

This paper reframes the debate as a comparison between **implicit world modeling (emergent protocols) vs. explicit world modeling (engineered protocols)**.

## Method

### Overall Architecture

The problem is formulated as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP), defined by the tuple $(\mathcal{I}, \mathcal{S}, \{\mathcal{A}_i\}, T, R, \{\Omega_i\}, O)$, where two agents in a partially observable grid world must each navigate to one of two distinct goals.

Two communication strategies are proposed and compared:
1. **Learned Direct Communication (LDC)**: End-to-end emergent communication.
2. **Intention Communication**: Engineered intention communication based on a world model.

### Key Designs

**Method 1: LDC (Emergent Communication)**

Each agent's policy network jointly outputs an action and a message: $\pi_{\theta_i}(a_t^{(i)}, m_t^{(i)} | o_t^{(i)}, m_{t-1}^{(j)})$

The message space is minimal—a single binary signal $m_t^{(i)} \in \{0, 1\}$ sampled via sigmoid. The entire system is trained end-to-end, with gradients flowing back from the shared task reward.

**Method 2: Intention Communication (Core Contribution)**

This approach consists of two key modules:

**(a) Imagined Trajectory Generation Module (ITGM)**: A lightweight learned world model.

1. Initial encoding: The current observation and received message are encoded into a latent state:
$$z_t^{(i)} = \text{ReLU}(\mathbf{W}_{enc}[o_t^{(i)} \oplus \text{embed}(m_{t-1}^{(j)})] + \mathbf{b}_{enc})$$

2. Latent-space rollout: Future trajectories are iteratively generated over a fixed horizon $H$:
$$\tilde{a}_{t+k} \sim \pi_{act}(\cdot | z_{t+k}^{(i)})$$
$$z_{t+k+1}^{(i)} = \text{ReLU}(\mathbf{W}_{trans}[z_{t+k}^{(i)} \oplus \text{embed}(\tilde{a}_{t+k})] + \mathbf{b}_{trans})$$

3. Output trajectory plan: $\tau^{(i)} = \langle z_{t+1}^{(i)}, z_{t+2}^{(i)}, \ldots, z_{t+H}^{(i)} \rangle$

**(b) Message Generation Network (MGN)**: Compresses the trajectory into a structured message.

A multi-head self-attention mechanism processes the trajectory sequence; the output is aggregated via mean pooling, then passed through a two-layer MLP with $N_{comp}$ independent classification heads to produce a structured message:

$$\pi_{msg}^{(k)} = \text{Softmax}(\mathbf{W}_{head}^{(k)} h_2 + b_{head}^{(k)})$$

The message is a composite vector of $N_{comp}$ independent categorical distributions, substantially more expressive than the single bit used in LDC.

### Loss & Training

The system is trained end-to-end using the Advantage Actor-Critic (A2C) objective, comprising:
- Actor loss: $L_{actor} = A_t \log \pi_{act}(a_t) + A_t \sum_k \log \pi_{msg}^{(k)}(m_t^{(k)})$
- Critic loss: TD error from the value network.
- Linear learning rate decay: $\text{lr}_e = \text{lr}_0 (1 - e / E_{total})$

Reward design:
- Success (both agents reach different goals): $+1.0$
- Collision (both agents at the same goal): $-0.10$
- Time penalty (per step): $-0.01$

## Key Experimental Results

### Main Results

Three methods are compared in a partially observable environment (vision\_range=2):

| Environment Scale | Method | Success Rate |
|---|---|---|
| 10×10 | No-communication baseline | 0.0% |
| 10×10 | LDC (Emergent Communication) | 30.8% |
| 10×10 | **Intention Communication** | **99.9%** |
| 15×15 | No-communication baseline | 0.0% |
| 15×15 | LDC (Emergent Communication) | 12.2% |
| 15×15 | **Intention Communication** | **96.5%** |

### Ablation Study

LDC communication ablation on a fully observable 6×6 grid:

| Condition | Success Rate | Avg. Steps |
|---|---|---|
| With emergent messages | 89.4% | 4.39 |
| Message ablated (set to 0) | 88.6% | 4.43 |

Partially observable 6×6 environment:

| Condition | Success Rate |
|---|---|
| With emergent messages | 31.89% |
| Message ablated | 30.26% |

Conditional probability analysis confirms that LDC messages do influence behavior: when Agent 1 receives message "0", its probability of moving upward is 13.54%, compared to only 4.53% when receiving "1".

### Key Findings

1. **Large scalability gap**: When the environment scales from 10×10 to 15×15, the LDC success rate plummets from 30.8% to 12.2%, whereas intention communication drops only marginally from 99.9% to 96.5%.
2. **Complete failure of the no-communication baseline**: Success rate is 0% across all partially observable settings, confirming the necessity of communication.
3. Although the single-bit LDC message produces meaningful statistical dependencies, it carries insufficient information to handle complex environments.
4. Intention communication decouples the learning problem—ITGM learns a simple forward dynamics model, while the policy learns to make decisions based on both agents' plans.

## Highlights & Insights

- **Core Insight**: In multi-agent coordination, **sharing plans is more effective than sharing percepts**. Agents should leverage world models to anticipate the future and communicate intentions, rather than passively transmitting raw observations.
- **Advantage of Decoupled Design**: The credit assignment problem is substantially simplified—messages are directly associated with imagined trajectories, enabling efficient gradient flow to optimize ITGM and MGN independently.
- **Practical Implication**: Under resource-constrained conditions (Google Colab), engineered inductive biases demonstrate greater sample efficiency and scalability than general-purpose emergent approaches.

## Limitations & Future Work

1. Validation is limited to simple deterministic grid worlds; complex environments such as continuous control or physics simulation are not explored.
2. The fixed two-agent, two-goal setup does not evaluate scalability to scenarios with more than two agents.
3. Experiments are conducted on Google Colab under limited computational resources, constraining model and environment scale.
4. ITGM employs simple linear layers with ReLU activations as the transition model, limiting its capacity to model complex dynamics.
5. No comparison is made against established MARL methods such as QMIX or MAPPO.

## Related Work & Insights

- **Relation to the Dreamer series**: ITGM can be viewed as a simplified variant of RSSM, demonstrating that even lightweight world models suffice to produce effective intention communication.
- **Relation to DIAL/CommNet**: LDC represents the classic emergent communication paradigm; this paper systematically exposes its scalability bottleneck.
- **Future Directions**: Extending intention communication to continuous action spaces and heterogeneous multi-agent scenarios with larger agent populations; integrating stronger world models such as RSSM to improve ITGM's predictive capacity.

## Rating

- ⭐ Novelty: 4/5 — World model-driven plan communication poses a compelling challenge to the emergent communication paradigm.
- ⭐ Value: 3/5 — The core idea is valuable, but the experimental scale is overly simplified and the path to practical deployment remains unclear.
- ⭐ Experimental Thoroughness: 3/5 — Ablations and comparisons are clear but confined to simple grid environments; comparisons with mainstream MARL methods are absent.
- ⭐ Writing Quality: 4/5 — Problem motivation and method presentation are clear, with well-designed figures and tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Emergent Coordination in Multi-Agent Language Models](../../ICLR2026/multi_agent/emergent_coordination_in_multi-agent_language_models.md)
- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](large_language_models_miss_the_multi-agent_mark.md)
- [\[NeurIPS 2025\] 3D-Agent: Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation](3d-agenttri-modal_multi-agent_collaboration_for_scalable_3d_object_annotation.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](../../ACL2026/multi_agent/silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)
- [\[CVPR 2025\] Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration](../../CVPR2025/multi_agent/collaborative_tree_search_for_enhancing_embodied_multi-agent_collaboration.md)

</div>

<!-- RELATED:END -->
