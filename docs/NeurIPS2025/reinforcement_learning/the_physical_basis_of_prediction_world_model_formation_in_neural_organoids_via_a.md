---
title: >-
  [Paper Note] The Physical Basis of Prediction: World Model Formation in Neural Organoids via an LLM-Generated Curriculum
description: >-
  [NeurIPS 2025 (Workshop: Scaling Environments for Agents / Embodied World Models)][Reinforcement Learning][Neural organoids] This paper proposes a framework for studying world model formation in human neural organoids, comprising three progressively complex virtual environments (conditioned avoidance, predator–prey, Pong) and a meta-learning approach in which an LLM automatically generates experimental protocols, complemented by a multi-scale biophysical evaluation strategy to quantify the physical basis of biological learning.
tags:
  - "NeurIPS 2025 (Workshop: Scaling Environments for Agents / Embodied World Models)"
  - Reinforcement Learning
  - Neural organoids
  - world models
  - curriculum learning
  - LLM-generated environments
  - synaptic plasticity
date: 2026-05-08
content_hash: 4ae8ed7c7e00b437
---

# The Physical Basis of Prediction: World Model Formation in Neural Organoids via an LLM-Generated Curriculum

**Conference**: NeurIPS 2025 (Workshop: Scaling Environments for Agents / Embodied World Models)
**arXiv**: [2509.04633](https://arxiv.org/abs/2509.04633)
**Code**: Unavailable
**Area**: Reinforcement Learning
**Keywords**: Neural organoids, world models, curriculum learning, LLM-generated environments, synaptic plasticity

## TL;DR

This paper proposes a framework for studying world model formation in human neural organoids, comprising three progressively complex virtual environments (conditioned avoidance, predator–prey, Pong) and a meta-learning approach in which an LLM automatically generates experimental protocols, complemented by a multi-scale biophysical evaluation strategy to quantify the physical basis of biological learning.

## Background & Motivation

World models are central to embodied intelligence—an agent's capacity to understand, predict, and interact with its environment fundamentally depends on an internal world model. Although substantial progress has been made on world models within AI, **the critical role of the environment itself as a driver of intelligence** remains a frontier area of investigation.

**Unique advantages of biological substrates**: Living neural organoids (3D neural networks cultured from human stem cells) offer an unprecedented opportunity to study learned world models not as abstract computational processes, but as **genuine biological phenomena** rooted in the physical principles of synaptic plasticity.

**The leap from 2D to 3D**: Early 2D dissociated neural cultures demonstrated that biological neural networks can establish instrumental relationships with external virtual worlds. 3D neural organoids replicate the complex cytoarchitecture of the human brain through developmental self-organization, providing a richer substrate.

**Bottleneck in environment design**: Manual protocol design is time-consuming and difficult to scale, motivating an automated method for large-scale generation and optimization of training curricula.

Core motivation: combining model-driven reinforcement learning concepts with physical biological systems—observing world model formation in living neural networks while using an LLM to automate the environment design process.

## Method

### Overall Architecture

The framework comprises three major components:
1. **Multi-electrode array (MEA) interface**: provides a bidirectional stimulation/recording interface that physically connects the virtual environment to the biological substrate.
2. **Three progressive virtual environments**: constitute the training curriculum, each requiring increasingly complex world models.
3. **LLM-generated experimental protocols**: a meta-learning framework in which an LLM automatically designs and optimizes experimental protocols.

### Key Designs

1. **Predictive-coding-based learning mechanism**:
   Drawing on the Free Energy Principle, biological agents have an intrinsic drive to minimize prediction error. Feedback signals are designed accordingly:
   - **Reward (model confirmation)**: predictable, low-entropy electrical stimulation (e.g., consistent low-frequency sine waves) that minimizes "surprise" and reinforces successful strategies. An alternative is UV-light-triggered release of caged dopamine to directly activate reward pathways.
   - **Punishment (model violation)**: unpredictable, high-entropy stimulation (white-noise electrical signals) that generates strong prediction errors, driving the network to update its strategy to avoid such states.

   This bipolar feedback translates abstract RL principles into concrete biophysical signals that drive synaptic change.

2. **Three progressive environment designs**:

   - **Environment 1 (Conditioned Avoidance)**: The agent moves on a 1D 8-position grid and learns to associate specific regions (positions 6–8) with negative outcomes and actively avoid them. Actions are decoded by comparing spike counts from recording groups A and B. Punishment intensity increases as a gradient with intrusion depth. Extensible to 2D/3D grids and maze navigation.
   - **Environment 2 (Predator–Prey)**: The agent (predator) must actively pursue a dynamic target (prey), requiring a goal-directed navigation world model. Two sensory channels are provided: prey position (exteroceptive) and self-position (proprioceptive). Extensible to include adversarial entities.
   - **Environment 3 (Pong)**: The classic Pong game, requiring the agent to model a continuous-time dynamical system. The agent must predict ball trajectories and move the paddle based on those predictions—the highest-level demand on the world model, requiring not only responses to the current state but also prediction of future states.

3. **LLM-automated protocol design**:
   An LLM serves as a meta-controller that automatically generates and optimizes new experimental protocols—including environmental parameters, stimulation patterns, and time windows—based on prior experimental results. This enables automated scaling of environment design, analogous to paradigms of autonomous discovery in chemistry and materials science.

### Evaluation Strategy: From Behavior to Synapse

A multi-scale evaluation framework that goes beyond simple task-performance metrics:

- **Functional network level**: Synaptic efficacy is quantified by measuring changes in field excitatory postsynaptic potentials (fEPSPs) before and after training. A sustained increase in slope indicates long-term potentiation (LTP); a decrease indicates long-term depression (LTD).
- **Cellular level**: Two-photon microscopy of GCaMP-expressing organoids longitudinally tracks the activity of hundreds of neurons to observe the emergence of stable neural ensembles.
- **Molecular level**: Post-training immunohistochemical staining targets AMPA/NMDA receptor distributions (changes in synaptic coding), Synapsin-1/PSD-95 (synaptic density), and c-Fos (marker of active neural ensembles).

## Key Experimental Results

### Environment Complexity Gradient Design

| Environment | State Space | Action Space | World Model Requirement | Core Challenge |
|---|---|---|---|---|
| Conditioned Avoidance | 1D × 8 positions | Binary (left/right) | Static state–outcome association | Learning to avoid specific regions |
| Predator–Prey | 2 × (1D × 8) | Binary (left/right) | Goal-directed navigation | Dynamic target tracking |
| Pong | Continuous 2D | Up/down movement | Continuous dynamical system modeling | Trajectory prediction + interception |

### Environment Scalability Comparison

| Extension Dimension | Conditioned Avoidance | Predator–Prey | Pong |
|---|---|---|---|
| Spatial extension | 1D → 2D grid → 3D volume | 1D → 2D search → 3D | Inherently 2D |
| Complexity extension | Simple boundary → maze navigation | Static prey → moving prey → + adversary | Single ball → multi-ball → acceleration |
| Dynamics extension | Static boundary → moving boundary | Random respawn → intelligent prey | Fixed physics → variable physics |

### Evaluation Scale Comparison

| Evaluation Scale | Measurement Method | Quantitative Metric | Interpretation for Learning |
|---|---|---|---|
| Functional network | fEPSP measurement | Slope change | LTP/LTD = change in synaptic strength |
| Cellular | Two-photon calcium imaging | Ensemble synchrony | Emergence of stable encoding ensembles |
| Molecular | Immunohistochemical staining | Receptor/synaptic protein density | Structural physical basis of learning |

### Key Findings

- This paper is a **framework proposal** rather than an experimental validation study; the primary contribution lies in formal design rather than experimental results.
- Operationalizing the Free Energy Principle as the learning-driving mechanism (predictable = reward, unpredictable = punishment) is theoretically rigorous.
- Using an LLM as a meta-learning experimental protocol generator represents a methodological innovation.
- The multi-scale evaluation strategy provides a complete bridge from behavior to physical substrate.

## Highlights & Insights

- **Unique interdisciplinary perspective**: directly maps the RL concept of world models onto synaptic plasticity research in computational neuroscience.
- **Elegant operationalization of the Free Energy Principle**: the predictability/unpredictability of electrical stimulation directly implements reward/punishment signals from RL.
- **LLM as a scientific experiment designer**: this meta-learning approach extends beyond conventional applications of LLMs in reasoning toward autonomous scientific discovery.
- **"Real" world models**—physically instantiated in living biological tissue and directly observable through molecular means.

## Limitations & Future Work

- As a workshop paper, it lacks experimental validation; all content consists of protocol design and theoretical analysis.
- Reproducibility and maturity challenges in neural organoids: high batch-to-batch variability and a required 60–90-day culture period.
- The action decoding scheme is overly simplistic (comparing spike counts from only two electrode groups) and may be insufficient to support complex behavior.
- LLM-generated protocols require domain-expert validation; reliability and safety require further assurance.
- Information transfer bandwidth for state encoding is severely limited by electrode count.

## Related Work & Insights

- **DishBrain (Kagan et al., 2022)**: first demonstrated that organoids can learn to play a simplified version of Pong.
- **Free Energy Principle (Friston, 2010)**: provides the theoretical foundation for the core learning mechanism.
- **Boiko et al., 2023**: pioneering work on LLMs for autonomous scientific discovery.
- Insight: implementing RL training in real biological neural networks may offer a unique perspective for understanding the physical basis of intelligence.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Combining the RL world-model concept with living neural networks; a uniquely conceived framework design.
- **Experimental Thoroughness**: ⭐⭐ Pure protocol-design paper with no experimental results.
- **Writing Quality**: ⭐⭐⭐⭐ Systematic and complete framework description with clear pseudocode, though with notable repetition.
- **Value**: ⭐⭐⭐ Forward-looking but lacking validation as a workshop paper; closer to a roadmap or proposal.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model](boundary-to-region_supervision_for_offline_safe_reinforcement_learning.md)
- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model (BOOM)](bootstrap_off-policy_with_world_model.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[NeurIPS 2025\] Learning Interestingness in Automated Mathematical Theory Formation](learning_interestingness_in_automated_mathematical_theory_formation.md)
- [\[NeurIPS 2025\] Bandit and Delayed Feedback in Online Structured Prediction](bandit_and_delayed_feedback_in_online_structured_prediction.md)

</div>

<!-- RELATED:END -->
