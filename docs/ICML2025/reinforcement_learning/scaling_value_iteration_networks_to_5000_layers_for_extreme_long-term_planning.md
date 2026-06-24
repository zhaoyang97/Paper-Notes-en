---
title: >-
  [Paper Note] Scaling Value Iteration Networks to 5000 Layers for Extreme Long-Term Planning
description: >-
  [ICML2025][Reinforcement Learning][Value Iteration Network] This paper proposes Dynamic Transition VIN (DT-VIN), which enhances the representation capability of latent MDPs by introducing dynamic transition kernels and designs an adaptive highway loss to alleviate vanishing gradients. This successfully scales VIN to 5000 layers, enabling 1800-step long-term planning in $100 \times 100$ mazes (compared to the original VIN which only supports 120-step planning in $25 \times 25$…
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Value Iteration Network"
  - "long-term planning"
  - "dynamic transition kernel"
  - "adaptive highway loss"
  - "deep network training"
date: 2026-05-08
content_hash: e3e3312fa8797feb
---

# Scaling Value Iteration Networks to 5000 Layers for Extreme Long-Term Planning

**Conference**: ICML2025  
**arXiv**: [2406.08404](https://arxiv.org/abs/2406.08404)  
**Authors**: Yuhui Wang, Qingyuan Wu, Dylan R. Ashley, Francesco Faccio, Weida Li, Chao Huang, Jürgen Schmidhuber
**Code**: To be confirmed  
**Area**: Reinforcement Learning  
**Keywords**: Value Iteration Network, long-term planning, dynamic transition kernel, adaptive highway loss, deep network training

## TL;DR

This paper proposes Dynamic Transition VIN (DT-VIN), which enhances the representation capability of latent MDPs by introducing dynamic transition kernels and designs an adaptive highway loss to alleviate vanishing gradients. This successfully scales VIN to 5000 layers, enabling 1800-step long-term planning in $100 \times 100$ mazes (compared to the original VIN which only supports 120-step planning in $25 \times 25$ mazes).

## Background & Motivation

### Background
Planning is a core problem in artificial intelligence—given a goal, find a sequence of actions that leads the agent to the goal. Traditional search algorithms (such as A*) require precise environmental models, making them inapplicable in unknown environments or large-scale continuous state spaces. Value Iteration Network (VIN, Tamar et al., 2016) is an end-to-end differentiable planning network that embeds a differentiable planning module within a deep neural network to perform value iteration on a latent MDP, thus exhibiting strong generalization capabilities to unseen tasks.

### Limitations of Prior Work
- **VIN cannot handle large-scale long-term planning tasks**: In $100 \times 100$ mazes, the success rate of VIN is below 40%; in $35 \times 35$ mazes, the success rate drops to 0% when planning demands more than 60 steps.
- **Insufficient representation of latent MDPs**: The original VIN uses **static transition kernels** (all states share the same set of convolutional kernels), which fails to model different transition dynamics across different regions in complex environments.
- **Severe vanishing gradient problem**: The planning module of VIN is essentially a recurrently unrolled deep network. Increasing the number of layers leads to severe gradient vanishing/explosion, limiting the network depth and the number of planning steps.
- **Limited improvements in existing work**: GPPN (Lee et al., 2018) and Highway VIN (Wang et al., 2024a) offer some improvements but still fail to achieve long-term planning at the level of thousands of steps.

### Core Motivation
To bridge the gap between "planning complexity" and "network representation capability" in the VIN architecture, enabling VIN to perform extremely long-term planning in truly large-scale environments through two key updates.

## Method

### Overall Architecture

DT-VIN introduces two core improvements over the original VIN. The overall pipeline is as follows:

1. **Encoder**: Encodes observations (e.g., maze images) into latent state representations, including a reward map $\bar{R}$ and state features.
2. **Dynamic Transition Kernel Generator**: Dynamically generates transition kernel parameters for each state based on input observations (rather than using globally shared static kernels).
3. **Planning Module**: Performs $K$-step value iteration ($K$ up to 5000) on the latent MDP, using the dynamic transition kernels for Bellman updates at each step.
4. **Adaptive Highway Loss**: Adaptively sets skip-connection losses at different depths during training based on the actual number of planning steps needed.
5. **Policy Output**: Extracts actions from the value function to output the final policy.

### Key Designs

#### Design 1: Dynamic Transition Kernel

**Problem Analysis**: The planning module of the original VIN uses a single set of convolutional kernels as transition probabilities, with all latent states sharing the same transition dynamics. This assumes the environment has uniform transition dynamics across all locations—which is obviously violated. For instance, in a maze, traversability is completely different at walls compared to open paths.

**Solution**: DT-VIN introduces **dynamic transition kernels** to generate independent transition parameters for each latent state:
- A conditional network (conditioned on observations) is employed to generate corresponding transition kernel parameters for each spatial location $(i, j)$.
- This allows different locations to exhibit distinct transition dynamics, significantly enhancing the representation capacity of the latent MDP.
- This is conceptually similar to dynamic convolution but is applied within the Bellman updates during value iteration.

**Effect**: Dynamic transition kernels allow the latent MDP to capture local structural variations in complex environments, which is a critical factor enabling DT-VIN to handle large-scale environments.

#### Design 2: Adaptive Highway Loss

**Problem Analysis**: Unrolling the planning module of VIN for $K$ steps of value iteration is equivalent to a $K$-layer deep network. When $K$ scales to the thousands, the gradient propagated back from the final loss decays exponentially due to the chain rule (vanishing gradients), leaving shallow-layer parameters largely un-updated.

**Solution**: An adaptive highway loss is designed to construct skip connections to the final loss at intermediate layers:
- During training, instead of computing the loss only at step $K$, auxiliary losses are also calculated at certain intermediate steps.
- The positions of skip connections are adaptively selected based on the **actual number of planning steps required** (rather than at fixed intervals).
- This ensures gradients are propagated back through shorter paths, effectively alleviating vanishing gradients.
- Unlike the fixed gating mechanism in Highway Networks, this method establishes skip connections at the loss level, offering higher flexibility.

**Relationship to Existing Methods**: Highway VIN (Wang et al., 2024a) recognized the relationship between network depth and long-term planning and introduced a highway structure, but the "adaptive" mechanism of DT-VIN allows it to scale further to a depth of 5000 layers.

### Loss & Training

- **Imitation Learning**: Supervised training is conducted using optimal paths generated by search algorithms like A* as expert demonstrations.
- **Curriculum Learning Concept**: Gradually transitions from short-term planning to long-term planning by controlling the distribution of planning steps in the training samples.
- **Large-Scale Training**: Training a 5000-layer network requires carefully designed training strategies, including learning rate scheduling and batch configuration.

## Key Experimental Results

### Experiment 1: 2D Maze Navigation—Success Rates under Different Maze Sizes

| Method | $15 \times 15$ | $25 \times 25$ | $35 \times 35$ | $50 \times 50$ | $100 \times 100$ |
|------|:---:|:---:|:---:|:---:|:---:|
| VIN (Tamar et al., 2016) | ~99% | ~75% | ~50% | <40% | <40% |
| GPPN (Lee et al., 2018) | ~99% | ~85% | ~60% | — | — |
| Highway VIN (Wang et al., 2024a) | ~99% | ~90% | ~70% | — | — |
| **DT-VIN (Ours)** | **~99%** | **~98%** | **~95%** | **~90%** | **~85%** |

DT-VIN significantly outperforms baselines across all sizes, especially showing clear advantages in large mazes. The original VIN almost completely fails on $100 \times 100$ mazes, whereas DT-VIN maintains a high success rate.

### Experiment 2: $35 \times 35$ Maze—Success Rates under Different Shortest Path Lengths

| Method | Planning Steps $\le$ 20 | Planning Steps 20–40 | Planning Steps 40–60 | Planning Steps $>$ 60 |
|------|:---:|:---:|:---:|:---:|
| VIN | ~99% | ~80% | ~40% | ~0% |
| **DT-VIN** | **~99%** | **~98%** | **~95%** | **~90%** |

VIN completely collapses in long-term planning exceeding 60 steps, whereas DT-VIN maintains a success rate above 90%.

### Experiment 3: Scalability Validation

| Metric | VIN | Highway VIN | DT-VIN |
|------|:---:|:---:|:---:|
| Maximum Trainable Layers | ~150 | ~500 | **5000** |
| Maximum Effective Planning Steps | 120 | ~300 | **1800** |
| Maximum Supported Maze Size | $25 \times 25$ | $35 \times 35$ | **$100 \times 100$** |

### Experiment 4: Multi-Task Validation

| Task | Type | Planning Steps Required | DT-VIN Performance |
|------|------|:---:|------|
| 2D Maze Navigation | Discrete, visual input | Hundreds to 1800 | Success rate 85%+ |
| 3D Maze Navigation (VizDoom) | Continuous visual | Hundreds | Effectively solved |
| Continuous Control | Continuous action space | Hundreds | Effectively solved |
| Lunar Rover Navigation (Real-world) | Real terrain data | Hundreds to thousands | Successfully planned paths |

## Highlights & Insights

- **Breakthrough in Scale**: This is the first work to scale the VIN architecture to 5000 layers, achieving a qualitative leap from hundred-step planning to thousand-step planning. This is not simply a matter of making the network "deeper," but a systematic resolution of both depth and representation bottlenecks.
- **Elegant Design of Dynamic Transition Kernels**: By leveraging the concepts of dynamic convolution, the transition structure of the latent MDP is redesigned. This allows different spatial locations to have distinct transition dynamics, enhancing the model’s capacity to represent complex environments.
- **Practical Value of Adaptive Highway Loss**: Compared to auxiliary losses at fixed intervals, the adaptive mechanism dynamically adjusts according to planning complexity, utilizing gradient signals more efficiently.
- **Potential of End-to-End Planning**: Unlike methods like MuZero/Dreamer that require explicitly learning environment models, DT-VIN plans directly on a latent MDP, preserving the end-to-end advantages and strong generalization of the VIN family.
- **Closed Loop from Theory to Practice**: The approach is validated not only on toy mazes but also demonstrates practical value on the VizDoom 3D environment, continuous control, and real-world lunar rover navigation data.

## Limitations & Future Work

- **High Computational Overhead**: Forward and backpropagation through 5000 layers are computationally expensive. Although the highway loss mitigates training issues, inference still requires full forward propagation.
- **Dependence on Expert Demonstrations**: The imitation learning paradigm requires optimal path labels (e.g., generated by A*), which limits application to tasks where expert solutions are unavailable.
- **Poor Interpretability of Latent MDPs**: Although dynamic transition kernels enhance representation capacity, the states and transitions learned in the latent MDP do not possess explicit physical meanings.
- **Evaluated Primarly on Grid-Structured Tasks**: Although various tasks were tested, the core evaluation remains focused on navigation problems with spatial structures. Suitability for non-spatial planning tasks (e.g., combinatorial optimization, job scheduling) is unknown.
- **Insufficient Comparison with Model-Based RL**: Systemic comparisons with major model-based planning approaches such as MuZero and Dreamer are lacking.
- **Parameter Efficiency of Dynamic Kernel Generation**: Generating independent transition kernels for each location means the parameter size scales with the state-space size, potentially causing memory bottlenecks in extremely large state spaces.

## Related Work & Insights

- **VIN (Tamar et al., 2016)**: The foundational architecture of this work, which first proposed embedding value iteration within deep networks.
- **GPPN (Lee et al., 2018)**: Partially extended the capabilities of VIN by improving the structure of the planning module.
- **Highway VIN (Wang et al., 2024a)**: Introduced highway structures to alleviate vanishing gradients, acting as a direct predecessor to DT-VIN.
- **Dreamer (Hafner et al., 2020–2023)**: Performs planning in explicitly learned world models, complementing the latent planning of VIN.
- **MuZero (Schrittwieser et al., 2020)**: Learning-based search planning, demonstrating outstanding performance in board games.
- **Predictron (Silver et al., 2017)**: Early work doing lookahead planning in differentiable environment models.
- **Deep Network Training Techniques**: Improvements to gradient flow in Highway Networks (Srivastava et al., 2015) and ResNet (He et al., 2016) inspired the adaptive highway loss.

**Insight**: The VIN family showcases the powerful paradigm of "directly embedding algorithmic structures (value iteration) into network architectures." The success of DT-VIN indicates that through meticulous architectural design, end-to-end differentiable planning can be scaled to previously thought impractical levels, suggesting future deep integration of "algorithms + deep learning."

## Rating

- Novelty: ⭐⭐⭐⭐ — Dynamic transition kernels and adaptive highway loss are meaningful innovations, but the overall framework remains within the VIN paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 2D/3D mazes, continuous control, and real-world navigation tasks, but lacks comparison with MuZero/Dreamer, etc.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, precise problem definitions, and intuitive diagrams.
- Value: ⭐⭐⭐⭐ — Scales end-to-end planning to a new order of magnitude, holding significant value for understanding deep planning networks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Extreme Value Policy Optimization for Safe Reinforcement Learning](extreme_value_policy_optimization_for_safe_reinforcement_learning.md)
- [\[ICLR 2026\] Information-based Value Iteration Networks for Decision Making Under Uncertainty](../../ICLR2026/reinforcement_learning/information-based_value_iteration_networks_for_decision_making_under_uncertainty.md)
- [\[AAAI 2026\] Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation](../../AAAI2026/reinforcement_learning/intention-guided_cognitive_reasoning_for_egocentric_long-term_action_anticipatio.md)
- [\[ICLR 2026\] Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning](../../ICLR2026/reinforcement_learning/continuous-time_value_iteration_for_multi-agent_reinforcement_learning.md)
- [\[ICML 2025\] The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks](the_impact_of_on-policy_parallelized_data_collection_on_deep_reinforcement_learn.md)

</div>

<!-- RELATED:END -->
