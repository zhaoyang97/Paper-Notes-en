---
title: >-
  [Paper Note] Quantifying Generalisation in Imitation Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][imitation learning] This paper proposes the Labyrinth benchmark environment, which achieves strict separation between training and evaluation data through controllable maze structure variations. It reveals severe deficiencies in the structural generalisation of current imitation learning methods (best method achieves only 5% success rate on the test set) and provides a systematic tool for evaluating generalisation in imitation learning.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - imitation learning
  - generalisation evaluation
  - benchmark environment
  - maze navigation
  - benchmark
date: 2026-05-08
content_hash: 4af6652cf7514742
---

# Quantifying Generalisation in Imitation Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.24784](https://arxiv.org/abs/2509.24784)
**Code**: [https://github.com/NathanGavenski/Labyrinth](https://github.com/NathanGavenski/Labyrinth)
**Area**: Reinforcement Learning
**Keywords**: imitation learning, generalisation evaluation, benchmark environment, maze navigation, benchmark

## TL;DR
This paper proposes the Labyrinth benchmark environment, which achieves strict separation between training and evaluation data through controllable maze structure variations. It reveals severe deficiencies in the structural generalisation of current imitation learning methods (best method achieves only 5% success rate on the test set) and provides a systematic tool for evaluating generalisation in imitation learning.

## Background & Motivation
Imitation learning sits at the intersection of reinforcement learning and supervised learning: it exploits observational data during training in a supervised manner and evaluates performance through environment interaction in an RL fashion. However, widely used imitation learning benchmarks suffer from a fundamental flaw — **insufficient differentiation between training and evaluation data** — making it impossible to effectively test generalisation.

Specific problems include:

**Classic control tasks (CartPole, MountainCar) are overly simplistic**: the state space is low-dimensional with limited discrete actions. The authors find that in MountainCar, simply replicating the action sequence of a single expert trajectory achieves the "solved" criterion across 100 different initialisations.

**Continuous control tasks (Hopper, HalfCheetah) lack state abstraction and behavioural interpretability**: the optimal action at an arbitrary state cannot be determined; the Manhattan distance between 100K initialisations is negligibly small (e.g., only 0.22 for Hopper), rendering training and testing virtually indistinguishable.

**Atari games use identical environments for training and testing**, making it impossible to separate training from test data.

Root Cause: High performance on existing benchmarks may reflect memorisation rather than generalisation. The authors argue that a qualified generalisation test environment must: (1) present sufficiently challenging tasks; (2) exhibit significant variation between training and evaluation; (3) allow precise control over such variation; and (4) support behavioural debugging and inspection.

## Method

### Overall Architecture
Labyrinth is a discrete maze navigation environment based on graph structures, formally defined as a graph $G$ where nodes represent cells and edges represent connected paths (removing an edge adds a wall). The agent navigates from a start state $s_0$ to a goal $g$ using discrete up/down/left/right actions. The environment supports both fully observable (complete maze visible) and partially observable (only surrounding region visible) modes.

### Key Designs
1. **Controllable Structural Variation and Data Separation**:

    - Three start/goal configurations: user-specified, biased (bottom-left to top-right), and unbiased (random but with guaranteed minimum distance $d(s_0, g) = |x_{s_0} - x_g| + |y_{s_0} - y_g|$)
    - Graph hashing ensures each structure appears uniquely in training/validation/test splits
    - Supports three generalisation tests: different structures with same start/goal, same structure with different start, same structure with different start/goal
    - Design Motivation: precisely control which aspect of the environment varies to isolate the specific causes of generalisation failure

2. **Task Variants and Complexity Scaling**:

    - **Key and Door**: the agent must first collect a key $g_k$ to unlock a door $g_d$ before reaching the goal $g$, testing sequential sub-goal execution
    - **Ice Floor**: stepping on an icy tile causes failure, testing path planning under safety constraints
    - **Partially Observable**: only the structure surrounding the agent is visible
    - Design Motivation: progressively increase difficulty to test generalisation along different dimensions

3. **Exact Optimal Behaviour Baseline**:

    - Core formulation — normalised reward function:
    $$r_i = \begin{cases} \frac{-0.1}{width \times height} & \text{goal not reached} \\ 1 + |\tau_s| \times \frac{0.1}{width \times height} & \text{goal reached} \end{cases}$$
    - Johnson's algorithm is used to compute all paths from $s_0$ to $g$, yielding the exact optimal action at every state
    - Cumulative reward along the shortest path is always 1, independent of maze size
    - Design Motivation: fully known optimal policy enables generalisation analysis to be precise at the level of individual state-action decisions

### Loss & Training
Labyrinth itself does not train models but provides the environment. It uses the gymnasium interface, compatible with various imitation learning methods:
- Supports Behavioural Cloning (BC), DAgger, GAIL, BCO, SQIL, IUPE, and others
- Datasets are hosted on HuggingFace under IL-Datasets
- Supports both vector and image-based state representations

## Key Experimental Results

### Main Results

| Method | Train AER | Train SR | Val AER | Val SR | Test AER | Test SR |
|--------|-----------|----------|---------|--------|----------|---------|
| BC | -2.11±2.41 | 37% | -3.70±1.18 | 6% | -3.90±0.70 | 2% |
| DAgger | -1.18±2.45 | 57% | -3.75±1.08 | 5% | -3.80±0.97 | 4% |
| GAIL | -0.98±1.89 | 61% | -3.57±1.58 | 9% | -3.85±0.85 | 3% |
| BCO | -0.53±2.23 | 70% | -3.90±0.69 | 2% | -3.85±0.85 | 3% |
| SQIL | -3.80±0.96 | 4% | -3.95±0.49 | 1% | -4.00±0.00 | 0% |
| IUPE | **0.27±2.39** | **75%** | **-2.80±2.12** | **21%** | -3.85±1.00 | **5%** |

### Ablation Study

| Configuration | Train SR | Val SR | Test SR | Notes |
|---------------|----------|--------|---------|-------|
| BC (1000 epochs, Atari CNN) | 37% | 6% | 2% | Original configuration |
| BC (10000 epochs, Atari CNN) | 100% | 41% | 34% | Longer training |
| BC (10000 epochs, ResNet-18) | 100% | **56%** | **53%** | Stronger encoder |

| Environment | Avg. Inter-initialisation Distance | Action Sequence Replication Feasibility |
|-------------|-----------------------------------|----------------------------------------|
| MountainCar | 0.001 | A single expert trajectory suffices to "solve" |
| CartPole | 0.0095 | Extremely similar |
| Hopper | 0.2175 | Nearest-neighbour actions match expert level |
| Labyrinth | High (structurally distinct) | Not feasible |

### Key Findings
- All imitation learning methods perform extremely poorly on unseen test structures (maximum 5% SR), indicating that no method genuinely learns the navigation task itself.
- IUPE achieves 21% SR on the validation set but only 5% on the test set, suggesting that validation-set "generalisation" may reflect chance rather than genuine understanding.
- Stronger network architectures (ResNet-18) substantially improve generalisation (2%→53%), yet the problem remains far from solved — indicating that the bottleneck lies in the algorithm rather than the representation.
- Pure imitation learning methods (BC-type) learn better state encodings than inverse reinforcement learning methods (GAIL, SQIL).

## Highlights & Insights
- **Well-founded and incisive critique of existing benchmarks**: the demonstrations that copying a single trajectory "solves" MountainCar and that nearest-neighbour state actions can substitute for expert behaviour in Hopper directly challenge the community's default assumptions about generalisation evaluation.
- **Environment design balances simplicity and controllability**: the discrete graph structure allows exact computation of optimal policies; biased/unbiased configurations enable quantification of variation in action distributions; key-and-door and ice-floor variants progressively introduce generalisation challenges along distinct dimensions.

## Limitations & Future Work
- Currently supports only discrete action spaces, precluding evaluation of methods designed exclusively for continuous actions (e.g., OPOLO, MAHALO).
- The maze task is relatively simple and may not adequately represent generalisation challenges arising from high-dimensional perception and complex physical interaction.
- Direct comparisons with existing procedurally generated environments such as Procgen and MiniGrid are absent.
- The partially observable setting is mutually exclusive with the key-and-door and ice-floor variants, limiting combinatorial testing.

## Related Work & Insights
- **vs MiniGrid**: MiniGrid is also a grid-based environment, but Labyrinth uniquely offers exact optimal action computation, graph-hash-based uniqueness guarantees, and structured data separation.
- **vs Procgen**: Procgen tests generalisation through procedural generation, but its tasks are too complex for precise failure analysis; Labyrinth offers substantially stronger controllability.
- **vs MuJoCo benchmarks**: MuJoCo exhibits minimal train-test differences (initialisation Manhattan distance <0.5), whereas Labyrinth provides fundamentally different structural variation.

## Rating
- Novelty: ⭐⭐⭐⭐ The environment design is conceptually novel, and the analysis of existing benchmark weaknesses is incisive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 6 imitation learning methods and multiple ablations, though in-depth experiments on additional task variants are lacking.
- Writing Quality: ⭐⭐⭐⭐ The argumentation is logically coherent, and the critique of existing benchmarks is data-driven.
- Value: ⭐⭐⭐⭐ Provides important insights for generalisation evaluation in imitation learning, though the complexity of Labyrinth itself may limit its adoption as a general-purpose benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Interactive and Hybrid Imitation Learning: Provably Beating Behavior Cloning](interactive_and_hybrid_imitation_learning_provably_beating_behavior_cloning.md)
- [\[NeurIPS 2025\] Massively Parallel Imitation Learning of Mouse Forelimb Musculoskeletal Reaching Dynamics](massively_parallel_imitation_learning_of_mouse_forelimb_musculoskeletal_reaching.md)
- [\[NeurIPS 2025\] EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](egobridge_domain_adaptation_for_generalizable_imitation_from_egocentric_human_da.md)
- [\[NeurIPS 2025\] BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning](beast_efficient_tokenization_of_b-splines_encoded_action_sequences_for_imitation.md)
- [\[NeurIPS 2025\] Inner Speech as Behavior Guides: Steerable Imitation of Diverse Behaviors for Human-AI Coordination](inner_speech_as_behavior_guides_steerable_imitation_of_diverse_behaviors_for_hum.md)

</div>

<!-- RELATED:END -->
