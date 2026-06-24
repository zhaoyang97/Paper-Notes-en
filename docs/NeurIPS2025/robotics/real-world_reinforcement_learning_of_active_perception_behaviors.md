---
title: >-
  [Paper Note] Real-World Reinforcement Learning of Active Perception Behaviors
description: >-
  [NeurIPS 2025][Robotics][Active Perception] This paper proposes Asymmetric Advantage-Weighted Regression (AAWR), which leverages additional privileged sensors during training to estimate more accurate advantage functions, enabling efficient learning of active perception policies in the real world. AAWR outperforms all baselines across 8 manipulation tasks spanning varying degrees of partial observability.
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Active Perception"
  - "Asymmetric Reinforcement Learning"
  - "Privileged Information"
  - "POMDP"
  - "Real-Robot"
date: 2026-05-08
content_hash: 090755d5754f1faa
---

# Real-World Reinforcement Learning of Active Perception Behaviors

**Conference**: NeurIPS 2025
**arXiv**: [2512.01188](https://arxiv.org/abs/2512.01188)  
**Code**: [https://penn-pal-lab.github.io/aawr/](https://penn-pal-lab.github.io/aawr/)  
**Area**: Reinforcement Learning
**Keywords**: Active Perception, Asymmetric Reinforcement Learning, Privileged Information, POMDP, Real-Robot

## TL;DR

This paper proposes Asymmetric Advantage-Weighted Regression (AAWR), which leverages additional privileged sensors during training to estimate more accurate advantage functions, enabling efficient learning of active perception policies in the real world. AAWR outperforms all baselines across 8 manipulation tasks spanning varying degrees of partial observability.

## Background & Motivation

**Background**: A robot's instantaneous sensor observations often fail to reveal the full state information required for task completion. In such partially observable settings, optimal policies typically require explicit information-gathering behaviors—such as scanning a scene to locate a target object or probing occluded regions with a wrist camera. These behaviors are referred to as active perception or interactive perception.

**Limitations of Prior Work**: Mainstream robot learning techniques struggle to produce effective active perception behaviors. Imitation learning is ill-suited because obtaining optimal active perception demonstrations is extremely difficult (e.g., forcing operators to teleoperate exclusively through a wrist-camera view). Standard RL is already sample-inefficient in fully observable settings, and partial observability exacerbates the problem further. Sim-to-real transfer is also inappropriate, since active perception is tightly coupled to sensor capabilities, and depth, RGB, and tactile sensors are all difficult to model accurately in simulation. State-of-the-art generalist policies (e.g., π₀), despite being trained on large amounts of teleoperation data, fail at even simple search tasks.

**Key Challenge**: RL can theoretically learn active perception through interaction, but is prohibitively sample-inefficient in practice. Privileged-information methods have succeeded in sim-to-real transfer, yet sim-to-real is unsuitable for active perception tasks. Offline RL can leverage suboptimal demonstrations, but value function estimation in POMDPs remains inaccurate.

**Goal**: How can active perception policies be learned efficiently on real robots, using only a small number of suboptimal demonstrations and an easily obtainable coarse initial policy?

**Key Insight**: The paper employs additional privileged sensors (e.g., object detectors, segmentation masks) during training to provide more accurate supervision signals to the critic and value networks. A key theoretical finding is that, when performing AWR-style policy improvement in a POMDP, using privileged advantage estimates that incorporate environment state is a mathematically correct approach.

**Core Idea**: Use privileged sensors available at training time to provide more accurate advantage estimates for RL than partial observations alone, thereby efficiently guiding policy learning toward information-gathering behaviors.

## Method

### Overall Architecture

AAWR follows an offline-to-online RL paradigm: a policy and privileged value function are first pretrained on a small set of suboptimal offline demonstrations, then fine-tuned online in the real environment. During training, the policy receives only partial observations (e.g., wrist-camera images), while the critic/value network additionally receives privileged observations (e.g., object positions, segmentation masks). At deployment, only the partial-observation policy is used; no privileged sensors are required.

### Key Designs

1. **Asymmetric Advantage-Weighted Regression (AAWR) Objective**:

    - **Function**: Implements policy improvement in POMDPs via correctly weighted behavioral cloning.
    - **Mechanism**: The POMDP is reformulated as an equivalent MDP whose state is $(s, z)$ (environment state + agent state). Deriving the KL-constrained policy improvement objective of AWR on this MDP yields the AAWR loss: $\mathcal{L}_{AAWR}(\pi) = \mathbb{E}_{(s,z) \sim d_\mu} \mathbb{E}_{a \sim \mu}[\exp(A^\mu(s,z,a)/\beta) \log \pi(a|z)]$. Here the advantage function $A^\mu(s,z,a) = Q^\mu(s,z,a) - V^\mu(s,z)$ depends on both the environment state $s$ and the agent state $z$, while the policy $\pi(a|z)$ conditions only on $z$. Theorem 1 proves that this objective is equivalent to the Lagrangian relaxation of maximizing expected policy improvement in the POMDP.
    - **Design Motivation**: The symmetric variant SAWR (which estimates advantages using only $z$, without $s$) cannot recover the optimal solution, because an advantage estimator based solely on agent state is insufficient to estimate advantages in the equivalent MDP. Furthermore, a non-privileged value function is not a fixed point of the corresponding Bellman equation, whereas the privileged value function is.

2. **IQL-Based Privileged Value Function Training**:

    - **Function**: Efficiently trains the privileged critic and value networks.
    - **Mechanism**: Implicit Q-Learning (IQL) with expectile regression is used to train $Q_\phi^\mu(s,z,a)$ and $V_\theta^\mu(s,z)$. IQL is well known for its stability in offline RL and offline-to-online fine-tuning. The critic receives $(o_t, s_t)$ or augmented observations $(o_t, o_t^p)$, while the policy receives only $o_t$.
    - **Design Motivation**: IQL avoids the need to maximize over out-of-policy actions as in standard Q-learning, making it well-suited for low-data offline-to-online settings. Privileged information enables accurate value estimation even within a POMDP.

3. **Offline-to-Online Training Pipeline**:

    - **Function**: Bootstraps from a small set of suboptimal demonstrations and improves autonomously through online interaction.
    - **Mechanism**: Training proceeds in two stages. In the offline stage, $Q$ and $V$ are updated on $\mathcal{D}_{off}$ using the IQL objective, and $\pi$ is updated using the AAWR objective. In the online stage, the policy collects trajectories stored in $\mathcal{D}_{on}$, and updates are performed on batches sampled equally from both buffers. Critically, the policy continues to be guided by the privileged value function during online fine-tuning.
    - **Design Motivation**: Suboptimal demonstrations provide initial coverage and avoid exploration from scratch. Online fine-tuning allows the policy to discover active perception behaviors not present in demonstrations through trial and error. In contrast, distillation methods stagnate because the privileged expert, unaware of camera field-of-view constraints, induces a suboptimal "rush to center" strategy in the distilled policy; AAWR, as a policy iteration algorithm, can discover scanning behaviors through online exploration.

### Loss & Training

Value functions are trained with IQL's expectile regression loss. The policy is trained with the AAWR weighted cross-entropy loss. Offline and online data are mixed at a 1:1 sampling ratio.

## Key Experimental Results

### Main Results

Simulation task performance (average over 10 seeds):

| Task | AAWR | AWR | BC |
|------|------|-----|----|
| Camouflage Pick | ~95% | ~45% | ~25% |
| Fully Obs. Pick | ~95% | ~30% | ~50% |
| AP Koch (final success rate) | **100%** | ~70% | ~40% |

Real-robot Koch interactive perception task:

| Method | Grasp Rate (%) | Pick Rate (%) |
|--------|---------------|--------------|
| On. AAWR (ours) | **94** | **89** |
| Off. AAWR | 88 | 71 |
| On. AWR | 71 | 55 |
| Off. AWR | 65 | 62 |
| BC | 47 | 41 |

### Ablation Study

π₀ generalist policy handoff tasks (real Franka robot):

| Method | Bookshelf-P Search | Completion | Shelf-Cabinet Search | Completion |
|--------|--------------------|------------|----------------------|------------|
| AAWR | **92.4** | **44.4** | **78.2** | **40.0** |
| AWR | 79.6 | 0.0 | 52.3 | 10.0 |
| BC | 29.9 | 20.0 | 3.8 | 0.0 |

### Key Findings

- AAWR outperforms its non-privileged counterpart AWR on all 8 tasks, including the fully observable Fully Obs. Pick task, indicating that the privileged critic helps not only with occlusion handling but also with extracting information more effectively from pixels.
- On the AP Koch task, the Distillation method plateaus at 80%—the privileged expert rushes directly toward the object without accounting for camera field-of-view constraints, causing the distilled policy to learn a suboptimal "rush to center" strategy. As a policy iteration algorithm, AAWR discovers scanning behaviors through online exploration.
- VIB (Variational Information Bottleneck) collapses at deployment when privileged information is no longer available.
- On the π₀ handoff tasks, AAWR learns to systematically search plausible hiding locations in heavily occluded scenes.

## Highlights & Insights

- **Elegant Theoretical Derivation**: The derivation chain from POMDP → equivalent MDP → constrained policy improvement → AAWR objective is clear and natural. Theorem 1 provides a rigorous theoretical foundation for AAWR while simultaneously establishing the inadequacy of SAWR. This is not a mere engineering trick of "feeding privileged information to the critic," but a principled approach with strict theoretical support.
- **Strong Real-World Deployment Capability**: The method is validated on 3 real robot platforms, requiring only 100–150 suboptimal demonstrations and limited online interaction (as few as 1,200 steps), with no simulation environment needed. Privileged sensors can be simple object detectors running on uncalibrated RGB cameras.
- **Novel Integration with Generalist VLA Policies**: AAWR trains a search policy that, upon locating the target, hands off to π₀ for grasping. This division of labor elegantly addresses the inability of generalist policies to handle active perception, and can be generalized to other VLA systems.

## Limitations & Future Work

- Task-specific reward functions must be designed, which may be non-trivial for certain tasks.
- The selection of privileged sensors requires domain knowledge—what information counts as "privileged" and which sensors to use must be manually specified.
- Privileged sensors must remain present during the online fine-tuning phase, limiting applicability in some scenarios.
- The current agent state representation uses a sliding window; more expressive history encodings (e.g., Transformers) are not explored, potentially limiting performance on tasks requiring long-horizon memory.
- The handoff mechanism between the search policy and the execution policy is relatively simple; more sophisticated integration strategies may yield further gains.

## Related Work & Insights

- **vs. Information-Theoretic Active Perception (e.g., next-best-view)**: Such methods optimize uncertainty reduction or information gain without regard for task constraints—they may identify the back of a shelf as highly "informative" even though the target object could never be located there. AAWR learns task-centric search behaviors through task rewards.
- **vs. Sim-to-Real Privileged Learning (e.g., RMA, DAgger-style)**: These methods train on billions of privileged state transitions in simulation; AAWR uses only hundreds of real-world trajectories and requires no accurate simulation of the sensors on which active perception depends.
- **vs. Distillation Methods**: Distillation first trains a privileged expert and then distills it into a partial-observation policy, but the privileged expert's ignorance of perceptual constraints produces suboptimal demonstrations. AAWR directly optimizes the policy under partial observability and leverages privileged information indirectly through the privileged value function, avoiding this issue entirely.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The theoretical derivation that combines privileged information with AWR in a POMDP is novel, though asymmetric RL itself is not a new concept.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely comprehensive: 8 tasks, 3 robot platforms, simulation and real-world evaluation, multiple baselines, and integration experiments with a generalist VLA.
- **Writing Quality**: ⭐⭐⭐⭐ The theoretical sections are clear and rigorous, and experimental descriptions are thorough, though the paper is lengthy.
- **Value**: ⭐⭐⭐⭐ Provides a practical and theoretically grounded approach to learning active perception in the real world; the integration with VLA policies has strong application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] APPLE: Toward General Active Perception via Reinforcement Learning](../../ICLR2026/robotics/apple_toward_general_active_perception_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[NeurIPS 2025\] To Distill or Decide? Understanding the Algorithmic Trade-off in Partially Observable Reinforcement Learning](to_distill_or_decide_understanding_the_algorithmic_trade-off_in_partially_observ.md)
- [\[CVPR 2025\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](../../CVPR2025/robotics/sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)
- [\[NeurIPS 2025\] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning](pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)

</div>

<!-- RELATED:END -->
