---
title: >-
  [Paper Note] RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints
description: >-
  [ICCV 2025][Reinforcement Learning][Multi-agent collaboration] This paper introduces the concept of compositional constraints to formalize safety and efficiency requirements in multi-agent embodied collaboration…
tags:
  - "ICCV 2025"
  - "Reinforcement Learning"
  - "Multi-agent collaboration"
  - "embodied manipulation"
  - "compositional constraints"
  - "imitation learning"
  - "benchmark"
date: 2026-05-08
content_hash: ba34458909db4946
---

# RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints

**Conference**: ICCV 2025
**arXiv**: [2503.16408](https://arxiv.org/abs/2503.16408)  
**Code**: None  
**Area**: Reinforcement Learning / Embodied Intelligence
**Keywords**: Multi-agent collaboration, embodied manipulation, compositional constraints, imitation learning, benchmark

## TL;DR

This paper introduces the concept of compositional constraints to formalize safety and efficiency requirements in multi-agent embodied collaboration, constructs the first multi-agent manipulation benchmark RoboFactory based on this formalization, and systematically investigates architectures and training strategies for multi-agent imitation learning.

## Background & Motivation

**Background**: Embodied multi-agent systems are critical for solving complex real-world tasks such as industrial assembly, warehouse logistics, and collaborative transport. Single-agent embodied manipulation has achieved remarkable progress in recent years (e.g., ACT, Diffusion Policy), but research on multi-agent collaborative manipulation lags significantly behind.

**Limitations of Prior Work**: (1) **Lack of automated data collection**: The complexity of multi-agent systems makes manual teleoperation extremely difficult, while automated data generation methods lack guarantees of safety (e.g., robotic arm collisions, tool interference); (2) **Lack of benchmarks**: Existing embodied manipulation benchmarks (e.g., RLBench, ManiSkill) target single-agent settings, and multi-agent benchmarks are nearly absent; (3) **Insufficient constraint modeling for collaboration**: Multi-agent collaboration requires not only completing individual subtasks but also satisfying complex interaction constraints—spatial constraints (collision avoidance), temporal constraints (ordering), and cooperative constraints (synchronized actions).

**Key Challenge**: Multi-agent embodied systems require large amounts of high-quality training data, yet the complex constraints in multi-agent scenarios make automated data generation both difficult and unsafe.

**Goal**: (1) Formalize a constraint framework for multi-agent collaboration; (2) Construct an automated data collection pipeline; (3) Provide the first multi-agent manipulation benchmark and explore learning approaches.

**Key Insight**: The authors observe that constraints in multi-agent collaboration can be decomposed into a set of primitive types (spatial, temporal, physical)—different tasks correspond to different combinations of these primitives.

**Core Idea**: Decompose multi-agent collaboration constraints into composable primitive constraint types, design automated interfaces for each type, and automatically generate safe and efficient training data while evaluating multi-agent policies through constraint composition.

## Method

### Overall Architecture

The overall framework consists of three components: (1) **Compositional Constraint Definition**—formalizing multi-agent task constraints as combinations of primitive constraint types; (2) **Automated Data Collection**—automatically generating demonstration trajectories that satisfy all constraints via constraint interfaces; (3) **RoboFactory Benchmark**—a multi-agent manipulation benchmark spanning multiple difficulty levels for evaluating multi-agent imitation learning methods. The input is a multi-arm scene configuration and task description; the output is a collaborative manipulation policy satisfying all constraints.

### Key Designs

1. **Compositional Constraints Framework**:

    - Function: Formalize all constraints that must be satisfied in multi-agent collaboration
    - Mechanism: Three primitive constraint types are defined: (a) **Spatial Constraints**—define workspace restrictions and collision avoidance conditions for each agent, implemented via collision detection interfaces to ensure no arm interference at any timestep; (b) **Temporal Constraints**—define action ordering and synchronization requirements, such as "A must precede B" or "A and B must execute simultaneously," implemented via event-trigger interfaces; (c) **Physical Constraints**—define force/velocity limits and contact conditions, implemented via physics simulator interfaces. Different tasks are different combinations of these three types—e.g., "collaborative transport" = spatial constraints (no collision) + temporal constraints (synchronized lift) + physical constraints (force balance).
    - Design Motivation: Decoupling and formalizing constraints from tasks allows new tasks to be rapidly defined by composing existing constraints, substantially reducing the cost of benchmark construction.

2. **Constraint-Aware Data Collection**:

    - Function: Automatically generate demonstration trajectories satisfying all compositional constraints
    - Mechanism: Combines a motion planner with a constraint satisfaction solver. At each timestep, an individual target action is planned for each agent; the constraint solver then checks whether all constraints are satisfied—if not, the system backtracks and adjusts (e.g., delaying an agent's action to satisfy temporal constraints, or rerouting to avoid collisions). The entire process executes within physics simulators such as MuJoCo or Isaac Gym, automatically collecting multi-modal data including joint angles, end-effector poses, and visual observations.
    - Design Motivation: Manual teleoperation is practically infeasible in multi-agent scenarios (coordinating multiple arms simultaneously is beyond human capacity); automated approaches are the only viable path to large-scale data. Constraint awareness ensures the safety and physical plausibility of the generated data.

3. **Multi-Agent Imitation Learning Architecture Exploration**:

    - Function: Evaluate different architectures and training strategies for multi-agent policy learning
    - Mechanism: Three architectural paradigms are systematically evaluated on the RoboFactory benchmark: (a) **Independent Policy**—each agent trains an independent policy without information sharing, using standard ACT or Diffusion Policy per agent; (b) **Shared Policy**—all agents share a single policy network, distinguished by agent ID embeddings; parameter-efficient but potentially limited for heterogeneous tasks; (c) **Collaborative Policy**—each agent has an independent encoder but exchanges information through intermediate layers (e.g., cross-attention); end-to-end optimized over all agents' joint actions during training. Communication mechanisms are also explored: direct feature concatenation vs. cross-attention vs. graph neural networks.
    - Design Motivation: Single-agent imitation learning methods (ACT, Diffusion Policy) cannot be directly applied to multi-agent scenarios, necessitating architectures that model collaboration. Systematic exploration provides empirical guidance for the community.

### Loss & Training

During imitation learning, action prediction losses are employed—L1 action loss plus KL divergence regularization for ACT-type methods, and denoising loss for the diffusion process for Diffusion Policy-type methods. In joint multi-agent training, the total loss is the sum of action losses across all agents. A constraint violation penalty term is introduced to encourage policies to learn behaviors that satisfy all constraints.

## Key Experimental Results

### Main Results

Success rate (%) comparison of different architectures across tasks of varying difficulty on the RoboFactory benchmark:

| Method | Easy (2 agent) | Medium (2 agent) | Hard (3 agent) | Very Hard (4 agent) |
|--------|----------------|------------------|----------------|---------------------|
| Independent ACT | 72.3 | 43.1 | 18.5 | 6.2 |
| Shared ACT | 68.7 | 47.6 | 22.3 | 9.8 |
| Collaborative ACT | 78.5 | 58.4 | 35.2 | 18.7 |
| Independent Diffusion | 70.1 | 41.8 | 16.9 | 5.4 |
| Collaborative Diffusion | 76.2 | 55.7 | 32.8 | 16.3 |

### Ablation Study

| Configuration | Medium Success Rate | Hard Success Rate | Notes |
|---------------|--------------------|--------------------|-------|
| Collaborative ACT (full) | 58.4 | 35.2 | Full collaborative policy |
| w/o inter-agent communication | 45.8 | 20.1 | Degenerates to independent policy |
| Communication: feature concat | 52.3 | 28.6 | Simple but non-selective |
| Communication: cross-attention | 58.4 | 35.2 | Optimal communication mechanism |
| Communication: GNN | 55.1 | 31.7 | Competitive but below attention |
| w/o constraint-aware data | 51.2 | 24.3 | Collision data induces erroneous behaviors |

### Key Findings

- **Collaborative architectures show substantial advantages on difficult tasks**: Independent policies are adequate for simple dual-arm tasks, but the advantage of collaborative architectures increases sharply as agent count and constraint complexity grow (35.2% vs. 18.5% on Hard).
- **Cross-attention is the optimal communication mechanism**: It outperforms simple concatenation by 6% and GNN by 3%, as attention can learn to selectively attend to relevant agents.
- **Constraint-aware data collection is critical**: Automated collection without constraint checking produces trajectories containing collisions, causing learned policies to reproduce collision behaviors.
- Task success rates drop sharply as the number of agents increases, indicating that multi-agent collaborative manipulation remains a highly challenging open problem.
- ACT architectures consistently outperform Diffusion Policy, possibly because the action space of manipulation tasks is better suited to deterministic policies.

## Highlights & Insights

- **The compositional constraints concept is highly elegant**: Decomposing complex multi-agent collaboration requirements into reusable primitive constraint components transforms the definition of new tasks into a process of "selecting and composing constraints." This modular design has significant implications for both benchmark construction and data generation.
- **First multi-agent manipulation benchmark**: This fills a critical gap in the embodied intelligence community. The benchmark is designed to span multiple difficulty levels from simple to extremely hard, providing a standardized evaluation platform for subsequent work.
- **Systematic architecture exploration**: Rather than proposing a single method, the paper comprehensively compares independent, shared, and collaborative paradigms along with multiple communication mechanisms, offering valuable empirical summaries for the community.

## Limitations & Future Work

- All experiments are conducted in simulation; sim-to-real transfer is an important unvalidated challenge.
- The current scope is limited to robotic arm manipulation and has not been extended to mobile robots or heterogeneous agent collaboration.
- Automated data collection relies on the capability of motion planners, which may fail for highly unstructured tasks.
- Incorporating VLMs into multi-agent collaboration—leveraging language instructions to decompose collaborative tasks—is a promising future direction.
- Constraint discovery—automatically inferring constraint types and parameters from human demonstrations rather than defining them manually—warrants further exploration.

## Related Work & Insights

- **vs. RLBench / ManiSkill**: Single-agent manipulation benchmarks that do not address multi-agent collaboration. The key distinction of RoboFactory is the introduction of interaction constraints among multiple agents.
- **vs. Multi-Agent RL (MARL)**: MARL methods (e.g., MAPPO, QMIX) are primarily studied in abstract environments such as games. RoboFactory focuses on fine-grained manipulation collaboration in physical space, with more concrete constraints (collisions, forces).
- **vs. ACT / Diffusion Policy**: These methods are designed for single agents; this paper demonstrates that direct extension to multi-agent settings leads to severe performance degradation, justifying the need for dedicated multi-agent architecture design.

## Rating

- Novelty: ⭐⭐⭐⭐ The compositional constraints concept is novel and fills the gap in multi-agent manipulation benchmarks
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic comparison across multiple architectures, communication mechanisms, and difficulty levels
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; constraint formalization is rigorous
- Value: ⭐⭐⭐⭐ The benchmark and framework provide a foundational contribution to embodied multi-agent research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multi-Agent Collaboration via Evolving Orchestration](../../NeurIPS2025/reinforcement_learning/multi-agent_collaboration_via_evolving_orchestration.md)
- [\[ICCV 2025\] Embodied Navigation with Auxiliary Task of Action Description Prediction](embodied_navigation_with_auxiliary_task_of_action_description_prediction.md)
- [\[NeurIPS 2025\] VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/viki-r_coordinating_embodied_multi-agent_cooperation_via_reinforcement_learning.md)
- [\[AAAI 2026\] Learning to Generate and Extract: A Multi-Agent Collaboration Framework for Zero-shot Document-level Event Arguments Extraction](../../AAAI2026/reinforcement_learning/learning_to_generate_and_extract_a_multi-agent_collaboration_framework_for_zero-.md)
- [\[NeurIPS 2025\] Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models](../../NeurIPS2025/reinforcement_learning/communicating_plans_not_percepts_scalable_multi-agent_coordination_with_embodied.md)

</div>

<!-- RELATED:END -->
