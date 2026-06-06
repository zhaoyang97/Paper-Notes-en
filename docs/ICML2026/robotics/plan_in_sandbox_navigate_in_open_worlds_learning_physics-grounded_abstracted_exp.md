---
title: >-
  [Paper Note] Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation
description: >-
  [ICML 2026][Robotics][Physical Sandbox] This paper proposes SAGE: automatically synthesizing mass navigation tasks and IF-THEN experience rules in a physics-constrained semantic sandbox…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Physical Sandbox"
  - "Generative Experience"
  - "GRPO"
  - "Asymmetric Clipping"
  - "A-EQA"
  - "GOAT-Bench"
date: 2026-05-08
content_hash: 90a6d7475693ccab
---

# Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation

**Conference**: ICML 2026  
**arXiv**: [2605.10118](https://arxiv.org/abs/2605.10118)  
**Code**: Undisclosed  
**Area**: Embodied Navigation / VLM Reinforcement Learning / Sim2Real  
**Keywords**: Physical Sandbox, Generative Experience, GRPO, Asymmetric Clipping, A-EQA, GOAT-Bench

## TL;DR
This paper proposes SAGE: automatically synthesizing mass navigation tasks and IF-THEN experience rules in a physics-constrained semantic sandbox, then distilling these experiences into a VLM policy using GRPO with mixed prompt sampling and Asymmetric Adaptive Clipping. It improves LLM-Match success rate on A-EQA from 43.5% to 53.2% (2B) / 60.2% (4B) and successfully transfers to real-world indoor robots.

## Background & Motivation
**Background**: VLMs (GPT-4o, Qwen3-VL, etc.) exhibit strong open-world perception and reasoning, leading to VLM-driven embodied navigation paradigms: object-oriented (ObjectNav, IIN) and question-answering oriented (A-EQA, OpenEQA). RL methods (SenseAct) attempt end-to-end policies, while modular methods (3D-Mem, Explore-EQA) use VLMs as high-level planners.

**Limitations of Prior Work**: (1) "Vision-robot control" data aligned with the real world is scarce, creating a massive modal gap between VLMs and continuous action spaces; training RL from scratch converges slowly and suffers severe Sim2Real degradation. (2) Forcibly trained policies often fail in real environments or rely on closed-source models like GPT-4o; open-source medium-scale VLMs show a significant performance gap.

**Key Challenge**: VLMs possess rich priors but cannot continuously learn low-level control online, while RL has learning mechanisms but suffers from low sample efficiency; a bridge to leverage both strengths is missing. Simulators that are photorealistic but physically inconsistent (or vice versa) do not solve the root problem.

**Goal**: (1) Provide VLMs with massive, diverse, and physically executable navigation experiences without relying on large-scale real-world data collection; (2) Design an RL algorithm to stably distill these experiences into the policy; (3) Enable policies learned in a sandbox to zero-shot transfer to the open world.

**Key Insight**: Humans rehearse in a "mental sandbox" before executing plans—abstract physical constraints and semantic scene graphs are sufficient, rather than realistic rendering. Thus, a VLM can generate tasks, record successful paths, and extract IF-THEN rules within a "physics-constrained + semantically abstracted" sandbox (HM3D/InteriorGS parsed into graphs with discrete semantic nodes and collision constraints).

**Core Idea**: Treat the sandbox as an "experience factory" for the VLM to generate structured task sets $\mathcal O$ and experience rule bases $\mathcal K_{exp}$. Then, use GRPO with "asymmetric clipping" (distinguishing augmented and standard samples) to internalize retrieved external experiences into the VLM's parametric policy.

## Method

### Overall Architecture
SAGE consists of three phases: (1) **Genesis**: Samples start/end points in a sandbox environment $\mathcal E_S=(\mathcal S,\mathcal A,\mathcal P)$, performs A* planning, and renders three-view observations $\mathcal V_t=\{v_{t,0°},v_{t,+120°},v_{t,-120°}\}$ at keypoints. A VLM synthesizes scene graphs and goal descriptions into natural language instructions $I$ and answers $a^*$, forming tasks $o=(I,\tau^*,a^*,\mathcal K)$. Simultaneously, the VLM encodes reasons for optimal view selection into "IF task X AND observation Y THEN priority path Z" rules in a vector database $\mathcal D_{exp}$. (2) **Evolution**: Optimizes policy $\pi_\theta$ on $\mathcal O$ using GRPO. Input retrieval experience $\mathcal K_{ret}$ is injected based on Bernoulli probability $\eta_t$. Advantages are calculated per homogeneous group, and PPO clip bounds are determined by masks. (3) **Navigation**: Deployment uses a "retrieval + VLM decision + geometric planner" pipeline: RGB-D and dynamic 3D scene graphs maintain a Memory Buffer $\mathcal M_t$ (seen objects) and a Frontier Buffer $\mathcal F_t$ (unexplored boundaries). The VLM selects target nodes from $\mathcal F_t\cup\mathcal M_t$, and Habitat-Sim/ROS planners execute.

### Key Designs

1. **Physical Sandbox Experience Generation (Genesis)**:
    - **Function**: Automatically synthesizes navigation tasks, optimal trajectories, and decision reasoning using VLMs and physical constraints to construct a structured experience library.
    - **Mechanism**: Parses HM3D/InteriorGS into "semantic state graphs" where each room is decomposed into discrete navigable nodes, and state transitions strictly follow traversability constraints. Task synthesis uses an A* + keypoint rendering + VLM captioning pipeline, with the forward view $v_{t,0°}$ as $a^*$. Rule synthesis asks the VLM to explain each step's view selection as IF-THEN rules, which are encoded into $\mathcal D_{exp}$.
    - **Design Motivation**: Abandoning photorealistic simulators reduces rendering overhead and Sim2Real degradation. Physical constraints + semantic abstraction are cheaper and naturally align with "3D scene graph + buffer" representations used during real deployment, reducing distribution shift.

2. **Mixed Prompt Sampling with Homogeneous Group Advantage Estimation**:
    - **Function**: Distinguishes "augmented samples with experience prompts" from "standard samples without prompts" during GRPO training to prevent the baseline of the latter from being contaminated by the former.
    - **Mechanism**: Dynamic injection probability $\eta_t=\max(\eta_{\min},\eta_{init}\cdot(1-\min(R_{val}^{(t)},R_{target})/R_{target}))$. As validation reward increases, $\eta_t$ decreases, transitioning from "imitating retrieval" to "autonomous exploration." For each input $x_i$, $G$ rollouts are sampled, but the mask $m_i$ is forced to be consistent within the group (homogeneous grouping): $x_t=[I_t,v_t,\mathcal K_{ret}]$ if $m=1$, else $[I_t,v_t]$. Advantages $A_{i,j}=(r_\phi(x_i,a_{i,j})-\mu)/(\sigma+\epsilon)$ are normalized within the group.
    - **Design Motivation**: Augmented samples naturally yield higher rewards. If mixed with standard samples for $\mu, \sigma$ calculation, standard sample advantages are depressed, causing good behavior to be misjudged as poor. Homogeneous grouping isolates the two distributions.

3. **Asymmetric Adaptive Clipping (AAC)**:
    - **Function**: Allows the policy to "update boldly when learning high-reward augmented samples" and remain "conservative and robust when learning standard samples," while avoiding over-punishment of augmented samples misjudged as low-reward due to noise.
    - **Mechanism**: Define $\rho_{i,t}(\theta)=\pi_\theta(a_{i,t}\mid x_{i,t})/\pi_{\theta_{old}}(a_{i,t}\mid x_{i,t})$. The upper bound is determined by the mask $\epsilon_{up}(m_i)=\epsilon_{exp}$ (augmented) or $\epsilon_{std}$ (standard), where $\epsilon_{exp}\gg\epsilon_{std}$. The lower bound is unified at a conservative $1-\epsilon_{std}$. The clipped loss is $L_{i,t}^{CLIP}=\min(\rho_{i,t}A_{i,t},\text{clip}(\rho_{i,t},1-\epsilon_{std},1+\epsilon_{up}(m_i))A_{i,t})$, with a KL constraint $J_\phi(\theta)=\mathbb E[L^{CLIP}-\beta\mathbb D_{KL}(\pi_\theta\|\pi_{ref})]$.
    - **Design Motivation**: Symmetrical clipping in classic PPO/GRPO implies "good behavior cannot update too much," contradicting the need to rapidly absorb high-quality experience. Asymmetric clipping provides more space upwards but remains conservative downwards to prevent policy collapse from noisy golden samples.

### Loss & Training
Reward $r_\phi(s_t,a_t)=w_f\mathbb I_f+w_{acc}(\mathbb I_m(1+\text{sim}(a_t,a_t^*))-\mathcal P_{err})$, including format compliance, correct image selection, text similarity, and error penalties. The optimizer is a GRPO variant with KL regularization (AAC). Training data: 14,526 valid trajectories (HM3D 7,988 + InteriorGS 6,538). Hyperparameters: $\eta_{init}=0.8,\eta_{min}=0,R_{target}=1.5, \epsilon_{exp}=1.0$, converging in 150 steps.

## Key Experimental Results

### Main Results
Two benchmarks: A-EQA (184 Q&A-oriented tasks, SR†/SPL† auto-scored by Qwen3-235B) and GOAT-Bench (278 target-oriented subtasks).

| Method | A-EQA SR† | A-EQA SPL† | GOAT SR | GOAT SPL |
|------|-----------|------------|---------|----------|
| SenseAct-NN Skill Chain (RL) | 24.7 | 13.3 | 29.5 | 11.3 |
| Explore-EQA (GPT-4o) | 46.9 | 23.4 | 55.0 | 37.9 |
| 3D-Mem (GPT-4o) | 52.6 | 42.0 | 69.1 | 48.9 |
| 3D-Mem (Qwen3-2B) | 44.3 | 19.4 | 46.4 | 20.3 |
| **SAGE (Qwen3-2B)** | **53.2** | **37.1** | **56.7** | **38.9** |
| **SAGE (Qwen3-4B)** | **60.2** | **47.2** | **64.8** | **44.9** |

Using the same backbone, SAGE-2B achieves +8.9% A-EQA SR† and +10.3% GOAT SR, with SPL nearly doubling, even surpassing the GPT-4o version of 3D-Mem on A-EQA SR†. SAGE-4B pushes A-EQA to a new SOTA of 60.2%.

### Ablation Study
**Cumulative Ablation of Main Components (Qwen3-VL-2B → SAGE Full):**

| Configuration | A-EQA SR† | A-EQA SPL† | GOAT SR |
|------|-----------|------------|---------|
| Zero-shot VLM | 43.51 | 27.53 | 49.17 |
| +$C_{ret}$ Retrieval Only | 46.47 | 30.72 | 50.58 |
| +Task Synthesis Training | 50.71 | 33.68 | 53.72 |
| +Task+Exp Experience Rules | 51.42 | 34.67 | 54.05 |
| +Task+Exp+AAC | 51.88 | 36.29 | 55.35 |
| **SAGE Full (with $C_{ret}$)** | **53.21** | **37.07** | **56.69** |

**Navigation Phase Ablation**: Training with Genesis+Evolution without retrieval already improves SR† by 6.29%. Adding random experience adds +1.93%, and correct retrieval adds +1.48%.

### Key Findings
- **Dynamic $\eta_t$** is significantly better than fixed values: fixed $\eta=0.0/0.5/0.8/1.0$ all underperform validation-driven annealing, confirming that a "imitation then exploration" curriculum is necessary.
- **$\epsilon_{exp}$ sweet spot at 1.0**: lower (0.4) results in underfitting, while higher (1.2) causes training collapse after 100 steps; AAC upper bound is not "the larger the better."
- **Sandbox data scaling**: Performance increases monotonically from 12.5% to 100% but shows diminishing returns; 12.5% already reaches 44.75% SR†, proving "cheap data from physical sandboxes" is highly scalable.
- **Input frames $v_t$**: Increasing from 2 to 4 gives significant gains; 5 frames causes a slight drop (visual tokens diluting attention). The optimal is 4 frames.
- **Real-world deployment**: Successful indoor robot deployment (Appendix J) demonstrates that the decoupling of "Sandbox Abstraction → Node Selection → ROS Planner" successfully traverses the Sim2Real gap.

## Highlights & Insights
- **"Rehearse in sandbox before hitting the road" analogy to mental simulation**: Moving beyond the photorealistic simulator mindset, using abstract physical + semantic graphs as a VLM training ground is cheap and aligns with deployment representations.
- **AAC is an attractive modification to GRPO/PPO**: "Adaptive upper bound, uniform conservative lower bound" is universally applicable to any "RLHF / self-iteration with high-quality demonstrations" scenario, such as code or math RL with expert traces.
- **Discrete action space with Frontier+Memory Buffers**: Simplifying continuous control to "select from enumerable nodes" allows VLM token-level reasoning to act directly as decisions, bypassing the uninterpretability of continuous actions.
- **Homogeneous Group Advantage Estimation**: A simple but easily overlooked detail when applying GRPO to mixed-distribution data; this paper provides a clear implementation.

## Limitations & Future Work
- The sandbox environment is based solely on existing datasets (HM3D / InteriorGS); generalization to new scenes still relies on the base VLM rather than true transfer learning. Dynamic environments (moving people, movable objects) are not covered.
- Real robot experiments are in the appendix rather than the main tables; deep deployment data is limited, and system-level metrics like long-term reliability or battery life are missing.
- Reward design relies on text similarity, posing a "format hack" risk for abstract spatial tasks (counting, spatial relationships).
- Experience rules are stored as IF-THEN strings; as scale increases, retrieval accuracy and noise management become concerns. Future work needs more structured knowledge graph formats.

## Related Work & Insights
- **vs 3D-Mem (yang2025b)**: Both maintain scene memory, but 3D-Mem does not train the VLM and relies on GPT-4o; SAGE trains medium-scale open-source VLMs to outperform closed-source models.
- **vs SenseAct-NN (khanna2024)**: Pure RL without VLM priors, resulting in significantly worse performance.
- **vs Explore-EQA (ren2024)**: Uses GPT-4o for exploration without an explicit experience base; SAGE distills experience into a retrievable structure via the sandbox.
- **vs Standard GRPO**: SAGE's AAC + homogeneous groups + mixed prompts can be seen as a more refined version of GRPO for "prior data + RL" scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The "physical + semantic abstract sandbox + experience rules" is innovative, though individual components are refined combinations of existing ideas.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ A-EQA + GOAT + 5 types of ablations + real-robot deployment provide comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ The three-phase narrative is clear with proper notation, though reward design descriptions are somewhat brief.
- **Value**: ⭐⭐⭐⭐ Provides a GPT-4o alternative for embodied navigation on medium-scale VLMs; the AAC approach is transferable to general RLHF.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] R2R2: Robust Representation for Intensive Experience Reuse via Redundancy Reduction in Self-Predictive Learning](r2r2_robust_representation_for_intensive_experience_reuse_via_redundancy_reducti.md)
- [\[ICLR 2026\] OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning](../../ICLR2026/robotics/omnieva_embodied_versatile_planner_via_task-adaptive_3d-grounded_and_embodiment-.md)
- [\[ICLR 2026\] ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning](../../ICLR2026/robotics/exopredicator_learning_abstract_models_of_dynamic_worlds_for_robot_planning.md)
- [\[ICML 2026\] DLO-Lab: Benchmarking Deformable Linear Object Manipulations with Differentiable Physics](dlo-lab_benchmarking_deformable_linear_object_manipulations_with_differentiable_.md)
- [\[ICML 2026\] Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation](dive_into_the_scene_breaking_the_perceptual_bottleneck_in_vision-language_decisi.md)

</div>

<!-- RELATED:END -->
