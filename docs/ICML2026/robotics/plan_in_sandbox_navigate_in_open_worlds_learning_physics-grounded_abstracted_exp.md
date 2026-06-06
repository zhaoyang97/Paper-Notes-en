---
title: >-
  [Paper Note] Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation
description: >-
  [ICML 2026][Robotics][Physics Sandbox] This paper proposes SAGE: automatically synthesizing large-scale navigation tasks and IF-THEN experience rules in a physics-constrained semantic sandbox…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Physics Sandbox"
  - "Generative Experience"
  - "GRPO"
  - "Asymmetric Clipping"
  - "A-EQA"
  - "GOAT-Bench"
date: 2026-05-08
content_hash: f36eadf76ef28e50
---

# Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation

**Conference**: ICML 2026  
**arXiv**: [2605.10118](https://arxiv.org/abs/2605.10118)  
**Code**: Not released  
**Area**: Embodied Navigation / VLM Reinforcement Learning / Sim2Real  
**Keywords**: Physics Sandbox, Generative Experience, GRPO, Asymmetric Clipping, A-EQA, GOAT-Bench

## TL;DR
This paper proposes SAGE: automatically synthesizing large-scale navigation tasks and IF-THEN experience rules in a physics-constrained semantic sandbox, then distilling these experiences into a VLM policy using hybrid prompt sampling and asymmetric adaptive clipping GRPO. This approach boosts LLM-Match success rate on A-EQA from 43.5% to 53.2% (2B) / 60.2% (4B), and enables transfer to real indoor robots.

## Background & Motivation
**Background**: VLMs (GPT-4o, Qwen3-VL, etc.) excel at open-world perception and reasoning, spurring a wave of VLM-driven embodied navigation: object-goal (ObjectNav, IIN) and question-goal (A-EQA, OpenEQA) paradigms. RL methods (SenseAct) attempt end-to-end policy learning, while modular approaches (3D-Mem, Explore-EQA) use VLMs as high-level planners.

**Limitations of Prior Work**: (1) Real-world aligned "vision-to-robot control" data is scarce, and there is a significant modality gap between VLMs and continuous action spaces. RL from scratch converges slowly and suffers severe Sim2Real degradation; (2) Policies trained by force either fail in real environments (high noise, unfamiliar layouts) or rely on closed-source large models like GPT-4o, with open-source mid-scale VLMs performing much worse in practice.

**Key Challenge**: VLMs possess rich priors but cannot continuously learn low-level control online; RL has learning mechanisms but suffers from low sample efficiency. There is a missing bridge to combine their strengths. Simulators with photorealism but inconsistent physics, or vice versa, cannot fundamentally solve the problem.

**Goal**: (1) Provide VLM policies with massive, diverse, physically executable navigation experience without relying on large-scale real-world data collection; (2) Design an RL algorithm to stably distill these experiences into the policy; (3) Ensure that policies learned in the sandbox can zero-shot transfer to open worlds.

**Key Insight**: Humans plan by first rehearsing in a "mental sandbox"—abstract physical constraints plus semantic scene graphs suffice, without the need for photorealistic rendering. Thus, let VLMs generate tasks, record successful paths, and extract IF-THEN rules in a "physics-constrained + semantic abstraction" sandbox (HM3D / InteriorGS parsed into discrete semantic nodes + collision-constrained graphs).

**Core Idea**: Treat the sandbox as an "experience factory" for the VLM, generating a structured task set $\mathcal O$ and experience rule base $\mathcal K_{exp}$, then use GRPO with asymmetric clipping that distinguishes enhanced and standard samples to internalize external retrieved experience into the VLM's parameterized policy.

## Method

### Overall Architecture
SAGE consists of three stages: (1) **Genesis** samples start and end points in the sandbox environment $\mathcal E_S=(\mathcal S,\mathcal A,\mathcal P)$, plans with A*, renders three-view observations at key points $\mathcal V_t=\{v_{t,0°},v_{t,+120°},v_{t,-120°}\}$, and uses the VLM to synthesize natural language instructions $I$ and answers $a^*$ from the scene graph and goal description, forming tasks $o=(I,\tau^*,a^*,\mathcal K)$. The VLM also encodes the rationale for optimal viewpoint selection at each step as "IF task X AND observation Y THEN prioritize path Z" rules, stored in the vector database $\mathcal D_{exp}$. (2) **Evolution** uses GRPO to optimize policy $\pi_\theta$ on $\mathcal O$, with input determined by Bernoulli probability $\eta_t$ for injecting retrieved experience $\mathcal K_{ret}$, advantage computed by homogeneous group, and PPO clip bounds determined by mask. (3) **Navigation** at deployment still follows the "retrieval + VLM decision + geometric planner" pipeline: RGB-D and dynamic 3D scene graphs maintain a Memory Buffer $\mathcal M_t$ (seen objects) and Frontier Buffer $\mathcal F_t$ (unexplored boundaries), VLM selects target nodes from $\mathcal F_t\cup\mathcal M_t$, and Habitat-Sim / ROS planners execute.

### Key Designs

1. **Physics Sandbox Experience Generation (Genesis)**:

    - **Function**: Automatically synthesize navigation tasks, optimal trajectories, and decision rationales using VLM + physical constraints, constructing a structured experience base.
    - **Mechanism**: Parse HM3D / InteriorGS into a "semantic state graph"—each room decomposed into discrete navigable nodes, with state transitions strictly following traversability constraints. Task synthesis uses A* + keypoint viewpoint rendering + VLM captioning pipeline, with the forward view $v_{t,0°}$ as $a^*$. Rule synthesis has the VLM explain "why this viewpoint was chosen" at each step as IF-THEN, encoded into $\mathcal D_{exp}$.
    - **Design Motivation**: Photorealistic simulators are abandoned mainly due to high rendering costs and severe Sim2Real degradation—physical constraints + semantic abstraction are cheaper and naturally aligned with the "3D scene graph + buffer" representation used in real deployment, reducing test-time distribution shift.

2. **Homogeneous Group Advantage Estimation with Hybrid Prompt Sampling**:

    - **Function**: Distinguish "enhanced samples with experience prompts" from "standard samples without prompts" during GRPO training, preventing baseline contamination.
    - **Mechanism**: Dynamic injection probability $\eta_t=\max(\eta_{\min},\eta_{init}\cdot(1-\min(R_{val}^{(t)},R_{target})/R_{target}))$; higher validation reward leads to lower $\eta_t$, gradually transitioning from "imitation of retrieval" to "autonomous exploration". Each input $x_i$ samples $G$ rollouts with enforced group-wise mask $m_i$ consistency (homogeneous grouping): $x_t=[I_t,v_t,\mathcal K_{ret}]$ when $m=1$, otherwise $[I_t,v_t]$. Advantage $A_{i,j}=(r_\phi(x_i,a_{i,j})-\mu)/(\sigma+\epsilon)$ is normalized within the group.
    - **Design Motivation**: Enhanced samples naturally have higher rewards (retrieved good experience directly copied), so mixing with standard samples for $\mu,\sigma$ would suppress or even invert the advantage of standard samples, misclassifying good behavior as bad; homogeneous grouping fundamentally isolates the two distributions.

3. **Asymmetric Adaptive Clipping (AAC)**:

    - **Function**: Allow the policy to update aggressively when learning from high-reward enhanced samples, but conservatively and robustly when learning from standard samples, while avoiding over-penalization of enhanced samples misjudged as low-reward due to noise.
    - **Mechanism**: Define $\rho_{i,t}(\theta)=\pi_\theta(a_{i,t}\mid x_{i,t})/\pi_{\theta_{old}}(a_{i,t}\mid x_{i,t})$, with the upper bound determined by mask: $\epsilon_{up}(m_i)=\epsilon_{exp}$ (enhanced) or $\epsilon_{std}$ (standard), $\epsilon_{exp}\gg\epsilon_{std}$; the lower bound is unified for all samples at a conservative $1-\epsilon_{std}$. The clipped loss is $L_{i,t}^{CLIP}=\min(\rho_{i,t}A_{i,t},\text{clip}(\rho_{i,t},1-\epsilon_{std},1+\epsilon_{up}(m_i))A_{i,t})$, with a KL constraint on the full objective $J_\phi(\theta)=\mathbb E[L^{CLIP}-\beta\mathbb D_{KL}(\pi_\theta\|\pi_{ref})]$.
    - **Design Motivation**: Classic PPO/GRPO symmetric clipping means "good behavior cannot be updated too much", which conflicts with the need to quickly absorb high-quality experience; asymmetric clipping allows more upward flexibility. The lower bound must remain conservative, otherwise a golden sample misclassified due to reward variance would be heavily downweighted, causing policy collapse.

### Loss & Training
Reward $r_\phi(s_t,a_t)=w_f\mathbb I_f+w_{acc}(\mathbb I_m(1+\text{sim}(a_t,a_t^*))-\mathcal P_{err})$ includes format compliance, correct image selection, text similarity reward, and error penalty. The optimizer is a GRPO variant with KL regularization (AAC). Training data: 14,526 valid synthetic trajectories (HM3D 7,988 + InteriorGS 6,538), $\eta_{init}=0.8,\eta_{min}=0,R_{target}=1.5$, $\epsilon_{exp}=1.0$ (optimal), converges in 150 steps.

## Key Experimental Results

### Main Results
Two benchmarks: A-EQA (184 question-goal tasks, SR†/SPL† auto-scored by Qwen3-235B), GOAT-Bench (278 subtasks, object-goal).

| Method | A-EQA SR† | A-EQA SPL† | GOAT SR | GOAT SPL |
|--------|-----------|------------|---------|----------|
| SenseAct-NN Skill Chain (RL) | 24.7 | 13.3 | 29.5 | 11.3 |
| Explore-EQA (GPT-4o) | 46.9 | 23.4 | 55.0 | 37.9 |
| 3D-Mem (GPT-4o) | 52.6 | 42.0 | 69.1 | 48.9 |
| 3D-Mem (Qwen3-2B) | 44.3 | 19.4 | 46.4 | 20.3 |
| **SAGE (Qwen3-2B)** | **53.2** | **37.1** | **56.7** | **38.9** |
| **SAGE (Qwen3-4B)** | **60.2** | **47.2** | **64.8** | **44.9** |

SAGE-2B achieves +8.9% A-EQA SR† and +10.3% GOAT SR over the same backbone, with SPL nearly doubled; A-EQA SR† even surpasses 3D-Mem (GPT-4o). SAGE-4B sets a new SOTA of 60.2% on A-EQA.

### Ablation Study
**Cumulative ablation of main components (Qwen3-VL-2B → SAGE Full):**

| Configuration | A-EQA SR† | A-EQA SPL† | GOAT SR |
|---------------|-----------|------------|---------|
| Zero-shot VLM | 43.51 | 27.53 | 49.17 |
| +$C_{ret}$ Retrieval Only | 46.47 | 30.72 | 50.58 |
| +Task Synthetic Task Training | 50.71 | 33.68 | 53.72 |
| +Task+Exp Add Experience Rules | 51.42 | 34.67 | 54.05 |
| +Task+Exp+AAC | 51.88 | 36.29 | 55.35 |
| **SAGE Full (+$C_{ret}$)** | **53.21** | **37.07** | **56.69** |

**Navigation phase ablation**: Training Genesis+Evolution without retrieval already improves SR† by 6.29%, adding random experience +1.93%, adding correct retrieval +1.48%.

### Key Findings
- Dynamic $\eta_t$ significantly outperforms fixed values: fixed $\eta=0.0/0.5/0.8/1.0$ all underperform validation-driven annealing, confirming the necessity of "imitation-then-exploration" curriculum.
- The sweet spot for $\epsilon_{exp}$ is 1.0: at 0.4, absorption is insufficient (underfitting); at 1.2, training collapses after 100 steps, indicating AAC's upper bound is not "the bigger the better".
- Sandbox data volume from 12.5% → 100% shows monotonic improvement but diminishing returns; 12.5% already achieves 44.75% SR†, indicating "cheap data generated by the physics sandbox" can be massively scaled.
- Number of input frames $v_t$: 2 → 4 yields significant improvement, while 5 slightly decreases performance (visual tokens dilute attention); 4 frames is optimal.
- Successful deployment on real indoor robots (Appendix J) demonstrates that the sandbox abstraction → node selection → ROS planner decoupling indeed bridges Sim2Real.

## Highlights & Insights
- **"Rehearse in sandbox before hitting the road" as a mental simulation analogy**: Moves beyond the photorealistic simulator mindset, using abstract physics + semantic graphs as the VLM's training ground—both cost-effective and aligned with deployment representations.
- **AAC is a compelling tweak to GRPO/PPO**: "Adaptive upper bound, unified conservative lower bound" can be generally applied to any RLHF/self-iterative scenario with high-quality demonstrations, e.g., code RL, math RL with expert traces.
- **Frontier+Memory Buffer as discrete action space**: Reduces continuous control to "selecting from enumerable nodes", allowing VLM token-level reasoning to directly serve as decision-making, avoiding the non-interpretability of continuous actions—a clever engineering decoupling.
- **Homogeneous group advantage estimation**: When applying GRPO to mixed-distribution data, this is a simple but often overlooked detail, clearly demonstrated in this work.

## Limitations & Future Work
- The sandbox environment is based only on existing datasets (HM3D / InteriorGS); generalization to new scenes still relies on the base VLM rather than true transfer learning. Dynamic environments (moving people, movable objects) are not covered.
- Real robot experiments are in the appendix rather than the main table, with limited deployment data; long-term reliability, battery life, and other system-level metrics are not presented.
- Reward design depends on text similarity, which may be susceptible to "format hacks" in abstract spatial tasks (counting, spatial relations).
- Experience rules are stored as IF-THEN strings; as scale increases, retrieval accuracy and noise management become concerns. More structured knowledge graph formats are needed in the future.

## Related Work & Insights
- **vs 3D-Mem (yang2025b)**: Also maintains scene memory, but 3D-Mem does not train the VLM, relying on GPT-4o's capabilities; SAGE trains mid-scale open-source VLMs to surpass closed-source large models.
- **vs SenseAct-NN (khanna2024)**: Pure RL without VLM priors, with much worse performance.
- **vs Explore-EQA (ren2024)**: Uses GPT-4o for exploration without an explicit experience base; SAGE consolidates experience into a retrievable structure via the sandbox.
- **vs Standard GRPO**: SAGE's AAC + homogeneous group + hybrid prompting can be seen as a more refined version of GRPO for "prior data + RL" scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ "Physics + semantic abstraction sandbox + experience rules" is creative, though each component is a refined combination of existing ideas.
- Experimental Thoroughness: ⭐⭐⭐⭐ A-EQA + GOAT + 5 types of ablation + real robot deployment, comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Three-stage narrative is clear, formulas and notation are present, but reward design description is somewhat brief.
- Value: ⭐⭐⭐⭐ Provides a GPT-4o alternative for embodied navigation on mid-scale VLMs; AAC approach is transferable to general RLHF.

## Related Papers

- [\[ICLR 2026\] ExGRPO: Learning to Reason from Experience](../../ICLR2026/reinforcement_learning/exgrpo_learning_to_reason_from_experience.md)
- [\[CVPR 2025\] CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos](../../CVPR2025/reinforcement_learning/citywalker_learning_embodied_urban_navigation_from_web-scale_videos.md)
- [\[ICML 2026\] Perceptual Flow Network for Visually Grounded Reasoning](perceptual_flow_network_for_visually_grounded_reasoning.md)
- [\[ICCV 2025\] Embodied Navigation with Auxiliary Task of Action Description Prediction](../../ICCV2025/reinforcement_learning/embodied_navigation_with_auxiliary_task_of_action_description_prediction.md)
- [\[ICML 2026\] R2R2: Robust Representation for Intensive Experience Reuse via Redundancy Reduction in Self-Predictive Learning](r2r2_robust_representation_for_intensive_experience_reuse_via_redundancy_reducti.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning](../../ICLR2026/robotics/omnieva_embodied_versatile_planner_via_task-adaptive_3d-grounded_and_embodiment-.md)
- [\[ICLR 2026\] ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning](../../ICLR2026/robotics/exopredicator_learning_abstract_models_of_dynamic_worlds_for_robot_planning.md)
- [\[ICLR 2026\] Experience-based Knowledge Correction for Robust Planning in Minecraft](../../ICLR2026/robotics/experience-based_knowledge_correction_for_robust_planning_in_minecraft.md)
- [\[CVPR 2026\] IGen: Scalable Data Generation for Robot Learning from Open-World Images](../../CVPR2026/robotics/igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)
- [\[CVPR 2026\] Towards Open Environments and Instructions: General Vision-Language Navigation via Fast-Slow Interactive Reasoning](../../CVPR2026/robotics/towards_open_environments_and_instructions_general_vision-language_navigation_vi.md)

</div>

<!-- RELATED:END -->
