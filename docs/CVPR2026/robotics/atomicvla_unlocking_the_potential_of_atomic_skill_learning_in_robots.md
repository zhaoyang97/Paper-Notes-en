---
title: >-
  [Paper Note] AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots
description: >-
  [CVPR 2026][Robotics][VLA] This paper proposes AtomicVLA, a unified planning-execution framework built upon π₀ that adaptively switches between Think and Act modes to generate atomic skill abstractions…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "VLA"
  - "Atomic Skills"
  - "Mixture-of-Experts"
  - "Continual Learning"
  - "Task Planning"
  - "Skill Routing"
date: 2026-05-08
content_hash: 195973144e6561b1
---

# AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots

**Conference**: CVPR 2026
**arXiv**: [2603.07648](https://arxiv.org/abs/2603.07648)  
**Institution**: Sun Yat-sen University, Peng Cheng Laboratory, Yinwang Intelligence
**Area**: Robotic Manipulation / Vision-Language-Action Models
**Keywords**: VLA, Atomic Skills, Mixture-of-Experts, Continual Learning, Task Planning, Skill Routing

## TL;DR

This paper proposes AtomicVLA, a unified planning-execution framework built upon π₀ that adaptively switches between Think and Act modes to generate atomic skill abstractions, and employs a Skill-Guided MoE (SG-MoE) to route actions to specialized experts. The approach improves LIBERO-LONG success rate from 85.2% to 95.2% (+10%), achieves +18.3% on real-world Franka long-horizon tasks, and +21% on continual learning benchmarks.

## Background & Motivation

- **Existing Bottleneck**: Current VLA models (π₀, OpenVLA, etc.) employ a single action decoder that conflates all skill knowledge within the same parameter space. They lack explicit planning capabilities for multi-step long-horizon tasks and suffer from catastrophic forgetting when learning new skills.
- **Limitations of Prior Work**: Two-stage methods such as SayCan and Inner Monologue rely on an external LLM for high-level planning and independent low-level controllers for execution, but the planner and executor lack mutual awareness, resulting in stale or irrelevant instructions and system-level latency.
- **Core Problem**: A robot model must simultaneously support (1) high-level reasoning and task planning, (2) fine-grained action generation, and (3) scalable continual learning — requirements that existing methods cannot jointly satisfy.
- **Mechanism**: Complex tasks are decomposed into reusable atomic skills (e.g., grasp, push, rotate), each handled by a dedicated expert, enabling continual expansion of the skill library through modular design.

## Method

### Overall Architecture

AtomicVLA constructs an end-to-end framework on top of π₀ that unifies thinking and acting modalities. Given multi-camera observations $O_t^{1:n}$ and a language instruction $\ell$, the model first adaptively predicts whether to enter a thinking or acting mode:

- **Thinking Mode**: Activated at task initiation or sub-skill transitions; produces three outputs — a task chain $C_{0 \to k}$ (decomposing the instruction into ordered sub-goals), current progress $C_t$, and an atomic skill abstraction $\sigma$.
- **Acting Mode**: Activated during normal execution; conditions on the latest skill abstraction $\sigma$ and proprioceptive state $s_t$ to generate an action chunk $A_t$ via SG-MoE.

### Key Designs

**1. Adaptive Think-Act Switching Mechanism**

- **Function**: Enables the model to autonomously decide "whether to reason or act," avoiding the computational overhead of planning at every step.
- **Core Idea**: Two special output tokens, [think] and [act], are introduced. At each decision step, the model first predicts the identifier: [think] triggers high-level planning to generate the task chain and atomic skill labels, while [act] triggers action chunk output.
- **Design Motivation**: Planning is only needed at task initiation and sub-skill transitions; intermediate execution steps can directly generate actions. This adaptive switching is more efficient than fixed-period planning and more robust than pure action prediction without any planning. When execution fails (e.g., object drop), the model automatically detects the anomaly, re-triggers thinking, generates a new skill abstraction, and resumes execution.

**2. Skill-Guided Mixture-of-Experts Architecture (SG-MoE)**

- **Function**: Routes action generation for different atomic skills to their respective specialized experts, minimizing inter-skill interference.
- **Core Idea**: SG-MoE comprises three components:
    - **Skill Router**: Makes routing decisions based on the atomic skill abstraction $\sigma$. A noise-schedule-style embedding is adopted: each atomic skill is mapped to a scalar noise level $\sigma \in [0, 100]$, which is transformed into a high-dimensional vector $Z_\sigma = E(\text{norm}(\log(\sigma)))$ via a learnable embedding function. The router computes an expert probability distribution over $Z_\sigma$ and activates the top-1 skill expert.
    - **Shared Expert**: Retains the pretrained π₀ weights to preserve general action generation capability; all tokens pass through this expert.
    - **Atomic Skill Experts**: Each expert specializes in one atomic skill, with specialization emerging naturally through training.
- **Design Motivation**: Conventional token-level MoE routing still causes each expert to learn a mixture of skills without explicit specialization. SG-MoE routes using semantically explicit atomic skill labels, ensuring that all action tokens belonging to the same skill are processed by the same expert, thereby reducing inter-skill interference.
- **Output Fusion**: The final action is a weighted combination of the shared expert and the activated skill expert: $F_{out} = (1-w_k) \cdot F_{share}(x_t) + w_k \cdot F_k(x_t)$

**3. Modular Continual Learning (Skill Expansion)**

- **Function**: Enables continual acquisition of new skills without forgetting existing ones.
- **Core Idea**: When a new atomic skill is introduced, only (a) a new skill expert (randomly initialized) is added, (b) the skill router's embedding space is extended to cover the new skill label, and (c) all existing expert parameters are frozen — only the new expert and the updated router branch are trained.
- **Design Motivation**: Full fine-tuning of conventional VLAs causes catastrophic forgetting (e.g., π₀.₅ suffers an average 15% decline on existing skills after learning new ones). The "add-only, no-modify" modular strategy avoids forgetting at the architectural level; the router is initialized by copying existing weights to ensure smooth integration.

**4. High-Quality Embodied Reasoning Data Generation**

- **Function**: Provides high-quality training data (task chains, progress annotations, and skill labels) for the thinking mode.
- **Core Idea**: A two-stage data construction pipeline is employed. First, principal-axis motion analysis is applied to robot demonstration trajectories (comparing the magnitudes of translational/rotational components and tracking gripper states) to automatically segment trajectories into atomic skill segments (e.g., sustained z-axis descent + gripper closing → "pick" action). Second, InternVideo2.5 is used to semantically annotate each segment, automatically correcting and enriching the initial atomic action labels.
- **Design Motivation**: Traditional approaches rely on VLMs for video understanding or optical flow features for segmentation, which are prone to ambiguity and noise and require extensive manual post-processing. Decomposition based on kinematic principal-axis analysis is more precise and less dependent on human annotation.

### Loss & Training

- **Think Mode**: Cross-entropy loss over the predicted token sequence of task chains, progress, and skill labels.
- **Act Mode**: Flow matching loss (inherited from π₀) for continuous action chunk prediction.
- **Total Loss**: $\mathcal{L}_{total} = \mathcal{L}_{think} + \mathcal{L}_{act}$
- **Expert Configuration**: 5 skill experts for LIBERO and real-robot experiments; 8 skill experts for CALVIN.
- **Two Variants**: AtomicVLA is built on π₀; AtomicVLA\* is built on π₀.₅.

## Key Experimental Results

### Main Results: LIBERO Benchmark (Success Rate %)

| Method | Spatial | Object | Goal | Long | Avg. |
|--------|---------|--------|------|------|------|
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA | 84.9 | 88.4 | 79.2 | 53.7 | 76.5 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| π₀ | 96.4 | 98.8 | 95.8 | 85.2 | 94.2 |
| π₀.₅ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| **AtomicVLA** | 96.8 | 98.0 | 96.4 | **95.2** | **96.6** |
| **AtomicVLA\*** | **98.8** | **98.8** | **97.2** | **96.2** | **97.8** |

### Main Results: CALVIN ABC→D (Long-Horizon Sequence Completion Rate %)

| Method | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg. Len↑ |
|--------|--------|--------|--------|--------|--------|-----------|
| π₀ | 94.3 | 87.0 | 77.9 | 68.5 | 59.4 | 3.87 |
| π₀.₅ | 91.9 | 84.6 | 79.4 | 75.5 | 71.0 | 4.02 |
| **AtomicVLA** | 95.0 | 87.8 | 81.9 | 75.0 | 69.1 | **4.09** |
| **AtomicVLA\*** | 94.1 | **88.7** | **85.2** | **81.7** | **77.6** | **4.27** |

### Real Franka Robot: Long-Horizon Tasks (Success Rate %)

| Method | Object-to-Plate | Object-to-Drawer | Object-to-Microwave | Avg. | ΔAvg. |
|--------|-----------------|------------------|---------------------|------|-------|
| π₀ | 45 | 55 | 10 | 36.7 | — |
| π₀.₅ | 65 | 35 | 35 | 45.0 | — |
| AtomicVLA | 65 | 60 | 45 | 56.7 | +20.0 |
| AtomicVLA\* | **75** | **60** | **55** | **63.3** | **+18.3** |

### Continual Learning: Skill Expansion Experiment (Success Rate %)

| Method | Grasp | Stack | Close | Press | Open (New) | Avg. | ΔAvg. |
|--------|-------|-------|-------|-------|------------|------|-------|
| π₀.₅ (Baseline) | 85 | 65 | 70 | 90 | — | 77.5 | — |
| π₀.₅ (Continual Learning) | 70 | 45 | 60 | 75 | 55 | 61.0 | **−15.0** |
| AtomicVLA\* (Baseline) | 95 | 80 | 70 | 100 | — | 86.3 | — |
| AtomicVLA\* (Continual Learning) | 90 | 80 | 80 | 100 | 70 | **82.0** | **−1.3** |

### Ablation Study: SG-MoE Routing Mechanism (LIBERO-LONG %)

| Method | LIBERO-LONG |
|--------|-------------|
| π₀ (No MoE) | 85.2 |
| + Standard Token-Level MoE | 88.6 (+3.4) |
| + MoDE (Denoising-Step Routing) | 89.5 (+4.3) |
| + **SG-MoE (Atomic Skill Routing)** | **95.2 (+10.0)** |

### Key Findings

- **Most Significant Gain on LIBERO-LONG**: AtomicVLA achieves a 10% improvement on the most challenging long-horizon task suite, demonstrating that explicit planning combined with skill decomposition is critical for multi-step tasks.
- **SG-MoE Substantially Outperforms Standard MoE**: Standard token-level MoE and MoDE yield only +3.4% and +4.3% respectively, whereas SG-MoE achieves +10% — because the former two still route at the token level, causing experts to learn mixed skills, while SG-MoE ensures all tokens of the same skill are processed by the same expert.
- **Near-Zero Forgetting in Continual Learning**: π₀.₅ experiences an average 15% decline on existing skills after learning a new one (Stack drops by 20%), whereas AtomicVLA\* declines by only 1.3%, resolving forgetting at the architectural level.
- **Inter-Skill Interference in Mixed Training**: Tasks with conflicting gripper state requirements interfere with each other when trained jointly (e.g., drawer opening does not require gripper closure, which interferes with grasping tasks); SG-MoE effectively mitigates this through skill isolation.
- **Error Recovery Capability**: AtomicVLA can automatically detect anomalies during execution and re-plan (e.g., regenerating a skill abstraction after an object drop), although the CALVIN evaluation protocol does not credit completions achieved after recovery, potentially underestimating the true capability.

## Highlights & Insights

- **Unified Think-Act Paradigm**: Rather than naively appending chain-of-thought, planning is deeply coupled with atomic skill abstraction — the output of thinking directly drives the MoE routing decision, forming a closed loop between planning and execution within a single model.
- **Noise-Schedule-Style Skill Embedding**: Borrowing from diffusion model design, discrete skill labels are mapped into a continuous embedding space for routing — an elegant design that experiments confirm to be superior to general token-level routing.
- **"Add-Only" Continual Learning**: Rather than relying on regularization (e.g., EWC) or experience replay, old experts are frozen and new experts are appended at the architectural level — a simple yet effective strategy.
- **Principal-Axis Analysis for Data Generation**: Kinematic principal-axis analysis replaces VLM-based video understanding for atomic action segmentation, leveraging physical priors for greater reliability without requiring manual annotation.

## Limitations & Future Work

- The quality of atomic skill labels depends on the generation capability of InternVideo2.5, which may produce inaccurate annotations for rare or highly specialized manipulations.
- SG-MoE adopts top-1 routing; actions requiring simultaneous coordination of multiple skills (e.g., "push while rotating") may benefit from top-k routing.
- Each new skill adds one expert, leading to linear parameter growth over time — no expert merging or pruning mechanism is provided.
- Gains on CALVIN (+0.22 avg. len) are relatively modest compared to LIBERO, potentially because CALVIN task granularity is less tightly aligned with atomic skills.
- Inter-skill interference from heterogeneous task mixing (e.g., conflicting gripper states) is mitigated but not fully eliminated by SG-MoE.

## Related Work & Insights

- **vs. π₀/π₀.₅**: These models rely on pure flow matching action prediction without explicit planning. AtomicVLA adds thinking and skill routing on top, with +10% on LIBERO-Long validating the value of planning.
- **vs. SayCan/Inner Monologue**: These approaches separate planning (external LLM) from execution (independent controller), creating a modality gap. AtomicVLA unifies both within a single model.
- **vs. MoDE**: MoDE uses the denoising timestep as the routing signal, which is still inherently token-level routing. SG-MoE routes using semantically explicit skill labels, with a +5.7% margin confirming the superiority of skill-level routing.
- **Generality of MoE Skill Routing**: The noise-schedule-style embedding routing is transferable to multi-task NLP settings, using task description embeddings to route experts.
- **Modular Continual Learning Paradigm**: The strategy of freezing old experts and appending new ones is applicable to domain-incremental continual pretraining of large vision models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The unified Think-Act paradigm, SG-MoE noise-schedule routing, and modular continual learning form three self-consistent innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers four LIBERO subsets, CALVIN, real Franka robot, ablation studies, and continual learning experiments with comprehensive data.
- Writing Quality: ⭐⭐⭐⭐ Motivation–method–experiment logic is clear; the SG-MoE architecture diagram is intuitive and the algorithmic pseudocode is well-structured.
- Value: ⭐⭐⭐⭐⭐ Makes important contributions to continual learning and long-horizon task planning for VLAs; the SG-MoE design offers broadly transferable insights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Policy Compatible Skill Incremental Learning via Lazy Learning Interface](../../NeurIPS2025/robotics/policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)
- [\[ICCV 2025\] iManip: Skill-Incremental Learning for Robotic Manipulation](../../ICCV2025/robotics/imanip_skill-incremental_learning_for_robotic_manipulation.md)
- [\[CVPR 2026\] MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent](mergevla_cross-skill_model_merging_toward_a_generalist_vision-language-action_ag.md)
- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [\[CVPR 2026\] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning](como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)

</div>

<!-- RELATED:END -->
