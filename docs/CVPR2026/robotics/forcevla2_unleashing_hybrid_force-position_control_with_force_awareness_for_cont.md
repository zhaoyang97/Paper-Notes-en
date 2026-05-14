---
title: >-
  [Paper Note] ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation
description: >-
  [CVPR 2026][Robotics][VLA] This paper proposes ForceVLA2, the first end-to-end model that unifies force awareness and hybrid force-position control within a VLA framework. Force-based Prompts injected into a VLM expert c…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "VLA"
  - "Force Control"
  - "Hybrid Force-Position Control"
  - "MoE"
  - "Contact-Rich Manipulation"
  - "Force Awareness"
date: 2026-05-08
content_hash: 2aeb558c8528edfa
---

# ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation

**Conference**: CVPR 2026
**arXiv**: [2603.15169](https://arxiv.org/abs/2603.15169)
**Code**: [Project Page](https://sites.google.com/view/force-vla2/home)
**Area**: Robotic Manipulation / Vision-Language-Action Models
**Keywords**: [VLA, Force Control, Hybrid Force-Position Control, MoE, Contact-Rich Manipulation, Force Awareness]

## TL;DR
This paper proposes ForceVLA2, the first end-to-end model that unifies force awareness and hybrid force-position control within a VLA framework. Force-based Prompts injected into a VLM expert construct cross-phase force-aware task concepts, while a Cross-Scale MoE adaptively fuses task semantics with real-time interaction forces to achieve closed-loop force-position regulation. The model achieves an average success rate of 66% across 5 contact-rich tasks, surpassing π₀ and π₀.5 by 48.0% and 35.0%, respectively.

## Background & Motivation
Current VLA models (e.g., π₀, OpenVLA, GR00T-N1) demonstrate strong semantic understanding and language following, yet suffer from a fundamental limitation:

1. **Lack of force reasoning and active force interaction**: Existing VLAs rely exclusively on position control and treat force only as an auxiliary perceptual input (e.g., ForceVLA), not as an active control signal. This leads to critical failure modes in contact-rich tasks (wiping, pressing, assembly), including arm overloading and unstable contact.
2. **Lack of phase-wise force awareness**: Contact-rich tasks typically comprise 3–5 sub-task phases (e.g., "approach → contact → apply force → release"), each imposing fundamentally different force requirements. Existing VLAs cannot reason about phase-specific force targets or use force feedback to assess sub-task completion progress.
3. **Absence of coupled force-position control**: From a control-theoretic perspective, the controllability matrix of a pure position control system has rank 6 (controlling only 6D pose), which is insufficient to independently control the 12-dimensional joint force-position space — force remains a passive output of environmental dynamics rather than an independently controllable variable.

**Key Challenge**: Humans rely on high-level visual-language reasoning to determine phase-specific goals while integrating real-time force feedback for adaptive force-position regulation. Existing VLAs lack a unified mechanism for this multi-scale force awareness and active force interaction.

## Core Problem
How to enable a VLA model to (1) build force-aware task concepts across different task phases and track phase progress, (2) adaptively fuse high-level task semantics with real-time interaction forces, and (3) realize genuine closed-loop hybrid force-position control rather than treating force merely as an auxiliary percept?

## Method

### Overall Architecture
ForceVLA2 adopts a two-level design inspired by human sensorimotor control. The **long-horizon level** injects Force Prompts into a VLM expert to construct force-aware task concepts; the **short-horizon level** uses a Cross-Scale MoE to fuse task knowledge with interaction forces and output hybrid force-position actions. Inputs include multi-view images, task prompts, force prompts, and proprioceptive state (EE 6D pose + 6D force/torque); outputs are EE pose delta $\Delta p \in \mathbb{R}^7$, target contact force $f \in \mathbb{R}^6$, and a sub-task transition indicator $s \in [0,1]$.

### Key Designs

1. **Long-Horizon Force Awareness — Force-based Prompts (Sec 3.1)**: Force information is injected into the VLM via textual prompts:

    - **Task Prompt $T_t$**: Describes the global task objective (e.g., "wipe the vase").
    - **Force Prompt $T_f$**: Encodes the current sub-task state and phase-specific force targets (e.g., "Phase 2: maintain 5N downward force while wiping the surface").
    - Each task predefines 3–5 sub-tasks; the force prompt acts as a discrete state machine determining whether to maintain the current sub-task or transition to the next phase.
    - Visual tokens $Z_v$ are processed through a visual encoder $f(\cdot)$; language tokens formed by concatenating the task and force prompts are processed through a text encoder $g(\cdot)$; the combined representation is fed into the VLM for token-to-token attention, yielding a fused multimodal representation $E$.
    - This enables the VLM to inherit pretrained knowledge while acquiring the ability to evaluate sub-task completion, manage cross-phase transitions, and explicitly update force cues.

2. **Short-Horizon Force-Control Closed Loop (Sec 3.2)**:

    - **Multimodal Encoding (Sec 3.2.1)**: EE 6D pose $p \in \mathbb{R}^7$ is encoded via a linear layer $\phi_P$ into $E_P$; raw 6D force/torque $f_{\text{raw}} \in \mathbb{R}^6$ is encoded via $\phi_F$ into $E_F$. Crucially, $E_F$ participates in two pathways: (a) concatenated with $E_P$ to form $E_{\text{state}}$, which interacts with the visual-language feature $E$ via cross-attention to inject global task semantics; and (b) a **bypass direct connection** to the MoE, skipping high-level fusion to preserve gradient fidelity of the raw force signal, forming a short-horizon reactive loop.
    - **Cross-Scale MoE (Sec 3.2.2)**: Comprises 3 modality-specific experts (visual expert, state expert, force expert), each implemented as a lightweight MLP. A dynamic gating network computes token-level routing weights $w = [w_V, w_S, w_F]$, automatically emphasizing visual reasoning during free-space motion and force or position cues during contact phases. The fused representation $E_{\text{MoE}}$ is passed to the flow matching policy head.

3. **Flow Matching for Hybrid Force-Position Control**:

    - Starting from a noise-initialized action $a_t(0) \sim \mathcal{N}(0, I)$, the learned conditional flow $F_\theta$ iteratively denoises conditioned on $E_{\text{MoE}}$.
    - The final output is $a_t = [\Delta p_t;\, f_t;\, s_t]$: pose delta (7D) + target force (6D) + sub-task transition probability (1D).
    - **Probabilistic Modeling of Sub-Task Transitions**: $s_t$ is computed as the joint probability of position proximity, pose alignment, and force satisfaction — modeled respectively as Beta, Exponential, and Uniform distributions. When the joint probability exceeds a threshold, a sub-task transition is triggered and states are reset.

4. **ForceVLA2-Dataset (Sec 4)**:

    - **Hardware**: Flexiv Rizon 4s 7-DOF robot arm + DH Robotics AG-95 adaptive gripper + 3 RGB cameras (2× Intel RealSense D455 third-person view + 1× D435 wrist-mounted) + 6D force/torque sensor (300 Hz).
    - **Data Collection**: Force-feedback GELLO teleoperation framework, preserving natural force application patterns.
    - **Tasks**: press bottle, clean vase, clean board, retrieve plate, assemble gears.
    - **Scale**: 1,000 trajectories, approximately 500K synchronized timesteps.
    - **Annotation**: Sub-task boundaries are automatically segmented based on force signal features, with additional force prompt annotations.

### Loss & Training
- 8× A100 GPUs, batch size 32, 30K steps, approximately 10 hours.
- AdamW optimizer with cosine decay learning rate schedule; EMA decay 0.99.
- Inference speed: 15 Hz on a 4090 GPU, chunk size 30.

## Key Experimental Results

### Main Results (Table 1)

| Method | Press bottle | Clean vase | Clean board | Retri. plate | Assem. gears | Avg. |
|--------|-------------|------------|-------------|--------------|-------------|------|
| π₀ (w/o force) | 35.0 | 20.0 | 35.0 | 0.0 | 0.0 | 18.0 |
| π₀.5 | 45.0 | 30.0 | 45.0 | 15.0 | 20.0 | 31.0 |
| ACP | 25.0 | 30.0 | 25.0 | 0.0 | 0.0 | 16.0 |
| π₀ w/ F (naive force input) | 30.0 | 25.0 | 20.0 | 10.0 | 0.0 | 17.0 |
| ForceVLA | 70.0 | 25.0 | 55.0 | 15.0 | 10.0 | 35.0 |
| **ForceVLA2** | **80.0** | **75.0** | **70.0** | **35.0** | **70.0** | **66.0** |

- vs. π₀: +48.0%; vs. π₀.5: +35.0%; vs. ForceVLA: +31.0%.
- Assemble gears: 70% vs. the second-best 10%, a 60-point improvement — the gap is largest on the most force-sensitive task.
- π₀ w/ F achieves only 17%, worse than the base π₀ (18%), demonstrating that naively concatenating force input as disturbance degrades pretrained representations.

### Ablation Study (Table 2)

| FP | ME | CM | Avg. |
|----|----|----|------|
| ✗ | ✗ | ✗ | 18.0 (π₀ baseline) |
| ✓ | ✗ | ✗ | 27.0 (+9) |
| ✓ | ✓ | ✗ | 40.0 (+13) |
| ✓ | ✓ | ✓ | **66.0** (+26) |

- The Cross-Scale MoE module contributes the largest gain (+26%).
- Force Prompt alone yields +9%, validating the value of force-aware task concepts.

### MoE Modality Fusion Ablation (Table 3)

| VM | FM | Avg. |
|----|-----|------|
| ✗ | ✗ | 36.0 |
| ✓ | ✗ | 50.0 (+14) |
| ✓ | ✓ | **66.0** (+16) |

- Visual and force modalities contribute roughly equally within the MoE; neither alone is sufficient.

### Force Injection Location Ablation (Table 5)
- VLM Pathway injection: 5.0% (severe degradation).
- Multimodal Encoder injection: 58.0%.
- State Fusion (ME + state fusion): **66.0%** — optimal design.

## Highlights & Insights
- **First hybrid force-position control framework for VLA**: Rather than treating force as auxiliary perception, ForceVLA2 enables the VLA to actively output force targets and perform closed-loop control — extending controllability from 6D toward 12D from a control-theoretic perspective.
- **Elegant multi-scale force injection design**: Long-horizon force-aware phase concepts are built via text prompts (inheriting VLM reasoning), while short-horizon bypass connections preserve force signal gradient fidelity (ensuring real-time reactivity); Cross-Scale MoE dynamically balances both.
- **Probabilistic modeling of sub-task transitions**: A Beta-Exponential-Uniform joint probability formulation elegantly models position/pose/force completion, providing a principled solution to phase-switching decisions.
- **Thorough ablation study**: Beyond module-level contributions, the paper carefully analyzes force injection location (VLM vs. ME vs. State Fusion) and MoE modality combinations, with experimental support for every design choice.
- **Instructive negative result**: π₀ w/ F at 17% (below the 18% base) powerfully demonstrates that "adding force" is insufficient — careful design of force injection is essential.

## Limitations & Future Work
- **Limited data scale**: 1,000 trajectories across 5 tasks; generalization to broader tasks and objects remains unknown — a force-control dataset at the scale of Open X-Embodiment is needed.
- **Predefined discrete force prompts**: Force prompts for unseen tasks require manual design; automatic force prompt generation capability is absent.
- **Distributional assumptions in transition modeling**: The probabilistic sub-task transition model assumes specific parametric distributions (Beta, Exponential, Uniform), which may not hold for real-world force signals.
- **Single hardware platform**: Validation is limited to the Flexiv platform; despite claims of hardware-agnosticism, no experiments on Franka, UR, or other platforms are reported.
- **Inference speed at 15 Hz**: This may be insufficient for high-frequency force control applications such as high-speed assembly.
- **Retrieve plate success rate of only 35%**: Force-guided probing under heavy occlusion still has considerable room for improvement.

## Related Work & Insights
- **vs. π₀ / π₀.5 (position-only VLA)**: These methods lack force control and achieve only 18%/31% on contact-rich tasks. ForceVLA2 achieves 66% via hybrid force-position control; the gap is most pronounced on force-sensitive tasks (assemble gears: 70% vs. 0%/20%).
- **vs. ForceVLA (force as auxiliary input)**: ForceVLA concatenates force as an additional modality while still outputting only position actions. ForceVLA2 not only perceives force but actively outputs target forces; the +31% improvement demonstrates the fundamental distinction between force perception and force interaction.
- **vs. ACP (adaptive compliance control)**: ACP implicitly incorporates force via admittance control but generalizes poorly (16%) and lacks the semantic reasoning capacity of a VLM for phase-level force planning.
- **vs. TLA / Tactile-VLA (tactile-based methods)**: Tactile sensors provide indirect, noisy force estimates and cannot substitute for direct 6D force/torque sensing, which ForceVLA2 uses for precise force feedback.

**Broader Implications**:
- **Force-guided active exploration**: ForceVLA2 demonstrates force-guided probing on the Retrieve plate task — autonomously retrying grasps via force feedback when vision fails. This "force as perception" paradigm is transferable to tactile manipulation, blind grasping, and related scenarios.
- **Multi-scale signal fusion paradigm**: The dual-channel design of long-horizon semantic fusion and short-horizon bypass direct connection can generalize to other domains requiring fast-slow signal integration — e.g., long-term path planning + short-term reactive obstacle avoidance in autonomous driving.
- **Complementarity with AtomicVLA**: AtomicVLA excels at multi-step task planning and skill decomposition; ForceVLA2 excels at force interaction. Future work could combine them: AtomicVLA's reasoning module for task planning and skill decomposition, ForceVLA2's force control module for executing contact-rich skills.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First hybrid force-position control framework for VLA; both the multi-scale force injection design and probabilistic sub-task transition model are novel contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 5 real-world tasks × 20 trials per condition + comprehensive ablations (module-level, modality-level, injection-location-level); cross-platform validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — A complete logical chain from control-theoretic motivation → dual-level architecture → probabilistic modeling → experimental validation, with a control-theoretic analysis in the appendix.
- **Value**: ⭐⭐⭐⭐⭐ — Opens a new direction for VLA force control; the paradigm shift from force perception to force interaction carries significant implications for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](force_transferable_visual_jailbreaking_attacks_via_feature_over_reliance_correct.md)
- [\[CVPR 2026\] DAWN: Pixel Motion Diffusion is What We Need for Robot Control](dawn_pixel_motion_diffusion_robot_control.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](lada_robotic_manipulation.md)
- [\[CVPR 2026\] PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors for Embodied Multi-Sensory Learning](pulse_privileged_knowledge_transfer_from_rich_to_deployable_sensors_for_embodied.md)
- [\[ICLR 2026\] On Entropy Control in LLM-RL Algorithms](../../ICLR2026/robotics/on_entropy_control_in_llm-rl_algorithms.md)

</div>

<!-- RELATED:END -->
