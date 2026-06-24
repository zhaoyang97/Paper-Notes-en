---
title: >-
  [Paper Note] InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions
description: >-
  [CVPR 2025][Image Generation][Physics Simulation] InterMimic proposes a curriculum-driven teacher-student distillation framework, achieving for the first time learning diverse whole-body physical human-object interaction skills from large-scale imperfect MoCap data using a single policy. It first "perfects" each motion subset through teacher policies, then distills them into a student policy, and leverages RL fine-tuning to transcend simple imitation…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Physics Simulation"
  - "Human-Object Interaction"
  - "Motion Mimicking"
  - "Teacher-Student Distillation"
  - "Whole-Body Control"
date: 2026-05-08
content_hash: 0e651c5deb055e09
---

# InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions

**Conference**: CVPR 2025  
**arXiv**: [2502.20390](https://arxiv.org/abs/2502.20390)  
**Code**: [https://sirui-xu.github.io/InterMimic](https://sirui-xu.github.io/InterMimic)  
**Area**: Image Generation  
**Keywords**: Physics Simulation, Human-Object Interaction, Motion Mimicking, Teacher-Student Distillation, Whole-Body Control

## TL;DR
InterMimic proposes a curriculum-driven teacher-student distillation framework, achieving for the first time learning diverse whole-body physical human-object interaction skills from large-scale imperfect MoCap data using a single policy. It first "perfects" each motion subset through teacher policies, then distills them into a student policy, and leverages RL fine-tuning to transcend simple imitation, ultimately supporting zero-shot generalization and seamless integration with motion generators.

## Background & Motivation

1. **Background**: Physics-based human motion mimicking reproduces reference MoCap data by training control policies in physical simulators, which has succeeded in simple movements. However, scaling it to complex human-object interaction (HOI) scenarios presents significant challenges.

2. **Limitations of Prior Work**:
    - **Imperfect MoCap**: Contact artifacts are common, such as unstable expected contact point distances and missing or inaccurate hand movements. Direct imitation of inaccurate movements leads to unrealistic dynamics.
    - **Scaling Difficulties**: Existing methods can only handle specific objects or interaction categories per policy, failing to scale to large-scale data containing diverse objects and interaction modes.
    - **Data Diversity**: HOI datasets contain different human shapes, requiring motion retargeting, which introduces new contact artifacts.

3. **Key Challenge**: Rich HOI datasets contain massive valuable interaction skills, but imperfect data makes direct learning with RL extremely difficult—either failing to learn well or failing to scale.

4. **Goal**: Train a universal physics simulation policy that can learn whole-body interaction skills from hours of imperfect and diverse MoCap data while correcting errors in the data.

5. **Key Insight**: Perfect first, then expand. A divide-and-conquer strategy—multiple teachers each learn a small batch of data and correct it, then distill into a single student.

6. **Core Idea**: Through a curriculum-based "perfecting before expanding" strategy, use multiple parallel teachers to correct imperfect MoCap data, then distill into a unified student, combined with RL fine-tuning to surpass simple imitation.

## Method

InterMimic's design philosophy resembles the alignment pipeline of LLMs: first pre-train using demonstration learning (teacher distillation), then fine-tune with RL. The entire pipeline consists of two stages: (1) training multiple teacher policies, each responsible for imitating and correcting a small subset of data; (2) freezing teachers and using their rollouts as high-quality references to train the student policy via a hybrid of DAgger and PPO.

### Overall Architecture

The system models HOI imitation as an MDP, where states contain human poses + object poses + distance/contact information, and actions are PD targets for 51 joints. The two-stage pipeline is:
- **Stage 1**: Each teacher policy (MLP) is trained on a small data subset, naturally correcting MoCap artifacts through physics simulation.
- **Stage 2**: Freeze the teachers, utilizing them to provide refined references and action supervision to train the student policy (Transformer).

### Key Designs

1. **Contact-Guided Imitation as Perfecting**:
    - Function: Allows teacher policies to automatically correct contact errors and hand inaccuracies in MoCap data while learning to imitate.
    - Mechanism: Designed both embodiment-aware and embodiment-agnostic reward components. The former uses a distance-adaptive weight $\boldsymbol{w}_d$ to emphasize rotation matching at a distance and position matching up close, achieving automatic retargeting for different body shapes. The latter tracks object poses and contact states. Contact labels are discretized into three levels—promoting (red), neutral (green), and penalizing (blue)—to tolerate contact distance fluctuations in MoCap. For missing hand data, contact labels are automatically activated when fingertips/palms approach objects, utilizing RL exploration to discover plausible hand interaction policies.
    - Design Motivation: Physics simulators naturally correct kinematically implausible contacts, unifying imitation and correction into the same optimization objective.

2. **Physics State Initialization (PSI) + Interaction Early Termination (IET)**:
    - Function: PSI solves the rollout initialization failure issue caused by imperfect reference states; IET avoids wasting computation on invalid interactions.
    - Mechanism: PSI maintains an initialization buffer that stores MoCap reference states and simulated states from prior rollouts. Trajectories of high-reward rollouts are added to the buffer using a FIFO strategy, replacing low-quality reference initializations. IET adds three HOI-specific conditions to standard early termination: object deviation from reference > 0.5m, weighted human-object distance deviation > 0.5m, and necessary contact lost for more than 10 frames.
    - Design Motivation: Directly using imperfect MoCap states for RSI (Reference State Initialization) leads to unrecoverable failures such as falling objects. PSI significantly alleviates this issue by employing simulation-corrected states.

3. **Reference Distillation + Policy Distillation + RL Fine-tuning**:
    - Function: Efficiently aggregates the expertise of multiple teacher policies into a single student policy.
    - Mechanism: Teacher policies provide dual supervision—(1) **Reference Distillation**: Teacher rollouts replace original MoCap as references for students, providing high-quality motion with contact correction and unified body shapes; (2) **Policy Distillation**: Actions of teachers are learned through a DAgger loss $J(\psi) = \|\boldsymbol{a}^{(S)} - \boldsymbol{a}^{(T)}\|$. Training progresses scheduled: DAgger dominates initial stages (weight $w = \min(t/\beta, 1)$), transitioning to PPO dominance in later stages. The student uses a 3-layer Transformer encoder (4 heads, hidden 256) to handle a longer observation window $K=\{1,2,4,16\}$.
    - Design Motivation: Pure BC leads to "averaged" sub-optimal behaviors when teachers conflict; RL fine-tuning helps the student converge to optimal solutions. This paradigm borrows from the SFT + RLHF approach in LLMs.

### Loss & Training

- Teacher: PPO + the aforementioned multi-component reward function, containing embodiment-aware/agnostic rewards + contact promotion/penalty + energy consumption regularization.
- Student: Gradient update is $\nabla_\psi(wL(\psi) + (1-w)J(\psi))$, where $w$ linearly increases from 0 to 1.
- Isaac Gym simulator, teacher using MLP (1024-1024-512), student using Transformer.

## Key Experimental Results

### Main Results

**Teacher Policy vs SkillMimic (BEHAVE yogamat)**

| Method | Duration(s)↑ | E_h(cm)↓ | E_o(cm)↓ |
|------|-------------|----------|----------|
| SkillMimic | 12.2 | 7.2 | 13.4 |
| InterMimic w/o IET | 40.3 | 6.7 | 9.9 |
| InterMimic w/o PSI | 36.1 | 6.6 | 10.2 |
| **InterMimic** | **42.6** | **6.4** | **9.2** |

**Large-scale Student Policy (OMOMO Dataset)**

| Configuration | Success Rate↑ | Duration↑ | E_h↓ | E_o↓ |
|------|--------|------|------|------|
| PPO only | 23.9 | 101.6 | 7.2 | 15.6 |
| DAgger only | 54.5 | 139.9 | 7.1 | 11.0 |
| PPO + Ref.Distill. | 71.7 | 152.8 | 8.9 | 12.7 |
| **Full (PPO+Ref+Policy)** | **90.7** | **168.0** | **5.5** | **9.7** |

### Ablation Study

| Configuration | OMOMO Train Success Rate | OMOMO Test Success Rate | Explanation |
|------|---------------|---------------|------|
| w/o Reference Distillation | 23.9% | 9.6% | Direct learning from MoCap is difficult |
| w/ Reference Distillation | 71.7% | 91.6% | Reference corrected by teacher brings massive gains |
| + Policy Distillation | 90.7% | 95.5% | DAgger guidance further improves performance |
| MLP vs Transformer | 90.7 vs 88.8 | 95.5 vs 98.1 | Transformer generalizes better |

### Key Findings
- **Reference distillation is the most critical component**: Improving success rate from 23.9% to 71.7% demonstrates that correcting imperfect MoCap is the core bottleneck.
- **RL fine-tuning is indispensable**: Pure DAgger distillation yields only 54.5%, reaching 90.7% when PPO is incorporated, resolving conflicts between teachers.
- **Successful zero-shot generalization**: The student policy can be directly applied to unseen objects (from BEHAVE and HODome).
- **Integration with motion generators**: It seamlessly drives motion outputs of HOI-Diff and InterDiff, extending from imitation to generation.
- Teachers can correct symmetric object rotation errors (e.g., objects sliding on the ground during MoCap $\rightarrow$ corrected to proper rotations).

## Highlights & Insights
- **Curriculum-based "Perfecting before Expanding"**: This design philosophy precisely addresses the dual challenges of "imperfect data" and "scaling difficulties", which can transition to any scenario requiring large-scale learning from imperfect demonstrations (e.g., robotic manipulation).
- **LLM Alignment Analogy**: The BC pre-training $\rightarrow$ RL fine-tuning paradigm borrows from the SFT $\rightarrow$ RLHF pipeline of LLMs, proving equally effective in physical simulation, suggesting this might be a more general learning paradigm.
- **Three-Level Contact Labels**: Labeling reference contact in three levels (red/green/blue) instead of binary yes/no elegantly handles inaccurate contact distances in MoCap.

## Limitations & Future Work
- Physical simulator (Isaac Gym) does not fully support soft bodies, ruling out scenarios like backpack straps.
- Some severely erroneous MoCap data cannot be corrected even in the teacher phase and must be discarded.
- Hand movements mainly rely on RL exploration to discover, lacking precise dexterous manipulation capabilities.
- Requires Isaac Gym environment and substantial GPU resources for training, presenting a high deployment barrier.
- Experiments mainly evaluate the complete teacher-student pipeline on the OMOMO dataset.

## Related Work & Insights
- **vs PhysHOI**: PhysHOI fails to track in complex interactions requiring multiple body parts, whereas InterMimic's contact-guided reward enables whole-body interactions.
- **vs SkillMimic**: SkillMimic performs poorly when dealing with imperfect MoCap data (Duration only 12.2s vs 42.6s) as it lacks contact correction mechanisms.
- **vs GRAB dataset methods**: Existing methods rely heavily on the high-quality but scene-limited GRAB dataset; InterMimic can utilize much larger but noisier datasets like OMOMO and BEHAVE.

## Rating
- Novelty: ⭐⭐⭐⭐ The curriculum framework design with teacher-student + RL fine-tuning is novel, and the problem definition (large-scale whole-body HOI from imperfect MoCap) is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough experiments with multi-dataset validation, detailed ablations, zero-shot generalization, and motion generator integration.
- Writing Quality: ⭐⭐⭐⭐ Detailed methodology descriptions, but with many components, taking readers longer to digest.
- Value: ⭐⭐⭐⭐⭐ Achieving universal whole-body physical HOI simulation for the first time, of great significance to robotic manipulation, character animation, and humanoid control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](../../ICCV2025/image_generation/dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [\[CVPR 2025\] Visual Persona: Foundation Model for Full-Body Human Customization](visual_persona_foundation_model_for_full-body_human_customization.md)
- [\[CVPR 2025\] InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation](interact_advancing_large-scale_versatile_3d_human-object_interaction_generation.md)
- [\[CVPR 2025\] DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image](decloth_decomposable_3d_cloth_and_human_body_reconstruction_from_a_single_image.md)
- [\[CVPR 2025\] Compass Control: Multi Object Orientation Control for Text-to-Image Generation](compass_control_multi_object_orientation_control_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
