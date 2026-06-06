---
title: >-
  [Paper Note] UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos
description: >-
  [CVPR 2026][Human Understanding][Dexterous Manipulation] This paper presents UniDex, a robot foundation suite comprising a large-scale dataset spanning 8 dexterous hands (50K+ trajectories / 9M frames)…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "Dexterous Manipulation"
  - "VLA Foundation Model"
  - "Unified Action Space"
  - "Learning from Human Videos"
  - "Cross-Hand Transfer"
date: 2026-05-08
content_hash: a827024219c00d50
---

# UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos

**Conference**: CVPR 2026
**arXiv**: [2603.22264](https://arxiv.org/abs/2603.22264)  
**Code**: [https://unidex-ai.github.io/](https://unidex-ai.github.io/)  
**Area**: Human Understanding
**Keywords**: Dexterous Manipulation, VLA Foundation Model, Unified Action Space, Learning from Human Videos, Cross-Hand Transfer

## TL;DR
This paper presents UniDex, a robot foundation suite comprising a large-scale dataset spanning 8 dexterous hands (50K+ trajectories / 9M frames), a Functionally-Aligned Actuator Space (FAAS), and a 3D VLA policy (UniDex-VLA). UniDex-VLA achieves 81% average task progress on real-world tool-use tasks (vs. 38% for π₀) and demonstrates spatial, object-level, and zero-shot cross-hand generalization.

## Background & Motivation

1. **Background**: Learning from Demonstrations is the dominant paradigm for visuomotor control. Vision-Language-Action (VLA) models perform well on tasks such as grasping, but most are designed for parallel grippers; foundation models for dexterous hands remain extremely scarce.

2. **Limitations of Prior Work**: Building foundation models for dexterous hands is significantly harder than for grippers, due to three challenges: (a) **Data scarcity**: Teleoperation data for dexterous hands is prohibitively expensive and difficult to collect at scale; (b) **Embodiment heterogeneity**: Dexterous hands vary widely (6–24 DoF, diverse kinematic structures), making data and policy transfer across hands difficult; (c) **High-dimensional control**: Dexterous hand action spaces are far higher-dimensional than grippers, requiring more expressive action representations and learning algorithms.

3. **Key Challenge**: Training dexterous hand policies requires large-scale, diverse data, yet teleoperation data is expensive and hand-specific. Humans naturally produce abundant manipulation data (egocentric videos), but a large kinematic and visual gap exists between human and robot hands.

4. **Goal**: (a) How to convert egocentric human videos into robot-executable dexterous hand trajectories? (b) How to design a unified action space that enables cross-hand transfer? (c) How to build a VLA foundation model for dexterous hands?

5. **Key Insight**: Dexterous hands are designed to mimic human hands and share a functional correspondence with them. This correspondence is exploited to retarget human hand motion to robot hands, while visually masking the human hand and replacing it with a robot hand point cloud, substantially reducing the kinematic and visual domain gap.

6. **Core Idea**: A human-to-robot data conversion pipeline is used to construct a large-scale, multi-hand pretraining dataset. A functionally-aligned unified action space (FAAS) enables cross-hand transfer, and a 3D VLA foundation model (UniDex-VLA) is trained for general dexterous manipulation.

## Method

### Overall Architecture
UniDex consists of three components: (1) **UniDex-Dataset** — a robot-centric dataset converted from egocentric human videos, spanning 8 hand types, 50K+ trajectories, and 9M frames; (2) **FAAS + UniDex-VLA** — a Functionally-Aligned Actuator Space and a 3D VLA policy trained on top of it; (3) **UniDex-Cap** — a portable human data collection device supporting human-robot co-training to reduce teleoperation costs.

### Key Designs

1. **Human-to-Robot Data Conversion Pipeline**:

    - **Function**: Converts egocentric RGB-D human videos into robot-executable dexterous hand trajectories.
    - **Mechanism**: A two-stage approach addresses both the kinematic and visual domain gaps. *Kinematic retargeting*: Fingertips serve as primary contact points. Human fingertip positions $X^* = [x_1^*, ..., x_m^*]$ are extracted, and a 6-DoF alignment offset $T_{\text{offset}}$ (dummy base) is introduced. IK is solved for joint angles $q$ such that robot fingertips align with human fingertips, while maintaining physically plausible contact. The process is **human-in-the-loop**: after automatic IK solving, a simple GUI allows the user to adjust $T_{\text{offset}}$ sliders, typically converging in a few iterations. *Visual alignment*: Point clouds are computed from RGB-D frames; human hands are masked using WiLoR+SAM2 and the corresponding points are removed; the retargeted robot hand mesh is rendered into the scene point cloud and projected back to RGB-D frames via pinhole projection.
    - **Design Motivation**: Fully automated retargeting is prone to errors in contact regions. Human-in-the-loop adjustment incurs minimal cost but significantly improves contact quality. Visual alignment eliminates the discrepancy of "observing a human hand while executing robot hand actions," making pretraining data consistent with downstream real-robot settings.

2. **Functionally-Aligned Actuator Space (FAAS)**:

    - **Function**: Defines a unified cross-embodiment action representation to enable data sharing and skill transfer across hand types.
    - **Mechanism**: FAAS is an 82-dimensional action vector. The first 18 dimensions encode bimanual wrist poses (9 per hand: 6D continuous rotation + 3D translation); the remaining 64 dimensions encode joint commands (32 slots per hand). The core design maps actuators by **functional role**: joints with analogous functions (e.g., thumb opposition joints across different hands) are assigned the same FAAS index regardless of the hand's URDF structure. For example, thumb and ring finger flexion and abduction joints are mapped to the same indices {0,1,3,5,6} across Oymotion (11 actuators), Allegro (16), Inspire (12), and Wuji (20). Additional slots are reserved for hand-specific degrees of freedom and future hand types.
    - **Design Motivation**: Prior unified action spaces (e.g., RDT-1B preserving semantic structure, π₀ using left-aligned representations) primarily target grippers. EgoVLA uses human hand parameters as the dexterous hand representation but requires a post-processing IK stage that introduces additional error. FAAS aligns directly at the functional level, requiring no post-processing and proving more robust for high-DoF dexterous hands.

3. **UniDex-VLA Policy**:

    - **Function**: A Vision-Language-Action foundation model that takes 3D point clouds as input and outputs actions in FAAS.
    - **Mechanism**: The architecture is based on π₀. Inputs are colored point cloud $P_t$, language instruction $\ell_t$, and proprioception $q_t$; outputs are an $H$-step action chunk $A_t = [a_t, ..., a_{t+H-1}]$. The SigLIP 2D visual encoder in π₀ is replaced with a Uni3D 3D point cloud encoder (ViT-based, initialized from a 2D pretrained ViT, with point cloud features aligned to image-text features). Wrist actions use relative representations (relative to the first frame of the action chunk). Training uses a conditional flow-matching objective. The model is first pretrained on UniDex-Dataset and then fine-tuned with a small number of task demonstrations.
    - **Design Motivation**: Dexterous tool use requires reasoning about 3D geometry and contact affordances, which 2D encoders cannot adequately capture. Point clouds directly preserve 3D information, and Uni3D's pretrained alignment makes it a strong 3D encoder. The flow-matching objective is well-suited for generative modeling in high-dimensional action spaces.

### Loss & Training
The model is trained with a conditional flow-matching objective; during inference, action chunks are generated via forward-Euler integration over the denoising trajectory. Pretraining is performed on UniDex-Dataset across 8 hand types; fine-tuning requires only 50 teleoperation demonstrations per task. UniDex-Cap supports human-robot co-training, with experiments showing a human-to-robot data exchange ratio of approximately 2:1 (two human demonstrations ≈ one robot demonstration).

## Key Experimental Results

### Main Results
Five real-world tool-use tasks (20 trials per task):

| Model | Avg. Task Progress | Final Success Rate |
|-------|-------------------|-------------------|
| Diffusion Policy | 29.0% | 22.0% |
| DP3 | 35.0% | 30.0% |
| π₀ | 38.0% | 35.0% |
| UniDex-VLA (No Pretrain) | 32.5% | 23.0% |
| **UniDex-VLA** | **81.0%** | **76.0%** |

On the most challenging task ("cutting a bag with scissors"), UniDex-VLA achieves an 84.6% relative improvement over the best baseline.

### Ablation Study

| Generalization Type | Experiment | Result |
|--------------------|------------|--------|
| Spatial generalization | OOD positions for kettle and dripper | UniDex-VLA maintains high success rate; near-perfect with DemoGen augmentation |
| Object generalization | Kettles of different colors/sizes | UniDex-VLA maintains strong performance |
| Zero-shot cross-hand | Inspire train → Oymotion deploy | 60% success vs. baseline ≈ 0% |
| Zero-shot cross-hand | Inspire train → Wuji deploy | 40% success vs. baseline ≈ 0% |
| Human-robot co-training | 50 robot + human demos | 2:1 exchange ratio; human data collection 5.2× faster |

### Key Findings
- Pretraining provides substantial gains: UniDex-VLA vs. No Pretrain (81% vs. 32.5%), demonstrating that large-scale multi-hand pretraining provides a strong motion prior for dexterous manipulation.
- FAAS enables zero-shot cross-hand transfer: Inspire→Oymotion 60%, →Wuji 40%, vs. baseline ≈ 0%, confirming that functionally-aligned action spaces preserve transferable skill semantics.
- Human-robot co-training is cost-effective: 1 robot demo ≈ 2 human demos, with human data collection being 5.2× faster, yielding an effective cost exchange ratio of approximately 1:2.6.
- 3D point cloud input naturally facilitates spatial generalization through geometry-based data augmentation.

## Highlights & Insights
- **Functional alignment in FAAS**: Mapping by functional role rather than URDF joint index is a simple yet highly effective design. This principle generalizes to unified control of other heterogeneous robot embodiments (e.g., dual-arm systems with different configurations).
- **Lightweight human-in-the-loop retargeting**: Rather than pursuing full automation, the pipeline uses GUI sliders to allow users to correct contact with minimal effort, achieving high-quality data at very low cost.
- **Fine-tuning with only 50 demonstrations**: Thanks to large-scale pretraining, downstream tasks require very few real demonstrations, substantially lowering the barrier to deployment.

## Limitations & Future Work
- Large-scale egocentric activity datasets without action annotations (e.g., Ego4D) are not yet exploited; despite lacking precise hand annotations, their visual diversity could further extend pretraining.
- Current evaluation focuses on tool-use tasks; fine-grained finger-level reorientation and in-hand manipulation are not addressed.
- The 32-slot joint reservation in FAAS may lack flexibility; higher-DoF future hand designs may require remapping.
- The benefits of human-robot co-training depend on retargeting quality; retargeting quality for non-contact manipulation (e.g., button pressing, sliding) remains to be verified.

## Related Work & Insights
- **vs. π₀**: π₀ achieves only 38% task progress on dexterous hands, as its gripper-focused pretraining data provides little benefit for dexterous manipulation. UniDex-VLA achieves 81% through dexterous-hand-specific pretraining, a substantial gap.
- **vs. EgoVLA**: EgoVLA uses human hand parameters as the dexterous hand representation but requires post-processing IK, introducing additional error. FAAS directly outputs joint angles and is post-processing free.
- **vs. RDT-1B**: RDT-1B's semantically-structured action space primarily targets grippers and does not address functional alignment for high-DoF dexterous hands.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First large-scale foundation suite spanning 8 dexterous hands; FAAS is an elegant design; human-robot co-training introduces genuine innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five real-world tasks, diverse generalization tests, and quantitative human-robot co-training analysis; experiments are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-organized paper; the system-level contribution is presented clearly and coherently.
- **Value**: ⭐⭐⭐⭐⭐ — A milestone contribution to dexterous hand foundation model research; the full open-source release of dataset, model, and collection hardware has enormous potential impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)
- [\[CVPR 2026\] EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR](egoposeformer_v2_accurate_egocentric_human_motion_estimation_for_arvr.md)
- [\[CVPR 2026\] Editing Physiological Signals in Videos Using Latent Representations](editing_physiological_signals_in_videos_using_latent_representations.md)
- [\[CVPR 2026\] TriLite: Efficient WSOL with Universal Visual Features and Tri-Region Disentanglement](trilite_efficient_weakly_supervised_object_localization_with_universal_visual_fe.md)
- [\[ICCV 2025\] AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning](../../ICCV2025/human_understanding/ar-vrm_imitating_human_motions_for_visual_robot_manipulation_with_analogical_rea.md)

</div>

<!-- RELATED:END -->
