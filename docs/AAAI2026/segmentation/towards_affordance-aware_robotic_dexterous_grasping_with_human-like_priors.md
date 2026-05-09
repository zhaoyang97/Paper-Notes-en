---
title: >-
  [Paper Note] Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors
description: >-
  [AAAI 2026][Segmentation][Dexterous Grasping] This paper proposes AffordDex, a two-stage framework: the first stage pre-trains human hand motion priors (natural motion trajectories) via imitation learning; the second stage refines the policy through reinforcement learning using a residual module and VLM-guided Negative Affordance Annotation (NAA), achieving dexterous robotic grasping that is both human-like in naturalness and functionally correct (e.g., avoiding the blade and grasping the handle of a knife). The method significantly outperforms state-of-the-art approaches across multiple generalization levels.
tags:
  - AAAI 2026
  - Segmentation
  - Dexterous Grasping
  - Functional Affordance
  - Human Motion Priors
  - Negative Affordance Segmentation
  - Reinforcement Learning
date: 2026-05-08
content_hash: aa6be460be9b7bca
---

# Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors

**Conference**: AAAI 2026
**arXiv**: [2508.08896](https://arxiv.org/abs/2508.08896)
**Code**: [afforddex.github.io](https://afforddex.github.io/)
**Area**: Segmentation
**Keywords**: Dexterous Grasping, Functional Affordance, Human Motion Priors, Negative Affordance Segmentation, Reinforcement Learning

## TL;DR

This paper proposes AffordDex, a two-stage framework: the first stage pre-trains human hand motion priors (natural motion trajectories) via imitation learning; the second stage refines the policy through reinforcement learning using a residual module and VLM-guided Negative Affordance Annotation (NAA), achieving dexterous robotic grasping that is both human-like in naturalness and functionally correct (e.g., avoiding the blade and grasping the handle of a knife). The method significantly outperforms state-of-the-art approaches across multiple generalization levels.

## Background & Motivation

Dexterous grasping is a foundational capability for robotic manipulation. Compared to simple parallel-jaw grippers, five-finger dexterous hands more closely resemble human hand anatomy, offering greater flexibility and task adaptability.

**Core Limitations of Existing Methods**: Prior work focuses exclusively on **low-level grasp stability metrics** (i.e., whether an object can be lifted), neglecting two critical dimensions:

**Affordance-Awareness**: Grasping is not merely about lifting objects. Functional correctness must be considered — for example, while the blade of a knife is geometrically graspable, any contact with the blade constitutes a functionally incorrect and unsafe grasp.

**Human-like Motion Postures**: Existing RL-based methods may produce kinematically unnatural joint configurations that, despite being successful, are inefficient and unpredictable in downstream tasks and detrimental to scenarios requiring fluent human-robot interaction.

**Core Idea**: Decouple naturalness and functional correctness and then integrate them — constraining motion priors with human data and understanding object functional properties with vision-language models.

## Method

### Overall Architecture

AffordDex adopts a two-stage training paradigm:

**Stage 1 — Human Hand Trajectory Imitating (HTI)**: A base policy $\pi^H$ is pre-trained on large-scale human hand motion data (OakInk2, ~2,200 sequences) to establish a strong prior over natural motion.

**Stage 2 — Affordance-aware Residual Learning**: The weights of $\pi^H$ are frozen, and a lightweight residual module is trained via RL to adapt the general human-like motion to object-specific interactions. This stage is guided by two key components:
- **Negative Affordance Annotation (NAA)**: Identifies object regions that should not be contacted functionally.
- **Teacher–Student Distillation**: Leverages privileged state information to improve the final vision-based policy.

### Key Designs

#### 1. **Human Hand Trajectory Imitating (HTI)**

The task is formulated as an RL problem, where the policy $\pi^H(a_t|S_t^H)$ generates dexterous hand actions conditioned on the current state. The state comprises the robot state $R_t$, object state $O_t$, and object point cloud $P_t$.

The **reward function** $r^H$ consists of two terms:

**Finger Imitation Reward**: Encourages the dexterous hand to track the reference human finger postures:

$$r^H_{finger} = \sum_{f=1}^F w_f \cdot \exp(-\lambda_f \|\mathbf{j}_{d,f} - \mathbf{j}_{h,f}\|_2^2)$$

where $\mathbf{j}_{d,f}$ is the position of the $f$-th keypoint on the dexterous hand and $\mathbf{j}_{h,f}$ is the corresponding target position from the reference human trajectory.

**Smoothness Reward**: Encourages energy-efficient motion by penalizing the product of joint velocities and torques.

**Design Motivation**: By training on large-scale human motion data, the policy is constrained to the manifold of natural motion, providing a well-initialized starting point for subsequent refinement.

#### 2. **Negative Affordance Annotation (NAA)**

The core innovation is to reformulate the segmentation problem as a **classification problem**, elegantly circumventing the weakness of VLMs in fine-grained spatial localization.

**Pipeline**:
1. **Procedural Texturing**: Texture-free 3D meshes are textured using TextPainter to ensure VLM interpretability.
2. **Multi-view Rendering**: The textured object is rendered from 6 canonical directions to obtain a full-coverage image set $I$.
3. **VLM Querying**: GPT-4V is queried to obtain detailed descriptions of negative affordance regions (e.g., "the blade portion").
4. **Mask Candidate Generation**: Each image is exhaustively segmented using SAM with a dense point grid, and duplicates are removed via NMS to produce a candidate mask set $M_i$:

$$M_i = \text{NMS}(\text{SAM}(I_i, G_i))$$

5. **CLIP-based Classification**: For each candidate mask, a visually highlighted image is created (blurring regions outside the mask), and CLIP computes the similarity with the negative affordance text description; the highest-scoring mask is selected.
6. **3D Projection**: The selected 2D mask is projected into 3D space to obtain the negative affordance point cloud $N_t$.

**Design Motivation**: Directly prompting CLIP to locate the "blade portion" in an image yields poor results, as VLMs excel at image-level understanding but struggle with precise spatial localization. By first generating accurate mask candidates with SAM and then using CLIP to select the best-matching one, the difficult segmentation problem is reduced to a simpler classification problem. NAA is an offline, one-time process requiring approximately 160 seconds per object.

#### 3. **Affordance-Aware Residual Learning and Distillation**

**State-based Teacher Policy**: Takes $S_t^T = \{R_t, O_t, P_t, N_t\}$ as input (including privileged object state) and learns a residual action:

$$a_t = \pi^H(S_t^T) + \pi^T(S_t^T)$$

Trained with PPO using the reward function $r^T$, which consists of four terms:
- $r_d^T$: Grasp distance penalty (distance between hand and object).
- $r_g^T$: Goal distance penalty (distance between object and target position).
- $r_s^T$: Success reward (bonus when the object reaches the target).
- $r_n^T$: **Negative affordance penalty** (penalty when the hand approaches negative affordance regions).

**Vision-based Student Policy**: Uses only information accessible in the real world, $S_t^S = \{R_t, P_t, N_t\}$ (without privileged object state), and is distilled from the teacher via DAgger:

$$\pi^S = \arg\min_{\pi^S} \|\pi^T(S_t^T) - \pi^S(S_t^S)\|$$

### Loss & Training

- Stage 1: PPO + imitation reward function.
- Stage 2: PPO + multi-term reward function (including negative affordance penalty) → DAgger distillation.
- Simulator: IsaacGym with 4,096 parallel environments on an RTX 4090.
- Policy Network: 4-layer MLP (1024, 1024, 512, 512) + PointNet+Transformer (vision mode).
- Shadow Hand: 24 active degrees of freedom.

## Key Experimental Results

### Main Results

Performance on the UniDexGrasp and OakInk2 datasets:

| Method | Seen Succ↑ | HLS↑ | AS↓ | Unseen (Seen Cat.) Succ↑ | HLS↑ | AS↓ | Unseen Cat. Succ↑ | HLS↑ | AS↓ |
|--------|-----------|------|-----|--------------------------|------|-----|-------------------|------|-----|
| UniDexGrasp | 73.7 | 6.2 | 16 | 68.6 | 6.1 | 18 | 65.1 | 6.0 | 17 |
| UniDexGrasp++ | 85.4 | 5.4 | 29 | 79.6 | 5.1 | 25 | 76.7 | 4.8 | 28 |
| **AffordDex** | **87.0** | **8.3** | **10** | **82.8** | **7.8** | **14** | **79.2** | **8.0** | **15** |

(Vision-based setting; table shows a subset of results.)

In the state-based setting, AffordDex achieves 89.2% Succ / 8.6 HLS / 4 AS, with substantial margins across all dimensions.

### Ablation Study

| Configuration | Succ↑ | HLS↑ | AS↓ | Note |
|---------------|-------|------|-----|------|
| Baseline (w/o HTI/NAA/Distill) | 70.1 | 5.0 | 27 | Vision-based, pure RL |
| +HTI | 84.9 | 5.6 | 28 | Large improvement in success rate |
| +HTI+Distill | 85.8 | 7.2 | 13 | Distillation significantly improves HLS |
| +HTI+NAA | 86.9 | 8.1 | 20 | NAA reduces AS |
| **+HTI+NAA+Distill** | **87.0** | **8.3** | **10** | All three components synergize optimally |

**Module Transferability** (applied to UniDexGrasp++):

| Configuration | Succ↑ | HLS↑ | AS↓ |
|---------------|-------|------|-----|
| UniDexGrasp++ | 87.9 | 5.4 | 28 |
| +HTI | 88.2 | **7.8** | 23 |
| +NAA | 88.0 | 5.9 | **19** |
| +HTI+NAA | **88.8** | 8.0 | **12** |

The HTI and NAA modules can be plugged into other RL-based methods as drop-in enhancements.

**NAA vs. Naïve GPT+SAM**: Direct coarse localization with an MLLM followed by SAM segmentation often results in the entire object being segmented rather than the specific target region. NAA reformulates segmentation as a classification problem, enabling precise part-level segmentation.

### Key Findings

1. The human motion prior introduced by HTI contributes the largest gain in success rate (from 70.1 to 84.9), demonstrating that natural motion is not only more aesthetically pleasing but also more effective.
2. NAA reduces AS from 27 to 10, confirming that the negative affordance constraint successfully guides the policy away from functionally inappropriate contact regions.
3. Teacher–student distillation contributes significantly to HLS (7.2 → 8.3), as privileged information facilitates more precise grasp learning.
4. The AffordDex modules transfer well to other methods (e.g., UniDexGrasp++), demonstrating strong generality.
5. UniDexGrasp++ achieves a high success rate but with extremely high AS (28–29), illustrating that task success does not equate to functional correctness.

## Highlights & Insights

1. **Upgraded Problem Definition**: The paper extends dexterous grasping evaluation from "can the object be lifted" to "is it safe + is it natural + does it facilitate downstream manipulation," aligning more closely with real-world application requirements.
2. **Elegant Modeling of Negative Affordance**: Rather than learning "where to grasp" (a positive, inherently difficult, and ambiguous problem), the method learns "what regions must not be contacted" (a negative constraint that is clear and well-defined), substantially simplifying the learning problem.
3. **Dimensionality Reduction via Segmentation→Classification**: This approach bypasses the spatial localization weakness of VLMs by combining SAM's precise segmentation with CLIP's semantic understanding, producing results that exceed the independent capabilities of either model.
4. **Elegant Two-Stage Training Design**: General motion priors are learned first, followed by object-specific adaptation; the residual module design ensures that previously learned natural motion is not disrupted.
5. **HLS Evaluation Innovation**: Gemini 2.5 Pro is employed as an automated evaluator to assess the human-likeness of grasps — imperfect but scalable.

## Limitations & Future Work

1. **Fixed 6-View Rendering**: For geometrically complex or concave objects, occlusion may lead to imprecise negative affordance segmentation.
2. **Sim-to-Real Gap**: All experiments are conducted in IsaacGym; real-world deployment results are not demonstrated.
3. **NAA Dependency on GPT-4V**: Reliance on a commercial API precludes local deployment, limiting reproducibility and scalability.
4. **Restricted to Grasping Tasks**: Negative affordance is task-dependent (e.g., grasping the blade may be necessary for certain cutting tasks); dynamic adjustment based on the specific task remains unexplored.
5. Future work could investigate volumetric affordance learning based on implicit 3D representations, which would be inherently robust to viewpoint occlusion.

## Related Work & Insights

- **UniDexGrasp / UniDexGrasp++**: The primary baselines for AffordDex, proposing geometry-aware curriculum learning while neglecting affordance.
- **DexGrasp Anything**: Generates static grasp poses via diffusion models but lacks motion trajectories, precluding human-likeness evaluation.
- **OakInk2**: Provides human hand manipulation sequence data, serving as the training source for the HTI stage.
- **GEAL**: Employs a dual-branch architecture for cross-modal affordance prediction but is task- and category-specific.
- The segmentation→classification paradigm of NAA is generalizable to other tasks that require fine-grained spatial reasoning with VLMs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Introduces affordance and human-likeness into dexterous grasping; the segmentation→classification conversion in NAA is highly inventive.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multi-setting evaluation with comprehensive ablations, but real-world experiments are absent.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Problem motivation is clearly articulated, framework figures are well-crafted, and ablation logic is rigorous.)
- Value: ⭐⭐⭐⭐⭐ (Redefines the evaluation criteria for dexterous grasping at the task level, with significant implications for the advancement of embodied intelligence.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RAGNet: Large-scale Reasoning-based Affordance Segmentation Benchmark towards General Grasping](../../ICCV2025/segmentation/ragnet_large-scale_reasoning-based_affordance_segmentation_benchmark_towards_gen.md)
- [\[AAAI 2026\] Vista: Scene-Aware Optimization for Streaming Video Question Answering Under Post-Hoc Queries](vista_scene-aware_optimization_for_streaming_video_question_answering_under_post.md)
- [\[AAAI 2026\] EAGLE: Episodic Appearance- and Geometry-Aware Memory for Unified 2D-3D Visual Query Localization](eagle_episodic_appearance-_and_geometry-aware_memory_for_unified_2d-3d_visual_qu.md)
- [\[NeurIPS 2025\] HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance](../../NeurIPS2025/segmentation/haodiff_human-aware_one-step_diffusion_via_dual-prompt_guidance.md)
- [\[AAAI 2026\] Guideline-Consistent Segmentation via Multi-Agent Refinement](guideline-consistent_segmentation_via_multi-agent_refinement.md)

</div>

<!-- RELATED:END -->
