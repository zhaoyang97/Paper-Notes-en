---
title: >-
  [Paper Note] iManip: Skill-Incremental Learning for Robotic Manipulation
description: >-
  [ICCV 2025][Robotics][Incremental learning] This paper proposes the iManip framework, which enables robots to continually acquire new manipulation skills without retraining through a temporal replay strategy and a scalab…
tags:
  - "ICCV 2025"
  - "Robotics"
  - "Incremental learning"
  - "robotic manipulation"
  - "catastrophic forgetting"
  - "temporal replay"
  - "scalable Transformer"
date: 2026-05-08
content_hash: 32dba3c0540da49a
---

# iManip: Skill-Incremental Learning for Robotic Manipulation

**Conference**: ICCV 2025
**arXiv**: [2503.07087](https://arxiv.org/abs/2503.07087)  
**Code**: Coming soon  
**Area**: Robotics
**Keywords**: Incremental learning, robotic manipulation, catastrophic forgetting, temporal replay, scalable Transformer

## TL;DR

This paper proposes the iManip framework, which enables robots to continually acquire new manipulation skills without retraining through a temporal replay strategy and a scalable PerceiverIO architecture, while mitigating catastrophic forgetting of previously learned skills. iManip achieves an average improvement of 9.4% over conventional incremental learning baselines on RLBench.

## Background & Motivation

**Background**: Mainstream methods in robotic manipulation focus on improving performance on single or multi-task manipulation (e.g., PerAct, ManiGaussian), or on transferring knowledge from pretrained large language/vision models to robotic tasks. Few works have investigated how robots can incrementally learn new skills.

**Limitations of Prior Work**: Existing incremental learning benchmarks (e.g., LIBERO) have begun exploring lifelong learning, but the tasks therein share the same underlying manipulation skills (e.g., "place the bowl on the plate" vs. "place the bowl on the stove"), differing only in object or spatial location. These benchmarks do not involve genuinely new skill acquisition. Directly applying conventional incremental learning methods (e.g., iCaRL, EEIL) to robotic manipulation still leads to severe catastrophic forgetting.

**Key Challenge**: Traditional incremental learning methods are primarily designed for image classification and overlook two unique complexities in robotic manipulation tasks: (1) **Temporal complexity**: the environment and robot state change dynamically over time, and each action influences subsequent actions; (2) **Action complexity**: robots must learn new action primitives (e.g., translation, rotation, grasping) whose representations in 3D space are highly complex.

**Goal**: Design a skill-incremental learning framework that allows robots to retain knowledge of prior skills when learning new manipulation skills, without retraining from scratch.

**Key Insight**: The authors observe that classical exemplar replay methods (e.g., herding, hard-exemplar sampling) select representative samples without accounting for the temporal structure of trajectory data, resulting in temporal imbalance. Moreover, classical methods focus exclusively on visual features while neglecting the need to expand the action space for new skills.

**Core Idea**: Maintain temporal data integrity through keyframe-based farthest-distance entropy sampling, while adapting action primitives for new skills using expandable weight matrices and skill-specific action prompts.

## Method

### Overall Architecture

iManip takes multi-view RGB-D images and language instructions as input, and outputs robot actions comprising 3D translation, rotation, gripper open/close, and collision avoidance. The framework consists of three main components:
- **Voxel Encoder**: Projects RGB-D images into 3D voxels and extracts scene features using a UNet-style 3D convolutional encoder.
- **Extendable PerceiverIO** (core): Receives multimodal inputs consisting of voxel tokens, language tokens, and action prompt tokens, and encodes them via cross-attention and extendable self-attention layers.
- **Policy Decoder**: Predicts the optimal robot action.

The learning pipeline first trains on a base skill set, then incrementally learns one new skill at each step, leveraging replay samples from a memory buffer of old skills and knowledge distillation to preserve prior knowledge.

### Key Designs

1. **Temporal Replay Strategy (TRS)**:

    - **Function**: Stores a fixed number of representative demonstration samples for old skills while preserving the temporal integrity of trajectory data.
    - **Mechanism**: Demonstration trajectories are first segmented by keyframes (i.e., moments when the end-effector state changes or velocity approaches zero). For each keyframe type, samples are selected using **farthest-distance entropy sampling**: the action prediction entropy of each sample is computed, a distance matrix $A[i][j] = \text{distance}(e_i, e_j)$ is constructed, and new samples are greedily selected to maximize the sum of entropy distances to the already-selected set: $j = \arg\max_{j \in E} \sum_{k \in S} A[j][k]$. Time complexity is $O(N^2)$.
    - **Design Motivation**: Classical replay methods (e.g., herding) select the most representative samples without considering temporal balance — samples from the same phase may be repeatedly selected, leading to execution instability. Uniform sampling by keyframe combined with entropy distance maximization ensures both temporal coverage and sample diversity.

2. **Extendable PerceiverIO (EPIO)**:

    - **Function**: Adapts action primitives for new skills via expandable weight matrices and skill-specific action prompts.
    - **Mechanism**: The input is $X = [X_{\text{voxel}}, X_{\text{language}}, X_{\text{action}}]$, where the action prompt $X_{\text{action}} = [X_{\text{action}}^{\text{old}}, X_{\text{action}}^{\text{new}}]$ concatenates prompts for old and new skills. In the self-attention layer, the weight matrices for Q and K are expandable: $W_Q^{\text{scale}} = [W_Q^{\text{old}}, W_Q^{\text{new}}]$. When learning a new skill, the old weights $W_Q^{\text{old}}$ are frozen, and only the new weights $W_Q^{\text{new}} \in \mathbb{R}^{d \times d_{\text{new}}}$ and new action prompts are learned.
    - **Design Motivation**: Different skills in robotic manipulation require different action primitives (e.g., pouring water vs. pressing a button). Shared weights are insufficient for this adaptation. Freezing old weights while extending new ones prevents overwriting prior knowledge while providing dedicated learning capacity for new skills.

3. **Knowledge Distillation (KD)**:

    - **Function**: Transfers knowledge between the old and new models to further prevent forgetting.
    - **Mechanism**: The output probability distributions of the old model are used to supervise training of the new model. The distillation loss is:
    $$\mathcal{L}_{\text{dis}} = \mathcal{L}_2(\mathcal{Q}_{\text{trans}}^{\text{old}}, \mathcal{Q}_{\text{trans}}^{\text{new}}) + \mathcal{L}_2(\mathcal{Q}_{\text{rot}}^{\text{old}}, \mathcal{Q}_{\text{rot}}^{\text{new}}) + |\mathcal{Q}_{\text{open}}^{\text{old}} - \mathcal{Q}_{\text{open}}^{\text{new}}| + |\mathcal{Q}_{\text{collide}}^{\text{old}} - \mathcal{Q}_{\text{collide}}^{\text{new}}|$$
    where MSE is applied to translation and rotation, and L1 is applied to gripper and collision branches.
    - **Design Motivation**: Replay and weight freezing alone may be insufficient; distillation provides an additional regularization signal to ensure that the action distributions of old skills are not corrupted.

### Loss & Training

The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{act}} + \lambda_{\text{dis}} \mathcal{L}_{\text{dis}}$, where $\mathcal{L}_{\text{act}}$ is the standard cross-entropy action loss covering four branches (translation, rotation, open/close, collision avoidance), and $\lambda_{\text{dis}} = 0.01$. When learning a new skill, the encoder and old PerceiverIO parameters are frozen; only the policy decoder, new action prompts, and new extended weights are fine-tuned, yielding faster convergence and fewer trainable parameters.

## Key Experimental Results

### Main Results

On RLBench under the B5-5N1 setting (pre-trained on 5 base skills, then incrementally learning 1 new skill per step over 5 steps for a total of 10 skills):

| Method | Base | Step1 All | Step2 All | Step3 All | Step4 All | Step5 All | Avg. |
|--------|------|-----------|-----------|-----------|-----------|-----------|------|
| PerAct (multi-task) | 44.0 | 7.3 | 5.1 | 9.0 | 6.7 | 1.6 | 5.9 |
| ManiGaussian (multi-task) | 55.2 | 20.7 | 12.0 | 15.5 | 9.3 | 5.2 | 12.5 |
| P-TIB (incremental) | 44.0 | 34.7 | 25.1 | 26.0 | 16.4 | 10.4 | 22.5 |
| M-TIB (incremental) | 55.2 | 45.3 | 37.1 | 39.5 | 31.6 | 26.8 | 36.1 |
| **iManip (Ours)** | **56.0** | **56.7** | **48.0** | **47.5** | **39.1** | **36.0** | **45.5** |

### Ablation Study

| Configuration | B5-1N1 | B5-5N1 | Description |
|---------------|--------|--------|-------------|
| R1: No incremental strategy | 20.7 | 5.2 | No forgetting mitigation |
| R2: +TRS | 49.3 | 27.6 | +Temporal replay, +22.4% |
| R3: +TRS+EPIO | 54.0 | 32.4 | +Extendable PerceiverIO, +4.8% |
| Full: +TRS+EPIO+DIS | 56.7 | 36.0 | +Distillation, +3.6%; full model is best |

### Key Findings

- **TRS contributes the most**: Removing TRS causes a 22.4% performance drop, demonstrating that preserving trajectory temporal integrity is critical in robotic manipulation.
- **Freezing the encoder + EPIO while training only the decoder is most effective**: Only 8M parameters are required (vs. 47M for full fine-tuning), convergence steps are reduced from 100k to 60k, and old skill retention is higher.
- **Classical replay methods fail**: Herding and hard-exemplar sampling achieve only ~15% success on old skills, whereas temporal replay reaches 57.6%.
- The method maintains its advantage across different incremental settings (B5-1N5, B2-4N2, B3-2N3).
- In real-robot experiments involving 5 everyday manipulation skills, the success rate on old skills after 4 incremental steps improves from a baseline of 0% to 40%.

## Highlights & Insights

- **Elegant temporal replay design**: Grouping samples by keyframes preserves temporal integrity of trajectories, while farthest-distance entropy sampling promotes sample diversity. This idea is transferable to any incremental learning scenario involving temporal data replay.
- **Expandable weight matrices**: Horizontally concatenating new weight columns to Q/K projection matrices is a simple yet effective approach; freezing old weights while learning only new ones achieves zero forgetting. A similar design pattern can be applied to other Transformer architectures requiring continual capability expansion.
- **Grad-CAM visualization**: Different skills activate distinct action prompt weights, confirming that action prompts successfully encode skill-specific action primitives.

## Limitations & Future Work

- Only one new skill is added per step; the setting of simultaneously introducing multiple complex skills is not evaluated.
- Skills are treated as relatively independent; knowledge sharing and composition across skills are not explored.
- Only single-view RGB-D input is used, without multi-view information.
- Evaluation is primarily conducted in the RLBench simulation environment; real-world experiments are limited in scale (only 10 trials per skill).
- Memory overhead grows linearly with the number of learned skills.

## Related Work & Insights

- **vs. LIBERO**: LIBERO addresses incremental learning over object/spatial variations, whereas this work targets genuinely new skill increments, which is more aligned with real-world requirements.
- **vs. iCaRL/EEIL**: These conventional incremental learning methods are effective for classification tasks but disregard the temporal and action complexity of robotic manipulation, resulting in poor direct transfer.
- **vs. PerAct/ManiGaussian**: These multi-task methods require retraining on all skills when new ones are introduced, making them highly inefficient.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to systematically define and address skill-incremental learning for robotic manipulation; however, the core techniques (replay + freezing + distillation) are improvements upon existing ideas.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablations, multiple settings compared, and real-robot experiments included, though the real-world evaluation scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear, method descriptions are detailed, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ — Provides a practical benchmark and baseline for continual learning in robotics with strong potential for real-world application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Policy Compatible Skill Incremental Learning via Lazy Learning Interface](../../NeurIPS2025/robotics/policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)
- [\[ICCV 2025\] PASG: A Closed-Loop Framework for Automated Geometric Primitive Extraction and Semantic Anchoring in Robotic Manipulation](pasg_a_closed-loop_framework_for_automated_geometric_primitive_extraction_and_se.md)
- [\[ICCV 2025\] Weakly-Supervised Learning of Dense Functional Correspondences](weakly-supervised_learning_of_dense_functional_correspondences.md)
- [\[ICCV 2025\] Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)
- [\[CVPR 2026\] Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](../../CVPR2026/robotics/lifelong_imitation_learning_multimodal_latent_rep.md)

</div>

<!-- RELATED:END -->
