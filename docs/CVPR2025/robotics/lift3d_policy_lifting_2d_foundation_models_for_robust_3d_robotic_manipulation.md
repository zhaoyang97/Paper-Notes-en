---
title: >-
  [Paper Note] Lift3D Foundation Policy: Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation
description: >-
  [CVPR 2025][Robotics][3D Robotic Manipulation] Lift3D proposes a two-stage framework: first, it enhances the implicit 3D perception of 2D foundation models via task-aware MAE depth reconstruction; second, it directly enables 2D models to encode point cloud data by projecting 3D point clouds onto virtual planes to establish a mapping with 2D position embeddings. It achieves an average success rate of 83.9% on MetaWorld (outperforming the previous SOTA DP3’s 65.3% by 18.6 perce…
tags:
  - "CVPR 2025"
  - "Robotics"
  - "3D Robotic Manipulation"
  - "2D Foundation Models"
  - "Point Cloud Coding"
  - "Position Embedding Lifting"
  - "Implicit 3D Representations"
date: 2026-05-08
content_hash: 44b3353f0b0951b7
---

# Lift3D Foundation Policy: Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation

**Conference**: CVPR 2025  
**arXiv**: [2411.18623](https://arxiv.org/abs/2411.18623)  
**Code**: [https://lift3d-web.github.io](https://lift3d-web.github.io)  
**Area**: 3D Vision/Robotic Manipulation  
**Keywords**: 3D Robotic Manipulation, 2D Foundation Models, Point Cloud Coding, Position Embedding Lifting, Implicit 3D Representations

## TL;DR

Lift3D proposes a two-stage framework: first, it enhances the implicit 3D perception of 2D foundation models via task-aware MAE depth reconstruction; second, it directly enables 2D models to encode point cloud data by projecting 3D point clouds onto virtual planes to establish a mapping with 2D position embeddings. It achieves an average success rate of 83.9% on MetaWorld (outperforming the previous SOTA DP3’s 65.3% by 18.6 percentage points).

## Background & Motivation

**Background**: Visual robot manipulation policies require understanding 3D scenes to complete complex manipulation tasks. Current approaches fall into two categories: one directly encodes point cloud data (PointNet++, PointNext, etc.) but suffers from limited generalization due to the lack of large-scale robotic 3D data and foundation models; the other performs modality conversion—projecting 3D point clouds into multi-view images as inputs for 2D models, or lifting 2D features into 3D space, which inevitably loses spatial information.

**Limitations of Prior Work**: Directly training 3D policy models faces challenges of data scarcity and high computational costs, while modality conversion methods lose spatial geometric information during the 3D-to-2D or 2D-to-3D transition, hindering the robot's understanding of 3D spatial relations. Existing 2D foundation models (e.g., CLIP, DINOv2) possess powerful pretrained knowledge but lack 3D spatial awareness.

**Key Challenge**: Large-scale pretrained knowledge resides in 2D models, whereas robotic manipulation requires 3D spatial understanding. How can 2D pretraining knowledge be leveraged for 3D manipulation without losing spatial information?

**Goal**: (1) Enhance the 3D spatial awareness of 2D foundation models; (2) allow 2D models to directly encode 3D point cloud data without modality conversion; (3) build a robust 3D manipulation policy that leverages large-scale pretrained knowledge.

**Key Insight**: The authors observe that the Position Embedding (PE) of Transformers serves as a critical bridge connecting 2D and 3D. If mapping relations between 3D points and 2D PE can be established, 2D models can directly understand 3D inputs. The model is first endowed with implicit 3D perception via MAE depth reconstruction, followed by explicit 3D encoding via PE mapping.

**Core Idea**: By projecting 3D points onto multiple virtual planes to obtain a mapping to the pretrained 2D position embeddings, the 2D foundation model can directly encode point cloud data for manipulation policy learning without modality conversion.

## Method

### Overall Architecture

A two-stage training framework. **Stage 1 (Implicit 3D Representation)**: Self-supervised fine-tuning of the 2D foundation model is performed using a task-aware MAE. A CLIP model is utilized to extract attention maps from the task text description to guide the masking strategy, masking task-relevant regions for depth reconstruction while preserving pretrained knowledge through a distillation loss. **Stage 2 (Explicit 3D Representation)**: The point cloud is encoded into 128 tokens by a lightweight 3D tokenizer, and projected onto 6 virtual planes to map them to the original 2D PE. The PEs from multiple planes are averaged to obtain a unified 3D position embedding ($PE_{3D}$), which is added to the 3D tokens and fed into the 2D foundation model for encoding. Finally, a 3-layer MLP policy head predicts the 7-DoF end-effector behavior.

### Key Designs

1. **Task-aware Masked Autoencoder (Task-aware MAE)**:

    - **Function**: Enhance the implicit 3D spatial perception of the 2D foundation model.
    - **Mechanism**: Utilize CLIP to generate image attention maps based on task text descriptions (e.g., "Robot arm take the red bowl"), and filter out task-relevant affordance regions using a threshold $\theta=0.5$ for focused masking (maintaining a 75% overall masking rate). The reconstruction target is depth information instead of RGB, as ablation studies demonstrate that depth reconstruction is more valuable for manipulation tasks than RGB reconstruction (+6 vs +1). Meanwhile, a distillation loss is introduced to constrain the output of visible tokens to align with the original pretrained model, preventing catastrophic forgetting.
    - **Design Motivation**: Traditional MAE random masking may heavily mask irrelevant backgrounds, which is inefficient. Focusing on task-relevant affordance areas allows for more efficient learning of manipulation-related geometric information.

2. **2D Model-Lifting Strategy**:

    - **Function**: Allow the 2D foundation model to directly encode 3D point cloud data.
    - **Mechanism**: Encode the input point cloud into 128 3D tokens using a lightweight 3D tokenizer (FPS downsampling + KNN local aggregation + linear layer). Then, project the coordinates of each 3D token onto 6 virtual planes (cube projection) to obtain corresponding 2D coordinates. These coordinates are used to locate the original pretrained 2D PE, which are averaged to obtain a unified 3D position embedding: $PE_{3D} = \frac{1}{n}\sum_{j=1}^{n} PE_{2D}(C_{2D}^{ij})$. Consequently, the projection process is only used to establish positional mapping rather than construct model inputs, thereby bypassing modality conversion.
    - **Design Motivation**: Introducing an entirely new 3D PE would cause semantic misalignment with the pretrained 2D model; reusing the original pretrained PE maximizes the preservation of large-scale pretrained knowledge.

3. **Distillation + Frozen Training Strategy**:

    - **Function**: Retain the pretrained knowledge of the 2D foundation model during fine-tuning.
    - **Mechanism**: Stage 1 uses an L1 distillation loss to match the output of visible tokens from the fine-tuned model with the original model. Stage 2 freezes the parameters of the 2D foundation model, updating only the 3D tokenizer, the injected adapter, and the policy head. Ablations show that distillation yields an +8% success rate improvement.
    - **Design Motivation**: Prevent catastrophic forgetting of large-scale pretrained knowledge when fine-tuning on small-scale robotic data.

### Loss & Training

Stage 1: $\mathcal{L}_{\text{implicit}} = \|2D_e(x_{\text{vis}}) - 2D_e^{\text{pre}}(x_{\text{vis}})\|_1 + \|2D_d(\cdot) - D_{\text{target}}\|_1$ (distillation + depth reconstruction). Stage 2: $\mathcal{L}_{\text{explicit}} = \text{MSE}(T) + (1 - \cos(R)) + \text{BCE}(G)$ (translation MSE + rotation cosine + gripper binary classification). Stage 1 is trained using self-supervised learning on 1 million samples, while Stage 2 performs imitation learning on 25-100 demos.

## Key Experimental Results

### Main Results

| Method | Type | Input | MetaWorld Mean S.R. | Adroit Mean S.R. |
|------|------|------|-------------------|-----------------|
| CLIP | 2D Rep. | RGB | 65.3 | 84.0 |
| R3M | 2D Rep. | RGB | 75.1 | 85.3 |
| PointNet++ | 3D Rep. | PC | 61.6 | 76.0 |
| SPA | 3D Rep. | RGB | 69.5 | 81.3 |
| DP3 | 3D Policy | PC | 65.3 | 66.7 |
| **Lift3D (CLIP)** | **Ours** | **PC** | **83.9** | **88.0** |

### Ablation Study

| Configuration | Mean Accuracy | Gain |
|------|:---:|---:|
| Baseline (w/o MAE, w/o Lifting) | 62 | +0 |
| + Depth reconstruction | 68 | +6 |
| + Affordance masking + Depth | 72 | +10 |
| + Visual distillation | 80 | +18 |
| + 2D Model-Lifting (our PE) | **96** | **+34** |
| + Learnable PE (Replacing our PE) | 90 | +28 |

### Key Findings

- 2D Model-Lifting is the most critical component (from 80 to 96, +16), proving that directly encoding 3D data with 2D pretrained PE is 6 percentage points more effective than introducing a new PE.
- Depth reconstruction is more crucial than RGB reconstruction (+6 vs +1), indicating that geometric information is vital for manipulation tasks.
- Affordance masking brings an additional 4% improvement compared to random masking, validating the value of focusing on task-relevant regions.
- Distillation prevents catastrophic forgetting, yielding an +8% improvement.
- The model learns new manipulation skills with only 30 real-world demos, demonstrating generalization to different objects, backgrounds, and lighting conditions.

## Highlights & Insights

- **The "projection only for map building, no modality conversion" design is highly elegant**—Projecting the point cloud onto virtual planes is only used for locating corresponding 2D PEs, while the actual input remains 3D tokens, entirely avoiding information loss. This design is transferable to any scenario requiring the use of pretrained 2D models for 3D data.
- **The two-stage progressive enhancement strategy is clear and effective**—It first implicitly enhances 3D perception (MAE + depth reconstruction) and then explicitly implements 3D inputs (PE mapping + point cloud encoding), with clear objectives and validation in each stage.
- **Task-aware masking** introduces semantic priors from CLIP to guide geometric learning; this cross-modal information integration paradigm is highly insightful.

## Limitations & Future Work

- Only a simple MLP is used as the policy head, without exploring stronger action decoders such as diffusion policies.
- Requires single-view RGB-D input; performance under pure RGB or multi-view settings remains unexplored.
- Stage 1 MAE training requires a large-scale dataset of 1 million samples, incurring non-trivial data preparation costs.
- The number and positions of virtual planes affect performance, but a sufficient sensitivity analysis is not provided in the paper.
- Integration with VLA models can be explored to achieve more flexible task generalization using language instructions.

## Related Work & Insights

- **vs. DP3 (3D Diffusion Policy)**: DP3 generates actions directly on point clouds using a diffusion model, but training the 3D encoder from scratch lacks pretraining knowledge; Lift3D leverages 2D pretraining knowledge to outperform DP3 by 18.6 percentage points on MetaWorld.
- **vs. RVT-2**: RVT-2 projects point clouds into multi-view images to feed into 2D models, losing spatial information in modality conversion; Lift3D avoids modality conversion through PE mapping.
- **vs. Act3D/ChainedDiffuser**: These methods lift 2D features to 3D space for multi-scale representation, requiring complex 3D feature construction; Lift3D more concisely reuses the 2D model itself.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Proposes for the first time the scheme of using PE mapping to allow 2D models to directly encode 3D point clouds; the MAE design is also innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three simulation benchmarks, real-world scenes, 30+ tasks, detailed ablations, generalization tests, and scalability analysis.
- Writing Quality: ⭐⭐⭐⭐ The methodological motivation is clearly derived, and the two-stage framework is thoroughly described.
- Value: ⭐⭐⭐⭐⭐ Solves the core challenge of leveraging 2D pretrained knowledge in 3D manipulation, holding significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] 3D-MVP: 3D Multiview Pretraining for Robotic Manipulation](3d-mvp_3d_multiview_pretraining_for_manipulation.md)
- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](../../NeurIPS2025/robotics/robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)
- [\[ICLR 2026\] Policy Contrastive Decoding for Robotic Foundation Models](../../ICLR2026/robotics/policy_contrastive_decoding_for_robotic_foundation_models.md)
- [\[CVPR 2026\] Localizing, Structuring, and Rendering: Bridging 3D and 2D Vision-Language-Action Models for Robotic Manipulation](../../CVPR2026/robotics/localizing_structuring_and_rendering_bridging_3d_and_2d_vision-language-action_m.md)
- [\[ICLR 2026\] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](../../ICLR2026/robotics/rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)

</div>

<!-- RELATED:END -->
