---
title: >-
  [Paper Note] PartSTAD: 2D-to-3D Part Segmentation Task Adaptation
description: >-
  [ECCV 2024][Segmentation][3D Part Segmentation] PartSTAD proposes a task adaptation method for 2D-to-3D part segmentation. By introducing a learnable weight prediction network for GLIP's 2D bounding boxes (optimized targeting 3D mRIoU) and integrating SAM to acquire precise foreground masks, it achieves a 7.0%p improvement in semantic segmentation mIoU and a 5.2%p improvement in instance segmentation mAP50 on PartNet-Mobility (relative to PartSLIP).
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "3D Part Segmentation"
  - "Task Adaptation"
  - "Few-shot"
  - "2D-to-3D lifting"
  - "SAM"
date: 2026-05-08
content_hash: b7326c3f05ecc430
---

# PartSTAD: 2D-to-3D Part Segmentation Task Adaptation

**Conference**: ECCV 2024  
**arXiv**: [2401.05906](https://arxiv.org/abs/2401.05906)  
**Code**: [https://github.com/KAIST-Visual-AI-Group/PartSTAD](https://github.com/KAIST-Visual-AI-Group/PartSTAD)  
**Area**: Segmentation  
**Keywords**: 3D Part Segmentation, Task Adaptation, Few-shot, 2D-to-3D lifting, SAM

## TL;DR

PartSTAD proposes a task adaptation method for 2D-to-3D part segmentation. By introducing a learnable weight prediction network for GLIP's 2D bounding boxes (optimized targeting 3D mRIoU) and integrating SAM to acquire precise foreground masks, it achieves a 7.0%p improvement in semantic segmentation mIoU and a 5.2%p improvement in instance segmentation mAP50 on PartNet-Mobility (relative to PartSLIP).

## Background & Motivation

3D part segmentation is a fundamental task for understanding the structure, function, and semantics of 3D shapes. However, 3D annotated data is extremely scarce—the largest part-annotated dataset, PartNet, contains fewer than 30,000 models, whereas 2D image annotations have reached the million scale.

**Prior Work**: Methods like PartSLIP utilize 2D vision-language models (GLIP) for multi-view rendering $\rightarrow$ 2D detection $\rightarrow$ voting aggregation to 3D, and fine-tune GLIP on synthetic data to adapt to rendered images and unnatural text prompts (domain adaptation).

**Limitations of Prior Work**:
- The fine-tuning of PartSLIP only performs **domain adaptation** (adapting GLIP to synthetic images and part name lists) instead of **task adaptation** (optimizing towards the final 3D segmentation quality).
- 2D bounding boxes inevitably contain noise; the key is how to control the influence of noise on the final 3D segmentation during multi-view integration.
- GLIP only outputs bounding boxes instead of segmentation masks, leading to imprecise segmentation boundaries.

**Key Insight**: To treat 2D-to-3D part segmentation as a **task adaptation** problem—while keeping the pre-trained weights frozen, a small weight prediction network is trained to optimize the aggregation of 2D boxes using 3D mIoU as the objective function; concurrently, SAM is introduced to obtain precise foreground masks to replace bounding boxes.

## Method

### Overall Architecture

The pipeline of PartSTAD is as follows:
1. Render the 3D point cloud into multi-view (10 fixed views) 2D images.
2. Extract 2D bounding boxes for each view using the fine-tuned GLIP.
3. Convert each bounding box into a foreground mask using SAM (SAM Mask Integration).
4. Predict a weight for each box/mask (Weight Prediction Network).
5. Aggregate to the 3D point cloud via weighted voting (2D-to-3D Task Adaptation).
6. Obtain the final segmentation labels based on superpoints.

Both GLIP and SAM are frozen, and only the weight prediction network is trained (per category, 8-shot objects).

### Key Designs

1. **3D mRIoU Loss Function**:

    - - Function: Directly use 3D segmentation quality as the adaptation objective.
    - - Mechanism: Since the standard mIoU is non-differentiable, relaxed IoU (mRIoU) is used to relax the predicted labels from $\{0,1\}$ to $[0,1]$.
    - - Formula: $\mathcal{L}_{\text{mRIoU}} = 1 - \frac{1}{M}\sum_{j=1}^{M} \frac{\mathbf{l}_j^\top \hat{\mathbf{l}}_j}{\|\mathbf{l}_j\|_1 + \|\hat{\mathbf{l}}_j\|_1 - \mathbf{l}_j^\top \hat{\mathbf{l}}_j}$
    - - Design Motivation: Cross-entropy loss is less effective than mRIoU in 3D segmentation tasks (as verified by supplementary experiments); mRIoU directly optimizes the evaluation metric itself.

2. **Bounding Box Weight Prediction**:

    - - Function: Predict a weight for each 2D bounding box to control its contribution to the 3D voting.
    - - Mechanism: Since mRIoU is non-differentiable with respect to the bounding box positions, the object positions are not directly adjusted; instead, a positive weight $W(b)$ is predicted to multiply the voting score.
    - - Modified Voting Formula: $\tilde{s}_{ij} = \frac{\sum_k \sum_{p \in P_i} V_k(p) \cdot \max_{b} I_b(p) \cdot W(b)}{\sum_k \sum_{p \in P_i} V_k(p)}$
    - - The final scores are normalized via softmax; the score for the null label is set as a learnable parameter (initial value of 10).
    - - Network Architecture: A two-layer shared MLP with context normalization in between to capture the global box context. The output is processed by a modified ReLU $\phi(x) = \max(\tau + x, 0)$ ($\tau=10$ ensures positive initial weights).
    - - Design Motivation: Significant improvements are achieved over the original PartSLIP voting framework with minimal modification (only multiplying weights); weight prediction can suppress noisy boxes and reinforce high-quality ones.

3. **SAM Mask Integration**:

    - - Function: Convert 2D bounding boxes into precise foreground masks using SAM's box-prompted segmentation capability.
    - - Mechanism: Use the bounding boxes as input prompts for SAM to obtain precise foreground segmentation regions, replacing the original rectangular boxes.
    - - Implementation: The point-to-bounding-box membership $I_b$ becomes point-to-mask membership, while weight prediction still utilizes GLIP's box features.
    - - Design Motivation: GLIP-generated bounding boxes contain a large amount of background; using SAM to extract the foreground significantly improves segmentation boundaries.

### Loss & Training

- - Loss: 3D mRIoU loss
- - Training Settings: Per-category training, with 8 annotated 3D objects per category (few-shot).
- - Trainable Parameters: Only the weight prediction MLP + null label score (an extremely small number of parameters).
- - GLIP and SAM are completely frozen.

## Key Experimental Results

### Main Results

Semantic segmentation mIoU (%) on PartNet-Mobility (10 representative categories):

| Method | Mean mIoU | Storage | Furniture | Table | Chair | Switch | Toilet | Laptop | USB | Remote | Scissors |
|------|----------|---------|-----------|-------|-------|--------|--------|--------|-----|--------|----------|
| SATR | 29.3 | 20.6 | 23.3 | 33.1 | 21.4 | 17.6 | 11.2 | 30.2 | 17.2 | 36.8 | - |
| SATR+SP | 34.8 | 28.9 | 28.0 | 37.7 | 37.0 | 22.1 | 12.4 | 33.4 | 28.0 | 43.0 | - |
| PartSLIP | 58.0 | 52.3 | 44.6 | 82.8 | 52.1 | 50.4 | 31.2 | 52.1 | 36.6 | 61.4 | - |
| **PartSTAD** | **65.0** | **59.5** | **47.8** | **85.3** | **57.9** | **57.5** | **34.6** | **59.9** | **53.4** | **68.5** | - |

Instance segmentation mAP50 (%):

| Method | Mean mAP | Storage | Furniture | Table | Chair | Toilet | Laptop | USB | Remote | Scissors |
|------|---------|---------|-----------|-------|-------|--------|--------|-----|--------|----------|
| PartSLIP | 41.6 | 29.1 | 32.6 | 82.2 | 21.2 | 36.2 | 17.8 | 20.9 | 19.9 | 23.6 |
| **PartSTAD** | **45.6** | **33.8** | **33.7** | **83.6** | **23.5** | **41.5** | **26.5** | **25.7** | **26.2** | **28.0** |

### Ablation Study

Ablation of semantic segmentation components (mean mIoU across 45 categories):

| Configuration | Mean mIoU | Description |
|------|----------|------|
| PartSLIP (Baseline) | 58.0 | No weight prediction, no SAM |
| w/o Weight Prediction | 61.9 | Only SAM added, +3.9 |
| w/o SAM Integration | 62.1 | Only weight prediction added, +4.1 |
| **PartSTAD (Full)** | **65.0** | Combination of both, +7.0 |

The contributions of each component are quantified through ablation:
- Removing weight prediction: mIoU decreases by 3.1%p ($65.0 \rightarrow 61.9$).
- Removing SAM mask: mIoU decreases by 2.9%p ($65.0 \rightarrow 62.1$).
- The contributions of the two components are complementary; the joint improvement is greater than the sum of individual improvements.

### Key Findings

1. **Task Adaptation > Domain Adaptation**: PartSLIP only performs domain adaptation (adapting to synthetic images), whereas PartSTAD performs task adaptation targeting 3D mIoU, yielding an additional 7.0%p improvement.
2. **Weight Prediction is the Core Contribution**: Simply by predicting a scalar weight for each bounding box (a minimal modification), a 4.1%p improvement is achieved.
3. **SAM Mask Significantly Improves Boundaries**: Especially effective for small parts (e.g., Camera, minor parts of Chair) and thin parts (e.g., Clock hands).
4. **Consistent Improvement across All Categories**: Regardless of the object type, PartSTAD exhibits improvements over PartSLIP, with some categories (e.g., Remote) exceeding 15%p.
5. **mRIoU Loss Outperforms Cross-Entropy**: Directly optimizing the evaluation metric itself is more effective.

## Highlights & Insights

- **The Perspective of 'Task Adaptation' is Highly Insightful**: It explicitly distinguishes between domain adaptation and task adaptation, the latter being critical in 2D-to-3D lifting scenarios.
- **Minimal yet Effective Design**: Formulates a simple MLP to predict bounding box weights without modifying any parameters of GLIP itself.
- **Clever Handling of mRIoU as the Objective Function**: It bypasses the non-differentiability of IoU with respect to discrete parameters, achieving end-to-end optimization by predicting weights instead of coordinates.
- **Complementary Combination of SAM and GLIP**: GLIP is responsible for semantic detection (identifying what is in each box), while SAM handles precise segmentation (tightening the boxes to foreground boundaries).

## Limitations & Future Work

- The weight prediction network is trained per-category and is not shared across different categories, which limits scalability.
- Experiments are conducted only on PartNet-Mobility (synthetic CAD models); generalization to real-scanned 3D data is not fully validated.
- 10 fixed views might be insufficient; occlusion and self-occlusion issues are not fully addressed.
- SAM may still generate imprecise masks for extremely small parts.
- The efficacy under extreme few-shot settings (1-2 objects) remains unknown given the 8-shot per category setup.

## Related Work & Insights

- **PartSLIP**: The direct foundation of PartSTAD, which proposes a 2D-to-3D pipeline using GLIP fine-tuning + superpoint voting.
- **SATR**: A similar approach but uses geodesic propagation instead of voting, which yields inferior performance.
- **SAM3D**: Directly lifts SAM masks into 3D, but lacks semantic labels.
- **LoRA / PEFT**: The concept of the weight prediction network of ours is highly aligned with PEFT—freezing the pre-trained model and training a minimal number of parameters.
- **Insight**: The task adaptation concept can be generalized to other 2D-to-3D lifting tasks (e.g., scene segmentation, semantic SLAM).

## Rating

- - Novelty: ⭐⭐⭐⭐ (The task adaptation perspective is insightful, and the weight prediction scheme is simple yet elegant)
- - Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive experiments on 45 categories plus ablation studies, but evaluated on only one dataset)
- - Writing Quality: ⭐⭐⭐⭐ (Clear logic, comprehensive mathematical derivations)
- - Value: ⭐⭐⭐⭐ (Provides a proper optimization direction for 2D-to-3D segmentation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] General and Task-Oriented Video Segmentation](general_and_task-oriented_video_segmentation.md)
- [\[ICCV 2025\] PartField: Learning 3D Feature Fields for Part Segmentation and Beyond](../../ICCV2025/segmentation/partfield_learning_3d_feature_fields_for_part_segmentation_and_beyond.md)
- [\[ACL 2025\] InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning](../../ACL2025/segmentation/instructpart_task-oriented_part_segmentation_with_instruction_reasoning.md)
- [\[ECCV 2024\] Part2Object: Hierarchical Unsupervised 3D Instance Segmentation](part2object_hierarchical_unsupervised_3d_instance_segmentation.md)
- [\[NeurIPS 2025\] PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding](../../NeurIPS2025/segmentation/partnext_a_next-generation_dataset_for_fine-grained_and_hierarchical_3d_part_und.md)

</div>

<!-- RELATED:END -->
