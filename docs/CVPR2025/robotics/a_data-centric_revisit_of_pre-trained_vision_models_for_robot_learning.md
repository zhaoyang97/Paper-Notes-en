---
title: >-
  [Paper Note] A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning
description: >-
  [CVPR 2025][Robotics][Pre-trained Vision Models] Through systematic evaluation, it is found that DINO/iBOT outperforms MAE in robot tasks but suffers performance degradation on non-object-centric (NOC) data due to the loss of object-centric representation capabilities. This paper proposes SlotMIM, which uses a semantic bottleneck (reducing prototype numbers to encourage the emergence of objectness), cross-view consistency regularization, and slot-level contrastive learning. T…
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Pre-trained Vision Models"
  - "Robot Learning"
  - "Object-Centric Representations"
  - "Self-Supervised Learning"
  - "Data Types"
date: 2026-05-08
content_hash: f547a3e44d22a4be
---

# A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.06960](https://arxiv.org/abs/2503.06960)  
**Code**: [https://github.com/CVMI-Lab/SlotMIM](https://github.com/CVMI-Lab/SlotMIM)  
**Area**: Robotics / Visual Pre-training  
**Keywords**: Pre-trained Vision Models, Robot Learning, Object-Centric Representations, Self-Supervised Learning, Data Types

## TL;DR

Through systematic evaluation, it is found that DINO/iBOT outperforms MAE in robot tasks but suffers performance degradation on non-object-centric (NOC) data due to the loss of object-centric representation capabilities. This paper proposes SlotMIM, which uses a semantic bottleneck (reducing prototype numbers to encourage the emergence of objectness), cross-view consistency regularization, and slot-level contrastive learning. This enables the model to learn object-centric representations from NOC data, outperforming MVP/VC-1 pre-trained on >1M samples using only 241K samples.

## Background & Motivation

**Background**: Pre-trained vision models (PVMs) have become fundamental building blocks for robot learning. The prevailing approach (e.g., MVP, VC-1) utilizes MAE pre-training on ego-centric data.

**Limitations of Prior Work**: Two unquestioned assumptions: (1) MAE is the optimal pre-training method; (2) ego-centric data is the best choice. Experiments reveal neither holds true: DINO/iBOT consistently outperforms MAE in both manipulation and perception tasks, and ImageNet (object-centric) data outperforms ego-centric data.

**Key Challenge**: Although DINO/iBOT performs best on object-centric data, its performance degrades severely on non-object-centric (NOC) data such as scene-centric or ego-centric data. This is because they struggle to learn object-centric representations from NOC data, whereas objectness is highly correlated with successful manipulation (correlation coefficient of $0.72$).

**Core Idea**: Design SlotMIM—it promotes the emergence of objectness by reducing the prototype counts (creating a semantic bottleneck), learns semantic prototypes via cross-view consistency, and improves discriminability using slot-level contrastive learning. This allows PVMs to learn object-centric representations across any type of pre-training data.

## Method

### Overall Architecture

Based on iBOT extensions: (1) reducing the number of prototypes ($8192 \to 512$) to construct an information bottleneck $\to$ patch clustering reveals emerging objectness; (2) introducing cross-view patch consistency loss $\to$ prototypes gain semantic meaning; (3) pooling patches into slots according to prototype assignments $\to$ slot-level MoCo contrastive learning.

### Key Designs

1. **Representation Bottleneck**:

    - **Function**: Drastically reduces the number of clustering prototypes.
    - **Key Finding**: iBOT uses 8192 prototypes to capture fine-grained patterns but lacks semantics. Reducing this to 512 allows objectness to naturally emerge (Fig.4a: transitioning from texture-level clustering to object-level clustering).
    - **Design Motivation**: The information bottleneck forces the model to learn compositional object concepts rather than low-level patterns.

2. **Cross-view Consistency**:

    - **Function**: Forces corresponding patches from different augmented views of the same image to share the same prototype.
    - **Mechanism**: Overlap regions of the two views are aligned using ROIAlign, computing cross-view cross-entropy for matched patch pairs: $\mathcal{L}_{patch}^{cross}$.
    - **Design Motivation**: iBOT's within-view MIM loss provides no view-invariance guidance, leading to a lack of semantic consistency in prototypes.

3. **Slot-level Contrastive Learning**:

    - **Function**: Pools patches into object-level slot features based on prototype assignment, and performs MoCo contrastive learning between slots.
    - **Mechanism**: $\mathbf{s}_{\theta,i} = h_\theta(\sum_j p_\theta(\mathbf{v}_j)_i \mathbf{z}_{\theta,j})$, where slots of the same prototype across two views form positive pairs.
    - **Design Motivation**: Patch-level learning is insufficient to distinguish objects, while slot-level contrastive learning enhances object-level discriminability.

### Key Findings: Inverse Scaling

While MAE performance scales up with data, DINO/iBOT's performance paradoxically drops when scaling data from 241K to 1.28M. Reason: Self-supervised learning objectives compress representations excessively, discarding low-level visual information crucial for manipulation tasks. SlotMIM circumvents this issue—it learns fine-grained parts instead of coarse-grained objects on ego-centric data, thereby avoiding over-compression.

## Key Experimental Results

### Main Results: 241K Pre-training Comparison

| Method | Data | Franka Kitchen | Meta-World | VOC Jacc | ADE mIoU |
|------|------|:-:|:-:|:-:|:-:|
| MAE | Ego-241K | 34.2 | 36.0 | 37.1 | 40.3 |
| DINO | INet-241K | 38.5 | 42.8 | 42.2 | 44.5 |
| iBOT | INet-241K | 40.1 | 43.3 | 43.2 | 48.2 |
| **SlotMIM** | **COCO+-241K** | **42.0** | **44.1** | **43.9** | **49.1** |

### Ablation Study: Contribution of Each Component

| Configuration | mask | cross | within | slot | k-NN | ADE | Jacc |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| DINO cross-view only | ✗ | ✓ | ✗ | ✗ | 45.1 | 47.4 | 42.5 |
| +MIM | ✓ | ✓ | ✗ | ✗ | 44.9 | 48.6 | 42.3 |
| +within-view+slot | ✓ | ✓ | ✓ | ✓ | **46.2** | **49.1** | **43.9** |

### Key Findings

- **241K SlotMIM > 1M+ MVP/VC-1**: Surpasses SOTA methods using million-scale ego-centric data with only 1/4 the data size.
- **Data types should match the task**: Ego-centric data fits manipulation best, scene-centric data fits navigation, and object-centric data fits perception—the "one-size-fits-all data" concept is suboptimal.
- **Objectness correlates with manipulation performance with a coefficient of 0.72**: This serves as a vital indicator for selecting pre-training methods.

## Highlights & Insights

- **Challenging two popular assumptions**: MAE is not the optimal PVM, and ego-centric data is not always the best choice. This offers significant guidance to the robot learning community.
- **Discovery and explanation of "inverse scaling"**: Performance decreases with more data due to over-compression in self-supervised learning, providing an important supplement to scaling laws.
- **Adaptability of SlotMIM**: It learns objectness at different granularities depending on the data type (object-centric $\to$ coarse-grained objects; ego-centric $\to$ fine-grained parts), demonstrating the flexibility of the method.

## Limitations & Future Work

- The number of slots (i.e., prototypes) must be manually configured, and the optimal values differ across datasets (ImageNet: 1024, COCO: 512).
- Cross-view matching utilizes ROIAlign to handle cropping alignment, which may not scale well to drastic deformations.
- Although the evaluation across 6 tasks is diverse, the experimental scale of each task is limited (3 seeds $\times$ limited demos).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Systematic research perspective + SlotMIM methodological innovation (bottleneck + cross-view + slot contrastive), challenging existing domain assumptions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 data types $\times$ 5 methods $\times$ 6 tasks $\times$ multiple scales, highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Figs 1, 2, 4, 6 are highly informative, with a clear logical chain from systematic analysis to method design.
- Value: ⭐⭐⭐⭐⭐ Provides a novel, data-centric perspective and an effective methodology for the PVM + robot learning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Mitigating the Human-Robot Domain Discrepancy in Visual Pre-training for Robotic Manipulation](mitigating_the_human-robot_domain_discrepancy_in_visual_pre-training_for_robotic.md)
- [\[AAAI 2026\] Object-Centric Latent Action Learning](../../AAAI2026/robotics/object-centric_latent_action_learning.md)
- [\[CVPR 2025\] Reasoning in Visual Navigation of End-to-end Trained Agents: A Dynamical Systems Approach](reasoning_in_visual_navigation_of_end-to-end_trained_agents_a_dynamical_systems_.md)
- [\[CVPR 2026\] IGen: Scalable Data Generation for Robot Learning from Open-World Images](../../CVPR2026/robotics/igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](../../NeurIPS2025/robotics/learning_interactive_world_model_for_object-centric_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
