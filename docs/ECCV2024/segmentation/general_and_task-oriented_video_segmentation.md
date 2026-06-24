---
title: >-
  [Paper Note] General and Task-Oriented Video Segmentation
description: >-
  [ECCV 2024][Segmentation][Video Segmentation] GvSeg proposes a universal video segmentation framework. By decoupling the segmentation target into three factors—appearance, shape, and position—and dynamically adjusting the involvement of these three factors in query initialization, matching, and sampling based on task requirements (VIS/VSS/VPS/EVS), it achieves state-of-the-art (SOTA) performance across four video segmentation tasks within a unified architecture.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Video Segmentation"
  - "Universal Framework"
  - "Task-Oriented"
  - "Query Matching"
  - "Temporal Contrastive Learning"
date: 2026-05-08
content_hash: 8342bbb1190a94d7
---

# General and Task-Oriented Video Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2407.06540](https://arxiv.org/abs/2407.06540)  
**Code**: [https://github.com/kagawa588/GvSeg](https://github.com/kagawa588/GvSeg)  
**Area**: Image Segmentation  
**Keywords**: Video Segmentation, Universal Framework, Task-Oriented, Query Matching, Temporal Contrastive Learning

## TL;DR

GvSeg proposes a universal video segmentation framework. By decoupling the segmentation target into three factors—appearance, shape, and position—and dynamically adjusting the involvement of these three factors in query initialization, matching, and sampling based on task requirements (VIS/VSS/VPS/EVS), it achieves state-of-the-art (SOTA) performance across four video segmentation tasks within a unified architecture.

## Background & Motivation

Video segmentation is a fundamental task in computer vision, comprising four major subtasks: Video Instance Segmentation (VIS), Video Semantic Segmentation (VSS), Video Panoptic Segmentation (VPS), and Exemplar-guided Video Segmentation (EVS). Currently, there are two trends in the industry: first, designing specialized models for each subtask, which perform well but are redundant and lack cross-task transferability; second, pursuing universal solutions, though existing universal frameworks (such as Video K-Net, TubeFormer, Tube-Link) are highly homogenized—applying the same query initialization, matching, and spatiotemporal learning strategies to all tasks, thus ignoring the intrinsic differences between different tasks and leading to suboptimal performance.

**Key Challenge**: The trade-off between universality and task specificity. The goal is to maintain a unified architecture while adapting to the unique characteristics of different tasks (e.g., VIS emphasizes instance discrimination, while VSS emphasizes semantic understanding).

**Key Insight**: Rethinking the constituent elements of segmentation targets—appearance, shape, and position—and flexibly adjusting the involvement of these three elements based on different task requirements.

**Core Idea**: Achieving adaptability to different video segmentation tasks without altering the network architecture, utilizing a Shape-Position Descriptor and task-oriented strategies.

## Method

### Overall Architecture

GvSeg adopts a semi-online query-based video segmentation paradigm, built upon Mask2Former. The input is a video clip containing three frames, from which frame-level queries are extracted via a Transformer encoder-decoder, and target association is achieved through cross-frame query matching. The key innovations lie in three architecture-agnostic designs: the Shape-Position Descriptor, task-oriented query initialization and association strategies, and task-oriented temporal contrastive learning.

### Key Designs

1. **Shape-Position Descriptor**

   **Function**: Constructing a compact descriptor for each target object that simultaneously encodes shape and relative spatial position information.

   **Design Motivation**: Existing methods rely solely on appearance features for cross-frame matching, ignoring shape and position cues. In scenarios with low light, occlusion, etc., appearance features become unreliable, and shape and position information can provide crucial complementary cues.

   **Mechanism**: Inspired by Shape Context, $M=200$ anchor points are uniformly sampled along the target contour and converted to a polar coordinate system centered at the target's center of mass. The polar coordinate space is partitioned into $u \times v = 36 \times 12$ bins, and the number of anchor points in each bin is counted to obtain a histogram $H$. Furthermore, bins whose center points fall outside the mask are assigned negative values of $-1/\sqrt{d_{model}}$, enabling the descriptor to encode both shape and position.

   **Key Formula**: $H_{ij}$ counts the number of anchor points in the polar coordinate bin. Objects with geographic shapes but different positions (such as B and C) are distinguished by different negative value region distributions.

   **Difference from Prior Work**: Unlike methods that solely use appearance features for matching, GvSeg's SPA query matching injects the descriptor $H$ prior to the SelfAttn of the Transformer decoder (analogous to APE encoding) and overlays $H$ onto the query embeddings during cross-frame matching: $S_{ij} = \text{cosine}(\hat{q}_i^t + H_i^t, \hat{q}_j^{t+1} + H_j^{t+1})$.

2. **Task-Oriented Query Initialization and Object Association**

   **Function**: Designing tailored query initialization and association strategies specifically for the characteristics of the four distinct tasks.

   **Design Motivation**: VIS requires strong instance discrimination (emphasizing shape/position), VSS requires strong semantic understanding (which should not excessively rely on shape/position), VPS needs to balance both, and EVS requires utilizing given annotation prompts. A unified strategy cannot satisfy all these requirements simultaneously.

   **Mechanism**:
    - **EVS**: Queries are initialized by sampling backbone features corresponding to the annotation prompts (points/boxes/masks), and SPA matching is applied to enhance instance discrimination.
    - **VIS**: Backbone features are divided into an $S \times S$ grid, and $N$ elements are randomly selected to initialize queries, encoding position and appearance information, followed by SPA matching.
    - **VSS**: Initialized utilizing semantic class queries $\bar{q}$ updated momentum-wise during training, without using SPA matching (as shape/position can impair semantic generalization).
    - **VPS**: Combining VIS (thing) and VSS (stuff) strategies.

   **Difference from Prior Work**: Prior universal plans (such as Video K-Net) either used a uniform strategy to handle all tasks or modified the architecture (e.g., dual-path designs) to adapt to different tasks. GvSeg achieves task adaptation without altering the architecture, solely by tuning the involvement of the three key elements.

3. **Task-Oriented Temporal Contrastive Learning (Task-Oriented TCL)**

   **Function**: Improving the cross-frame positive and negative sample sampling strategy.

   **Design Motivation**: Existing TCL only selects reference frames from temporal neighborhoods, ignoring distant frames, which severely limits the quantity of positive/negative samples. Additionally, the same instance in distant frames might undergo significant changes in shape/position, making it unsuitable as a positive sample.

   **Mechanism**:
    - **Instance-level Tasks (VIS/EVS/VPS-thing)**: By calculating the shape-position descriptor variation $\Delta H = \|H^{t+n} - H^t\|_2 / \|H^t\|_2$, a threshold $\tau = 0.2$ is set to determine whether a distant-frame sample is positive or negative. $\Delta H < \tau$ indicates a positive sample, otherwise it is classified as a negative sample.
    - **Semantic-level Tasks (VSS/VPS-stuff)**: Maintaining a FIFO queue (containing $N_Q = 100$ queries per class) and sampling from the entire training set to enrich semantic descriptions.

### Loss & Training

- Follows standard Mask2Former training loss (including Hungarian matching, mask loss, classification loss).
- Temporal contrastive learning loss is used for cross-frame query embedding contrastive learning.
- Training iterations: 10K for OVIS/VSPW/VIPSeg/KITTI, 15K for YouTube-VOS18/YouTube-VIS21.
- Optimizer: AdamW, initial $lr=0.001$, step decay.
- Data augmentation: Flipping, random scaling, and cropping.
- MS COCO pseudo-video pre-training is utilized for the YouTube series datasets.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (R-50) | Prev. SOTA | Gain |
|--------|------|-------------|----------|------|
| VIPSeg (VPS) | VPQ | 44.0 | 39.2 (Tube-Link) | +4.8 |
| VIPSeg (VPS) | STQ | 44.9 | 39.5 (Tube-Link) | +5.4 |
| VSPW (VSS) | mIoU | 44.5 | 43.4 (Tube-Link) | +1.1 |
| Occluded-VIS | AP | 35.9 | 31.1 (TarVIS) | +4.8 |
| YouTube-VIS21 | AP | 49.6 | 48.3 (TarVIS) | +1.3 |
| YouTube-VOS18 (EVS) | G | 81.5 | 79.2 (TarVIS) | +2.3 |
| BURST (EVS) | H_all | 35.9 | 32.3 (XMem) | +3.6 |

When using a stronger Swin-L backbone, GvSeg further sets new SOTA results across all datasets.

### Ablation Study

| Configuration | VPQ | STQ | Description |
|------|-----|-----|------|
| Baseline | 36.0 | 37.3 | No improvements |
| + SPA query matching | 37.8 | 38.5 | Shape-position aware matching is effective |
| + Task-oriented init & asso. | 40.1 | 40.7 | Task-oriented initialization & association |
| + Task-oriented TCL | 41.2 | 42.0 | Task-oriented contrastive learning |
| GvSeg (All) | 44.0 | 44.9 | Components are complementary |

### Key Findings

- The shape-position descriptor brings a 1.8% VPQ improvement for cross-frame matching, verifying the importance of shape/position cues.
- Task-oriented query initialization is the most critical, contributing the largest performance gain.
- In temporal contrastive learning, the $\Delta H$ threshold $\tau=0.2$ and the queue length $N_Q=100$ are close to optimal, introducing almost no additional training overhead.
- The improvement is particularly pronounced on the heavily occluded OVIS dataset (+4.8% AP), demonstrating that the shape-position descriptor effectively handles occlusion.

## Highlights & Insights

- **Elegant Decoupling Concept**: Decoupling the segmentation target into three mutually orthogonal factors (appearance, shape, and position) allows both independent modeling and flexible combination, presenting an elegant design concept.
- **Architecture-Agnostic**: All innovative designs do not modify the network architecture, maintaining a unified framework to handle four distinct tasks, which simplifies practical deployment.
- **Deep Understanding of Task Characteristics**: Designs such as skipping SPA matching for VSS and combining thing/stuff strategies for VPS reflect an accurate grasp of the essential differences among tasks.
- **Dual Purpose of the Shape-Position Descriptor**: It is used not only for enhancing query matching but also for guiding sampling decisions in TCL.

## Limitations & Future Work

- The shape-position descriptor relies on accurate mask predictions, which may introduce noise when the first frame's prediction quality is poor.
- The number of bins in the polar coordinate histogram (36×12) is a manually designed hyperparameter, which may not be optimal for all scenarios.
- The semi-online mode requires frame-by-frame inference, which limits processing efficiency on long videos.
- Only Mask2Former was verified as the base segmenter; integration with other architectures (such as SAM) was not explored.
- The positive/negative sample threshold $\tau=0.2$ in TCL may require adjustment for different datasets.

## Related Work & Insights

- **Mask2Former**: The foundation segmenter of GvSeg, which proves the potential of query-based paradigms in multi-task scenarios.
- **Shape Context**: The inspiration source for the shape-position descriptor, elegantly transferring traditional shape description methods to deep learning queries.
- **Tube-Link**: The previously strongest universal video segmentation solution, which is significantly outperformed by GvSeg.
- **Insights**: Decoupled shape+position modeling can be extended to other visual tracking/matching tasks; task-adaptive strategies can inspire task-specialization designs in multi-task learning.

## Rating

- Novelty: ⭐⭐⭐⭐ Decoupling the segmentation target into three factors to achieve task adaptation is novel, though query matching and contrastive learning themselves are not completely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four tasks, seven datasets, and comprehensive ablations and visualizations, which is exceptionally thorough.
- Writing Quality: ⭐⭐⭐⭐ Logically clear, but features many formulas, requiring careful reading for some details.
- Value: ⭐⭐⭐⭐⭐ High practical value, as the unified framework comprehensively outperforms both specialized and universal solutions across all tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning](../../ACL2025/segmentation/instructpart_task-oriented_part_segmentation_with_instruction_reasoning.md)
- [\[ECCV 2024\] PartSTAD: 2D-to-3D Part Segmentation Task Adaptation](partstad_2d-to-3d_part_segmentation_task_adaptation.md)
- [\[ECCV 2024\] VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement](visage_video_instance_segmentation_with_appearance-guided_enhancement.md)
- [\[CVPR 2026\] Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation](../../CVPR2026/segmentation/task-oriented_data_synthesis_and_control-rectify_sampling_for_remote_sensing_sem.md)
- [\[ECCV 2024\] ActionVOS: Actions as Prompts for Video Object Segmentation](actionvos_actions_as_prompts_for_video_object_segmentation.md)

</div>

<!-- RELATED:END -->
