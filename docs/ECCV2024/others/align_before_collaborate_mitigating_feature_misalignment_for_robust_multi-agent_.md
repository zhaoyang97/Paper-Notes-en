---
title: >-
  [Paper Note] Align before Collaborate: Mitigating Feature Misalignment for Robust Multi-Agent Perception
description: >-
  [ECCV 2024][Collaborative Perception] Proposes NEAT, a model-agnostic and lightweight plug-in that explicitly addresses feature-level spatial misalignment caused by pose errors and communication delays in collaborative perception, using three modules: importance-guided query proposals, deformable feature alignment, and region cross-attention reinforcement. It delivers consistent gains for multiple baseline methods under noisy settings across four collaborative 3D detection da…
tags:
  - "ECCV 2024"
  - "Collaborative Perception"
  - "Feature Alignment"
  - "Multi-Agent"
  - "3D Object Detection"
  - "Pose Error Robustness"
date: 2026-05-08
content_hash: d789223d78e6b52c
---

# Align before Collaborate: Mitigating Feature Misalignment for Robust Multi-Agent Perception

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Others  
**Keywords**: Collaborative Perception, Feature Alignment, Multi-Agent, 3D Object Detection, Pose Error Robustness

## TL;DR

Proposes NEAT, a model-agnostic and lightweight plug-in that explicitly addresses feature-level spatial misalignment caused by pose errors and communication delays in collaborative perception, using three modules: importance-guided query proposals, deformable feature alignment, and region cross-attention reinforcement. It delivers consistent gains for multiple baseline methods under noisy settings across four collaborative 3D detection datasets.

## Background & Motivation

**Background**: Collaborative perception has attracted significant attention in recent years. Its core idea is to enhance the perception capability of autonomous vehicles through information sharing among multiple agents. Compared to single-vehicle perception, collaborative perception can expand the perception range and reduce occlusion blind spots, which is crucial for safe driving. Existing methods mainly focus on how to efficiently fuse features from different agents, adopting a pipeline from BEV (Bird's Eye View) features to 3D detection.

**Limitations of Prior Work**: The performance of existing collaborative perception systems is severely affected by inevitable "collaboration noise". Specifically, there are two main types of noise: (1) Pose error—inaccurate positioning of GPS/IMU leads to spatial coordinate transformation biases between agents; (2) Communication delay—due to network bandwidth limitations, there is a time lag in the information sent by different agents, causing the feature locations of moving objects to be inconsistent with the current timestamp. These noises manifest as spatial misalignment at the feature level, meaning that shared features from collaborators cannot be precisely aligned with ego-vehicle features in space.

**Key Challenge**: Collaborative perception needs to fuse features from different agents to improve performance, but pose errors and communication delays in reality inevitably lead to feature spatial misalignment. If misaligned features are fused directly, it not only fails to improve performance but may also introduce noise. Most existing methods assume ideal conditions and lack explicit handling of feature misalignment.

**Goal**: 1) How to explicitly correct feature-level spatial misalignment without changing the original collaborative perception architecture; 2) How to achieve effective feature alignment while keeping it lightweight; 3) How to make the alignment method universally applicable to collaborative perception methods with different structures.

**Key Insight**: The authors propose an "Align Before Collaborate" strategy. Before feature fusion, an independent alignment module is first used to correct the spatial misalignment of shared features from collaborators. This module is designed as a model-agnostic plug-in that can be added to various existing methods in a plug-and-play manner.

**Core Idea**: Use a lightweight, plug-and-play feature alignment plug-in (NEAT) to explicitly correct spatial misalignment caused by pose errors and communication delays before feature fusion in collaborative perception.

## Method

### Overall Architecture

The NEAT plug-in is positioned between "feature sharing" and "feature fusion" in the collaborative perception system. The inputs are the BEV feature maps from collaborators and the ego-vehicle's BEV feature map, and the output is the spatially aligned collaborator features. The entire pipeline consists of three steps: first, key regions likely containing foreground objects are filtered using importance-guided query proposals; second, explicit spatial correction is performed in these key regions via deformable feature alignment; finally, semantic diffusion and global enhancement of the aligned features are conducted through region cross-attention reinforcement.

### Key Designs

1. **Importance-Guided Query Proposal**:

    - **Function**: Predict and filter regions likely containing foreground targets from collaborator-shared BEV features to eliminate environmental redundancy.
    - **Mechanism**: Perform spatial-channel semantic analysis on the collaborator's BEV features to predict importance scores for each location. Specifically, a lightweight convolutional head is used to predict a foreground probability heatmap, and then the top-K high-scoring locations are selected as query proposals. Each query proposal encodes both spatial location information and channel-level semantic information for subsequent alignment operations. Through this filtering, subsequent alignment operations only need to be executed on key regions, significantly reducing computational overhead.
    - **Design Motivation**: Aligning the entire BEV feature map is computationally expensive and unnecessary—even if misalignment exists in background regions, its impact on detection results is minimal. Precisely aligning only the foreground regions is both efficient and targeted.

2. **Deformable Feature Alignment**:

    - **Function**: Explicitly align misalignments in collaborator-shared features through query-aware spatial association.
    - **Mechanism**: Based on the deformable attention mechanism. For each query proposal, a set of deformable sampling offsets is learned to perform adaptive sampling at corresponding locations on the collaborator's feature map. These offsets implicitly encode spatial displacements caused by pose errors and communication delays. Through multi-scale sampling and aggregation, the module can collect visual cues of different granularities and correct varying degrees of misalignment. The key is that the offsets are query-aware; since the degree of misalignment in different regions may vary, the module can adaptively learn different correction strategies for each region.
    - **Design Motivation**: Spatial misalignment caused by pose errors and communication delays may manifest differently in different regions of the feature map (e.g., angular errors have a greater impact on distant targets), thus requiring an adaptive, position-dependent alignment method. Deformable attention naturally supports this spatial adaptive characteristic.

3. **Region Cross-Attention Reinforcement**:

    - **Function**: Facilitate spatial diffusion of aligned feature representations to achieve global feature semantic enhancement.
    - **Mechanism**: Perform cross-attention operations between the aligned query features and the ego-vehicle's BEV features. The aligned query features serve as keys/values, and the ego-vehicle features serve as queries. Through cross-attention, the aligned collaborator information is propagated to relevant regions of the ego-vehicle feature map. This step not only fuses complementary collaborator information but also eliminates potential local inconsistencies remaining from the alignment process via the global attention mechanism.
    - **Design Motivation**: The previous alignment step is performed in local query regions, and the aligned features need to be further fused globally with the ego-vehicle features. Cross-attention can establish associations between aligned features and ego-vehicle features globally to fully utilize the information.

### Loss & Training

As a plug-in, NEAT is trained end-to-end jointly with the original collaborative perception system. The total loss function includes the original detection loss (classification loss + regression loss) and the foreground prediction loss of the importance-guided query proposal module. Foreground prediction uses binary cross-entropy loss, where positive samples are BEV grid locations covered by ground-truth bounding boxes. During training, pose errors are simulated by adding random Gaussian noise to pose parameters, and communication delays are simulated by randomly rolling back to features of previous frames.

## Key Experimental Results

### Main Results

| Dataset | Metric | NEAT Gain | Noise Setting | Description |
|--------|------|------|----------|------|
| OPV2V | AP@0.7 | +2~5% | Pose noise $\sigma=0.4$m | Consistent improvements across multiple baseline methods |
| V2XSet | AP@0.7 | +3~6% | Pose noise + 200ms delay | The larger the noise, the more pronounced the gain |
| DAIR-V2X | AP@0.7 | +1~3% | Real-world pose error | Still effective under real-world noise scenarios |
| V2V4Real | AP@0.7 | +2~4% | Real-world scenario | Real-world data validates generalization |

### Ablation Study

| Config | Key Metric | Description |
|------|---------|------|
| Full NEAT | Best AP | Complete module |
| w/o Query Proposal | AP drops 1~2% | No foreground region filtering, full-map alignment |
| w/o Deformable Alignment | AP drops 3~4% | No spatial correction, direct fusion |
| w/o Cross-Attention Reinforcement | AP drops 1~2% | No global enhancement after alignment |
| Different $K$ queries | Smooth changes | Balanced performance and computation when $K=256$ |

### Key Findings
- Deformable feature alignment is the most critical component, with its removal leading to the largest performance drop, verifying the importance of explicit spatial correction against noise.
- NEAT brings consistent gains across baseline methods of different structures, proving the effectiveness of the model-agnostic design.
- The larger the noise (larger pose errors or longer delays), the more significant the gain brought by NEAT, indicating that alignment is particularly crucial in high-noise environments.
- The extra parameter size and computational cost introduced by NEAT are very small (<5% increase), making it suitable for practical deployment.

## Highlights & Insights

- **Plug-and-play design philosophy**: NEAT does not change the architecture of the original collaborative perception system, but only adds a lightweight plug-in before feature fusion. This design allows it to be seamlessly integrated into any existing method, greatly improving practicality.
- **Filter-then-align efficiency strategy**: By filtering foregrounds to restrict alignment operations to key regions, it not only improves the accuracy of alignment (avoiding background interference) but also significantly reduces the computational load. This idea can be transferred to other tasks requiring spatial alignment.
- **Noise-aware training strategy**: Purposely injecting pose noise and delays during training to enhance robustness is a simple yet effective data augmentation method.

## Limitations & Future Work

- Heavy reliance on pose estimation values as initial alignment references—if pose estimation is completely unavailable (rather than noisy), the method cannot work.
- Currently only addresses feature-level alignment; semantic-level conflicts (e.g., differing occlusion relationships observed by different agents) are not explicitly modeled.
- Foreground prediction of query proposals relies on bounding boxes to generate pseudo-labels, which might be affected in scenarios with sparse or inaccurate annotations.
- Further research can be conducted on utilizing temporal information—using the temporal consistency of multi-frame features to assist alignment.
- For large-scale scenarios (e.g., city-level multi-vehicle collaboration), the limitations on communication bandwidth may require more aggressive feature compression strategies.

## Related Work & Insights

- **vs V2VNet**: V2VNet uses simple spatial transformation to align features, assuming accurate poses; NEAT achieves adaptive spatial correction through deformable attention, which is more robust to noise.
- **vs DiscoNet**: DiscoNet focuses on knowledge distillation for communication efficiency; NEAT orthogonally solves the feature alignment problem, and the two can be combined.
- **vs Where2comm**: Where2comm optimizes the decision of "where to communicate"; NEAT solves "how to align communicated features," representing a complementary relationship.
- The "align-before-fuse" paradigm of this work provides insights for any task involving multi-source feature fusion, such as multi-modal fusion, multi-view reconstruction, etc.

## Rating
- Novelty: ⭐⭐⭐ The idea is intuitive and reasonable (align before fusion), the technical components (deformable attention, etc.) are not entirely brand new, but the combination is clever.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively validated across four datasets, multiple baseline methods, and various noise settings.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, the motivation is fully elaborated, and the experimental comparison is detailed.
- Value: ⭐⭐⭐⭐ The plug-and-play design has direct practical value for the collaborative perception field, addressing key pain points in real deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception](../../CVPR2025/others/traf-align_trajectory-aware_feature_alignment_for_asynchronous_multi-agent_perce.md)
- [\[ECCV 2024\] HPFF: Hierarchical Locally Supervised Learning with Patch Feature Fusion](hpff_hierarchical_locally_supervised_learning_with_patch_feature_fusion.md)
- [\[ECCV 2024\] Mahalanobis Distance-Based Multi-View Optimal Transport for Multi-View Crowd Localization](mahalanobis_distance-based_multi-view_optimal_transport_for_multi-view_crowd_loc.md)
- [\[ECCV 2024\] MemBN: Robust Test-Time Adaptation via Batch Norm with Statistics Memory](membn_robust_test-time_adaptation_via_batch_norm_with_statistics_memory.md)
- [\[ECCV 2024\] Superpixel-Informed Implicit Neural Representation for Multi-Dimensional Data](superpixel-informed_implicit_neural_representation_for_multi-dimensional_data.md)

</div>

<!-- RELATED:END -->
