---
title: >-
  [Paper Note] Upper-Body Hierarchical Graph for Skeleton Based Emotion Recognition in Assistive Driving
description: >-
  [ECCV 2024][Human Understanding][Emotion Recognition] This paper proposes UbH-GCN for assistive driving scenarios. It utilizes upper-body skeleton sequences to construct a hierarchical graph structure (UbH-Graph) that dynamically models the relationship between joint movements and emotions. It also introduces a class-specific variation mechanism to balance the uneven data distribution, outperforming existing multimodal methods on the AIDE assistive driving dataset.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Emotion Recognition"
  - "Skeleton Sequences"
  - "Graph Convolutional Networks"
  - "Assistive Driving"
  - "Hierarchical Graph Structure"
date: 2026-05-08
content_hash: 8ec1273d33542ff4
---

# Upper-Body Hierarchical Graph for Skeleton Based Emotion Recognition in Assistive Driving

**Conference**: ECCV 2024  
**PDF**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03697.pdf)
**Code**: [https://github.com/jerry-wjh/UbH-GCN](https://github.com/jerry-wjh/UbH-GCN)  
**Area**: Human Understanding  
**Keywords**: Emotion Recognition, Skeleton Sequences, Graph Convolutional Networks, Assistive Driving, Hierarchical Graph Structure

## TL;DR

This paper proposes UbH-GCN for assistive driving scenarios. It utilizes upper-body skeleton sequences to construct a hierarchical graph structure (UbH-Graph) that dynamically models the relationship between joint movements and emotions. It also introduces a class-specific variation mechanism to balance the uneven data distribution, outperforming existing multimodal methods on the AIDE assistive driving dataset.

## Background & Motivation

**Background**: Emotion recognition is crucial in assistive driving—understanding the driver's emotional state (such as anger, fatigue, and anxiety) helps improve driving safety and human-computer interaction experiences. Current emotion recognition research mainly relies on facial expressions, speech signals, and physiological signals (such as EEG and heart rate), while methods based on body gestures are relatively scarce.

**Limitations of Prior Work**: (1) Facial and speech methods are limited in driving scenarios—drivers may wear masks/sunglasses, a vehicle cabin can be very noisy, and lighting conditions can change dramatically; (2) Most existing skeleton-based methods use full-body poses, but in driving scenarios, the lower body is obscured, making only the upper-body skeleton accessible; (3) Traditional GCN methods use pre-defined and fixed adjacency matrices, failing to adaptively model the dynamic relationships between joints under different emotional states; (4) Data in driving scenarios is highly unbalanced across different emotion categories, which limits the generalization capability of the model.

**Key Challenge**: The available body information in driving scenarios is limited (only the upper body) and the range of movement is restricted (sitting posture), resulting in subtle differences in skeleton movements across different emotions. Static graph structures and standard classification training strategies struggle to capture these delicate variances.

**Goal**: (1) How to effectively extract emotional features only from the upper-body skeleton? (2) How to dynamically model the relationship between joints and emotions? (3) How to handle the class imbalance problem in driving data?

**Key Insight**: The authors observe that joint groups at different levels in the upper body play different roles in emotional expression—for example, hand movements express anxiety, while head postures express fatigue. By modeling individual joints, joint groups, and global upper-body movements hierarchically using a hierarchical graph structure, multi-scale emotional cues can be better captured.

**Core Idea**: Extract multi-scale emotional movement features hierarchically through an upper-body hierarchical graph structure, and couple this with a class-specific variation enhancement mechanism to balance the training data, achieving precise emotion recognition in driving scenarios.

## Method

### Overall Architecture

UbH-GCN takes upper-body skeleton joint sequences as input (including 2D/3D coordinates of joints such as head, shoulders, arms, and hands over time). The skeletal data first goes through the UbH-Graph module to construct a multi-level hierarchical graph representation. Then, spatial-temporal features are extracted through multi-layer graph convolutions, which are modulated by the class-specific variation enhancement module. Finally, the classification head outputs the emotion category. Multi-modal late fusion is supported (joint coordinates, bone vectors, joint motion).

### Key Designs

1. **Upper-Body Hierarchical Graph (UbH-Graph)**:

    - Function: Dynamically models the hierarchical topological relations among upper-body joints and captures multi-scale emotional movement patterns.
    - Mechanism: Organizes upper-body joints into a three-layer hierarchical structure. The bottom layer consists of individual joint nodes (e.g., left wrist, right elbow); the middle layer represents functional joint groups (e.g., left arm group, head group, torso group); the top layer is the global upper-body representation. The layers are connected and aggregated through learnable connections. At each layer, the adjacency matrix is constructed dynamically from two parts: $A = A_{phys} + A_{learn}$, where $A_{phys}$ is the fixed topology based on physical connections, and $A_{learn}$ represents dynamic relationships learned through the attention mechanism.
    - Design Motivation: Different emotions involve different combinations of joints—tension associates with stiff hand and shoulder movements, while fatigue leads to a drooping head. The hierarchical graph structure can separately capture local (individual joint), mesoscopic (joint group), and global (overall body pose) emotional cues.

2. **Dynamic Graph Convolution**:

    - Function: Dynamically adjusts the graph structure in the temporal dimension to capture the temporal dynamics of emotional expression.
    - Mechanism: At each time step, attention weights are calculated based on the joint features of the current frame to update the learnable part of the adjacency matrix. Specifically, for any two joints $i, j$, their dynamic connection weight is computed as $a_{ij}^t = \text{softmax}(\phi(h_i^t)^T \psi(h_j^t) / \sqrt{d})$, where $\phi, \psi$ are linear transformations. Temporal convolution employs a multi-scale TCN to extract motion patterns across different time spans.
    - Design Motivation: Emotional expression is time-varying—for example, during the transition from calm to angry, the coordination relationships between joints change. Static adjacency matrices cannot capture such dynamic changes.

3. **Class-Specific Variation Enhancement**:

    - Function: Balances the feature distribution of different emotional categories and alleviates the class imbalance problem.
    - Mechanism: During training, the feature mean and covariance of each emotional category are maintained. For minority class samples, variations are increased by applying Gaussian perturbations in the feature space along class-specific directions: $h_{aug} = h + \epsilon \cdot \sigma_c$, where $\sigma_c$ is the feature standard deviation of class $c$, and $\epsilon \sim \mathcal{N}(0, 1)$. This expands the coverage of minority classes in the feature space, reducing the risk of overfitting.
    - Design Motivation: In driving data, samples for certain emotions (such as anger) are far fewer than other categories (such as calm). Traditional oversampling or class weighting strategies only operate at the data or loss level, whereas feature-level enhancement is more direct and does not introduce noisy samples.

### Loss & Training

The cross-entropy loss with class weight balancing is used: $L = -\sum_c w_c \cdot y_c \log(\hat{y}_c)$. A four-modality late fusion strategy is applied: four separate models for joint coordinates (joint), bone vectors (bone), joint motion (joint motion), and bone motion (bone motion) are trained. During inference, their softmax outputs are weighted and summed.

## Key Experimental Results

### Main Results

| Dataset | Method | Accuracy | F1 Score | Modality |
|--------|------|--------|---------|------|
| AIDE | UbH-GCN (4-way) | **62.7%** | **0.584** | Skeleton |
| AIDE | AIDE-baseline (multimodal) | 58.3% | 0.541 | Face+Skeleton+Speech |
| AIDE | CTR-GCN | 56.8% | 0.519 | Skeleton |
| AIDE | HD-GCN | 57.2% | 0.525 | Skeleton |
| AIDE | 2s-AGCN | 55.1% | 0.503 | Skeleton |
| Emliya | UbH-GCN (4-way) | **78.4%** | **0.762** | Skeleton |

### Ablation Study

| Configuration | Accuracy | Description |
|------|--------|------|
| Full UbH-GCN | **62.7%** | Full model |
| w/o Hierarchical Graph (Single-layer graph) | 59.1% | Hierarchical structure contributes ~3.6% |
| w/o Dynamic Adjacency Matrix (Fixed topology) | 58.5% | Dynamic graph contributes ~4.2% |
| w/o Class-Specific Variation | 60.3% | Variation enhancement contributes ~2.4% |
| Full-Body Skeleton (with lower-body estimation) | 61.2% | Upper-body is more suitable for driving scenarios |
| Single Modality (joint only) | 58.9% | Multimodal fusion contributes ~3.8% |

### Key Findings
- UbH-GCN outclasses the multimodal baseline that uses facial+skeleton+speech signals using only skeletal information, which indicates that the potential for emotional expression via skeleton in driving scenarios is severely underestimated.
- The dynamic adjacency matrix is the most crucial component, suggesting that the dynamic coordination relationships between joints in emotion recognition are more important than a fixed topology.
- It also performs well on the Emliya daily action dataset, demonstrating the cross-scenario generalization capability of the method.

## Highlights & Insights
- **Upper-body tailored design** precisely matches the actual constraints of driving scenarios. This paradigm of "tailoring input according to the application scenario" can be transferred to other restricted scenarios, such as wheelchair user recognition, and doctor posture analysis in operating rooms.
- **Class-specific variation enhancement** performs data augmentation in the feature space rather than the data space. This avoids the repeated sample problem generated by traditional oversampling and maintains the rationality of the feature distribution better than methods like SMOTE.
- The fact that the skeleton-only modality surpasses multimodal methods is inspiring. This shows that in specific scenarios, a meticulously designed single-modality method can be more effective than brute-force multimodal fusion.

## Limitations & Future Work
- The absolute accuracy of emotion recognition is still not highly satisfactory (62.7%), which highlights that recognizing emotions purely from the upper-body skeleton remains incredibly challenging.
- The scale of the AIDE dataset is relatively small and only includes specific populations and driving scenarios; its generalizability remains to be verified.
- Advanced architectures for temporal modeling (such as Transformers) were not introduced, potentially missing long-range temporal dependencies.
- Future work could consider fusing upper-body skeleton with vehicle behavior data (such as steering wheel angle, braking frequency) to provide richer emotional cues.

## Related Work & Insights
- **vs CTR-GCN**: CTR-GCN utilizes channel-level topology refinement but does not consider hierarchical structures. UbH-GCN introduces a hierarchical graph structure to model joint relationships from multiple scales, achieving better performance in driving scenarios.
- **vs HD-GCN**: HD-GCN uses hierarchical decomposition but relies on a pre-defined hierarchy and is designed for action recognition. UbH-GCN designs a hierarchy specifically for upper-body emotion recognition, and its adjacency matrix is dynamically learned.
- **vs ST-GCN**: ST-GCN is a pioneering work in GCN action recognition that uses a fixed adjacency matrix. UbH-GCN introduces dynamicity, hierarchy, and upper-body specialization on top of it.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of applying hierarchical graph structures and class-specific enhancement to assistive driving emotion recognition is novel.
- Experimental Thoroughness: ⭐⭐⭐ The datasets used are limited (only two), and the AIDE dataset is small in scale.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, and the motivation is thoroughly explained.
- Value: ⭐⭐⭐⭐ It holds practical value for emotion recognition in assistive driving, and the finding that skeleton-only outperforms multimodal signals is inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Modeling and Driving Human Body Soundfields through Acoustic Primitives](modeling_and_driving_human_body_soundfields_through_acoustic_primitives.md)
- [\[CVPR 2026\] CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation](../../CVPR2026/human_understanding/cigpose_causal_intervention_graph_neural_network_for_whole-body_pose_estimation.md)
- [\[ECCV 2024\] HUMOS: Human Motion Model Conditioned on Body Shape](humos_human_motion_model_conditioned_on_body_shape.md)
- [\[ECCV 2024\] Generalizable Facial Expression Recognition](generalizable_facial_expression_recognition.md)
- [\[AAAI 2026\] Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis](../../AAAI2026/human_understanding/facial-r1_aligning_reasoning_and_recognition_for_facial_emotion_analysis.md)

</div>

<!-- RELATED:END -->
