---
title: >-
  [Paper Note] DG-PIC: Domain Generalized Point-In-Context Learning for Point Cloud Understanding
description: >-
  [ECCV 2024][3D Vision][Point Cloud Understanding] DG-PIC is proposed, representing the first point cloud understanding framework that simultaneously addresses multi-domain and multi-task learning in a unified model. Through dual-level source prototype estimation and a test-time feature shifting mechanism, it enhances generalization capability to unseen domains without requiring model updates.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Point Cloud Understanding"
  - "Domain Generalization"
  - "In-Context Learning"
  - "Multi-Task Learning"
  - "Test-Time Adaptation"
date: 2026-05-08
content_hash: e2237922c0e156a7
---

# DG-PIC: Domain Generalized Point-In-Context Learning for Point Cloud Understanding

**Conference**: ECCV 2024  
**arXiv**: [2407.08801](https://arxiv.org/abs/2407.08801)  
**Code**: [https://github.com/Jinec98/DG-PIC](https://github.com/Jinec98/DG-PIC)  
**Area**: 3D Vision  
**Keywords**: Point Cloud Understanding, Domain Generalization, In-Context Learning, Multi-Task Learning, Test-Time Adaptation

## TL;DR

DG-PIC is proposed, representing the first point cloud understanding framework that simultaneously addresses multi-domain and multi-task learning in a unified model. Through dual-level source prototype estimation and a test-time feature shifting mechanism, it enhances generalization capability to unseen domains without requiring model updates.

## Background & Motivation

**Background**: Point cloud understanding is widely used in fields such as autonomous driving and robotics, but models are typically trained and tested on a single dataset. Performance degrades significantly when facing new data with different distributions (e.g., from synthetic ModelNet40 to real-scan ScanObjectNN).

**Limitations of Prior Work**:
   - **Domain Generalization (DG) methods** are generally designed for a single task (e.g., classification), lack multi-task capabilities, and ignore the valuable utility of the test data itself.
   - **In-Context Learning (ICL) methods** (such as PIC) can handle multiple tasks but are restricted to a single dataset, exhibiting poor cross-domain generalization.
   - Neither type of method can simultaneously solve the "multi-domain" and "multi-task" problems.

**Key Challenge**: A unified model needs to reconcile task generalization (multi-task) and domain generalization (multi-domain), whereas existing methods can only address one of them.

**Goal**: To handle point cloud understanding across multiple domains and tasks in a unified model, and generalize to unseen domains during testing without updating model parameters.

**Key Insight**: Combining the multi-task ICL of PIC with test-time domain generalization—learning cross-domain generalizable information via PIC during pre-training, and shifting target domain features towards the source domain during testing.

**Core Idea**: Dual-level source prototypes + dual-level test-time feature shifting, which aligns unseen test domain data to known source domains without requiring model updates.

## Method

### Overall Architecture

DG-PIC consists of two stages: (1) Pre-training stage—training a PIC model on multiple source domains based on the Masked Point Modeling (MPM) framework, using a cross-domain prompt pairing strategy; (2) Testing stage—freezing the model, determining the distance between test samples and each source domain via dual-level source prototype estimation, and then aligning test data to the source domains through dual-level feature shifting before selecting the most similar samples from the nearest source domain as prompts.

### Key Designs

1. **Multi-domain Prompt Pairing**:

    - **Function**: Randomly selects prompt samples from different source domains during pre-training to enhance cross-domain associations.
    - **Mechanism**: Assuming the query is from domain $D_s^i$ and the prompt is from domain $D_s^j (j \neq i)$, the predicted masked patch is:
    $P \sim (D_s^i, D_s^j) = Trans([F_\theta(I_i) \oplus F_\theta(T_i^k) \oplus F_\theta(I_j) \oplus F_\theta(T_j^k)], Mask)$
      The training loss uses Chamfer Distance: $\text{CD}(P,G) = \frac{1}{|P|}\sum_{x \in P}\min_{y \in G}\|x-y\|^2 + \frac{1}{|G|}\sum_{y \in G}\min_{x \in P}\|y-x\|^2$
    - **Design Motivation**: Cross-domain pairing forces the model to learn domain-invariant feature representations.

2. **Dual-level Source Prototype Estimation**:

    - **Function**: Computes global and local prototypes for each source domain to act as anchors for test-time feature alignment.
    - **Mechanism**:
        - **Local prototype** $Z_{local}^{i,m}$: Averages the patch-level features of all samples in domain $D_s^i$: $Z_{local}^{i,m} = \frac{1}{N_{D_s^i}} \sum_{n=1}^{N_{D_s^i}} F_\theta(P_m)$
        - **Global prototype** $Z_{global}^i$: Averages max-pooled patch features: $Z_{global}^i = \frac{1}{N_{D_s^i}} \sum_{n=1}^{N_{D_s^i}} max(F_\theta(P_m))$
        - Computes the Euclidean distance of test samples to each source domain prototype: $\mathcal{E}_{global}^i = \|F_{global} - Z_{global}^i\|$, $\mathcal{E}_{local}^{i,m} = \|F_{local}^m - Z_{local}^{i,m}\|$
    - **Design Motivation**: Global features capture shape context, whereas local features capture geometric structural details. Dual-level representation is required to comprehensively represent the source domains.

3. **Dual-level Test-time Feature Shifting**:

    - **Function**: Shifts target domain features towards the source domain direction during test time without updating model parameters.
    - **Mechanism**:
        - **Macro semantic coefficient $\alpha$**: Derived from the global distance, controlling the contribution level of each source domain to the feature shift: $\alpha = softmax(\mathcal{E}_{global})$
        - **Micro positional coefficient $\beta^i$**: Derived from the local distance, considering the alignment of patch positions: $\beta^i = softmax(\mathcal{E}_{local}^i)$
        - Final feature shifting formula:
    $F'_{local} = \frac{1}{R}\sum_{i=1}^{R}\alpha_i\left(\frac{1}{M}\sum_{m=1}^{M}\beta^{i,m}F_{local}^m\right) + \frac{1}{R}\sum_{i=1}^{R}(1-\alpha_i)\left(\frac{1}{M}\sum_{m=1}^{M}(1-\beta^{i,m})Z_{local}^{i,m}\right)$
    - **Design Motivation**: $\alpha$ utilizes cross-domain semantic similarity to regulate the overall shifting intensity, while $\beta$ utilizes geometric similarity of co-located patches for fine-grained alignment. Even across domains, patches at the same relative position should share similar geometric structures (e.g., table edges versus a flat surface).

4. **Test-time Prompt Selection**:

    - **Function**: Selects the most similar sample from the nearest source domain as the prompt.
    - **Mechanism**: Determines the nearest source domain by combining global and local distances: $\mathcal{E}^i = \lambda \cdot \mathcal{E}_{global}^i + (1-\lambda) \cdot \frac{1}{M}\sum_{m=1}^{M}\mathcal{E}_{local}^{i,m}$ ($\lambda=0.5$), and then finds the sample with the minimum feature distance in that domain to serve as the prompt.

### Loss & Training

- Pre-training uses Chamfer Distance loss
- AdamW optimizer, lr=0.001, cosine learning rate schedule
- Trained for 300 epochs, batch size 128
- Each point cloud is sampled with 1024 points, divided into 64 patches (32 points each), with a mask ratio of 0.7
- **Absolutely no model parameter updates during test time**

## Key Experimental Results

### Main Results (ScanObjectNN as target domain, CD×10⁻³ ↓)

| Method | Setting | Reconstruction | Denoising | Registration |
|------|------|------|------|------|
| **DG-PIC (Ours)** | Test-time DG | **4.1** | **15.2** | **5.8** |
| PIC | Supervised | 72.9 | 80.0 | 12.7 |
| Point-MAE (task-specific) | Supervised | 30.4 | 36.0 | 31.2 |
| PCT (multi-task) | Supervised | 31.5 | 36.5 | 34.9 |
| PointCutMix (task-specific) | Train-time DG | 44.8 | 43.5 | 41.3 |

### Ablation Study (CD×10⁻³ ↓)

| Model | Prototype Estimation | Feature Shifting | Anchor Domain | Reconstruction | Denoising | Registration |
|------|---------|---------|--------|------|------|------|
| Model A| Random | Mean | Single Domain | 8.4 | 40.5 | 6.7 |
| Model B | Global Only | Mean | Single Domain | 7.2 | 38.3 | 6.4 |
| Model C | Local Only | Mean | Single Domain | 7.3 | 36.7 | 6.7 |
| Model D | Global+Local | Mean | Single Domain | 6.8 | 35.1 | 6.2 |
| Model E | Global+Local | Mean | All Domains | 6.3 | 32.4 | 6.5 |
| Model F | Global+Local | Macro Only | All Domains | 5.2 | 22.7 | 6.0 |
| Model G | Global+Local | Micro Only | All Domains | 4.9 | 25.6 | 6.2 |
| **Ours** | **Global+Local** | **Macro+Micro** | **All Domains** | **4.1** | **15.2** | **5.8** |

### Key Findings

- DG-PIC significantly outperforms all baseline methods across all three tasks; on the reconstruction task, its CD is only 5.6% of PIC's (4.1 vs 72.9).
- Traditional methods (PointNet, DGCNN, etc.) exhibit poor cross-domain generalization, with CD values generally falling within the 30-45 range.
- Although DG methods (Pointmixup, PointCutMix) introduce domain generalization, the variance of results across tasks is small, indicating they focus solely on domain differences while ignoring task differences.
- PIC can handle multiple tasks but fails in cross-domain scenarios (CD 72.9-80.0), proving that ICL alone is insufficient to bridge domain gaps.
- Each component of the dual-level design (global+local prototypes, macro+micro shifting) independently contributes to performance, with the denoising task benefiting the most.

## Highlights & Insights

- **Pioneering Setting**: This paper introduces the novel setting of multi-domain multi-task point cloud understanding and establishes a benchmark containing 4 datasets (2 synthetic + 2 real), 7 object categories, 3 tasks, and 30,954 samples.
- **Test-Time Generalization Without Model Updates**: Domain adaptation is achieved via feature-space shifting rather than fine-tuning, maintaining a manageable computational overhead.
- **Elegant Design Intuition for Micro Positional Coefficients**: Exploits the effective prior that patches at the same relative position on an object should exhibit similar geometric structures across domains.
- **A Unified Model for Three Tasks**: Reconstruction, denoising, and registration share a single network, with tasks toggled via prompts.

## Limitations & Future Work

- The benchmark only contains 7 shared classes, with limited scale and diversity.
- The framework only considers regression-like tasks for coordinates (reconstruction/denoising/registration) and does not cover discriminative tasks like classification or segmentation.
- Source domain prototypes are calculated as a simple average of all samples, potentially losing intra-class multi-modal distribution information.
- The formulation for feature shifting is somewhat heuristic and lacks rigorous theoretical analysis.
- Ablation studies and thorough experiments were primarily conducted with ScanObjectNN as the target domain, leaving other target domain configurations insufficiently explored.

## Related Work & Insights

- **vs PIC**: PIC is a single-domain multi-task ICL model. DG-PIC introduces test-time domain generalization on top of it, achieving massive performance improvements (CD: 72.9 $\rightarrow$ 4.1 on the reconstruction task).
- **vs Point-BERT / Point-MAE**: Although these self-supervised features learn rich representations, they do not account for cross-domain generalization, yielding poor direct transfer performance.
- **vs Pointmixup / PointCutMix**: Train-time DG methods focus on "domain invariance" via mixed data augmentation but ignore "task invariance", leading to mediocre results.
- **vs DGLSS / SemanticSTF**: Pioneering works in point cloud DG, but they are task-specific and do not support a unified multi-task model.

## Rating

- Novelty: ⭐⭐⭐⭐ Proposes the multi-domain multi-task point cloud understanding setting for the first time; the dual-level design is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation studies and various baseline comparisons, though the choice of target domain configurations is somewhat simple.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-justified motivation, and intuitive diagrams.
- Value: ⭐⭐⭐⭐ The new setting and benchmark make a strong contribution to advancing point cloud generalization research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding](../../CVPR2026/3d_vision/mamba_learns_in_context_structure-aware_domain_generalization_for_multi-task_poi.md)
- [\[ECCV 2024\] GPSFormer: A Global Perception and Local Structure Fitting-Based Transformer for Point Cloud Understanding](gpsformer_a_global_perception_and_local_structure_fitting-based_transformer_for_.md)
- [\[CVPR 2026\] Deformation-based In-Context Learning for Point Cloud Understanding](../../CVPR2026/3d_vision/deformation-based_in-context_learning_for_point_cloud_understanding.md)
- [\[ECCV 2024\] Heterogeneous Graph Learning for Scene Graph Prediction in 3D Point Clouds](heterogeneous_graph_learning_for_scene_graph_prediction_in_3d_point_clouds.md)
- [\[ECCV 2024\] T-MAE: Temporal Masked Autoencoders for Point Cloud Representation Learning](t-mae_temporal_masked_autoencoders_for_point_cloud_representation_learning.md)

</div>

<!-- RELATED:END -->
