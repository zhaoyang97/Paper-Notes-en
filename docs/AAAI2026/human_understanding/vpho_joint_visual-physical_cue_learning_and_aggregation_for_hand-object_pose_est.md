---
title: >-
  [Paper Note] VPHO: Joint Visual-Physical Cue Learning and Aggregation for Hand-Object Pose Estimation
description: >-
  [AAAI 2026][Human Understanding][Hand-object pose estimation] This paper proposes VPHO, a framework for hand-object pose estimation that jointly leverages visual and physical cues. It introduces a force prediction module to learn 3D physical cues and designs a two-stage candidate pose aggregation strategy (visual-guided + physics-guided) to achieve physical plausibility while preserving visual consistency. VPHO attains state-of-the-art performance in both pose accuracy and ph…
tags:
  - "AAAI 2026"
  - "Human Understanding"
  - "Hand-object pose estimation"
  - "visual-physical cues"
  - "diffusion model"
  - "force prediction"
  - "candidate pose aggregation"
date: 2026-05-08
content_hash: 53068459015ce93c
---

# VPHO: Joint Visual-Physical Cue Learning and Aggregation for Hand-Object Pose Estimation

**Conference**: AAAI 2026
**arXiv**: [2511.12030](https://arxiv.org/abs/2511.12030)  
**Code**: [github.com/zhoujun-7/VPHO](https://github.com/zhoujun-7/VPHO)  
**Area**: Human Understanding
**Keywords**: Hand-object pose estimation, visual-physical cues, diffusion model, force prediction, candidate pose aggregation

## TL;DR

This paper proposes VPHO, a framework for hand-object pose estimation that jointly leverages visual and physical cues. It introduces a force prediction module to learn 3D physical cues and designs a two-stage candidate pose aggregation strategy (visual-guided + physics-guided) to achieve physical plausibility while preserving visual consistency. VPHO attains state-of-the-art performance in both pose accuracy and physical plausibility on the DexYCB and HO3D benchmarks simultaneously.

## Background & Motivation

### Problem Definition

Estimating the 3D poses of hands and objects from a single RGB image is a fundamental yet challenging problem with broad applications in augmented reality and human-computer interaction. The core difficulty lies in simultaneously satisfying two criteria:

**Visual consistency**: The 2D projection of the 3D pose should be consistent with image observations.

**Physical plausibility**: Hand-object interactions should conform to physical constraints (no interpenetration, contact maintained, force equilibrium).

### Root Cause

Existing methods cannot simultaneously guarantee visual consistency and physical plausibility:

| Method Type | Representative Work | Visual Consistency | Physical Plausibility | Issue |
|-------------|--------------------|--------------------|----------------------|-------|
| Pure visual | HFL | ✓ Good | ✗ Poor | Ignores physical constraints; produces artifacts such as interpenetration and lack of contact |
| Post-optimization | CPF, ContactOpt | ✗ Unstable | ✓ Good | Highly dependent on initial pose quality; diverges when initialization is inaccurate |
| End-to-end physical | DeepSimHO | ✗ Degraded | ✓ Good | Visual fidelity degrades when physical constraints are introduced |

**Key observation** (Figure 1): HFL appears plausible from the original camera viewpoint, but rendering from a side view reveals an inappropriate grasp; DeepSimHO improves physical plausibility but the object pose deviates from image observations.

### Core Problem

**Difficulty in predicting 3D interaction forces**: High dimensionality, complex contact dynamics, and lack of ground-truth force annotations.

**Difficulty in pose aggregation**: Hands have high degrees of freedom (articulated joints), and hand-object interactions are mutually dependent.

## Method

### Overall Architecture

VPHO consists of four modules:
1. **Feature extraction**: Extracts visual features and heatmaps for the hand and object.
2. **Force prediction**: Predicts hand-object interaction forces (physical cues).
3. **Candidate generation**: Generates multiple candidate poses using a diffusion model.
4. **Pose aggregation**: Two-stage aggregation (visual → physical) to produce the final pose.

### Key Designs

#### 1. **Force Prediction Module**: Learning 3D Physical Cues

**Core Idea**: Contact forces are represented using a friction cone model, and a local-to-global coordinate transformation is achieved via the MANO hand model.

**Local force modeling**:
- 32 hand-surface anchor points $\{O_k^a\}_{k=1}^{32}$ are used to represent contact.
- According to Coulomb's friction law, contact forces must lie within the friction cone.
- The friction cone is parameterized by $N_v$ basis vectors $v_j = (\mu\sin(\frac{2\pi j}{N_v}), \mu\cos(\frac{2\pi j}{N_v}), 1)$.
- The local force is a weighted sum of basis vectors: $F_k' = s_k \sum_{j=1}^{N_v} w_{k,j} v_j$.

**Global force transformation**:
Leveraging the linear blend skinning process of the MANO hand model, local forces are transformed to global forces via rotation matrix $\mathbf{R}_k^{L2G}$: $F_k = \mathbf{R}_k^{L2G} F_k'$. The rotation matrix is computed from the three vertices of the triangle face to which each anchor point is attached.

**Physical constraints** (static equilibrium assumption):

| Constraint | Formula | Meaning |
|------------|---------|---------|
| Force balance | $\mathcal{L}_{force} = \|\sum_{k=1}^{32}F_k + \mathbf{G}\|_2^2$ | Sum of all forces equals zero |
| Torque balance | $\mathcal{L}_{torque} = \|\sum_{k=1}^{32}F_k \times r_k\|_2^2$ | Sum of all torques equals zero |
| Contact-force relationship | $\mathcal{L}_{contact} = \sum_{k=1}^{32}\|F_k\|_2 \cdot |d_k|$ | Larger force corresponds to smaller distance |

**Semi-supervised training**: In the absence of ground-truth force annotations, the module is trained using physical constraints and pseudo force labels. Pseudo labels are generated via two-stage optimization: first optimizing force directions so that the resultant force cancels gravity, then jointly optimizing force magnitudes and torques.

**Design Motivation**: The friction cone model ensures physical correctness (forces lie within the cone) while remaining differentiable (suitable for end-to-end training). The 32 anchor points strike a balance between accuracy and complexity.

#### 2. **Visual-based Aggregation**: Hierarchical Aggregation Along the Kinematic Chain

**Core Idea**: Heatmaps serve as visual guidance to progressively aggregate candidate poses along the hand's kinematic chain, from lower to higher levels.

Hand joints are divided into 4 levels (Figure 5a), and aggregation proceeds iteratively from level 1 to level 4. After each level's aggregation, the refined joint parameters overwrite the corresponding joints across all candidates, reducing error propagation to higher levels.

For level $l$, joint $j$, the visual score of each candidate is:

$$s_i^h = \sum_{c \in Children_j} H_c^h(\text{Proj2D}^h(c, \theta_i, \beta_{reg}))$$

This queries the heatmap at the projected 2D location of each candidate joint. The top-K candidates are selected and averaged with weighting:

$$\tilde{\theta}[j,:] = \frac{\sum_{i \in K^h} s_i^h \theta_i[j,:]}{\sum_{i \in K^h} s_i^h}$$

**Design Motivation**: Hierarchical aggregation addresses error accumulation along the kinematic chain in articulated systems—lower-level joints are refined first, providing a reliable basis for optimizing higher-level joints.

#### 3. **Physics-based Aggregation**: Re-ranking Using Force and Torque Constraints

**Core Idea**: Building on visual aggregation, physical constraints are applied to further select physically plausible candidates.

- Physical score for the hand: $s_{phy}^h = -\mathcal{L}_{force} \cdot \mathcal{L}_{contact}$ (force balance × contact quality).
- Physical score for the object: $s_{phy}^o = -\mathcal{L}_{torque} \cdot \mathcal{L}_{contact}$ (torque balance × contact quality).

For the hand, only the highest-level joints $L_4$ are refined (as high-level joints most strongly affect physical plausibility). For the object, top-K translation and top-K rotation candidates are combined, re-ranked by physical score, and the top-K are averaged.

**Design Motivation**: Applying physical constraints only at high-level joints improves physical plausibility without disrupting the visual consistency established for low-level joints during visual aggregation.

### Loss & Training

Physical network loss:

$$\mathcal{L}_{phy} = \lambda_F \mathcal{L}_{mse}(F', \tilde{F}') + \lambda_c \mathcal{L}_{mse}(c, \bar{c}) + \lambda_{force} \mathcal{L}_{force} + \lambda_{torque} \mathcal{L}_{torque}$$

where $\tilde{F}'$ denotes the precomputed pseudo force labels. Candidate poses are generated by a score-based diffusion model (100 candidates; hand top-K = 30/5, object top-K = 10/5). Optional pretraining on DexYCB for 5 epochs is supported.

## Key Experimental Results

### Main Results

**DexYCB Full pose accuracy** (mm):

| Method | MJE ↓ | PA-MJE ↓ | MCE ↓ | OCE ↓ | ADDS ↓ |
|--------|-------|----------|-------|-------|--------|
| HFL | 12.6 | 5.47 | 48.0 | 42.7 | 33.8 |
| HOISDF | 10.1 | 5.13 | 35.8 | 27.6 | 18.6 |
| **VPHO** | **10.0** | **5.08** | **26.2** | **23.7** | **13.5** |

Object metrics show substantial improvement: MCE reduced by 26.8%, OCE by 14.1%, and ADDS by 27.4% relative to the second-best method HOISDF.

**HO3Dv2 Full pose accuracy** (mm):

| Method | PA-MJE ↓ | MJE ↓ | OCE ↓ | ADDS ↓ |
|--------|----------|-------|-------|--------|
| HandBooster | 8.5 | 21.1 | - | - |
| HOISDF | 9.2 | 19.0 | 35.5 | 14.4 |
| VPHO w/o pretrain | 8.9 | 21.1 | 29.3 | 15.2 |
| **VPHO** | **8.5** | **19.9** | **27.1** | **14.3** |

**DexYCB Phy physical plausibility** (mm):

| Method | MJE ↓ | SMCE ↓ | CP(%) ↑ | PD ↓ | SD ↓ |
|--------|-------|--------|---------|------|------|
| ArtiBoost | 10.7 | 16.0 | 94.23 | 15.0 | 27.8 |
| DeepSimHO | 11.2 | 17.3 | 95.90 | 14.8 | 24.2 |
| **VPHO** | **8.5** | **15.1** | **98.85** | **13.4** | **17.3** |

Contact percentage increases from 95.9% to 98.85%, penetration depth decreases from 14.8 to 13.4, and simulation displacement decreases from 24.2 to 17.3—while hand MJE also drops substantially from 11.2 to 8.5 (DeepSimHO's hand accuracy is notably worse).

**HO3Dv2 Phy physical plausibility** (cm):

| Method | SMCE ↓ | CP(%) ↑ | PD ↓ | SD ↓ |
|--------|--------|---------|------|------|
| CPF | 5.74 | 96.47 | 1.65 | 3.16 |
| DeepSimHO | 5.28 | 96.64 | 1.17 | 2.42 |
| **VPHO** | **3.12** | **98.80** | **0.96** | **2.21** |

### Ablation Study

| Force Pred | Visual Agg | Physics Agg | MJE ↓ | PA-MJE ↓ | OCE ↓ | ADDS ↓ | CP ↑ | PD ↓ | SD ↓ |
|-----------|-----------|------------|-------|----------|-------|--------|------|------|------|
| ✗ | ✗ | ✗ | 12.54 | 5.45 | 35.30 | 20.14 | 95.42 | 14.4 | 30.4 |
| ✓ | ✗ | ✗ | 12.16 | 5.33 | 29.81 | 16.66 | 96.51 | 14.1 | 25.9 |
| ✓ | ✓ | ✗ | 10.05 | 5.09 | 26.12 | 15.21 | 97.95 | 13.9 | 20.0 |
| **✓** | **✓** | **✓** | **10.01** | **5.08** | **23.72** | **13.47** | **98.85** | **13.4** | **17.3** |

### Key Findings

1. **Force prediction module contributes significantly**: Adding force prediction alone (without aggregation) reduces OCE from 35.30 to 29.81 (−15.6%) and ADDS from 20.14 to 16.66 (−17.3%), indicating that learning physical cues itself benefits pose estimation.
2. **Visual aggregation is central to accuracy gains**: Its addition reduces MJE from 12.16 to 10.05 (−17.4%), demonstrating that hierarchical aggregation effectively mitigates error accumulation along the kinematic chain.
3. **Physics aggregation further improves plausibility**: CP improves from 97.95% to 98.85%, and SD decreases from 20.0 to 17.3 (−13.5%), validating the effectiveness of physics-constrained candidate re-ranking.
4. **First simultaneous SOTA in accuracy and physical plausibility**: Prior methods (e.g., DeepSimHO) improved physical plausibility at the cost of pose accuracy; VPHO breaks this trade-off.

## Highlights & Insights

1. **Elegant two-stage aggregation**: Visual cues are used first to ensure visual consistency, followed by physical cues to ensure physical plausibility; the two filtering stages do not conflict.
2. **Ingenious pseudo force label generation**: In the absence of ground-truth force annotations, pseudo labels are generated via physics-constrained optimization, enabling semi-supervised force prediction.
3. **Joint modeling of friction cone and MANO**: The classical friction cone model is seamlessly integrated with the skeletal structure of the MANO hand model, yielding rigorous physical modeling.
4. **Thorough analysis of candidate count**: 100 candidates provide a favorable accuracy-efficiency trade-off (50 candidates perform slightly worse; 200 yield marginal gains at twice the cost).

## Limitations & Future Work

1. **Static equilibrium assumption**: The model assumes a static equilibrium between hand and object, precluding handling of accelerations during dynamic grasping.
2. **Absence of temporal information**: Only single-frame images are used; temporal cues from video are not exploited.
3. **Candidate generation overhead**: Inference time for generating 100 candidate poses via the diffusion model may be substantial.
4. **Object shape dependency**: Known object CAD models and camera intrinsics are required as inputs.
5. **Gravity direction approximation**: At inference time, the camera's y-axis is used to approximate the gravity direction, which may be inaccurate under non-standard camera orientations.

## Related Work & Insights

- **HFL**: A representative pure-visual method with good pose accuracy but poor physical plausibility.
- **DeepSimHO**: A representative physics-engine-based method with good physical plausibility but at the cost of visual accuracy.
- **MANO hand model**: The standard parametric representation for hand pose.
- **Score-based diffusion model**: Used for generating diverse candidate poses.
- **CPF**: The inspiration for friction cone modeling, using 32 anchor points to approximate contact.
- **Insight**: Aggregating physical and visual cues at inference time (rather than hard-fusing them during training) is an effective strategy for balancing both objectives.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Joint visual-physical learning combined with two-stage candidate aggregation is both novel and effective)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive evaluation of accuracy and physical plausibility on two benchmarks, with complete ablation studies)
- Writing Quality: ⭐⭐⭐⭐ (Clear method description and rigorous mathematical derivations)
- Value: ⭐⭐⭐⭐⭐ (First to achieve simultaneous SOTA in both accuracy and physical plausibility, breaking a long-standing trade-off)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation](../../CVPR2025/human_understanding/unihope_a_unified_approach_for_hand-only_and_hand-object_pose_estimation.md)
- [\[AAAI 2026\] New Synthetic Goldmine: Hand Joint Angle-Driven EMG Data Generation Framework for Micro-Gesture Recognition](new_synthetic_goldmine_hand_joint_angle-driven_emg_data_generation_framework_for.md)
- [\[ICLR 2026\] Pose Prior Learner: Unsupervised Categorical Prior Learning for Pose Estimation](../../ICLR2026/human_understanding/pose_prior_learner_unsupervised_categorical_prior_learning_for_pose_estimation.md)
- [\[ICCV 2025\] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation](../../ICCV2025/human_understanding/cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)
- [\[NeurIPS 2025\] Learning Dense Hand Contact Estimation from Imbalanced Data](../../NeurIPS2025/human_understanding/learning_dense_hand_contact_estimation_from_imbalanced_data.md)

</div>

<!-- RELATED:END -->
