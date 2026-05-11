---
title: >-
  [Paper Note] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training
description: >-
  [NeurIPS 2025][Robotics][sim-to-real] This paper proposes a sim-and-real policy co-training framework based on Unbalanced Optimal Transport (UOT)…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "sim-to-real"
  - "optimal transport"
  - "domain adaptation"
  - "behavior cloning"
  - "robotic manipulation"
date: 2026-05-08
content_hash: d86f0a6dd852c7c9
---

# Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training

**Conference**: NeurIPS 2025
**arXiv**: [2509.18631](https://arxiv.org/abs/2509.18631)
**Code**: [Project Page](https://ot-sim2real.github.io/)
**Area**: Robotics / Sim-to-Real / Domain Adaptation
**Keywords**: sim-to-real, optimal transport, domain adaptation, behavior cloning, robotic manipulation

## TL;DR
This paper proposes a sim-and-real policy co-training framework based on Unbalanced Optimal Transport (UOT), which aligns the joint observation-action distribution (rather than only the marginal observation distribution), and incorporates a temporally aligned sampling strategy to handle data imbalance, achieving a 30% improvement in OOD generalization on robotic manipulation tasks.

## Background & Motivation
**Background**: Behavior cloning (BC) requires large amounts of costly real-world demonstrations, while simulators can generate data at low cost but suffer from visual/sensor gaps between simulation and reality.

**Limitations of Prior Work**: Naively mixing sim and real data for co-training lacks explicit feature-space constraints, leading to severe performance degradation in OOD scenarios; marginal distribution alignment methods such as MMD are too coarse and disrupt task-relevant feature structure.

**Key Challenge**: Simulated data is abundant but distribution-mismatched, real data is scarce but distribution-accurate, and the two sources are severely imbalanced in quantity ($N_{sim} \gg N_{real}$).

**Key Insight**: Align the joint observation-action distribution via optimal transport rather than aligning observations alone, thereby preserving action-relevant feature structure.

**Core Idea**: Apply unbalanced OT to handle partial overlap and data imbalance between sim and real joint distributions, augmented with DTW-based temporal alignment sampling to improve mini-batch quality.

## Method

### Overall Architecture
Behavior cloning and UOT regularization are trained in parallel: $L = L_{BC}(f_\phi, \pi_\theta) + \lambda \cdot L_{UOT}(f_\phi)$. The encoder $f_\phi$ is shared; the BC loss trains the policy while the UOT loss aligns the sim-real feature space.

### Key Designs

1. **Action-Aware Feature Alignment via Optimal Transport**:

    - Function: Aligns the joint distribution $(z, x)$ rather than $z$ alone ($z$ = features, $x$ = proprioceptive state/action).
    - Ground cost: $C_\phi = \alpha_1 \cdot d_\mathcal{Z}(z^i_{src}, z^j_{tgt}) + \alpha_2 \cdot d_\mathcal{A}(x^i_{src}, x^j_{tgt})$
    - Design Motivation: The joint distribution preserves the structure of "which features correspond to which actions," providing finer-grained alignment than marginal matching.

2. **Unbalanced Optimal Transport (UOT)**:

    - Function: Relaxes the strict marginal constraints of standard OT to handle the $N_{sim} \gg N_{real}$ data imbalance.
    - Mechanism: $L_{UOT} = \min_\Pi \langle\Pi, \hat{C}\rangle_F + \epsilon\Omega(\Pi) + \tau \text{KL}(\Pi\mathbf{1}||\mathbf{p}) + \tau \text{KL}(\Pi^\top\mathbf{1}||\mathbf{q})$
    - Design Motivation: Standard OT forces every simulated sample to be matched to a real sample; however, many simulated states are not covered in the real domain. UOT allows partial mass to remain untransported.

3. **Temporally Aligned Sampling Strategy**:

    - Function: Uses DTW to compute trajectory similarity and constructs mini-batches via similarity-weighted sampling.
    - Weight: $w(\xi_{src}, \xi_{tgt}) = \frac{1}{1+e^{10\cdot(\bar{d}-0.01)}}$
    - Design Motivation: Randomly sampled mini-batches yield poorly matched sim-real pairs; DTW alignment produces more semantically coherent sample pairings.

## Key Experimental Results

### Sim-to-Sim Domain Adaptation

| Method | In-Distribution Success Rate | OOD Success Rate |
|--------|------------------------------|-----------------|
| Co-training | 0.71 | 0.28 |
| MMD Alignment | — | 0.50 (but hurts source domain) |
| **Ours (UOT)** | **0.78** | **0.36** |
| Target-only | — | 0.0 |

### Sim-to-Real Domain Adaptation

| Method | Image Policy | Point Cloud Policy |
|--------|--------------|--------------------|
| Co-training | 0.55 | 0.60 |
| **Ours** | **0.73** | **0.77** |

### Key Findings
- Joint distribution alignment outperforms marginal alignment (MMD); t-SNE visualizations clearly show better mixing of source and target features.
- Scaling from 100 to 1000 simulated trajectories yields a 25% improvement in OOD performance, confirming the value of additional simulated data.
- The framework supports both RGB image and point cloud modalities.

## Highlights & Insights
- **Theoretical Advantage of Joint Distribution Alignment**: Preserving the action-feature structure yields richer alignment signal than marginal matching.
- **Elegant Handling of Data Imbalance via UOT**: UOT does not force every simulated sample to find a real counterpart, allowing partial mass to remain untransported.
- **Cross-Modal Validation**: The same framework is effective for both RGB and point cloud inputs, demonstrating the generality of the approach.

## Limitations & Future Work
- Only addresses the visual gap; dynamic discrepancies are not handled (quasi-static assumption).
- Requires a small amount of structured real demonstrations rather than unlabeled data.
- Tasks are limited to quasi-static grasping and manipulation; dynamically contact-rich tasks remain untested.

## Related Work & Insights
- **vs. DeepJDOT**: DeepJDOT uses pseudo-labels for joint distribution alignment, whereas this work uses ground-truth actions/states combined with UOT to handle imbalance.
- **vs. Co-training**: Co-training lacks explicit constraints and suffers severe OOD collapse; the proposed method resolves this via explicit OT alignment.
- **vs. MMD**: MMD aligns only marginal distributions, which is insufficiently fine-grained; joint distribution alignment achieves higher precision.

## Rating
- Novelty: ⭐⭐⭐⭐ The UOT + DTW sampling combination is novel, and the joint distribution alignment perspective is theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers sim-to-sim and sim-to-real settings, image and point cloud modalities, with sufficient ablations and visualizations.
- Writing Quality: ⭐⭐⭐⭐ Method motivation is clear and the modular design is easy to follow.
- Value: ⭐⭐⭐⭐ Offers significant practical value for sim-to-real robot learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning](pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](../../CVPR2026/robotics/gecosrt_geometryaware_continual_adaptation_for_rob.md)
- [\[NeurIPS 2025\] Policy Compatible Skill Incremental Learning via Lazy Learning Interface](policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)
- [\[NeurIPS 2025\] Operation Veja: Fixing Fundamental Concepts Missing from Modern Roleplaying Training Paradigms](operation_veja_fixing_fundamental_concepts_missing_from_modern_roleplaying_train.md)
- [\[ICCV 2025\] AnyBimanual: Transferring Unimanual Policy for General Bimanual Manipulation](../../ICCV2025/robotics/anybimanual_transferring_unimanual_policy_for_general_bimanual_manipulation.md)

</div>

<!-- RELATED:END -->
