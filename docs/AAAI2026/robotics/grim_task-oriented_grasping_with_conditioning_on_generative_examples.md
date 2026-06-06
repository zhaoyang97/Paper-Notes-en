---
title: >-
  [Paper Note] GRIM: Task-Oriented Grasping with Conditioning on Generative Examples
description: >-
  [AAAI 2026][Robotics][Task-oriented grasping] This paper proposes GRIM (Grasp Re-alignment via Iterative Matching), a **training-free** task-oriented grasping (TOG) framework that employs a **retrieve–align–transfer** pi…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "Task-oriented grasping"
  - "training-free"
  - "video generation model"
  - "semantic alignment"
  - "grasp transfer"
date: 2026-05-08
content_hash: 02014bba3bd86021
---

# GRIM: Task-Oriented Grasping with Conditioning on Generative Examples

**Conference**: AAAI 2026
**arXiv**: [2506.15607](https://arxiv.org/abs/2506.15607)  
**Code**: [Project Page](https://grim-tog.github.io/)  
**Area**: Robotics
**Keywords**: Task-oriented grasping, training-free, video generation model, semantic alignment, grasp transfer

## TL;DR

This paper proposes GRIM (Grasp Re-alignment via Iterative Matching), a **training-free** task-oriented grasping (TOG) framework that employs a **retrieve–align–transfer** pipeline combining video generation models with a multi-source memory bank. By leveraging DINO-feature-based semantic 3D alignment, GRIM achieves functional grasp transfer across objects, surpassing GraspMolmo—trained on 379K samples—using only 210 memory instances.

## Background & Motivation

### From Geometric Grasping to Functional Grasping

Traditional grasp synthesis focuses primarily on geometric stability—"can the object be picked up?" True manipulation intelligence, however, requires selecting **functionally appropriate** grasps—"how should the object be grasped to accomplish task X?" For instance, a hammer must be grasped by the handle to drive a nail, not by the head. This is the core problem that Task-Oriented Grasping (TOG) addresses.

### The Data Bottleneck

The central bottleneck of TOG is **data scarcity**:
- **Supervised learning methods** (e.g., TaskGrasp, GraspGPT) rely on large-scale manually annotated datasets labeling which grasps are appropriate for which tasks.
- **Knowledge graph methods** require extensive engineering effort to construct and maintain.
- Even approaches that leverage open-world knowledge from LLMs/VLMs (e.g., GraspGPT, GraspMolmo) still require training on predefined task–grasp datasets.

### Core Idea of GRIM

Entirely training-free. The data bottleneck is circumvented through:
1. Constructing a compact memory bank from diverse low-cost sources (AI-generated videos, web images, human demonstrations).
2. Performing cross-object alignment via semantic features rather than geometric shape.
3. Fusing transferred grasp poses with geometrically stable candidate grasps.

## Method

### Overall Architecture

GRIM follows a three-stage **Retrieve → Align → Transfer** pipeline:

1. **Retrieve**: Query the memory bank for the most relevant prior experience (based on DINO visual similarity + CLIP task semantic similarity).
2. **Align**: Perform 3D semantic alignment between the retrieved memory object and the scene object.
3. **Transfer & Refine**: Transfer the task grasp pose to the scene object and fuse it with geometrically stable candidate grasps.

### Key Designs

#### 1. **Multi-Source Memory Construction Pipeline**

Each memory instance is a quadruple $(F_M, G_t, T, O)$: a feature mesh, a 6D task grasp pose, a task description, and an object name.

Memory sources include:
- **AI-generated videos**: VLM (Gemini Pro) generates textual descriptions → VGM (VEO2) generates videos → frames are sampled to extract grasps. Scalable and low-cost.
- **Web images**: Images depicting grasping actions are crawled from the web; VLM generates corresponding task descriptions.
- **Expert demonstrations**: When the robot fails, a human provides a single demonstration image that is seamlessly added to the memory bank.

Grasp extraction from images/video frames: a hand-object reconstruction model extracts the object mesh and hand mesh; a 6D parallel-jaw grasp pose is then derived from the hand mesh using the centroids of the thumb, index/middle fingers, and palm to determine the closing direction and approach direction.

The feature mesh $F_M$ is constructed by sampling points on the object mesh surface and computing dense DINOv2 feature vectors, forming a semantic descriptor field.

#### 2. **Joint-Similarity-Based Memory Retrieval**

Given the point-cloud DINO features $\bar{F}_{SO}^D$ of the scene object and the task CLIP embedding $E_{T_S}$, the retrieval score is:

$$S_{\text{joint}}(i) = \alpha \cdot \text{sim}_{\cos}(\bar{F}_{SO}^D, \bar{F}_{MO,i}^D) + (1-\alpha) \cdot \text{sim}_{\cos}(E_{T_S}, E_{T_{M,i}})$$

where $\alpha=0.5$ balances visual and task semantic similarity. This design allows the system to trade off between "looks similar" and "task-compatible."

#### 3. **Semantic 3D Alignment (Core Innovation)**

Conventional ICP aligns objects based purely on geometry and fails when object shapes differ (e.g., a metal spatula vs. a plastic spatula). GRIM proposes a coarse-to-fine semantically guided alignment:

**Coarse Alignment**:
- DINO features are reduced to 4 dimensions via PCA.
- 8 steps are sampled along each of the three Euler angles ($8^3 = 512$ candidate rotations).
- For each candidate rotation, a joint feature–geometry cost is computed ($w_f=100, w_g=10$, heavily favoring semantic features).
- The 10 lowest-cost candidates are selected.

**Fine Alignment**:
- The best coarse alignment result initializes standard ICP for geometric refinement.
- The final transformation $T_{\text{final}}$ is output.

**Design Motivation**: Semantically guided initialization + geometric refinement = robust alignment even when objects are "semantically similar but geometrically different."

#### 4. **Grasp Transfer and Refinement**

Transfer: $G_S = T_{\text{final}} \cdot G_M$

AnyGrasp is used to generate $N$ geometrically stable candidate grasps $\{G_{A,i}\}$ on the scene object. The task compatibility score for each candidate is:

$$S_{\text{task},i} = \underbrace{(\mathbf{v}_{\text{target}} \cdot \mathbf{v}_{A,i})}_{\text{direction similarity}} + \underbrace{\exp\!\left(-\frac{\|\mathbf{t}_{A,i} - \mathbf{t}_S\|^2}{2\sigma^2}\right)}_{\text{position similarity}}$$

The final score is a weighted sum: $S_i = w_{\text{task}} S_{\text{task},i} + w_{\text{geo}} S_{\text{geo},i}$, where $w_{\text{task}}=0.95$ and $w_{\text{geo}}=0.05$ (heavily favoring task compatibility, since geometric quality is already ensured by AnyGrasp).

## Key Experimental Results

### Main Results: mAP on the TaskGrasp Dataset

| Method | All Data | Held-out Objects | Held-out Tasks |
|--------|----------|-----------------|----------------|
| Random | 0.49 | 0.41 | 0.43 |
| RTAGrasp (training-free SOTA) | 0.58 | 0.52 | 0.51 |
| GraspMolmo (trained on 379K) | 0.62 | 0.57 | 0.55 |
| **GRIM (training-free, 210 instances)** | **0.67** | **0.65** | **0.64** |

- GRIM outperforms GraspMolmo by 5 points on the full dataset—despite GraspMolmo being trained on 379K annotated samples.
- The advantage is more pronounced in held-out generalization scenarios: GRIM drops by approximately 3%, whereas RTAGrasp drops by more than 10%.

### Ablation Study

| Configuration | mAP (All Data) | Note |
|---------------|----------------|------|
| GRIM w/o Semantic Alignment | 0.50 | Near-random; confirms semantic alignment is the most critical component |
| GRIM w/o Grasp Refinement | 0.59 | Reasonable but insufficient; refinement translates functional intent into physical feasibility |
| **GRIM (Full Model)** | **0.67** | Both components are indispensable |

### Real-Robot Validation

Tested on a Kinova Gen3 Lite with two RGB-D cameras across 5 novel objects × 10 trials each:

| Metric | Value | Note |
|--------|-------|------|
| Success rate | 39/50 (78%) | Failures attributed to point cloud noise and calibration error, not incorrect grasp selection |
| Inference time | ~10 seconds | Efficient for online inference |
| Memory construction | ~7 min/instance | One-time offline cost |

### Key Findings

1. **Semantic alignment is the core**: Removing it reduces performance to near-random. DINO features enable the system to handle objects that are "functionally similar but geometrically different."
2. **Training-free outperforms large-scale training**: 210 multi-source memory instances outperform 379K annotated samples, demonstrating the effectiveness of the retrieve–align paradigm for grasp transfer.
3. **Strong generalization robustness**: Performance degrades minimally in held-out scenarios, confirming the generalizability of semantic alignment.
4. **Failure modes are well-defined**: The system depends on input point cloud quality—semantic correspondence fails under severe sensor noise or extreme sparsity.

## Highlights & Insights

- **Exceptional data efficiency**: 210 uncurated memory instances outperform a model trained on 379K carefully annotated samples.
- **Elegant application of generative AI in robotics**: Rather than using VGMs for direct decision-making, GRIM employs them to "imagine" grasping scenarios and create memory data—a particularly insightful design.
- **Semantic-first, geometry-refined** design philosophy: semantic weight in 3D alignment is set 10× higher than geometric weight; task compatibility weight in final scoring is set 19× higher than geometric quality—these strong priors are well-justified.
- **Lifelong learning capability**: The memory bank can be dynamically expanded at runtime via human demonstrations.

## Limitations & Future Work

- Reliance on multiple upstream pretrained models (Gemini Pro, VEO2, SAM, hand-object reconstruction, etc.) may inherit their hallucinations and biases.
- Offline memory construction requires ~7 minutes per instance, limiting scalability to large memory banks.
- Real-robot failures are primarily caused by point cloud quality—a better perception front-end is needed.
- Grasping of non-rigid or deformable objects is not addressed.

## Related Work & Insights

- The key distinction from RTAGrasp is that GRIM performs 3D semantic feature alignment rather than 2D feature matching, yielding greater robustness to viewpoint changes.
- The key distinction from GraspGPT is that GRIM is entirely training-free, avoiding the data acquisition bottleneck altogether.
- Insight: In robotic manipulation, the **retrieve + align + refine** paradigm may represent a universally efficient learning strategy.

## Rating

- Novelty: ⭐⭐⭐⭐ — Semantic 3D alignment combined with generative memory construction constitutes the core innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage across simulation benchmarks, ablation studies, and real-robot experiments.
- Writing Quality: ⭐⭐⭐⭐ — Framework description is clear and formulations are well-presented.
- Value: ⭐⭐⭐⭐⭐ — Demonstrating that a training-free approach can surpass large-scale trained methods carries significant implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Lagrangian Perturbation Diffusion Steering: Latent Reinforcement Learning for Generative Policies](../../ICML2026/robotics/lagrangian_perturbation_diffusion_steering_latent_reinforcement_learning_for_gen.md)
- [\[ACL 2026\] Capability-Oriented Failure Attribution for Vision-Language Navigation Agents](../../ACL2026/robotics/where_did_it_go_wrong_capability-oriented_failure_attribution_for_vision-and-lan.md)
- [\[ICLR 2026\] Grounding Generative Planners in Verifiable Logic: A Hybrid Architecture for Trustworthy Embodied AI](../../ICLR2026/robotics/grounding_generative_planners_in_verifiable_logic_a_hybrid_architecture_for_trus.md)
- [\[ICML 2026\] Plug-and-Play Label Map Diffusion for Universal Goal-Oriented Navigation](../../ICML2026/robotics/plug-and-play_label_map_diffusion_for_universal_goal-oriented_navigation.md)
- [\[CVPR 2026\] ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation](../../CVPR2026/robotics/maniparena_comprehensive_real-world_evaluation_of_reasoning-oriented_generalist_.md)

</div>

<!-- RELATED:END -->
