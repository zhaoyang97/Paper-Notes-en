---
title: >-
  [Paper Note] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer
description: >-
  [CVPR 2026][Robotics][Sim-to-Real Transfer] GeCo-SRT proposes a continual cross-task Sim-to-Real transfer paradigm that exploits the domain-invariance and task-invariance of local geometric features. Through a geometry-a…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "Sim-to-Real Transfer"
  - "Continual Learning"
  - "Geometry-aware MoE"
  - "Point Cloud Representation"
  - "Experience Replay"
date: 2026-05-08
content_hash: e3a6f1291f399db5
---

# GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer

**Conference**: CVPR 2026
**arXiv**: [2602.20871](https://arxiv.org/abs/2602.20871)  
**Code**: N/A  
**Area**: Robotics / Embodied Intelligence
**Keywords**: Sim-to-Real Transfer, Continual Learning, Geometry-aware MoE, Point Cloud Representation, Experience Replay

## TL;DR
GeCo-SRT proposes a continual cross-task Sim-to-Real transfer paradigm that exploits the domain-invariance and task-invariance of local geometric features. Through a geometry-aware MoE module for reusable geometric knowledge extraction and expert-guided prioritized experience replay for forgetting prevention, the method achieves a 52% improvement in average success rate over baselines across four manipulation tasks while requiring only 1/6 of the data.

## Background & Motivation
Conventional Sim-to-Real methods (system identification, domain randomization, data-driven transfer) treat each transfer as an independent process—every new task requires re-tuning and data collection from scratch, incurring high costs and discarding prior experience entirely. The root cause is that the Sim-to-Real gap across different tasks actually shares substantial structured cross-domain knowledge (e.g., geometry is consistent between simulation and reality), yet existing methods cannot accumulate or reuse this knowledge across tasks.

## Core Problem
How can **transferable knowledge be continually accumulated** across multiple Sim-to-Real tasks so that each new task's transfer is faster and better, rather than restarting from zero? What kind of knowledge carrier is simultaneously cross-domain and cross-task?

## Method

### Overall Architecture
A Human-in-the-Loop Sim-to-Real pipeline proceeds in three stages: a base diffusion policy (point cloud encoder + diffusion policy head) is first trained in simulation on 2,000 expert trajectories; upon real-world deployment, a human operator corrects impending failures via SpaceMouse (60 correction trajectories collected); correction data and simulation data are then mixed to train a shared perceptual residual module. The residual module is shared and continually updated across tasks, enabling knowledge accumulation.

### Key Designs
1. **Geometry-aware Mixture-of-Experts (Geo-MoE)**: Acts as the perceptual residual module. Local point groups are sampled from the input point cloud via kNN; PCA extracts local geometric features (planarity, linearity, saliency); these features drive a gating network to route point groups to different experts, each specializing in particular geometric knowledge (edges, corners, planes). The output residual vector is concatenated with frozen base encoder features and fed to the diffusion policy head. Core insight: local geometric features exhibit **dual invariance**—domain-invariant (geometric structures are consistent between simulation and reality) and task-invariant (different manipulation tasks share basic geometric elements), enabling reuse of learned expert knowledge on new tasks.

2. **Geometry Expert-guided Prioritized Experience Replay (Geo-PER)**: Standard PER samples by task loss, neglecting idle experts and causing forgetting. Geo-PER shifts priority to **expert utilization**: if an expert is underutilized on the current task (low $u_j^{\text{new}}$), historical samples that strongly activated that expert (high $w_{i,j}$) are prioritized: $P_i \propto \sum_{j=1}^{M} w_{i,j} \cdot \frac{1}{u_j^{\text{new}} + \epsilon}$. This inverse-hedging strategy ensures all experts are periodically refreshed and is tailored specifically to MoE architectures.

3. **Human-in-the-Loop Correction Pipeline**: Quantifies the Sim-to-Real gap as human correction trajectories—when the operator anticipates failure, they take over control. Correction and simulation data are mixed; only the shared Geo-MoE module is updated (base policy frozen), providing a clear knowledge accumulation pathway.

### Loss & Training
$\mathcal{L}_{\text{total}} = \text{MSE}(\hat{a}, a) + \alpha \mathcal{L}_{\text{balance}}$. The balance loss prevents gating collapse. Base policy training lr = $3 \times 10^{-4}$; residual learning lr = $1 \times 10^{-3}$; Geo-PER priority parameter 0.6; EMA update coefficient 0.4. Sixty correction trajectories per task are required.

## Key Experimental Results

| Setting | Metric | GeCo-SRT | Transic+PER | Geo-MoE+PER | Direct Deploy |
|--------|------|------|----------|------|------|
| Single-task transfer | Avg SR (%) | **50.0** | 38.3 | — | 3.1 |
| Continual 4-task transfer | Avg SR (%) | **63.3** | 40.0 | 55.7 | 3.1 |
| Continual 4-task transfer | Avg N-NBT (%) | **26.5** | 48.2 | 36.3 | — |
| Data efficiency | Data to match baseline | **1/6** | — | — | — |

### Ablation Study Highlights
- Observation residual (point cloud encoder) is the most critical component: adding it raises SR from 3.1% to 45.8%.
- Adding MoE alone without observation residual is nearly ineffective (geometric routing requires meaningful features as a prerequisite).
- Observation residual + MoE together achieve the best performance (55.8% SR).
- Geo-PER vs. standard PER: 63.3% vs. 55.7%, confirming expert-level priority outperforms task-level loss priority.
- Task similarity affects transfer: Pick Cube → Stack Cube yields positive transfer (40%), Plug Insert → Stack Cube yields negative transfer (16.7%).
- $N=3$ experts is optimal; $N=2$ and $N=8$ are also robust (60–65%).
- On new tasks (Faucet/Tidying): continual learning (83.3% / 56.7%) far outperforms zero-shot (53.3% / 30%) and training from scratch (76.6% / 43.3%).

## Highlights & Insights
- First work to extend Sim-to-Real transfer from isolated task adaptation to a continual cross-task knowledge accumulation paradigm.
- The "dual invariance" of local geometric features is a novel and experimentally validated insight—experts demonstrably specialize in edges, corners, and planes spontaneously.
- Shifting experience replay priority from task-loss level to MoE expert utilization level is an elegant, architecture-specific design that yields a uniquely effective anti-forgetting strategy.
- Strong data efficiency: 20 trajectories under continual learning approach the performance of 60-trajectory from-scratch training.
- MoE is interpretable: visualizations confirm expert specialization by geometric primitive type.

## Limitations & Future Work
- Primarily addresses the observation gap (visual level); limited effectiveness against complex dynamics gaps (physical level).
- Relies on Human-in-the-Loop correction data collection; although only 60 trajectories per task are needed, human participation is still required.
- Validated on only 4 tasks; scalability to much longer task sequences remains unexplored.
- Uses only point cloud input; RGB performance is notably lower (40% vs. 80%).

## Related Work & Insights
- **vs. Transic**: Both use human correction trajectories, but Transic employs a single residual network without MoE or continual learning. Single-task: 38.3% vs. 50%; the gap widens further in continual learning (Transic+PER 40% vs. GeCo-SRT 63.3%).
- **vs. Domain Randomization**: Requires manual randomization range specification and per-task independent tuning; GeCo-SRT automatically accumulates cross-task knowledge from data.
- **vs. LIBERO/LOTUS and similar continual learning methods**: Designed for purely imitative continual learning without addressing the Sim-to-Real gap; GeCo-SRT is the first to introduce continual learning into Sim-to-Real transfer.

## Inspiration and Connections
- The use of geometric features as cross-domain invariants complements prior work on 3D dynamic pretraining (e.g., AFRO)—AFRO learns dynamics while GeCo-SRT uses geometry for transfer.
- The expert-level prioritized experience replay design is generalizable to other MoE-based continual learning systems.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Continual cross-task Sim-to-Real is an entirely new problem setting; the Geo-MoE + Geo-PER combination is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four plus three real-robot tasks, with comprehensive ablations, transfer analysis, data efficiency studies, and interpretability visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Problem-driven structure is clear; method is presented in a well-organized, hierarchical manner.
- **Value**: ⭐⭐⭐⭐ Provides a new continual learning perspective on Sim-to-Real transfer; high data efficiency; strong practical value for resource-constrained real-robot scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [\[NeurIPS 2025\] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](../../NeurIPS2025/robotics/generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [\[ICLR 2026\] D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping](../../ICLR2026/robotics/d-rex_differentiable_real-to-sim-to-real_engine_for_learning_dexterous_grasping.md)
- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)
- [\[ICML 2026\] Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation](../../ICML2026/robotics/decompose_and_recompose_reasoning_new_skills_from_existing_abilities_for_cross-t.md)

</div>

<!-- RELATED:END -->
