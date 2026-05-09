---
title: >-
  [Paper Note] ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving
description: >-
  [ICLR 2026][Autonomous Driving][Temporal Residual] ResWorld proposes a Temporal Residual World Model (TR-World) that extracts dynamic object information by computing temporal residuals of BEV scene representations—without relying on detection or tracking—thereby avoiding redundant modeling of static regions. Combined with a Future-Guided Trajectory Refinement (FGTR) module that leverages predicted future BEV features to refine planned trajectories, ResWorld achieves state-of-the-art planning performance on nuScenes and NAVSIM.
tags:
  - ICLR 2026
  - Autonomous Driving
  - Temporal Residual
  - World Model
  - End-to-End Autonomous Driving
  - BEV Features
  - Trajectory Refinement
date: 2026-05-08
content_hash: ba60bd15153b4e8c
---

# ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving

**Conference**: ICLR 2026
**arXiv**: [2602.10884](https://arxiv.org/abs/2602.10884)
**Code**: [https://github.com/mengtan00/ResWorld](https://github.com/mengtan00/ResWorld)
**Area**: Autonomous Driving / World Models
**Keywords**: Temporal Residual, World Model, End-to-End Autonomous Driving, BEV Features, Trajectory Refinement

## TL;DR
ResWorld proposes a Temporal Residual World Model (TR-World) that extracts dynamic object information by computing temporal residuals of BEV scene representations—without relying on detection or tracking—thereby avoiding redundant modeling of static regions. Combined with a Future-Guided Trajectory Refinement (FGTR) module that leverages predicted future BEV features to refine planned trajectories, ResWorld achieves state-of-the-art planning performance on nuScenes and NAVSIM.

## Background & Motivation

**Background**: In end-to-end autonomous driving frameworks, world models serve as surrogate tasks to enhance scene understanding, replacing traditional auxiliary modules such as detection, tracking, and prediction. A common approach is to predict future scene representations (BEV features) to indirectly improve planning accuracy.

**Limitations of Prior Work**: (a) The vast majority of information in scene representations corresponds to static objects (ground, buildings), which world models redundantly model—since static objects do not change position across future frames, predicting them is unnecessary. (b) Dynamic objects (vehicles, pedestrians) are critical for planning but are difficult to identify from the scene without relying on perception tasks. (c) The predicted future scene representations lack deep interaction with planned trajectories and are therefore underutilized.

**Key Challenge**: World models waste computational capacity on static objects, inadequately model dynamic objects, and fail to effectively feed predicted future information back into trajectory planning.

**Goal**: (a) Focus world model capacity on dynamic object modeling; (b) distinguish dynamic from static objects without auxiliary perception tasks; (c) directly leverage predicted future BEV features to refine trajectories.

**Key Insight**: Aligning BEV features from different timestamps to a common coordinate frame and computing their difference—the temporal residual—naturally captures the changes attributable to dynamic objects.

**Core Idea**: "Subtract to simplify"—temporal residuals naturally decouple dynamic and static objects, allowing the world model to predict only the future distribution of dynamic components.

## Method

### Overall Architecture
Multi-view images → GeoBEV extracts multi-frame BEV features → fused features predict a prior trajectory → temporal residuals are simultaneously computed → TR-World predicts the future spatial distribution of dynamic objects → overlaid onto the current BEV to obtain future BEV → FGTR module refines the prior trajectory using future BEV → final planning output.

### Key Designs

1. **Temporal Residual Extraction**:

    - Function: Extracts dynamic object information from aligned multi-frame BEV features via differencing.
    - Mechanism: Past $k$ BEV feature frames $\{B_t, B_{t-1}, \dots, B_{t-k}\}$ are aligned to the current frame's coordinate system. A shared spatial attention map is generated from the fused BEV feature $B_{\text{fuse}}$ to focus on dynamic regions, from which sparse scene queries $S_i$ are extracted per frame. Adjacent-frame differences $R_i = S_i - S_{i-1}$ are then computed.
    - Design Motivation: Static objects produce identical BEV features across frames in the same coordinate system, so differencing cancels them out automatically—yielding dynamic object information without any detection or tracking annotations. The shared spatial attention map ensures features are extracted from the same locations across frames, making the difference meaningful.

2. **Temporal Residual World Model (TR-World)**:

    - Function: Processes only temporal residuals to predict the future spatial distribution of dynamic objects.
    - Mechanism: Each residual $R_i$ is processed via self-attention and accumulated across timestamps to yield $\hat{R}$. A TokenFuser then maps $\hat{R}$ onto $B_{\text{fuse}}$: $B_{\text{future}} = \text{MLP}(B_{\text{fuse}}) \otimes \hat{R} + B_{\text{fuse}}$. Since $B_{\text{fuse}}$ already encodes static object information, adding the dynamic prediction produces a complete future BEV.
    - Design Motivation: Prevents the world model from wasting capacity on static regions that require no prediction, concentrating all modeling power on the dynamic changes that actually matter.

3. **Future-Guided Trajectory Refinement (FGTR)**:

    - Function: Refines the prior trajectory using predicted future BEV features.
    - Mechanism: Deformable Attention allows trajectory queries $W$ to sample information from $B_{\text{future}}$ using the prior trajectory $T_{\text{prior}}$ as reference points, checking whether the prior trajectory leads to collisions or deviations from the drivable area, and decoding a corrected final trajectory $T_{\text{final}}$.
    - Design Motivation: Dual benefit—(a) directly optimizes the trajectory using future information rather than doing so indirectly through surrogate tasks; (b) provides sparse spatiotemporal supervision to the future BEV features (reference points supply spatial signals; multi-timestamp queries supply temporal signals), preventing the world model from collapsing.
    - Key Design Detail: No direct ground-truth supervision is applied to the future BEV. Supervising specific timestamps would cause the model to lose dynamic object distribution information at other timestamps; allowing the model to self-organize retains the most important information across timestamps.

### Loss & Training
Only L1 losses are used to supervise the prior and final trajectories: $\mathcal{L} = L1(T_{\text{prior}}, T_{\text{GT}}) + L1(T_{\text{final}}, T_{\text{GT}})$. No direct supervision is applied to the future BEV features—a counterintuitive yet effective design choice.

## Key Experimental Results

### Main Results
nuScenes planning evaluation (no auxiliary tasks, zero annotation dependency):

| Method | Auxiliary Tasks | L2 Avg (m)↓ | Collision Rate Avg (%)↓ |
|--------|----------------|-------------|------------------------|
| UniAD | Det&Track&Map&Motion&Occ | 1.03 | 0.31 |
| SSR | None | 0.74 | 0.31 |
| GenAD | Det&Map&Motion | 0.91 | 0.43 |
| **ResWorld** | **None** | **0.65** | **0.23** |
| **ResWorld (w/ ego status)** | **None** | **0.59** | **0.17** |

NAVSIM evaluation:

| Method | PDMS↑ |
|--------|-------|
| LAW | 84.6 |
| World4Drive | 85.1 |
| DiffusionDrive | 88.1 |
| **ResWorld** | **87.3** (no auxiliary tasks) |

### Ablation Study

| Configuration | L2 Avg↓ | Collision Rate↓ | Notes |
|---------------|---------|----------------|-------|
| Baseline (SSR) | 0.74 | 0.31 | No world model |
| + TR-World | 0.69 | 0.27 | Temporal residual world model |
| + TR-World + FGTR | **0.65** | **0.23** | With future-guided trajectory refinement |
| Direct supervision on $B_{\text{future}}$ | 0.68 | 0.26 | Performance degrades |

### Key Findings
- TR-World reduces L2 from 0.74 to 0.69; FGTR further reduces it to 0.65—the two modules contribute complementarily.
- Applying direct ground-truth supervision to future BEV underperforms the unsupervised variant (0.68 vs. 0.65), validating the design of letting the model self-organize future representations.
- ResWorld requires no auxiliary perception tasks (no detection, tracking, or map annotations) yet outperforms all methods that rely on multi-task supervision.
- Compared to full-scene world models, TR-World—which models only temporal residuals—is both more efficient and more effective.

## Highlights & Insights
- The **"subtract to isolate dynamics"** approach is remarkably simple and elegant: no detectors, no trackers, no segmentation masks—only differencing of aligned BEV features. This technique is directly transferable to any scene representation task requiring dynamic/static separation.
- The finding that **no supervision outperforms direct supervision** is highly instructive: imposing ground-truth supervision at specific timestamps constrains what the world model can learn (locking it to a single temporal state), whereas gradient signals propagated through the downstream trajectory refinement task allow the future BEV to encode the most useful information across all timestamps.
- The **dual-role design of FGTR** is elegant: it simultaneously leverages future information to refine trajectories and provides backward gradient signals to prevent world model collapse—achieving two objectives with a single module.

## Limitations & Future Work
- Evaluation is limited to open-loop benchmarks (nuScenes, NAVSIM); closed-loop evaluation (e.g., Bench2Drive) is absent—open-loop performance does not necessarily transfer to closed-loop settings.
- Temporal residual computation assumes perfect BEV feature alignment; in practice, sensor pose estimation errors will degrade residual quality.
- The residual accumulation scheme (Eq. 6) is relatively simple; more expressive temporal modeling (e.g., Transformers) could be explored.
- The world model predicts only a single-step future BEV; the feasibility of multi-step recursive prediction remains an open question.

## Related Work & Insights
- **vs. SSR (Li & Cui, 2025)**: ResWorld extends SSR with TR-World and FGTR, reducing L2 from 0.74 to 0.65 and collision rate from 0.31 to 0.23.
- **vs. UniAD (Hu et al., 2023)**: UniAD requires five auxiliary tasks (detection + tracking + map + motion + occupancy); ResWorld requires none yet achieves superior performance.
- **vs. World4Drive (Zheng et al., 2025)**: Both are world-model-based end-to-end methods; ResWorld outperforms World4Drive on NAVSIM (87.3 vs. 85.1 PDMS).

## Rating
- Novelty: ⭐⭐⭐⭐ — The temporal residual idea for dynamic object extraction is simple yet effective; the dual-role FGTR design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation on two benchmarks with detailed ablations, though closed-loop evaluation is missing.
- Writing Quality: ⭐⭐⭐⭐ — Architecture diagrams are clear, method descriptions are fluent, and motivations are well articulated.
- Value: ⭐⭐⭐⭐ — Establishes an efficient world model design paradigm: model only what changes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](../../ICCV2025/autonomous_driving/world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [\[NeurIPS 2025\] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)
- [\[CVPR 2026\] Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems](../../CVPR2026/autonomous_driving/scaling-aware_data_selection_for_end-to-end_autonomous_driving_systems.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [\[CVPR 2026\] CausalVAD: De-confounding End-to-End Autonomous Driving via Causal Intervention](../../CVPR2026/autonomous_driving/causalvad_de-confounding_end-to-end_autonomous_driving_via_causal_intervention.md)

</div>

<!-- RELATED:END -->
