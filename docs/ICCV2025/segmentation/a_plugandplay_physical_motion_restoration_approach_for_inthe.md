---
title: >-
  [Paper Note] A Plug-and-Play Physical Motion Restoration Approach for In-the-Wild High-Difficulty Motions
description: >-
  [ICCV 2025][Segmentation][physical motion restoration] A plug-and-play two-stage physical motion restoration method is proposed: defective frames in video motion capture are first corrected by a mask-conditioned diffusion model (MCM), then the corrected motion is transferred into a physically plausible form via a pretrained RL controller with test-time adaptation (PTM). This work achieves, for the first time, physically plausible enhancement of in-the-wild high-difficulty act…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "physical motion restoration"
  - "motion imitation"
  - "diffusion-based motion completion"
  - "test-time adaptation"
  - "high-difficulty motions"
date: 2026-05-08
content_hash: 56f1d35d3d479f33
---

# A Plug-and-Play Physical Motion Restoration Approach for In-the-Wild High-Difficulty Motions

**Conference**: ICCV 2025
**arXiv**: [2412.17377](https://arxiv.org/abs/2412.17377)  
**Code**: [Project Page](https://physicalmotionrestoration.github.io/)  
**Area**: Human Motion / Physics Simulation
**Keywords**: physical motion restoration, motion imitation, test-time adaptation, diffusion model, high-difficulty motions

## TL;DR
A plug-and-play physical motion restoration framework is proposed that repairs defective frames in video-based motion capture via a Mask-conditioned Motion Correction Module (MCM), and subsequently transfers the corrected motion into a physically plausible simulation through a Physics-based Motion Transfer Module (PTM) with RL-based test-time adaptation. This work is the first to achieve physics-based simulation restoration for in-the-wild high-difficulty motions such as gymnastics and martial arts back-flips.

## Background & Motivation

**Background**: Extracting 3D human motion from monocular video is a practical route for obtaining motion assets. Video motion capture (VMC) methods (e.g., GVHMR, TRAM) can rapidly estimate 3D motions but lack dynamics modeling, leading to physically implausible artifacts such as floating, foot sliding, self-penetration, and ground penetration. Physics-based motion imitation methods (e.g., PHC, UHC) can serve as post-processing modules to improve physical realism.

**Limitations of Prior Work**: Current physics-based simulation methods are limited to everyday motions (walking, running, jumping) and fail on high-difficulty motions (gymnastics, martial arts, street dance). Two reasons account for this: (a) VMC algorithms produce defective frames under complex motions (extreme poses cause body localization failure), and these frames directly cause simulation failure; (b) the force control involved in high-difficulty motions is extremely complex and follows a long-tail distribution, making it difficult for a single controller to generalize.

**Key Challenge**: High-difficulty motions are extremely rare in existing MoCap datasets (long-tail distribution) and involve complex mechanical interactions such as aerial flips and elastic surface assistance. Pretrained controllers lack the necessary prior knowledge and are subject to catastrophic forgetting.

**Goal**: How can physics-based restoration be extended from everyday motions to in-the-wild high-difficulty motions?

**Key Insight**: Divide and conquer — (a) defective motions are repaired into "simulation-friendly" motions via segmentation mask-guided diffusion; (b) high-difficulty motions are addressed per-instance through pretraining followed by test-time RL adaptation.

**Core Idea**: Large-scale pretraining provides general motion priors; test-time RL adaptation optimizes per-sequence dynamics — together enabling robust handling of in-the-wild high-difficulty motions.

## Method

### Overall Architecture
**Input**: Video motion capture results (noisy 3D motion sequences) + corresponding video. The pipeline proceeds in two stages: (1) MCM detects and repairs defective frames to produce "simulation-friendly" motion; (2) PTM imitates the corrected motion in a physics simulation environment and outputs physically plausible motion. PTM is pretrained on large-scale data and then performs RL adaptation for each test sequence.

### Key Designs

1. **Mask-conditioned Motion Correction Module (MCM)**:

    - **Function**: Detects and replaces defective frames in video motion capture results.
    - **Mechanism**: Two steps — **Defect Detection**: Projects 3D reference motion joints onto 2D and computes OKS similarity with 2D keypoints detected from the video; frames below a threshold are marked as defective. Alternatively, the SMPL mesh is projected to 2D and the overlap ratio with SAM segmentation masks is used for detection. **Defect Repair**: A diffusion model-based motion in-betweening approach regenerates the defective segments. Conditioning signals include: (a) contextual motion frames (keyframe signal $\mathbf{c}$), and (b) human segmentation masks extracted by SAM, encoded by a pretrained ViT and used as conditions. The masks provide approximate human body location information that remains accessible even in ambiguous motion frames.
    - **Design Motivation**: Segmentation naturally separates foreground from background and is more robust to motion blur; defective segments are typically short and surrounded by rich context, making them well-suited for in-betweening repair.

2. **Physics-based Motion Transfer Module (PTM)**:

    - **Function**: Tracks and reproduces reference motions within a physics simulation environment.
    - **Mechanism**: **Pretraining stage** — A motion imitation controller is trained via PPO on AMASS, Human3.6M, AIST++, and the Kungfu subset of Motion-X to acquire rich motion priors. **Test-Time Adaptation (TTA)** — Independent RL fine-tuning is performed for each test sequence (updating network parameters) for a limited number of steps, leveraging the trial-and-error nature of RL to solve the dynamics per instance. Key adaptation design choices include:
        - **Relative Reward**: Absolute root position is ignored ($rela()$ removes the gravity axis) to avoid the influence of accumulated root errors in video motion capture.
        - **Relative Termination Condition**: Termination is determined based on relative joint distances rather than absolute positions, combined with height and contact conditions.
        - **Residual Forces**: External residual forces are introduced to compensate for missing environmental factors (e.g., elastic surfaces) in the simulation.
    - **Design Motivation**: The pretrain+adapt paradigm naturally addresses the long-tail distribution and domain gap issues — pretraining provides fast initialization, and per-instance adaptation requires no similar motions in the training dataset.

3. **AMP Discriminator + Energy Penalty**:

    - **Function**: Constrains the naturalness of generated motions and prevents jitter.
    - **Mechanism**: Total reward $r_t = r_t^g + r_t^{amp} + r_t^{energy}$, where the reconstruction reward tracks the reference motion, the AMP discriminator distinguishes real from generated motions to ensure naturalness, and the energy penalty prevents joint jitter.

### Loss & Training
- **MCM Training**: Random motion segments are masked and reconstructed via the diffusion model; 10% of training uses null conditioning to support classifier-free guidance.
- **PTM Pretraining**: Strict reconstruction reward with early termination conditions.
- **PTM TTA**: Relative reward + relative termination conditions + residual forces; each sequence is adapted independently.

## Key Experimental Results

### Main Results

| Method | Dataset | GP↓ | Float↓ | FS↓ | SP↓ | MPJPE↓ |
|------|--------|------|--------|------|------|---------|
| GVHMR | EMDB | 82.3 | 510.3 | 0.69 | 0.006 | 109.1 |
| GVHMR+Ours | EMDB | **0.25** | **3.6** | **0.17** | **0.002** | **91.2** |
| TRAM | Kungfu | 199.7 | 161.2 | 17.4 | 0.073 | 230.6 |
| TRAM+Ours | Kungfu | **1.4** | **4.6** | **1.5** | **0.045** | **224.0** |
| GVHMR+Ours | In-the-wild 206 videos | 0.33 | 14.9 | 0.72 | 0.12 | - |
| GVHMR+PhysPT | In-the-wild 206 videos | 6.62 | 54.0 | 5.63 | - | - |

### Ablation Study

| Configuration | OKS↑ | MPS↑ | Success Rate↑ | Notes |
|------|------|------|---------|------|
| No PTM settings | 0.811 | 0.673 | 37% | Directly uses pretrained model |
| +Early Termination | 0.784 | 0.652 | 52% | Avoids premature failure |
| +Residual Force | 0.823 | 0.673 | 61% | Compensates environmental forces |
| +TTA | 0.850 | 0.706 | 85% | Test-time adaptation |
| +Relative Reward | 0.853 | 0.710 | 87% | Full method |
| PTM vs UHC (Kungfu) | 98.16% vs 42.91% SR | - | - | Significantly surpasses SOTA |
| PTM vs PHC+ (Kungfu) | 98.16% vs 76.41% SR | - | - | |

### Key Findings
- TTA is the most critical component: success rate improves from 37% to 85%, demonstrating that per-instance adaptation is essential for handling out-of-distribution motions.
- Physical metrics improve substantially: ground penetration drops from 82.3 to 0.25 on EMDB; self-penetration is reduced by more than 50%.
- Mask conditioning in MCM outperforms pure motion context: masks provide reliable human location information even in ambiguous frames.
- MPJPE in camera coordinates shows minor change (since the method operates in physical space without considering camera parameters), while world-coordinate metrics improve markedly.

## Highlights & Insights
- **Pretrain+TTA paradigm**: The combination of large-scale pretraining with per-instance RL adaptation is elegant — it reframes "generalizing to new motions" as "rapidly fine-tuning to a single motion sequence," entirely bypassing the long-tail distribution problem. This paradigm is transferable to any RL control system that must handle out-of-distribution data.
- **Segmentation masks as surrogate signals for defective frames**: A practically important observation — when motion is rapid, keypoint detection fails, but segmentation (foreground/background separation) remains stable. Using segmentation masks as diffusion in-betweening conditions is more reliable than relying solely on motion context.
- **Relative reward design**: By discarding absolute root position along the gravity axis and retaining only relative joint positions, rotations, and velocities, the method cleverly circumvents the problem of accumulated root errors in video-based MoCap.

## Limitations & Future Work
- **Camera parameters are not considered**: Operating in physical space leads to imperfect recovery in the camera coordinate system.
- **TTA computational cost**: Each sequence requires an independent RL adaptation phase, significantly increasing inference time.
- **Residual forces lack physical constraints**: Although the introduction of external forces solves practical problems, it sacrifices strict physical plausibility.
- **Human-object interaction is not handled**: All scenes involving person-object interaction are excluded.
- Potential improvements include online adaptation with visual feedback and more efficient few-shot TTA strategies.

## Related Work & Insights
- **vs PHC/PHC+**: PHC can simulate nearly all motions in AMASS but fails on in-the-wild high-difficulty motions; the proposed TTA overcomes this limitation.
- **vs PhysPT**: PhysPT employs a physics-aware Transformer for self-supervised learning but lacks understanding of high-difficulty motion distributions, resulting in poor restoration quality.
- **vs DiffPhy/SimPoE**: These methods require careful parameter tuning and are sensitive to motion type, making generalization difficult.

## Rating
- Novelty: ⭐⭐⭐⭐ The pretrain+TTA paradigm for motion simulation is novel, and the mask-guided correction in MCM is also a distinctive contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dataset evaluation + self-constructed 206-video in-the-wild benchmark + detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear, and the two core challenges are well analyzed.
- Value: ⭐⭐⭐⭐ First work to extend physics-based simulation restoration to high-difficulty motions, with strong practical application value.

**Area**: Human Understanding
**Keywords**: physical motion restoration, motion imitation, diffusion-based motion completion, test-time adaptation, high-difficulty motions

## TL;DR

A plug-and-play two-stage physical motion restoration method is proposed: defective frames in video motion capture are first corrected by a mask-conditioned diffusion model (MCM), then the corrected motion is transferred into a physically plausible form via a pretrained RL controller with test-time adaptation (PTM). This work achieves, for the first time, physically plausible enhancement of in-the-wild high-difficulty actions (gymnastics/martial arts/dance).

## Background & Motivation

Extracting physically plausible 3D human motion from monocular video is a core requirement in virtual reality, game animation, and robotics. Existing methods suffer from two major bottlenecks:

1. **Video motion capture (VMC) methods** (e.g., TRAM, GVHMR): While capable of rapidly obtaining 3D motion from video, they lack dynamics modeling, leading to physical artifacts such as floating, foot sliding, self-penetration, and ground penetration — particularly severe for high-difficulty actions.
2. **Physics-based motion imitation methods** (e.g., PhysPT, PHC+): These can improve physical quality for everyday motions (walking, running) but fail on high-difficulty motions such as gymnastics and martial arts — because (a) VMC produces **defective frames** during rapid extreme poses, causing the imitation process to collapse; and (b) the **long-tail distribution and complex force control** of high-difficulty motions prevent a single controller from generalizing, leading to catastrophic forgetting.

## Core Problem

How can physical plausibility be restored for VMC results of in-the-wild high-difficulty motions (gymnastics, martial arts, dance, etc.) while preserving the original motion patterns? Two challenges must be simultaneously addressed: (1) repairing short-duration defective frames in VMC results; and (2) successfully tracking complex motions in physics simulation.

## Method

### Overall Architecture

The method is designed as a **plug-and-play post-processing module** that can be appended to any VMC method. It consists of two cascaded modules:
- **MCM (Mask-conditioned Motion Correction Module)**: Detects and repairs defective frames in the reference motion.
- **PTM (Physics-based Motion Transfer Module)**: Transforms the corrected motion into physically plausible motion via physics simulation.

Motion is represented in an SMPL-compatible format: joint positions $\mathbf{p}_t \in \mathbb{R}^{J \times 3}$ and rotations $\boldsymbol{\theta}_t \in \mathbb{R}^{J \times 6}$ (6D continuous rotation representation).

### Key Designs

#### MCM Module

**Mismatch Detection**: Two detection strategies identify defective frames:
- **Keypoint-based detection**: The 3D reference motion is projected to 2D and OKS similarity is computed with 2D keypoint detection results from the video; frames below a threshold are marked as defective.
- **Mask-based detection**: The SMPL mesh is projected onto the 2D plane and the proportion of projected points falling within the SAM segmentation mask is computed.

$$OKS = \frac{\sum_i \exp(-d_i^2 / 2\epsilon_i^2) \cdot \delta(v_i > 0)}{\sum_i \delta(v_i > 0)}$$

**Mask-conditioned Diffusion Completion**: A pretrained ViT extracts human pose features from segmentation masks. These mask features, together with motion context, serve as conditions for a GMD-based UNet diffusion model to regenerate the defective segments. During training, 10% of samples use null conditioning to support unconditional generation.

#### PTM Module

**Pretraining Stage**: A motion imitation controller is trained on AMASS, Human3.6M, AIST++, and the Kungfu subset of Motion-X using PPO optimization and an AMP discriminator to ensure stylistic naturalness. Reward function:

$$r_t = r_t^g + r_t^{amp} + r_t^{energy}$$

**Test-Time Adaptation (TTA)**: The core innovation — independent RL fine-tuning is applied to each test motion sequence:

- **Relative Reward**: Absolute root position is ignored; only relative joint positions, rotations, and velocities are considered, avoiding the influence of root node errors in noisy reference motions:
$$r_t^g = e^{w_p \|\text{rela}(\hat{\mathbf{p}}_t) - \text{rela}(\mathbf{p}_t)\|} + e^{w_r \|\hat{\boldsymbol{\theta}}_t \ominus \boldsymbol{\theta}_t\|} + e^{w_v \|\hat{\mathbf{v}}_t - \mathbf{v}_t\|} + e^{w_\omega \|\hat{\boldsymbol{\omega}}_t - \boldsymbol{\omega}_t\|}$$

- **Relative Termination Condition**: Termination is judged based on average relative joint distance rather than absolute distance, with additional height and ground contact conditions to prevent falls and erroneous contacts:
$$\mathcal{F}_t = (\frac{1}{J}\sum_{i=1}^J \|\text{rela}(\hat{\mathbf{p}}_t^i) - \text{rela}(\mathbf{p}_t^i)\| > d_{term}) \lor \mathcal{F}_t^h \lor \mathcal{F}_t^c$$

- **Residual Forces**: External residual forces are introduced during TTA to compensate for dynamics mismatches — high-difficulty actions (e.g., gymnastics tumbling) rely on elastic trampolines/mats in practice, and external forces bridge this gap in simulation.

### Loss & Training

- **MCM Training**: Standard diffusion model training; random motion segments are selected as generation targets; conditioning signals are dropped with 10% probability.
- **PTM Pretraining**: Strict reconstruction reward and early termination conditions; trained on 4 datasets for approximately 2–3 days on a single A100.
- **TTA Inference**: Everyday motions converge in fewer than 500 steps; high-difficulty motions require 2000–4000 steps; each motion sequence is adapted independently.

## Key Experimental Results

### Evaluation on Public Datasets (Table 1)

| Dataset | Method | SP↓ | GP↓ | FS↓ |
|--------|------|------------|------------|----------|
| AIST++ | GVHMR | 0.072 | 12.390 | 2.232 |
| AIST++ | GVHMR+PhysPT | – | 4.978 | 2.468 |
| AIST++ | **GVHMR+Ours** | **0.046** | **0.498** | **0.587** |
| Kungfu | GVHMR | 0.079 | 10.368 | 2.217 |
| Kungfu | **GVHMR+Ours** | **0.018** | **0.290** | **0.257** |
| EMDB | GVHMR | 0.006 | 82.266 | 0.693 |
| EMDB | **GVHMR+Ours** | **0.002** | **0.248** | **0.173** |

Physical metrics improve substantially: ground penetration on EMDB drops from 82 to 0.24; self-penetration is reduced by more than 50%.

### Evaluation on In-the-Wild High-Difficulty Test Set (Table 2, 206 videos)

| Method | OKS↑ | MPS↑ | SP↓ | GP↓ | Float↓ | FS↓ |
|------|------|------|-----|-----|--------|-----|
| GVHMR | 0.837 | 0.704 | 0.289 | 9.999 | 137.969 | 3.006 |
| GVHMR+PhysPT | 0.806 | 0.685 | – | 6.616 | 54.032 | 5.630 |
| **GVHMR+Ours** | **0.854** | **0.710** | **0.120** | **0.334** | **14.921** | **0.717** |

### Physical Transfer Capability (Table 3, Kungfu Dataset)

| Method | SR | MPJPEg↓ | MPJPE↓ | PA-MPJPE↓ |
|------|-----------|---------|--------|-----------|
| UHC | 42.91% | 86.23 | 48.91 | 39.73 |
| PHC+ | 76.41% | 84.86 | 47.98 | 39.43 |
| **PTM (Ours)** | **98.16%** | **82.13** | **33.45** | **26.12** |

### Ablation Study Highlights

**Effect of TTA strategy components (Table 4, in-the-wild dataset)**:
- Pretrained controller only: success rate 37%
- +Early Termination: 52% (relaxed termination avoids premature failure on low-quality motions)
- +Relative Reward: 61%
- +Residual Forces: 85% (aerial actions such as gymnastics tumbling require external force compensation)
- +TTA Adaptation: **87%** (per-sequence parameter updates contribute the largest single gain)

**MCM module ablation (Table 5)**:
- Direct simulation without MCM: success rate 78%
- Mask conditioning outperforms keypoint conditioning (masks carry richer shape and motion information and are more stable under complex motions)
- Mask + keypoint + mask detection: **OKS 0.853, MPS 0.710, SR 87%**

## Highlights & Insights

1. **Plug-and-play design**: The framework integrates seamlessly with any VMC method without additional training, making it practically elegant.
2. **Pretrain+TTA paradigm**: By leveraging the trial-and-error nature of RL and treating each test motion as an independent instance for fine-tuning, the method naturally resolves long-tail distribution and domain transfer issues while avoiding catastrophic forgetting.
3. **Relative design philosophy**: Relative reward and relative termination conditions embody the core insight that root nodes in in-the-wild motions are inherently noisy and should not be pursued for exact absolute matching.
4. **Mask-assisted repair**: The method leverages the fact that SAM segmentation masks are more robust than keypoint detection under complex motions, guiding the diffusion model to repair defective frames.
5. **Residual force compensation**: Recognizing that high-difficulty actions in practice rely on trampolines or mats, external forces are introduced in simulation to bridge the environmental gap.

## Limitations & Future Work

1. **Single-person motions only**: Multi-person interaction scenarios (e.g., partner dance, sparring) are not supported — a limitation explicitly acknowledged by the authors.
2. **Inference speed**: High-difficulty motions require 2000–4000 TTA steps; efficiency needs improvement.
3. **No camera parameter modeling**: Restoration is performed in physical space without considering camera parameters, potentially degrading metrics in the camera coordinate system.
4. **Residual force plausibility**: The introduction of external forces, while practical, lacks physical constraints that would restrict force directions to physically reasonable ranges.
5. **MCM depends on segmentation quality**: Scenes where SAM segmentation fails may lead to correction failures.

## Related Work & Insights

| Method | Type | High-Difficulty Motions | Plug-and-Play | Defective Frame Repair |
|------|------|----------|---------|-----------|
| PhysPT | Physics-aware Transformer | ✗ | ✓ | ✗ |
| PHC+ | Motion imitation | ✗ (76% SR) | ✗ | ✗ |
| SimPoE | Simulation + visual fusion | ✗ | ✗ | ✗ |
| PhysCap | Numerical optimization + physics | ✗ | ✗ | ✗ |
| **Ours** | **MCM+PTM** | **✓ (98% SR)** | **✓** | **✓** |

Key differentiators: (1) this work performs restoration rather than reconstruction from scratch; (2) TTA enables a single model to generalize to high-difficulty actions; (3) MCM is the first to incorporate video segmentation signals to assist motion correction.

- **TTA in other motion tasks**: The pretrain+test-time adaptation paradigm is generalizable to motion generation, human mesh recovery, and other tasks — particularly suited for handling OOD data.
- **Segmentation models as motion cues**: Foundation segmentation models such as SAM can provide additional 2D supervision signals for motion understanding and are worth exploring in more human-centric tasks.
- **Combining physics simulation with diffusion models**: The complementary design of using diffusion models for motion completion (MCM) and RL for physics simulation (PTM) offers an inspiring framework for future work.

## Rating

| Dimension | Score (1–5) | Notes |
|------|-----------|------|
| Novelty | 4 | The pretrain+TTA paradigm and mask-conditioned diffusion repair combination is novel and practical. |
| Technical Depth | 4 | Spans diffusion models, RL, and physics simulation with rich design details. |
| Experimental Thoroughness | 4.5 | Multi-dataset validation + self-constructed 206-video benchmark + comprehensive ablations. |
| Writing Quality | 4 | Clear motivation, systematic method description, and intuitive figures. |
| Practical Value | 4.5 | Plug-and-play design offers strong engineering utility. |
| **Overall** | **4.2** | Strong practical utility; a significant advance in high-difficulty motion restoration. |
- A new in-the-wild high-difficulty motion benchmark is constructed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild](correspondence_as_video_test-time_adaption_on_sam2_for_reference_segmentation_in.md)
- [\[ECCV 2024\] OLAF: A Plug-and-Play Framework for Enhanced Multi-object Multi-part Scene Parsing](../../ECCV2024/segmentation/olaf_a_plug-and-play_framework_for_enhanced_multi-object_multi-part_scene_parsin.md)
- [\[ICCV 2025\] What If: Understanding Motion Through Sparse Interactions](what_if_understanding_motion_through_sparse_interactions.md)
- [\[ICCV 2025\] Temporal Rate Reduction Clustering for Human Motion Segmentation](temporal_rate_reduction_clustering_for_human_motion_segmentation.md)
- [\[ICCV 2025\] Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP](know_no_better_a_data-driven_approach_for_enhancing_negation_awareness_in_clip.md)

</div>

<!-- RELATED:END -->
