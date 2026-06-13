---
title: >-
  [Paper Note] Astra: General Interactive World Model with Autoregressive Denoising
description: >-
  [ICLR 2026][Autonomous Driving][world model] This paper proposes Astra, a general interactive world model that enables action-conditioned long-horizon video prediction on top of a pretrained video diffusion model via an…
tags:
  - "ICLR 2026"
  - "Autonomous Driving"
  - "world model"
  - "autoregressive denoising"
  - "action control"
  - "interactive video"
  - "mixture of experts"
date: 2026-05-08
content_hash: 7bf67e23499f828d
---

# Astra: General Interactive World Model with Autoregressive Denoising

**Conference**: ICLR 2026
**arXiv**: [2512.08931](https://arxiv.org/abs/2512.08931)  
**Code**: [https://github.com/EternalEvan/Astra](https://github.com/EternalEvan/Astra)  
**Area**: Autonomous Driving / Video Generation
**Keywords**: world model, autoregressive denoising, action control, interactive video, mixture of experts

## TL;DR
This paper proposes Astra, a general interactive world model that enables action-conditioned long-horizon video prediction on top of a pretrained video diffusion model via an autoregressive denoising framework. Three key contributions are introduced: ACT-Adapter (action injection), noise-augmented history memory (to mitigate visual inertia), and Mixture of Action Experts (to unify heterogeneous action modalities). Astra achieves state-of-the-art fidelity and action-following capability across autonomous driving, robotic manipulation, and scene exploration scenarios.

## Background & Motivation
**Background**: Video diffusion models (e.g., Wan-2.1) can generate high-quality short videos but lack interactivity — they cannot dynamically adjust generation based on action inputs. A true world model must respond to arbitrary actions at any time step.

**Limitations of Prior Work**: (1) Standard T2V/I2V models generate fixed clips without long-horizon rollout; (2) hybrid autoregressive-diffusion methods suffer from error accumulation and temporal drift; (3) extending history context improves temporal consistency but weakens action responsiveness — the "visual inertia" problem; (4) real-world environments involve heterogeneous action modalities (camera poses, robot joints, keyboard commands) that are difficult for a single model to unify.

**Key Challenge**: A fundamental tension exists between long-horizon temporal consistency and action responsiveness — models tend to extrapolate smoothly from past frames while ignoring new action control signals.

**Goal**: To build a general world model capable of generating interactive long-horizon videos conditioned on diverse action types across multiple real-world scenarios.

**Key Insight**: Attach lightweight adapters to a pretrained video diffusion model for action injection, augment history frames with noise to mitigate visual inertia, and route heterogeneous actions through a Mixture of Experts.

**Core Idea**: Reduce the dominance of history frames via noise augmentation, inject action signals via adapters, and unify multimodal actions via MoE — transforming a video diffusion model into an interactive world model.

## Method

### Overall Architecture
Astra is built upon the pretrained Wan-2.1 video diffusion model and employs chunk-wise autoregressive generation: at each step, the next video chunk (33 frames) is predicted, appended to the history, and used as context for the next prediction. History is aggregated via causal temporal attention.

### Key Designs

1. **ACT-Adapter (Action-Aware Adapter)**:

    - **Function**: Injects action signals into the denoising process of the video diffusion model.
    - **Mechanism**: An action encoder projects actions into a feature space aligned with the video latent space; the projected features are added element-wise at each DiT block. Most pretrained parameters are frozen; only self-attention layers and a single linear adapter (initialized as the identity matrix) are fine-tuned.
    - **Design Motivation**: The influence of actions on video can be interpreted as feature shifts in the latent space (analogous to optical flow); element-wise addition is the most direct realization of such shifts.

2. **Noise-as-Mask (Noise-Augmented History Memory)**:

    - **Function**: Random noise is injected into history condition frames during training to reduce their informational dominance.
    - **Mechanism**: The noise is independent of the diffusion noise and is used purely to degrade history frames — forcing the model to rely on action signals rather than simply copying past frames. Clean history frames are used at inference time.
    - **vs. YUME's masking**: YUME randomly masks visual tokens; Astra blurs history frames with noise — requiring no architectural modification or additional parameters.

3. **Mixture of Action Experts (MoAE)**:

    - **Function**: Unifies heterogeneous action modalities (7D camera pose, 7D robot joint angles, keyboard/mouse commands).
    - **Mechanism**: Each modality is first mapped to a shared space via modality-specific projectors; a routing network then computes gating scores to select the top-K experts (independent MLPs), whose outputs are aggregated with learned weights.
    - **Design Motivation**: Different action modalities exhibit large structural and scale differences, making a single encoder insufficient for unification.

4. **Action-Free Guidance (AFG)**:

    - Analogous to CFG: action conditions are randomly dropped during training, and at inference the guided velocity is computed as $v_{guided} = v_\emptyset + s \cdot (v_a - v_\emptyset)$ to amplify the effect of action conditioning.

### Loss & Training
Flow matching loss. Built upon Wan-2.1 pretraining; trained for 30 epochs on 8 GPUs (~24 hours). Training data: ~397K videos (360 hours) covering nuScenes, Sekai, SpatialVID, RT-1, and Multi-Cam Video.

## Key Experimental Results

### Main Results (Astra-Bench, 480×832, 96 frames)

| Method | Instruction Following↑ | Subject Consistency↑ | Motion Smoothness↑ |
|--------|----------------------|--------------------|--------------------|
| Wan-2.1 | 0.061 | 0.854 | 0.958 |
| MatrixGame | 0.268 | 0.916 | 0.981 |
| YUME | 0.652 | 0.936 | 0.985 |
| **Astra** | **0.669** | **0.939** | **0.989** |

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| w/o ACT-Adapter (replace with cross-attn) | Significant drop in action following |
| w/o AFG | Weakened action responsiveness |
| w/o noise-as-mask | Increased visual inertia; action signals ignored |
| w/o MoAE | Inability to handle multimodal actions; performance degradation |

### Key Findings
- Astra outperforms prior SOTA on all 6 metrics, with a particularly large margin on Instruction Following (0.669 vs. 0.061 for Wan-2.1).
- Astra maintains stability in long-horizon rollouts (96+ frames) where competing methods exhibit drift and degradation.
- The noise augmentation strategy is simpler than token masking (no architectural changes required) and yields better performance.
- MoAE enables a single model to handle both autonomous driving (camera pose) and robotic manipulation (joint angles).
- AFG effectively amplifies the influence of action conditioning, analogous to CFG in unconditional generation.

## Highlights & Insights
- **"Visual inertia" is a core challenge in world models**: This paper is the first to name and systematically address the tension between long-horizon consistency and action responsiveness.
- **Elegance of noise-augmented history**: No architectural modification, no additional parameters — noise is injected only during training to rebalance the relative influence of different information sources.
- **Ambition of multi-scenario unification**: A single model handles autonomous driving, robotic manipulation, and first-person exploration, enabled by MoAE.

## Limitations & Future Work
- Training data is dominated by driving and exploration scenarios; complex physical interactions (e.g., fluids, collisions) may be underrepresented.
- Error accumulation may still emerge in extremely long rollouts (hundreds of frames or more).
- Whether the MoAE routing mechanism performs meaningful modality-level specialization requires further analysis.
- The Instruction Following metric relies on human evaluation, limiting scalability.

## Related Work & Insights
- **vs. YUME**: YUME employs a masked video diffusion transformer; Astra uses noise-as-mask, which is simpler and requires no architectural changes.
- **vs. MatrixGame**: MatrixGame uses causal action guidance; Astra's ACT-Adapter injects action signals more directly.
- **vs. Genie2/UniSim (large-scale world models)**: Astra achieves competitive performance with substantially less data (~400K vs. millions of samples).

## Rating
- Novelty: ⭐⭐⭐⭐ — Noise-augmented history, ACT-Adapter, and MoAE each represent creative contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Five datasets, multiple scenarios, comprehensive ablations, and human evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Architecture diagrams are clear; the "visual inertia" concept is vivid and well-motivated.
- Value: ⭐⭐⭐⭐⭐ — A practical framework for general interactive world models, with open-sourced code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](../../ICCV2025/autonomous_driving/epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[ICLR 2026\] ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)
- [\[CVPR 2026\] SparseWorld-TC: Trajectory-Conditioned Sparse Occupancy World Model](../../CVPR2026/autonomous_driving/sparseworld_tc_trajectory_conditioned_sparse_occupancy_world_model.md)
- [\[AAAI 2026\] WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving](../../AAAI2026/autonomous_driving/worldrft_latent_world_model_planning_with_reinforcement_fine-tuning_for_autonomo.md)
- [\[AAAI 2026\] Difficulty-Aware Label-Guided Denoising for Monocular 3D Object Detection](../../AAAI2026/autonomous_driving/difficulty-aware_label-guided_denoising_for_monocular_3d_object_detection.md)

</div>

<!-- RELATED:END -->
