---
title: >-
  [Paper Note] DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving
description: >-
  [CVPR 2025][Autonomous Driving][Diffusion Models] This paper proposes DiffusionDrive, which successfully applies diffusion models to real-time multimodal trajectory planning in end-to-end autonomous driving for the first time. By introducing a truncated diffusion policy (reducing denoising steps from 20 to 2) and a cascade diffusion decoder, it achieves a record-breaking 88.1 PDMS on the NAVSIM dataset while maintaining a real-time speed of 45 FPS.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Diffusion Models"
  - "End-to-End Autonomous Driving"
  - "Multimodal Trajectory Planning"
  - "Truncated Diffusion Strategy"
  - "Real-Time Planning"
date: 2026-05-08
content_hash: f7fa88b6c16c2062
---

# DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving

**Conference**: CVPR 2025  
**arXiv**: [2411.15139](https://arxiv.org/abs/2411.15139)  
**Code**: [hustvl/DiffusionDrive](https://github.com/hustvl/DiffusionDrive)  
**Area**: Autonomous Driving  
**Keywords**: Diffusion Models, End-to-End Autonomous Driving, Multimodal Trajectory Planning, Truncated Diffusion Strategy, Real-Time Planning

## TL;DR
This paper proposes DiffusionDrive, which successfully applies diffusion models to real-time multimodal trajectory planning in end-to-end autonomous driving for the first time. By introducing a truncated diffusion policy (reducing denoising steps from 20 to 2) and a cascade diffusion decoder, it achieves a record-breaking 88.1 PDMS on the NAVSIM dataset while maintaining a real-time speed of 45 FPS.

## Background & Motivation

End-to-end autonomous driving has made significant progress in recent years. Mainstream methods (e.g., Transfuser, UniAD, VAD) typically regress a single trajectory from an ego-query. However, this paradigm ignores the inherent uncertainty and multimodal nature of driving behavior. VADv2 introduces a massive vocabulary of fixed anchors (4096-8192 anchors) to discretize the continuous action space, but it is limited by the quantity and quality of anchors, struggles to cover out-of-vocabulary scenarios, and incurs enormous computational overhead.

Diffusion models have demonstrated powerful capabilities in modeling multimodal action distributions for robotic policy learning. Nevertheless, directly applying vanilla diffusion policies to autonomous driving faces two major challenges: (1) requiring 20 denoising steps, which reduces FPS from 60 to 7, failing to meet real-time requirements; (2) trajectories sampled from different Gaussian noises severely overlap, leading to mode collapse.

The Key Insight is: Unlike denoising from random Gaussian noise, human driving follows established driving patterns while dynamically adjusting based on real-time traffic conditions. Therefore, prior driving patterns can be embedded into the diffusion policy, allowing denoising to start from an anchored Gaussian distribution (instead of a standard Gaussian distribution), thereby drastically reducing the denoising steps.

## Method

### Overall Architecture
DiffusionDrive consists of a perception module and a diffusion decoder. The perception module can integrate various existing end-to-end perception architectures (such as UniAD, VAD, Transfuser) and receive inputs from different sensors (camera, LiDAR). The diffusion decoder samples noisy trajectories from anchored Gaussian distributions and progressively denoises them through enhanced scene context interactions to generate the final multimodal planned trajectories.

### Key Designs

1. **Truncated Diffusion Policy**:

    - **Mechanism**: Instead of starting from pure Gaussian noise, denoising starts from an anchored Gaussian distribution.
    - A small number of anchors (only 20, a 400x reduction compared to VADv2's 8192) are obtained by applying K-Means clustering to the training trajectories. A small amount of Gaussian noise is then injected around these anchors to form the anchored Gaussian distributions.
    - During training, the diffusion schedule is truncated (50/1000), performing diffusion only near the anchors.
    - During inference, the process starts from the anchored Gaussian distributions, requiring only 2 denoising steps (a 10x reduction compared to 20 steps in vanilla diffusion).
    - Each anchor simultaneously predicts a classification score and a denoised trajectory, and the trajectory with the highest score is ultimately selected as the output.
    - Inference flexibility: The number of sampled trajectories during inference can be dynamically adjusted and does not need to equal the number of anchors used during training.

2. **Cascade Diffusion Decoder**:

    - Based on a Transformer architecture, replacing the UNet.
    - Interacts with BEV and perspective view features via Deformable Spatial Cross-attention.
    - Conducts cross-attention with agent/map queries output by the perception module.
    - Uses a Timestep Modulation layer to encode diffusion timestamp information.
    - Cascade mechanism: Multi-layer decoders are stacked to step-by-step refine trajectory reconstruction.
    - Parameters are shared across denoising timesteps, reducing parameter size (from 102M to 60M).

3. **Training Objective**:

    - Trajectory reconstruction loss: L1 reconstruction loss, calculated only for the anchor closest to the ground truth (positive sample).
    - Classification loss: BCE loss to distinguish positive and negative samples.
    - Total Loss: 
      $$\text{Total Loss} = \sum [y_k \cdot L_{\text{rec}} + \lambda \cdot \text{BCE}]$$

### Loss & Training
- For each training sample, the anchor closest to the ground-truth trajectory is identified as the positive sample.
- The positive sample computes the trajectory L1 reconstruction loss, and all samples compute the BCE classification loss.
- Uses the AdamW optimizer with a learning rate of $6 \times 10^{-4}$, training for 100 epochs on 8 RTX 4090 GPUs.
- Total batch size is 512, with no test-time augmentation.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| NAVSIM navtest | PDMS | 88.1 | 86.5 (Hydra-MDP-W-EP) | +1.6 |
| NAVSIM navtest | EP | 82.2 | 78.7 (Hydra-MDP-W-EP) | +3.5 |
| NAVSIM navtest | DAC | 96.2 | 96.0 (Hydra-MDP-W-EP) | +0.2 |
| nuScenes | Avg L2 (m) | 0.57 | 0.61 (SparseDrive) | -6.6% |
| nuScenes | Avg Collision (%) | 0.08 | 0.08 (SparseDrive) | Flat |

### Ablation Study

| Configuration | Key Metric (PDMS) | Description |
|------|---------|------|
| Transfuser (baseline) | 84.0 | Single-mode regression MLP |
| TransfuserDP (vanilla diffusion) | 84.6 (+0.6) | 20-step denoising, FPS=7, mode diversity 11% |
| TransfuserTD (truncated diffusion) | 85.7 (+1.7) | 2-step denoising, FPS=27, mode diversity 70% |
| DiffusionDrive (Full) | 88.1 (+4.1) | 2-step denoising, FPS=45, mode diversity 74% |
| W/o spatial cross-attention | 55.1 | Performance severely degrades, indicating spatial interaction scaling is crucial |
| Only 1-step denoising | 87.9 | Only 1 step already achieves good performance |
| 10 sampling noises | 84.9 | Small number of samplings yields decent results |
| 40 sampling noises | 88.2 | More samplings cover more potential action spaces |

### Key Findings
- The truncated diffusion policy simultaneously resolves the dual issues of mode collapse and computational overhead: mode diversity improves from 11% to 70%, while denoising steps decrease from 20 to 2.
- The diffusion decoder has fewer parameters than UNet (60M vs. 102M) but achieves better performance (88.1 vs. 85.7 PDMS).
- Spatial cross-attention is the most critical design element; removing it causes the PDMS to plunge from 87.1 to 55.1.
- DiffusionDrive can generate high-quality multimodal trajectories (e.g., lane changes, obstacle avoidance), which is impossible with single-modal methods.

## Highlights & Insights
- Successfully applies diffusion models to real-time end-to-end autonomous driving planning for the first time, overcoming the fundamental bottleneck of "diffusion models being too slow."
- The design intuition of the truncated diffusion policy is highly natural: human driving does not start from randomness but is fine-tuned based on established patterns.
- Surpassing the VADv2-series methods (which use 8192 anchors) with only 20 anchors demonstrates that generative modeling is much more efficient than discretization.
- Inference flexibility: The number of training anchors is decoupled from the number of inference samples, permitting dynamic scaling based on computational resources.
- Real-time performance: 45 FPS on an RTX 4090, far superior to the 7 FPS of the vanilla diffusion policy.

## Limitations & Future Work
- Primarily evaluated in non-reactive simulations and has not been validated in real-world closed-loop driving.
- Anchors are statically clustered via K-Means; future work could explore adaptive anchor generation.
- The diffusion decoder currently only interacts with BEV features (under the Transfuser setup), which can be extended to richer scene representations.
- Zero-shot generalization capabilities (e.g., to new cities, new weather conditions) have not been fully validated.
- Experiments on nuScenes indicate limited improvements in simple scenarios; the method's advantages are more pronounced in complex scenarios.

## Related Work & Insights
- Relation to Diffusion Policy (robotics domain): The proposed truncated diffusion policy is a significant improvement over vanilla diffusion, incorporating domain-specific driving priors.
- Comparison with VADv2/Hydra-MDP (vocabulary-sampling paradigm): This work demonstrates that continuous generative modeling is more efficient than discretization.
- TDPM (truncated denoising in image generation) inspired the truncation concept, but this work introduces explicit driving priors (anchors) rather than implicit intermediate distributions.
- The core idea of this approach (substituting pure noise starting points with domain priors) can be generalized to other robotic tasks requiring real-time decision-making.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The design intuition of the truncated diffusion policy is ingenious, successfully bringing diffusion models to real-time autonomous driving for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative and qualitative results on NAVSIM and nuScenes are comprehensive, but real-world deployment results are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ The progressive explanations from Transfuser to DiffusionDrive are logically coherent, accompanied by polished figures.
- Value: ⭐⭐⭐⭐⭐ Resolves the main bottleneck of diffusion models in autonomous driving, demonstrating both theoretical innovation and practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)
- [\[CVPR 2025\] SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving](solve_synergy_of_language-vision_and_end-to-end_networks_for_autonomous_driving.md)
- [\[ICCV 2025\] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](../../ICCV2025/autonomous_driving/world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [\[ICLR 2026\] ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](../../ICLR2026/autonomous_driving/resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)
- [\[ICCV 2025\] Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](../../ICCV2025/autonomous_driving/unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)

</div>

<!-- RELATED:END -->
