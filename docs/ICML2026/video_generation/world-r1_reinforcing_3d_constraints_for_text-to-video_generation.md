---
title: >-
  [Paper Note] World-R1: Reinforcing 3D Constraints for Text-to-Video Generation
description: >-
  [ICML 2026][Video Generation][Text-to-Video Generation] World-R1 formulates the 3D consistency problem of text-to-video models as reinforcement learning (RL) post-training. By employing implicit camera conditioning and 3…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Text-to-Video Generation"
  - "3D Consistency"
  - "Reinforcement Learning"
  - "Flow-GRPO"
  - "Camera Control"
date: 2026-05-08
content_hash: 48da9d6c620a6622
---

# World-R1: Reinforcing 3D Constraints for Text-to-Video Generation

**Conference**: ICML 2026  
**arXiv**: [2604.24764](https://arxiv.org/abs/2406.24764)  
**Code**: None  
**Area**: Video Generation / World Models  
**Keywords**: Text-to-Video Generation, 3D Consistency, Reinforcement Learning, Flow-GRPO, Camera Control  

## TL;DR
World-R1 formulates the 3D consistency problem of text-to-video models as reinforcement learning (RL) post-training. By employing implicit camera conditioning and 3D-aware rewards for Flow-GRPO alignment on video foundation models like Wan 2.1, it significantly reduces geometric hallucinations while maintaining general video generation quality without altering the model architecture or inference pipeline.

## Background & Motivation
**Background**: Large-scale video generation models are capable of generating high-fidelity short videos and are increasingly viewed as foundations for world models. However, their training objectives primarily match visual distributions in image/video space, lacking explicit 3D geometric constraints. This issue is less apparent for static shots or minor movements, but once prompts require large camera motions like orbiting an object, moving through corridors, or zooming into buildings, object shapes, wall structures, and scene layouts tend to drift.

**Limitations of Prior Work**: Existing 3D-aware video generation often incorporates 3D modules, point clouds/3DGS constraints, or auxiliary camera-control networks during inference. While these methods improve consistency, they introduce architectural changes, extra inputs, expensive inference, and task range limitations, with many favoring image-to-video over pure text-to-video. Conversely, training solely on more video data does not guarantee that the model will internalize rigid geometric laws.

**Key Challenge**: Video foundation models may have already learned certain implicit 3D knowledge during pre-training, but standard generation objectives do not force them to utilize this knowledge during large viewpoint changes. To transform a model into a generator that behaves more like a world simulator, geometric feedback is necessary; however, if the feedback is too rigid, it may suppress dynamic objects and visual diversity.

**Goal**: The authors aim to internalize 3D geometric constraints into text-to-video foundation models without introducing explicit 3D reasoning modules, relying on large-scale 3D supervised data, or changing the inference pipeline. The objectives include better camera trajectory following, object persistence, and 3D reconstruction consistency, all while preserving general video quality on VBench.

**Key Insight**: The paper adopts an analysis-by-synthesis reward design. After generating a video, a 3D foundation model is first used to lift the video into 3D Gaussian Splatting (3DGS) and a camera trajectory. Subsequently, views are rendered from new perspectives to compare reconstruction quality, check trajectory deviations, and use a VLM to evaluate the structural reliability of the meta-view. In this way, the model does not learn directly from 3D annotations but understands through rewards which videos are 3D-consistent.

**Core Idea**: Use Flow-GRPO to combine 3DGS reconstruction, meta-view semantic evaluation, trajectory alignment, and general visual quality into a reward system. This RL alignment on existing T2V models makes geometric consistency an inherent generation preference of the model rather than an external hard constraint at inference time.

## Method

### Overall Architecture
The base model for World-R1 is Wan 2.1 T2V. Given a text prompt, the system identifies camera motion keywords (e.g., push in, pan left, orbit left) and generates a corresponding 3D camera extrinsic trajectory. It then projects this trajectory into 2D optical flow and injects camera motion priors into the initial latent noise using a Go-with-the-Flow style noise wrapping. The video foundation model generates candidate videos under this latent condition.

After candidate videos are generated, World-R1 calculates a composite reward. The 3D-aware reward consists of meta-view structural evaluation, 3DGS reconstruction fidelity, and camera trajectory alignment. The general generation reward uses HPSv3 to evaluate the general aesthetic and visual quality of the initial frames. During training, Flow-GRPO-Fast is used, treating the video sampling process as a stochastic policy rollout and updating the model using the advantage normalized by intra-group rewards.

### Key Designs
1. **Implicit Camera Conditioning: Embedding Motion Priors into Initial Noise**:
	- **Function**: Allows the T2V model to perceive user-requested camera trajectories without adding a camera-control module.
	- **Mechanism**: A sequence of camera extrinsics is generated from movement keywords in the prompt, and then 3D motion is projected into optical flow between adjacent frames using planar homography. Since directly warping diffusion noise destroys the standard normal distribution, the method uses discrete noise transport: summing noises that fall into the same target grid and normalizing by the number of incidents keeps camera-induced spatial structures in the initial noise while maintaining unit variance.
	- **Design Motivation**: Explicit control networks increase structural and training costs; pure text prompts make it difficult for foundation models to stably execute complex camera motions. Noise wrapping provides a better initial inductive bias for RL, making it easier for the model to learn trajectory following.

2. **3D-aware analysis-by-synthesis reward**:
	- **Function**: Converts the "appearance" of a video into optimizable feedback regarding its "3D self-consistency."
	- **Mechanism**: Depth Anything 3 is used to recover the 3DGS representation $\Phi_{GS}$ and estimate the camera trajectory $\hat{E}$ from the generated video. The 3D reward is formulated as $R_{3D} = S_{meta} + S_{recon} + S_{traj}$: $S_{meta}$ renders the 3DGS from an offset perspective for Qwen3-VL to judge structural reliability, $S_{recon}$ measures the consistency between the original video and 3DGS re-rendering using $1 - \text{LPIPS}$, and $S_{traj}$ penalizes translation/rotation deviations between target and estimated trajectories.
	- **Design Motivation**: Examining original video frames alone can mask 3D errors like "flatness," "floaters," or "texture stretching." Meta-view and reconstruction consistency expose these hidden defects, while the trajectory term prevents the model from tricking reconstruction metrics with static videos.

3. **Composite Reward and Periodic Decoupled Training to Prevent Over-Rigidity**:
	- **Function**: Retains dynamic scenes and general visual quality while reinforcing geometric consistency.
	- **Mechanism**: The total reward is $R(x,c) = R_{3D}(x,E,c) + \lambda_{gen}R_{gen}(x,c)$, where $R_{gen}$ evaluates visual preference using HPSv3. The training data also includes approximately 500 highly dynamic scene prompts; every 100 steps, the 3D-aware reward is temporarily disabled, and the model is trained on the dynamic subset using only the general reward.
	- **Design Motivation**: If only easily reconstructible rigid scenes are rewarded, the model might engage in reward hacking, generating videos that are too static or stiff. Periodic relaxation of 3D constraints acts as regularization, allowing the model to learn geometric consistency for static environments while permitting non-rigid motions such as water, fire, or crowds.

### Loss & Training
World-R1 utilizes Flow-GRPO-Fast for online RL post-training. The deterministic ODE sampling of flow matching is rewritten as a noise-augmented reverse-time SDE to form an explorable policy. A set of videos is sampled for each prompt, advantages are normalized based on intra-group reward means and standard deviations, and the model is updated using a clipped objective similar to PPO/GRPO with a KL constraint. Two versions are trained: World-R1-Small based on Wan2.1-T2V-1.3B using 48 H200 GPUs, and World-R1-Large based on Wan2.1-T2V-14B using 96 H200 GPUs. Training resolution is 832×480, with a GRPO group size of 8 and 48 parallel groups.

## Key Experimental Results

### Main Results
The main results are categorized into VBench for general video quality and 3DGS reconstruction for geometric consistency. World-R1 does not sacrifice VBench performance; instead, it surpasses the base models in aesthetics, imaging, and subject consistency.

| Method | Aesthetic Quality | Imaging Quality | Motion Smoothness | Subject Consistency | Background Consistency |
|------|-------------------|-----------------|-------------------|---------------------|------------------------|
| CogVideoX-1.5-5B | 62.07 | 65.34 | 98.15 | 96.56 | 96.81 |
| Wan2.1-T2V-1.3B | 62.43 | 66.51 | 97.44 | 96.34 | 97.29 |
| ReCamMaster | 42.70 | 53.97 | 99.28 | 92.05 | 93.83 |
| World-R1-Small | 65.74 | 67.53 | 98.55 | 97.58 | 96.67 |

| Method | PSNR | SSIM | LPIPS | Description |
|------|------|------|-------|------|
| CogVideoX-1.5-5B | 24.44 | 0.783 | 0.242 | Strong video baseline |
| Wan2.2-T2V-14B | 23.47 | 0.779 | 0.253 | Larger Wan series baseline |
| Wan2.1-T2V-14B | 19.76 | 0.629 | 0.405 | Base for World-R1-Large |
| Wan2.1-T2V-1.3B | 17.40 | 0.550 | 0.467 | Base for World-R1-Small |
| World-R1-Small | 27.63 | 0.858 | 0.201 | +10.23 dB PSNR vs. 1.3B base |
| World-R1-Large | 27.67 | 0.865 | 0.162 | +7.91 dB PSNR vs. 14B base |

### Ablation Study
Ablations of reward components and training strategies demonstrate that each component is not redundant but rather balances geometric consistency, trajectory control, and visual quality.

| Reward Component Ablation | PSNR | SSIM | LPIPS | VBench AVG | Conclusion |
|-----------------|------|------|-------|------------|------|
| Full pipeline | 27.63 | 0.858 | 0.201 | 85.21 | Best balance of geometry and quality |
| w/o meta-view score | 26.91 | 0.841 | 0.218 | 83.67 | Hidden structural defects harder to penalize |
| w/o reconstruction score| 25.14 | 0.798 | 0.271 | 84.35 | 3D reconstruction consistency drops significantly |
| w/o trajectory score | 26.27 | 0.829 | 0.237 | 84.53 | Camera trajectory following weakens |

| Training/Condition Ablation | PSNR | SSIM | LPIPS | VBench AVG | Key Impact |
|----------------|------|------|-------|------------|----------|
| Full | 27.63 | 0.858 | 0.201 | 85.21 | Most stable overall |
| w/o noise wrapping | 24.46 | 0.745 | 0.298 | 76.39 | Loss of trajectory inductive bias, poor convergence |
| w/o periodic decoupled training | 27.89 | 0.898 | 0.192 | 82.64 | Higher recon but lower quality (too rigid) |
| w/o 3D-aware reward | 18.93 | 0.502 | 0.496 | 84.96 | Maintains quality but geometric constraints fail |
| w/o general reward | 27.57 | 0.849 | 0.206 | 83.44 | Geometry remains strong, perceptual quality drops |

| Analysis Item | Result | Implication |
|--------|------|------|
| User Study Geometric Consistency Win Rate | 92% | Human preference for World-R1 structural stability |
| User Study Camera Control Win Rate | 76% | Complex trajectory following superior to Wan 2.1 |
| User Study Overall Preference | 86% | Geometric constraints do not harm overall look |
| Auto 3D metric vs Human Preference Agreement | 91.17% | Reconstruction metrics align with subjective judgment |
| MVCS small backbone | 0.974 → 0.989 | Multi-view consistency improves without 3DGS |
| MVCS large backbone | 0.963 → 0.993 | Large models benefit similarly |
| 121-frame long-video PSNR | 18.32 → 26.32 | 3D alignment generalizes to longer videos |

### Key Findings
- 3D consistency is the strongest result: World-R1-Small improved from a PSNR of 17.40 on Wan2.1-1.3B to 27.63, and World-R1-Large improved from 19.76 on Wan2.1-14B to 27.67, with significantly lower LPIPS.
- General video quality was not crushed by 3D constraints. In VBench, World-R1-Small outperformed Wan2.1-1.3B in Aesthetics, Imaging, Motion Smoothness, and Subject Consistency.
- Reward ablation shows the 3D reward is a necessary condition for geometric improvement; without it, PSNR is only 18.93, nearly regressing to the baseline. Removing the general reward preserves geometry but lowers VBench scores, illustrating the necessity of a balanced composite reward.
- Periodic decoupled training is critical to prevent reward hacking. Without it, reconstruction metrics are slightly higher, but VBench drops from 85.21 to 82.64, indicating the model may become overly static or rigid.

## Highlights & Insights
- The paper avoids making 3D consistency an external module during inference; instead, it internalizes it as a model preference through RL. Thus, once training is complete, the inference pipeline remains as simple as a standard T2V model.
- The reward design is comprehensive: meta-view checks for hidden geometric issues, reconstruction checks self-consistency, trajectory checks control, and the general reward checks visual quality. Each metric blocks a potential path for shortcutting.
- Using pure text data for post-training is compelling. It allows the model to learn geometric laws from a massive combination of scene descriptions and camera movements without relying on expensive ground-truth 3D video annotations.
- The paper acknowledges that 3D constraints can suppress dynamic content and addresses this with periodic decoupled training. This detail makes the method more like a practical world generator rather than one that only generates static, reconstructible scenes.

## Limitations & Future Work
- Training costs are high. World-R1-Small requires 48 H200s, while Large requires 96, and online RL involves repeated video generation and 3D/reward evaluation, which is more expensive than standard SFT.
- The method is limited by the upper bounds of the base video model. Complex interactions between multiple objects, hand details, long-term non-rigid motion, and extremely long-horizon scenes may still inherit defects from the Wan base.
- The 3D reward relies on external evaluators like Depth Anything 3, 3DGS, Qwen3-VL, and HPSv3; bias in these evaluators in certain scenes could be learned by the RL policy.
- Currently, camera trajectories come from keywords and preset motion primitives. Future work could support free-form trajectory inputs, continuous control signals, or interfaces with real trajectories from robotics/autonomous driving simulators.

## Related Work & Insights
- **vs CameraCtrl / ReCamMaster**: These methods control trajectories through explicit camera-control modules or conditional inputs. World-R1 achieves control via latent noise wrapping and RL rewards without adding inference modules.
- **vs 3D-aware video generation**: Explicit 3D representations or 3D decoders often require architectural modifications and high inference costs. World-R1 uses a 3D model as a training critic to distill constraints into the video generator.
- **vs Flow-GRPO**: Flow-GRPO provides an RL framework for visual generation. World-R1's contribution lies in designing usable rewards for 3D consistency and solving reward hacking in video geometric constraints.
- **vs Go-with-the-Flow**: Go-with-the-Flow uses noise wrapping for camera motion priors. World-R1 uses it as the conditional basis for RL post-training to learn geometric consistency via rewards.
- **Insight**: World modeling for generative models does not necessarily require collecting large-scale 3D annotations. Strong 3D foundation models and VLMs can be used as training discriminators, transferring physical constraints to the generator via RL or preference optimization.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Treating 3D consistency as an RL post-training reward is quite inspiring, combining existing 3D/VLM/RL tools with a clear objective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Main experiments, user studies, MVCS, long video analysis, reward ablation, and component ablations are all comprehensive.
- Writing Quality: ⭐⭐⭐⭐☆ The framework description is clear, and the appendix provides sufficient supplementation; the main text relies on the appendix for some specific ablation values.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for the transition of T2V toward world models, particularly for simulations, robotics, and autonomous driving video generation requiring camera motion and geometric stability.

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] 3D4D: An Interactive Editable 4D World Model via 3D Video Generation](../../AAAI2026/video_generation/3d4d_an_interactive_editable_4d_world_model_via_3d_video_generation.md)
- [\[CVPR 2025\] GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control](../../CVPR2025/video_generation/gen3c_3d-informed_world-consistent_video_generation_with_precise_camera_control.md)
- [\[ICML 2026\] OLAF-World: Orienting Latent Actions for Video World Modeling](olaf-world_orienting_latent_actions_for_video_world_modeling.md)
- [\[ICML 2026\] CamGeo: Sparse Camera-Conditioned Image-to-Video Generation with 3D Geometry Prior](camgeo_sparse_camera-conditioned_image-to-video_generation_with_3d_geometry_prio.md)
- [\[CVPR 2025\] World-Consistent Video Diffusion with Explicit 3D Modeling](../../CVPR2025/video_generation/world-consistent_video_diffusion_with_explicit_3d_modeling.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] 3D4D: An Interactive Editable 4D World Model via 3D Video Generation](../../AAAI2026/video_generation/3d4d_an_interactive_editable_4d_world_model_via_3d_video_generation.md)
- [\[ICML 2026\] CamGeo: Sparse Camera-Conditioned Image-to-Video Generation with 3D Geometry Prior](camgeo_sparse_camera-conditioned_image-to-video_generation_with_3d_geometry_prio.md)
- [\[ICML 2026\] OLAF-World: Orienting Latent Actions for Video World Modeling](olaf-world_orienting_latent_actions_for_video_world_modeling.md)
- [\[ICML 2026\] T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation](t2av-compass_towards_unified_evaluation_for_text-to-audio-video_generation.md)
- [\[ICML 2026\] LocoT2V-Bench: Benchmarking Long-form and Complex Text-to-Video Generation](locot2v-bench_benchmarking_long-form_and_complex_text-to-video_generation.md)

</div>

<!-- RELATED:END -->
