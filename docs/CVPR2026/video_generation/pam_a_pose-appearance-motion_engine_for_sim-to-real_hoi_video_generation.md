---
title: >-
  [Paper Note] PAM: A Pose-Appearance-Motion Engine for Sim-to-Real HOI Video Generation
description: >-
  [CVPR 2026][Video Generation][hand-object interaction] This paper proposes PAM — the first engine capable of generating realistic hand-object interaction (HOI) videos given only initial/target hand poses and object geometry. Through a three-stage decoupled architecture of pose generation, appearance generation, and motion generation, PAM achieves FVD 29.13 (vs. InterDyn 38.83) and MPJPE 19.37 mm (vs. CosHand 30.05 mm) on DexYCB. The generated synthetic data also effectively augments downstream hand pose estimation tasks.
tags:
  - CVPR 2026
  - Video Generation
  - hand-object interaction
  - sim-to-real
  - controllable video generation
  - diffusion models
  - data augmentation
date: 2026-05-08
content_hash: 75fae327c9f754a6
---

# PAM: A Pose-Appearance-Motion Engine for Sim-to-Real HOI Video Generation

**Conference**: CVPR 2026
**arXiv**: [2603.22193](https://arxiv.org/abs/2603.22193)
**Code**: [https://gasaiyu.github.io/PAM.github.io/](https://gasaiyu.github.io/PAM.github.io/)
**Area**: 3D Vision / Diffusion Models / Video Generation
**Keywords**: hand-object interaction, sim-to-real, controllable video generation, diffusion models, data augmentation

## TL;DR
This paper proposes PAM — the first engine capable of generating realistic hand-object interaction (HOI) videos given only initial/target hand poses and object geometry. Through a three-stage decoupled architecture of pose generation, appearance generation, and motion generation, PAM achieves FVD 29.13 (vs. InterDyn 38.83) and MPJPE 19.37 mm (vs. CosHand 30.05 mm) on DexYCB. The generated synthetic data also effectively augments downstream hand pose estimation tasks.

## Background & Motivation

1. **State of the Field**: Reconstruction and synthesis of hand-object interactions (HOI) are increasingly important for embodied AI and AR/VR. Data-driven approaches require large-scale annotated HOI datasets, but the high cost of manual annotation severely limits scalability.

2. **Limitations of Prior Work**: Existing HOI generation methods fall into three disconnected directions: (1) pose synthesis methods (e.g., GraspXL) predict MANO trajectories without generating pixels; (2) single-image generation methods (e.g., affordance-based approaches) synthesize appearance from masks or 2D cues but lack dynamics; (3) video generation methods (e.g., InterDyn, ManiVideo) require **complete pose sequences and a real reference frame** as input, precluding true sim-to-real deployment since simulators cannot provide realistic initial frames.

3. **Root Cause**: No unified framework simultaneously handles pose, appearance, and motion. In particular, video generation methods depend on real reference frames — a critical bottleneck in the sim-to-real pipeline, as simulators can only produce geometric and pose data without photorealistic initial frames.

4. **Paper Goals**: Design a minimally conditioned HOI video generation engine that requires only initial and target hand poses plus object geometry to produce temporally consistent, photorealistic HOI videos, thereby completing the sim-to-real pipeline.

5. **Starting Point**: Decompose the problem into three independently optimizable stages: (1) generate pose sequences via RL-based policy; (2) synthesize the initial-frame appearance using a controllable image diffusion model; (3) generate the full video using a controllable video diffusion model. Multi-modal conditions (depth maps, segmentation masks, hand keypoints) serve as triple constraints encoding geometry, semantics, and fine-grained detail.

6. **Core Idea**: Construct a sim-to-real HOI video generation engine that requires no real reference frame, via a three-stage decoupled architecture (pose → appearance → motion) with multi-modal conditional control.

## Method

### Overall Architecture
Given an initial MANO hand pose $\mathbf{h}_0$, object mesh $\mathbf{m}$, initial object pose $\mathbf{o}_0$, and target hand pose $\mathbf{h}_T$, the generative model $f_\theta: (\mathbf{h}_0, \mathbf{m}, \mathbf{o}_0, \mathbf{h}_T) \rightarrow \{I_t\}_{t=0}^T$ outputs a photorealistic video sequence. The three stages are: (1) **Pose Generation** — a pretrained GraspXL model generates the full hand-object trajectory $\{\mathbf{h}_t, \mathbf{o}_t\}$; (2) **Appearance Generation** — Flux + ControlNet synthesizes the initial frame $I_0$ from multi-modal conditions; (3) **Motion Generation** — CogVideoX + ControlNet generates the complete video from the initial frame and condition sequences.

### Key Designs

1. **Stage I: Pose Generation**

    - **Function**: Generate intermediate hand-object interaction dynamics from initial and target poses.
    - **Mechanism**: Directly employs the pretrained GraspXL model, which takes the initial hand pose $\mathbf{h}_0$, object pose $\mathbf{o}_0$, and object mesh $\mathbf{m}$ as inputs, and produces temporally coherent hand-object trajectories via a reinforcement learning policy. GraspXL's key advantage is strong generalization without requiring predefined reference grasps.
    - **Design Motivation**: RL-based methods can generate physically plausible interaction data within simulators, without being constrained by costly annotated datasets as in supervised learning approaches.

2. **Stage II: Appearance Generation**

    - **Function**: Synthesize a photorealistic RGB initial frame from geometric and pose conditions, replacing the unavailable real reference frame.
    - **Mechanism**: A Flux image diffusion model is fine-tuned with a ControlNet that accepts three condition types — depth map $D_0$, segmentation mask $S_0$, and hand keypoint map $K_0$ (each of shape $H \times W \times 3$). The conditions are VAE-encoded into $\frac{H}{8} \times \frac{W}{8} \times 16$ latents, concatenated channel-wise, and injected into the first two layers of the DiT blocks via zero-convolution initialization. Only the ControlNet parameters are updated during training.
    - **Design Motivation**: Depth maps capture global geometry and segmentation masks provide semantic information, but these two alone are insufficient for accurate hand detail synthesis (finger count, individual finger poses, etc.). Hand keypoints serve as a third explicit structural constraint, and all three modalities together enable synthesis that is both geometrically accurate and detail-rich.

3. **Stage III: Motion Generation**

    - **Function**: Render the generated initial frame and pose sequence into a complete video.
    - **Mechanism**: CogVideoX serves as the base video diffusion model, again paired with a ControlNet. Per-frame depth maps, segmentation masks, and keypoint maps are rendered as condition sequences, VAE-encoded into $\frac{T+1}{4} \times \frac{H}{8} \times \frac{W}{8} \times 16$ latents, concatenated channel-wise, and injected into 12 duplicate DiT blocks of CogVideoX (also initialized with zero-convolution). During training, each condition type is randomly masked with probability 0.2 to prevent over-reliance on any single modality.
    - **Design Motivation**: CogVideoX's temporal attention mechanism naturally ensures inter-frame consistency, outperforming CosHand's frame-by-frame generation. Using the same condition types as Stage II (depth + segmentation + keypoints) ensures stylistic consistency between the video and the initial frame. Random mask training improves robustness and generalization.

### Loss & Training
Both the appearance and motion stages are trained with the standard diffusion denoising objective. The appearance stage uses the DexYCB s0-split (6,400 training / 1,600 validation samples); the motion stage uses the same data. During ControlNet training, only ControlNet parameters are updated while the base model weights remain frozen.

## Key Experimental Results

### Main Results
Comparison on DexYCB:

| Method | FVD↓ | MF↑ | LPIPS↓ | SSIM↑ | PSNR↑ | MPJPE↓(mm) | Resolution |
|--------|------|-----|--------|-------|-------|------------|------------|
| CosHand | 58.51 | 0.591 | 0.139 | 0.767 | 23.20 | 30.05 | 256×256 |
| InterDyn | 38.83 | 0.680 | 0.119 | 0.848 | 24.86 | - | 256×384 |
| **PAM(all)** | **29.13** | **0.712** | **0.069** | **0.914** | **30.17** | **19.37** | **480×720** |

On OAKINK2: FVD decreases from 68.76 (CosHand) to 46.31, and MPJPE from 14.49 to 7.01.

### Ablation Study (Condition Combinations, DexYCB)

| Condition | FVD↓ | MF↑ | MPJPE↓(mm) | Note |
|-----------|------|-----|------------|------|
| Seg only | 33.23 | 0.695 | 21.14 | Segmentation mask only |
| Depth only | 30.00 | 0.703 | 23.16 | Depth map only |
| Hand only | 33.41 | 0.713 | 20.70 | Keypoints only; lowest MPJPE but worse on other metrics |
| Depth+Seg | 29.32 | 0.712 | 22.51 | Geometry + semantics |
| All three | **29.13** | **0.712** | **19.37** | Three conditions optimal |

### Key Findings
- The three-condition combination achieves across-the-board optimality — keypoints alone yield the lowest MPJPE (explicit pose constraint) but degrade appearance quality, while depth and segmentation provide global context; the three modalities are mutually complementary.
- Downstream validation: augmenting training with 3,400 generated videos (207K frames) shows that 50% real data + all synthetic data matches the 100% real data baseline (PA-MPJPE: 5.5 vs. 5.5), demonstrating the practical value of synthetic data.
- Zero-shot cross-dataset transfer: a model trained on DexYCB directly produces reasonable results on OAKINK2 (bimanual interactions), benefiting from the generalization capacity of the pretrained video diffusion model.
- Compared to CosHand (conditioned only on hand masks), the multi-condition approach combined with a video diffusion base model yields comprehensive improvements across all metrics.

## Highlights & Insights
- **Decoupled three-stage design**: Pose, appearance, and motion are optimized separately, leveraging the strengths of each component (RL for physically plausible poses, diffusion models for photorealistic appearance and dynamics), avoiding the difficulties of end-to-end training. This decoupling paradigm is transferable to other sim-to-real generation tasks.
- **No real reference frame required**: Prior methods all depend on ground-truth initial frames. The appearance generation stage eliminates this requirement, realizing for the first time a complete simulator → real video pipeline.
- **Downstream value of synthetic data**: Beyond evaluating generation quality, the paper validates the utility of generated videos as training data for downstream task augmentation — 50% real data + synthetic achieves the 100% real data baseline.

## Limitations & Future Work
- Generation quality depends on GraspXL's pose outputs; physically implausible poses will degrade downstream video quality.
- Errors may accumulate across the three serial stages (inaccurate pose → inaccurate condition rendering → degraded video quality).
- The current framework handles only single-hand, single-object scenarios; bimanual or multi-object interactions require framework extensions.
- Appearance diversity in Stage II is bounded by the capabilities of Flux and ControlNet; complex backgrounds and lighting variations may not be adequately handled.
- Future work may explore unifying the three stages into an end-to-end model to reduce information loss across intermediate steps.

## Related Work & Insights
- **vs. InterDyn**: InterDyn applies ControlNet with hand mask sequences for video generation but underutilizes conditional information and requires a real initial frame. PAM's multi-modal conditioning and reference-frame-free design reduce FVD from 38.83 to 29.13.
- **vs. CosHand**: CosHand relies solely on hand mask conditioning and lacks explicit temporal modeling; frame-by-frame generation leads to inter-frame inconsistency. PAM employs a video diffusion base model with temporal attention to ensure coherence.
- **vs. ManiVideo**: ManiVideo introduces occlusion-aware representations but requires human appearance data unavailable from simulators, making it unsuitable for sim-to-real deployment.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The three-stage decoupled framework with multi-modal conditional control is innovative in its combination strategy and the reference-frame-free formulation, even though individual components are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on DexYCB and OAKINK2 with condition ablations, downstream task validation, and zero-shot transfer — a highly rigorous experimental suite.
- **Writing Quality**: ⭐⭐⭐⭐ Illustrations are clear and problem formulation is well-defined.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical value for synthetic data generation in embodied AI; demonstrates the feasibility of sim-to-real HOI video generation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)
- [\[ICLR 2026\] MoSA: Motion-Coherent Human Video Generation via Structure-Appearance Decoupling](../../ICLR2026/video_generation/mosa_motion-coherent_human_video_generation_via_structure-appearance_decoupling.md)
- [\[CVPR 2026\] StreamDiT: Real-Time Streaming Text-to-Video Generation](streamdit_real-time_streaming_text-to-video_generation.md)
- [\[ICLR 2026\] MotionStream: Real-Time Video Generation with Interactive Motion Controls](../../ICLR2026/video_generation/motionstream_real-time_video_generation_with_interactive_motion_controls.md)
- [\[CVPR 2026\] PerformRecast: Expression and Head Pose Disentanglement for Portrait Video Editing](performrecast_expression_and_head_pose_disentanglement_for_portrait_video_editin.md)

<!-- RELATED:END -->
