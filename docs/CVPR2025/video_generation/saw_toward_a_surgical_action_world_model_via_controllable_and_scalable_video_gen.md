---
title: >-
  [Paper Note] SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation
description: >-
  [CVPR2025][Video Generation][surgical video generation] Proposes SAW (Surgical Action World), which drives a video diffusion model using four lightweight conditioning signals (language prompts, reference frames, tissue functional maps, and tool trajectories) to achieve controllable and scalable surgical action video generation for rare action augmentation and surgical simulation.
tags:
  - "CVPR2025"
  - "Video Generation"
  - "surgical video generation"
  - "world model"
  - "video diffusion"
  - "tool-tissue interaction"
  - "action recognition"
  - "surgical simulation"
date: 2026-05-08
content_hash: fd8b95d9adecc68e
---

# SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation

**Conference**: CVPR2025  
**arXiv**: [2603.13024](https://arxiv.org/abs/2603.13024)  
**Code**: To be confirmed  
**Area**: Video Generation  
**Keywords**: surgical video generation, world model, video diffusion, tool-tissue interaction, action recognition, surgical simulation

## TL;DR

Proposes SAW (Surgical Action World), which drives a video diffusion model using four lightweight conditioning signals (language prompts, reference frames, tissue functional maps, and tool trajectories) to achieve controllable and scalable surgical action video generation for rare action augmentation and surgical simulation.

## Background & Motivation

- **Data Bottleneck in Surgical AI**: Surgical perception models suffer heavily from data scarcity, particularly for rare yet clinically critical surgical actions (e.g., clipping, cutting), which incur extremely high annotation and collection costs.
- **Sim-to-Real Gap in Surgical Simulation**: Physics-based simulators struggle to accurately model complex tool-tissue interactions and tissue deformations.
- **Limitations of Prior Work in Surgical Video Generation**:
    - HieraSurg relies on expensive frame-by-frame segmentation annotations.
    - SG2VID depends on complex intermediate representations like spatio-temporal scene graphs.
    - SurgSora has a short inference window (only 21 frames) and poor temporal consistency.
- **Goal**: Construct a surgical action world model that leverages lightweight conditioning signals to achieve controllability and scalability during inference.

## Method

### Video Diffusion Backbone
- Utilizes LTX-Video as the backbone, which is a Transformer-based latent space diffusion model natively supporting multimodal conditioning (text, video, and images).
- Employs a VAE for spatio-temporal downsampling, and the diffusion transformer is trained via flow-matching.
- Fine-tuned using IC-LoRA (In-Context Low Rank Adaptation).

### Four Lightweight Conditioning Signals
1. **Language Prompt ($z^a$)**: Templated text encoding the tool-action context, e.g., "A robotic da Vinci {tool} performs {action} during a cholecystectomy..."
2. **Reference Frame ($z^f$)**: The first frame anchors the scene appearance, ensuring anatomical consistency.
3. **Tissue Functional Map ($z^\gamma$)**: 2D binary maps annotating tool-tissue interaction regions, encoded via VAE.
4. **Tool Trajectory ($z^p$)**: A sequence of 2D coordinates of the tool tip over time, where each coordinate is encoded as a circle with a fixed radius in each frame, and tool categories are encoded via the R/G channels.

### Depth Consistency Loss ($L_{DC}$)
- Since all spatial conditions are 2D, depth control (the Z-dimension) is learned implicitly.
- During training, depth maps are generated using Depth Anything V2; cross-attention layers and a projection head are introduced to reconstruct masked depth tokens from denoised RGB tokens.
- A Smooth L1 loss is used to enforce geometric consistency, though no depth input is required during inference.

### Dataset Construction
- Curated 12,044 clips cropped from 101 surgical videos: 21 YouTube videos (2,760 clips) + 80 videos from public datasets (HeiChole 1,794 + Cholec80 6,042 + SurgVU 4,278 + CRCD 764).
- Each clip is annotated with video-level action categories (clipping/grasping/cutting/dissecting), tool types (grasper/hook/clipper/scissors), tissue functional zones, and frame-by-frame tool tip positions.
- Training set: 11,502 clips (dissecting heavily dominates with 8,822 clips); testing set: 542 clips.
- Standardized to 81 frames @ 25fps, with a resolution of $1024 \times 576$, removing black borders and text overlays.

## Key Experimental Results

### Training Setup
- A single NVIDIA A100 GPU, fine-tuning via IC-LoRA for 7,500 steps, with $\alpha=128$, $\text{lr}=2 \times 10^{-4}$, AdamW optimizer, and bfloat16 mixed precision.
- Classifier-free guidance: the reference frame conditioning is dropped with a 20% probability during training; inference uses 50 denoising steps with a guidance scale of 3.5.

### Video Generation Quality

| Model | FVD↓ | CD-FVD↓ | SSIM↑ | PSNR↑ | LPIPS↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| WAN | 439.60 | 429.67 | 0.575 | 15.82 | 0.448 |
| SurgSora | 541.61 | 546.82 | 0.416 | 17.10 | 0.383 |
| **SAW** | **224.28** | **199.19** | **0.595** | **17.36** | 0.410 |

- CD-FVD is reduced by 63.6% compared to SurgSora (199.19 vs. 546.82), showing a substantial lead in temporal consistency.
- FVD is reduced by 29.8% compared to the second-best (LTX baseline of 319.37).

### Ablation Study
- Removing the reference frame: FVD surges to 1096.21, and CD-FVD rises to 338.75 (making it the most critical conditioning signal).
- Removing trajectories: CD-FVD increases to 344.79 (causing massive degradation in temporal consistency).
- Removing the depth consistency loss: CD-FVD rises to 207.59 (demonstrating its contribution to temporal consistency).
- Removing language prompts / tissue functional maps: relatively minor performance impact.

### Downstream Application 1: Rare Action Augmentation
- Generated 287 synthetic videos (110 clipping + 177 cutting).
- Spatio-temporal CNN results:
    - Clipping F1: 20.93% $\rightarrow$ **43.14%** (+22.21 pp)
    - Cutting F1: 0.00% $\rightarrow$ **8.33%** (from zero to something)
- ViT results: Cutting F1: 30.77% $\rightarrow$ **47.06%**

### Downstream Application 2: Surgical Simulation
- Built a simulator using Isaac Lab, and exported tool segmentations, trajectories, and functional maps from the simulator.
- Superimposed the simulator trajectories onto real surgical backgrounds to generate realistic tool-tissue interaction videos (as a proof of concept).

## Highlights & Insights

1. **Extremely Lightweight Conditioning Design**: Requires only text, a single reference frame, a binary functional map, and 2D trajectory points. No segmentation annotations or scene graphs are needed during inference.
2. **Substantial Lead in Temporal Consistency**: CD-FVD of 199.19 vs. 429.67 for the second-best, yielding a leap-bound improvement in temporal continuity for generated videos.
3. **Ingeniously Designed Depth Consistency Loss**: Leverages depth information during training but does not require it for inference, achieving "free" geometric constraints.
4. **Dual Downstream Validation**: Demonstrates not only generation quality but also functional utility gains on real downstream tasks (action recognition).
5. **Large-Scale Data Curation**: The curated surgical dataset of 12,044 annotated clips is highly valuable to the community.

## Limitations & Future Work

1. Only validated on cholecystectomy (laparoscopy) scenarios, lacking generalization experiments across other surgical types (such as orthopedics, neurosurgery, or robot-assisted surgery).
2. Language and functional map conditions show relatively small impacts in ablation, and their individual contributions warrant deeper investigation due to potential redundancy.
3. The simulation application is purely a proof of concept, lacking real-time inference and closed-loop control, presenting a gap before acting as a practical simulation engine.
4. A fixed inference window of 81 frames (approx. 3.2 seconds) prevents generating longer surgical sequences to cover entire surgical phases.
5. When training ViT with synthetic videos, clipping F1 dropped slightly (84.62% $\rightarrow$ 83.64%), indicating potential marginal negative impacts on already well-represented classes.
6. Part of the dataset comes from YouTube (21 videos), which may introduce selection bias and inconsistent video quality.
7. Lacks direct comparison with methods like HieraSurg on the same dataset (although SurgSora was retrained on the paper's dataset).

## Rating
- Novelty: 4/5 — The concept of a surgical world model is novel, the four-signal conditional video diffusion is creative, and the depth consistency loss is clever (used in training but not inference).
- Experimental Thoroughness: 4/5 — Comprehensive coverage of generation quality (5 metrics) + complete ablation studies (removing conditions one by one) + validation on two downstream applications, though limited to a single surgical type (cholecystectomy).
- Writing Quality: 4/5 — Well-structured with intuitive figures (using red overlay annotations to clearly point out generation defects) and effective conceptual explanations.
- Value: 4/5 — Opens up highly promising directions for data augmentation in surgical AI and surgical simulation, with the curated 12K-clip annotated dataset holding independent value for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] World2Act: Latent Action Post-Training via Skill-Compositional World Models](world2act_latent_action_post-training_via_skill-compositional_world_models.md)
- [\[CVPR 2026\] Physical Object Understanding with a Physically Controllable World Model](../../CVPR2026/video_generation/physical_object_understanding_with_a_physically_controllable_world_model.md)
- [\[ICML 2025\] How Far is Video Generation from World Model: A Physical Law Perspective](../../ICML2025/video_generation/how_far_is_video_generation_from_world_model_a_physical_law_perspective.md)
- [\[CVPR 2025\] DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes](dynamicscaler_seamless_and_scalable_video_generation_for_panoramic_scenes.md)
- [\[CVPR 2026\] Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout](../../CVPR2026/video_generation/infinity-rope_action-controllable_infinite_video_generation_emerges_from_autoreg.md)

</div>

<!-- RELATED:END -->
