---
title: >-
  [Paper Note] Motion Prompting: Controlling Video Generation with Motion Trajectories
description: >-
  [CVPR 2025][Video Generation][motion prompting] By training ControlNet with spatio-temporally sparse/dense point trajectories as "motion prompts," a single model achieves diverse motion control capabilities—including object control, camera control, motion transfer, and drag-and-drop editing—while demonstrating the emergence of realistic physical behaviors.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "motion prompting"
  - "video diffusion"
  - "point trajectories"
  - "ControlNet"
  - "camera control"
  - "object control"
  - "motion transfer"
date: 2026-05-08
content_hash: d97ef088f0c313e4
---

# Motion Prompting: Controlling Video Generation with Motion Trajectories

**Conference**: CVPR 2025  
**arXiv**: [2412.02700](https://arxiv.org/abs/2412.02700)  
**Code**: [https://motion-prompting.github.io/](https://motion-prompting.github.io/)  
**Area**: Video Generation  
**Keywords**: motion prompting, video diffusion, point trajectories, ControlNet, camera control, object control, motion transfer

## TL;DR

By training ControlNet with spatio-temporally sparse/dense point trajectories as "motion prompts," a single model achieves diverse motion control capabilities—including object control, camera control, motion transfer, and drag-and-drop editing—while demonstrating the emergence of realistic physical behaviors.

## Background & Motivation

**Background**: Video generation models primarily rely on text prompts for control, but text is fundamentally limited in describing precise motion details—"a bear quickly turns its head" can be interpreted in countless ways. Subtle nuances in motion timing, trajectory, acceleration, and deceleration require more direct control signals.

**Limitations of Prior Work**:
1. **High engineering complexity**: Methods like Tora, MotionCtrl, and DragNUWA require two-stage training, task-specific loss functions, customized architectures, or joint fine-tuning of multiple modules.
2. **Restricted control signals**: Entity-level signals such as bounding boxes, segmentation masks, and human poses can only control specific types of motion.
3. **Inadequate optical flow representations**: Optical flow cannot link features across multiple frames (leading to error accumulation), does not handle occlusions, and is ill-suited for generative tasks.
4. **Different motion types require different adapters**: For instance, MOFA-Video needs separate training for camera and object motions.

**Key Insight**: Point trajectories (point tracks) can encode an arbitrary number of trajectories, object or global motions, and temporally sparse motions. This makes them a sufficiently flexible "motion language"—analogous to how text serves as a "semantic language," motion prompts act as a "motion language."

## Method

### Overall Architecture

ControlNet is trained on a pre-trained video diffusion model (Lumiere) using point trajectories as conditional signals. The training is highly minimal: single-stage, with uniformly sampled dense trajectories and no specialized engineering. During inference, high-level user intents are converted into detailed trajectories through motion prompt expansion.

### Key Designs

#### Module 1: Motion Prompt Representation & Encoding

- **Representation**: $N$ point trajectories $p \in \mathbb{R}^{N \times T \times 2}$ + visibility flags $v \in \mathbb{R}^{N \times T}$.
- **Encoding**: Each trajectory is assigned a randomly sampled unique embedding vector $\phi^n \in \mathbb{R}^C$ (from a fixed pool). This embedding is placed at the spatio-temporal positions through which the trajectory passes, with all other locations set to zero.
- **Mathematical formulation**: $c[t, x_t^n, y_t^n] = v[n,t] \cdot \phi_n$.
- Embeddings are summed when multiple trajectories pass through the same position; fully dense trajectories are equivalent to forward-warping a dense grid of embeddings.

Key advantage: This approach can encode trajectories of arbitrary density, temporal span, and spatial distribution, providing a unified conditional input for ControlNet.

#### Module 2: Motion Prompt Expansion

The process of transforming high-level user requests into detailed trajectories (analogous to text prompt expansion):
- **Image "Interaction"**: Mouse dragging $\rightarrow$ generating local grid trajectories around the drag point, supporting temporal sparseness (multiple drags) and background static constraints.
- **Geometric Control**: Mapping mouse movements onto geometric proxies (e.g., spheres) to achieve complex motions like rotations.
- **Camera Control**: Monocular depth estimation $\rightarrow$ point cloud $\rightarrow$ re-projection according to the camera trajectory $\rightarrow$ 2D trajectories (including z-buffer occlusions).
- **Motion Transfer**: Extracting trajectories from a source video and applying them to a new first frame.
- **Motion Composition**: Point track displacements of objects are combined with the camera trajectory $\rightarrow$ simultaneously controlling both camera and objects.

#### Module 3: Training Strategy

- Data: Extracting 16,384 dense trajectories per video for 2.2M videos using BootsTAP.
- During training, the number of trajectories is randomly sampled (log-uniformly distributed from $2^0$ to $2^{13}$).
- Standard ControlNet training: single-stage, with no data filtering.
- **Key Finding**: Although trained solely on dense trajectories, the model generalizes well to sparse trajectories, spatially local trajectories, and trajectories starting at non-first frames.

### Loss & Training

Standard diffusion loss (denoising score matching) without any additional losses. ControlNet's zero convolutions ensure that the pre-trained model is not disrupted in the early stages of training—which is one of the reasons why training is so simple.

## Key Experimental Results

### Main Results

**Quantitative evaluation on the DAVIS validation set**:

| Tracks | Method | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ | EPE↓ |
|--------|------|-------|-------|--------|------|------|
| N=16 | ImageConductor | 12.184 | 0.175 | 0.502 | 1838.9 | 24.263 |
| N=16 | DragAnything | 15.119 | 0.305 | 0.378 | 1282.8 | 9.800 |
| N=16 | **Ours** | **16.618** | **0.405** | **0.319** | **1322.0** | **8.319** |
| N=2048 | DragAnything | 14.845 | 0.286 | 0.397 | 1468.4 | 12.485 |
| N=2048 | **Ours** | **19.327** | **0.608** | **0.227** | **655.9** | **3.887** |

**Human Evaluation** (2AFC, win rate%):

| vs. | Motion Consistency | Motion Quality | Visual Quality |
|-----|----------|---------|---------|
| ImageConductor | 74.3% | 80.5% | 77.3% |
| DragAnything | 74.5% | 75.7% | 73.7% |

### Ablation Study

**Ablation on Training Trajectory Density** (evaluated with 4 tracks / 2048 tracks):

| Training Strategy | PSNR (N=4) | PSNR (N=2048) | EPE (N=2048) |
|----------|------------|---------------|--------------|
| Sparse only | 15.075 | 15.697 | 26.724 |
| Dense + Sparse | 15.162 | 15.294 | 27.931 |
| **Dense only** | **15.638** | **19.197** | **4.806** |

### Key Findings

1. **Dense training is optimal**: Training solely on dense trajectories yields the best performance even during sparse inference—because sparse trajectory training provides signals that are too weak.
2. **More trajectories, better results**: As $N$ increases from 1 to 2048, PSNR improves by approximately 4 dB, and FVD decreases by roughly threefold.
3. **Emergence of physical behavior**: Dragging hair leads to natural swaying, and poking sand results in physically plausible scattering—the model has successfully learned motion priors.
4. **Strong generalization**: Although trained exclusively on uniformly distributed trajectories, the model generalizes to out-of-distribution conditions during inference, such as spatially local, temporally sparse, or non-first-frame-starting trajectories.
5. **No camera annotations needed**: Camera control is achieved indirectly through motion trajectories, despite the model not being explicitly trained on camera motion parameters.

## Highlights & Insights

1. **The concept of a "motion language"**: Unifying motion control into a general trajectory condition allows a single model to cover multiple control tasks—this is a highly elegant, unified framework.
2. **Minimalist training**: Single-stage, without specialized losses or data filtering—standing in stark contrast to the complex engineering of prior methods.
3. **Motion Prompt Expansion**: Analogous to text prompt rewriting, high-level intent is converted into low-level trajectories via a computer vision pipeline, effectively bridging user intent with model inputs.
4. **Probing the video model's physical understanding**: Motion prompts can be used to "interrogate" the model's understanding of the physical world—e.g., what happens when chess pieces are dragged? What happens when hair is pulled?
5. **Practicality**: The authors provide a mouse-interactive GUI. Although not real-time (~12 minutes per video), it demonstrates a future direction for interacting with generative world models.

## Limitations & Future Work

1. Generation is not real-time (~12 minutes per video), which is far from interactive applications.
2. Non-causal generation: The model requires complete trajectory inputs to generate, thereby lacking support for streaming interactive feedback.
3. Motion conditioning can sometimes introduce artifacts, such as a cow's horn being incorrectly "locked" to the background.
4. Limitations of the underlying video model: for instance, moving a chess piece might abruptly generate a new piece.

## Related Work & Insights

- **DragAnything**: Entity-level track control that utilizes latent warping to achieve accurate motion, but at the cost of visual artifacts.
- **MotionCtrl**: Explicitly decouples camera and object motion control, but requires dedicated designs.
- **MOFA-Video**: Different motion types require different adapters $\rightarrow$ Motion Prompting resolves this with a unified representation.
- **CoTracker/BootsTAP**: Advances in dense trajectory estimation make large-scale acquisition of high-quality training data feasible.
- **Insights**: Future world models might natively support motion prompts as an interactive interface for visual planning in embodied AI.

## Rating

⭐⭐⭐⭐⭐ — Conceptually elegant (motion as a motion language), minimalist in design (single-stage ControlNet), broadly applicable (one model for multiple controls), and showcasing surprising emergent behaviors. A solid piece of work from Google DeepMind that majorly impacts the steering paradigms in video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation](through-the-mask_mask-based_motion_trajectories_for_image-to-video_generation.md)
- [\[CVPR 2025\] ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer](conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)
- [\[CVPR 2025\] DiTFlow: Video Motion Transfer with Diffusion Transformers](video_motion_transfer_with_diffusion_transformers.md)
- [\[ICCV 2025\] Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation](../../ICCV2025/video_generation/free-form_motion_control_controlling_the_6d_poses_of_camera_and_objects_in_video.md)
- [\[CVPR 2025\] MotionPro: A Precise Motion Controller for Image-to-Video Generation](motionpro_a_precise_motion_controller_for_image-to-video_generation.md)

</div>

<!-- RELATED:END -->
