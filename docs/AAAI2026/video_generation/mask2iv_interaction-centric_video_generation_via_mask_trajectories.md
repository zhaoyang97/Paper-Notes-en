---
title: >-
  [Paper Note] Mask2IV: Interaction-Centric Video Generation via Mask Trajectories
description: >-
  [AAAI 2026][Video Generation][Interaction video generation] This paper proposes Mask2IV, a two-stage decoupled framework that first predicts mask motion trajectories of the interactor and object…
tags:
  - "AAAI 2026"
  - "Video Generation"
  - "Interaction video generation"
  - "mask trajectories"
  - "human-object interaction"
  - "robot manipulation"
  - "two-stage diffusion"
date: 2026-05-08
content_hash: dc5de1fba40385e6
---

# Mask2IV: Interaction-Centric Video Generation via Mask Trajectories

**Conference**: AAAI 2026
**arXiv**: [2510.03135](https://arxiv.org/abs/2510.03135)  
**Code**: [Project Page](https://reagan1311.github.io/mask2iv)  
**Area**: Video Understanding / Video Generation
**Keywords**: Interaction video generation, mask trajectories, human-object interaction, robot manipulation, two-stage diffusion

## TL;DR

This paper proposes Mask2IV, a two-stage decoupled framework that first predicts mask motion trajectories of the interactor and object, then generates video conditioned on these trajectories. The approach enables controllable, interaction-centric video generation without dense mask annotations, supporting both human-object interaction and robot manipulation scenarios.

## Background & Motivation

### State of the Field

Diffusion models have achieved remarkable progress in video generation, producing high-quality videos from text or image prompts. In the context of embodied AI, generating realistic human-object or robot-object interaction video sequences holds significant value, providing visual priors for downstream tasks such as imitation learning and affordance learning.

### Limitations of Prior Work

**Imprecise text-conditioned control**: Existing text-conditioned methods (e.g., EgoVid, LEGO) lack fine-grained control over interaction details — they cannot specify which object to interact with or where the hand should be positioned.

**Mask-conditioned methods suffer from two critical drawbacks**:
   - **Poor practicality**: Methods such as InterDyn require users to provide **dense, per-frame hand mask sequences** as control signals. Obtaining these masks requires recording or synthesizing the very interaction video one wishes to control — creating a chicken-and-egg paradox.
   - **Hand-only focus**: Relying solely on hand masks limits interaction modeling scope, making it impossible to precisely specify target objects or capture fine-grained hand-object contact information.

**Lack of a unified framework**: Human-object interaction and robot manipulation are typically studied as separate problems, with no unified solution.

### Root Cause

Masks are effective interaction control signals — geometrically explicit and motion-trackable — yet **the cost of obtaining dense mask annotations is as high as that of generating the video itself**. The core technical challenge lies in leveraging the advantages of masks while eliminating their annotation dependency.

### Starting Point

**Decoupling trajectory prediction from video generation**: The first stage automatically predicts interaction trajectories (mask sequences), and the second stage generates video conditioned on these predicted trajectories. Users need only provide an initial image, a target object mask, and text or position conditions — no dense annotations are required.

## Method

### Overall Architecture

Mask2IV decomposes interaction video generation into two stages:

**Stage 1: Interaction Trajectory Generation**
- Input: initial frame $I$, object mask $M$, conditioning signal (text $T$ or target position mask $P$)
- Output: mask trajectory sequence $S \in \mathbb{R}^{N \times H \times W \times 3}$

**Stage 2: Trajectory-conditioned Video Generation**
- Input: initial frame $I$, predicted mask trajectory $S$
- Output: interaction video $V \in \mathbb{R}^{N \times H \times W \times 3}$

### Key Designs

#### 1. Interaction Trajectory Generation (Stage 1)

**Function**: Predict the joint motion trajectories of the interactor (hand/robotic arm) and the object.

**Mechanism**:
- The initial frame $I$ and object mask $M$ are encoded into latent space features via a VAE encoder.
- The object mask is first **color-encoded** into RGB format (as the VAE requires three-channel input).
- If the initial frame contains an interactor (hand/robotic arm), GroundedSAM is used to segment it and assign distinct colors, enabling the model to distinguish between roles.
- The encoded latent features are concatenated with the noisy latent variable and fed into a video diffusion model.
- **Temporal attention layers are frozen** to preserve motion priors, while remaining parameters are fine-tuned.

**Two conditioning variants**:

**(a) Text-conditioned Trajectory Generation (TT-Gen)**:
- Text prompts are encoded via CLIP and injected into the model through cross-attention.
- Capable of distinguishing subtle interaction intents, e.g., "pick up" vs. "put down," "push" vs. "pull."

**(b) Position-conditioned Trajectory Generation (PT-Gen)**:
- The target position mask $P$ is encoded and inserted into the slot of the last frame.
- The initial object mask latent is assigned to the first frame.
- Intermediate frames are filled with zeros, and the model automatically interpolates to produce a coherent trajectory.

**Design Motivation**: Simplifying the problem to predicting mask motion first allows the model to focus solely on motion dynamics without handling appearance details, substantially reducing the difficulty of directly generating complex interaction videos.

#### 2. Trajectory-conditioned Video Generation (Stage 2)

**Function**: Synthesize the final video conditioned on predicted mask trajectories.

**Mechanism**:
- The trajectory $S$ is encoded into a feature tensor $f_s$ via the VAE.
- $f_s$ is concatenated with the noisy latent variable and the first-frame features before being fed into the diffusion model.

**Two targeted designs**:

**(a) Random perturbation for robustness**:
- During training, masks are randomly dilated or eroded with probability $p=0.2$ (kernel size randomly selected from {3, 5, 7}).
- This prevents the model from over-relying on the precise shape of masks, improving generalization.

**(b) Contact region-weighted loss**:
- A contact map is defined as $m_c = (\delta(m_h) \cap m_o) \cup (m_h \cap \delta(m_o))$, where $\delta(\cdot)$ denotes dilation.
- The contact map is used to re-weight the diffusion objective:

$$w = (1 - m_c) + \lambda \cdot m_c$$

$$\mathcal{L} = \mathbb{E}_{z,S,\epsilon,t}[\|w \odot (\epsilon - \epsilon_\theta(z, f_\psi(S), t))\|_2^2]$$

where $\lambda=5$, assigning five times the loss weight to contact regions relative to non-contact regions.

**Design Motivation**:
- Random perturbation addresses the distribution gap between training (ground-truth masks) and inference (predicted masks).
- Contact-weighted loss resolves the synthesis difficulty at hand-object boundary regions — precisely the most critical area for interaction modeling.

#### 3. Benchmark Construction

**Function**: Construct training/evaluation benchmarks with per-frame segmentation annotations.

- **HOI4D** (human-object interaction): Video clips are trimmed using timestamps; low-dynamic videos are filtered based on a motion score derived from hand and object displacement (bottom 5% removed); text annotations follow the template "a hand {verbing} an {object}."
- **BridgeData V2** (robot manipulation): GroundingDINO is used for object detection and SAM2 for video segmentation to extract robotic arm and object masks; manipulated objects are identified by low inter-frame mIoU over time (indicating changing position and shape), requiring no additional annotation.

### Loss & Training

- Built upon DynamiCrafter with additional convolutional channels to accommodate mask latents.
- 16 frames at 320×512 resolution.
- AdamW optimizer, learning rate 1e-5, batch size 8.
- DDIM sampler with 50 denoising steps at inference.
- Contact weight $\lambda = 5$.

## Key Experimental Results

### Main Results

| Method | Conference | FVD↓ (HOI/Robot) | LPIPS↓ | PSNR↑ | SSIM↑ | V2V-Sim↑ | T2V-Sim↑ |
|--------|------------|-----------------|--------|-------|-------|----------|----------|
| DynamiCrafter | ECCV24 | 554/861 | 0.516/0.375 | 13.48/14.21 | 0.553/0.571 | 0.473/0.867 | 0.146/0.215 |
| DynamiCrafter-ft | ECCV24 | 169/198 | 0.206/0.166 | 20.49/19.80 | 0.721/0.775 | 0.814/0.957 | 0.199/0.223 |
| CosHand | ECCV24 | 163/175 | 0.209/0.123 | 20.67/21.81 | 0.725/0.809 | 0.837/0.969 | 0.191/0.220 |
| InterDyn | CVPR25 | 172/208 | 0.207/0.145 | 20.71/21.16 | 0.730/0.802 | 0.794/0.955 | 0.172/0.219 |
| **Mask2IV** | **Ours** | **150/156** | **0.178/0.111** | **21.48/22.30** | **0.741/0.815** | **0.847/0.971** | **0.200/0.220** |

- FVD is reduced by 8.7% on HOI4D (150 vs. 163) and by 10.9% on BridgeData V2 (156 vs. 175).
- Mask2IV outperforms all baselines across all metrics.

### Ablation Study

| Configuration | FVD↓ | LPIPS↓ | PSNR↑ | SSIM↑ | Note |
|---------------|------|--------|-------|-------|------|
| ControlNet | 157.38 | 0.182 | 21.49 | 0.747 | Auxiliary network approach |
| MaskLatent | 130.07 | 0.157 | 22.33 | 0.760 | Direct latent concatenation, superior |
| +object mask | 115.14 | 0.132 | 23.85 | 0.802 | Adding object trajectory, large gain |
| +random d/e | 108.80 | 0.124 | 24.16 | 0.802 | Random dilation/erosion augmentation |
| +contact loss | **104.61** | **0.126** | **24.37** | **0.804** | Contact-weighted loss |

(Ablation conducted on HOI4D using ground-truth mask trajectories.)

### Key Findings

1. **Direct mask latent concatenation outperforms ControlNet**: Training is more stable with faster early convergence.
2. **Adding object trajectories yields the largest gain**: FVD drops from 130 to 115 (−11.5%), demonstrating that modeling hand motion alone is insufficient — joint hand-object motion modeling is necessary.
3. **Random perturbation genuinely improves robustness**: FVD drops from 115 to 109.
4. **Contact-weighted loss further improves quality**: FVD drops from 109 to 105.
5. **Flexible object specification**: Different interactions with different objects can be generated within the same scene by modifying the mask.
6. **Text and position conditions are complementary**: Text is better suited for describing action types, while position conditions enable precise spatial control.

## Highlights & Insights

1. **Dual advantage of decoupled design**: Reduces generation difficulty while providing more flexible control (users can modify predicted trajectories).
2. **Innovation of contact region-weighted loss**: Precisely focuses on the most critical region for interaction (hand-object contact boundary), using the geometric information of masks to define the weighting map.
3. **Unified framework covering both humans and robots**: The same method handles two interaction scenarios, differing only in condition type (text vs. position).
4. **Clever object identification strategy**: In BridgeData V2, manipulated objects are identified by low inter-frame mIoU over time, requiring no additional annotation.
5. **Color-encoding design detail**: Masks are converted into differently colored RGB images for VAE input, allowing the model to distinguish between interactor and object roles.

## Limitations & Future Work

1. **Resolution limitation**: 320×512 is relatively low; computational cost would increase substantially at higher resolutions.
2. **Only 16 frames**: Adequate for short interactions, but insufficient to cover long-horizon manipulation (e.g., multi-step assembly).
3. **Error accumulation in two-stage inference**: The quality of trajectories predicted in Stage 1 directly affects final video quality.
4. **No support for multi-step interactions**: Sequential actions such as "pick up then put down" are not modeled.
5. **Absence of human evaluation**: The method relies entirely on automatic metrics, with no perceptual user study on video quality.
6. **Dataset scale**: HOI4D covers a limited range of action categories (primarily grasping); more diverse interaction types remain to be explored.

## Related Work & Insights

- **Masks as intermediate representations**: More semantically meaningful than optical flow and more precise than bounding boxes, making them an ideal control signal for interaction generation.
- **Decoupling trajectory prediction from video generation**: This paradigm is transferable to other control signals (e.g., skeletons, keypoints).
- **Contact map definition**: The dilation-intersection approach offers a simple yet effective definition of contact regions applicable to other hand-object interaction tasks.
- **Integration with robotics**: Generated interaction videos can be directly used to train visuomotor policies, offering an alternative data acquisition pathway beyond sim-to-real transfer.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The two-stage decoupled design and contact-weighted loss are novel contributions, though the overall framework builds on existing diffusion models.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two datasets and complete ablation studies, but human evaluation is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation and methodology are articulated with exceptional clarity; figures and tables are well-designed.
- **Value**: ⭐⭐⭐⭐ — Offers practical value for data generation in embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](../../CVPR2026/video_generation/u-mind_a_unified_framework_for_real-time_multimodal_interaction_with_audiovisual.md)
- [\[CVPR 2026\] Chain of Event-Centric Causal Thought for Physically Plausible Video Generation](../../CVPR2026/video_generation/chain_of_event-centric_causal_thought_for_physically_plausible_video_generation.md)
- [\[ICCV 2025\] DreamRelation: Relation-Centric Video Customization](../../ICCV2025/video_generation/dreamrelation_relation-centric_video_customization.md)
- [\[ICLR 2026\] LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](../../ICLR2026/video_generation/lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)
- [\[ICML 2026\] CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal](../../ICML2026/video_generation/clear_context-aware_learning_with_end-to-end_mask-free_inference_for_adaptive_vi.md)

</div>

<!-- RELATED:END -->
