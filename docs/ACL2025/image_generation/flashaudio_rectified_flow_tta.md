---
title: >-
  [Paper Note] FlashAudio: Rectified Flows for Fast and High-Fidelity Text-to-Audio Generation
description: >-
  [ACL 2025][Image Generation][Text-to-Audio] This paper introduces Rectified Flow to text-to-audio generation. By leveraging bifocal samplers to optimize timestep distribution, immiscible flow to minimize total data-noise distance, and anchored optimization to correct CFG guidance errors, the proposed method achieves single-step generation with a FAD of 1.49, outperforming 100-step diffusion models while reaching a generation speed of 400x real-time.
tags:
  - "ACL 2025"
  - "Image Generation"
  - "Text-to-Audio"
  - "Rectified Flow"
  - "Fast Generation"
  - "Diffusion Model Acceleration"
  - "Single-Step Generation"
date: 2026-05-08
content_hash: 74c16ecd811286e1
---

# FlashAudio: Rectified Flows for Fast and High-Fidelity Text-to-Audio Generation

**Conference**: ACL 2025  
**arXiv**: [2410.12266](https://arxiv.org/abs/2410.12266)  
**Code**: [https://github.com/liuhuadai/FlashAudio](https://github.com/liuhuadai/FlashAudio)  
**Area**: Image Generation  
**Keywords**: Text-to-Audio, Rectified Flow, Fast Generation, Diffusion Model Acceleration, Single-Step Generation  

## TL;DR
This paper introduces Rectified Flow to text-to-audio generation. By leveraging bifocal samplers to optimize timestep distribution, immiscible flow to minimize total data-noise distance, and anchored optimization to correct CFG guidance errors, the proposed method achieves single-step generation with a FAD of 1.49, outperforming 100-step diffusion models while reaching a generation speed of 400x real-time.

## Background & Motivation
**Background**: Text-to-audio (TTA) generation based on latent diffusion models (LDMs) has made remarkable progress. Models such as AudioLDM2 and TANGO2 typically require 100 iterative sampling steps to generate high-quality audio.

**Limitations of Prior Work**: (a) Iterative sampling is computationally expensive, limiting real-time deployment; (b) consistency distillation methods (e.g., AudioLCM) reduce step counts, but curved trajectories lead to error accumulation, preventing single-step performance from outperforming diffusion models; (c) Classifier-Free Guidance (CFG) amplifies cumulative errors in few-step generation.

**Key Challenge**: Curved trajectories require multi-step integration for accurate simulation, whereas straight trajectories theoretically need only one step—but how can a truly straight flow trajectory be achieved in the TTA domain?

**Goal**: To apply rectified flows to TTA, addressing practical issues during training such as suboptimal timestep distribution, suboptimal noise-data pairing, and CFG error amplification.

**Key Insight**: Rectified flow has been successfully applied in image generation (Stable Diffusion 3/InstaFlow) and TTS, but remains unexplored in the TTA domain. Initializing from a pre-trained flow matching model can accelerate convergence.

**Core Idea**: To learn straight paths using rectified flows for fast simulation, combined with three training optimization techniques to enable single-step generation quality to exceed that of 100-step diffusion models.

## Method

### Overall Architecture
Starting from a pre-trained conditional flow matching (CFM) model, the optimization involves three stages: (1) training with an improved 1-rectified flow to learn straight trajectories; (2) performing reflow to further straighten trajectories, resulting in a 2-rectified flow; (3) distilling the 2-rectified flow to obtain a single-step generation model. During inference, only 1-4 steps are required to generate high-quality audio.

### Key Designs

1. **Bifocal Samplers**:

    - **Function**: Optimizes the sampling distribution of training timesteps.
    - **Mechanism**: Rectified flows are relatively easy to learn near $t=0$ (pure noise) and $t=1$ (pure data), while the intermediate timesteps are the most challenging. A logit-normal distribution is used to increase the sampling frequency of intermediate timesteps, combined with a Beta distribution to increase boundary timestep sampling, forming two focal points (middle and boundaries).
    - **Design Motivation**: Uniform sampling wastes computational resources on easy timesteps; bifocal samplers focus on the most challenging regions.

2. **Immiscible Flow**:

    - **Function**: Optimizes the assignment of data-noise pairs within a batch.
    - **Mechanism**: Optimal transport (linear assignment) is employed to minimize the total transport distance between data points and noise points within the same batch, pairing each data point with its nearest noise point.
    - **Design Motivation**: Random pairing may lead to distant data-noise pairs, causing intersecting flow trajectories. Immiscible flow reduces intersections, making the trajectories straighter.

3. **Anchored Optimization**:

    - **Function**: Corrects the error amplification caused by CFG in rectified flows.
    - **Mechanism**: When reflow generates noise-data pairs, instead of directly using the CFG-guided trajectory, the guidance scale is anchored to a reference trajectory with $\omega=1$ (no guidance), and a first-order correction is performed via $\hat{v}_\theta = v_\theta + (\omega-1)\frac{\partial v_\theta}{\partial \omega}\bigg|_{\omega=1}$.
    - **Design Motivation**: CFG alters the marginal distribution, which violates the straightness assumption of reflow. Anchoring to the reference trajectory reduces distribution shift.

### Loss & Training
- Standard MSE loss for rectified flow: $\min_v \mathbb{E}[\|(z_1-z_0) - v_\theta(z_t,t)\|^2]$
- Pre-trained CFM initialization $\rightarrow$ 1-rectified flow training $\rightarrow$ Reflow to obtain 2-rectified flow $\rightarrow$ Distillation to achieve a single-step model
- Both training and inference are conducted in the latent space (using VAE encoders/decoders)

## Key Experimental Results

### Main Results (AudioCaps Test Set)

| Model | Steps(NFE) | FAD(↓) | KL(↓) | CLAP(↑) | RTF(↓) | MOS-Q |
|------|---------|--------|-------|---------|--------|-------|
| AudioLDM 2 | 100 | 1.90 | 1.48 | 0.622 | 1.250 | 73.38 |
| TANGO 2 | 100 | 2.84 | 1.20 | 0.680 | 0.800 | 73.46 |
| AudioLCM | 2 | 1.67 | 1.37 | 0.617 | 0.003 | 76.48 |
| ConsistencyTTA | 1 | 2.13 | 1.33 | 0.655 | 0.004 | 73.19 |
| **FlashAudio (24 steps)** | 24 | **1.18** | **1.28** | **0.658** | 0.054 | **78.86** |
| **FlashAudio (4 steps)** | 4 | **1.26** | 1.30 | **0.652** | 0.014 | 78.23 |
| **FlashAudio (1 step + distillation)** | 1 | **1.49** | **1.32** | 0.648 | **0.0025** | **77.56** |

### Ablation Study

| Configuration | FAD(↓) | KL(↓) | CLAP(↑) | Description |
|------|--------|-------|---------|------|
| 1-RF w/ Logit-Normal | 1.12 | 1.25 | 0.659 | Full model |
| 1-RF w/o Logit-Normal | 1.08 | 1.27 | 0.649 | Worse KL and CLAP |
| w/o Immiscible Flow | Performance degradation | | | Confirms the importance of immiscible flow |
| w/o CFM Initialization | Significant degradation | | | Pre-training initialization is critical |
| CFG without Anchored Optimization | FAD 1.43 | | | CLAP 0.639 |
| CFG + Anchored Optimization | FAD 1.26 | | | CLAP 0.652 |

### Key Findings
- FlashAudio single-step generation (FAD=1.49) outperforms all 100-step diffusion models, achieving this milestone for the first time in the TTA domain.
- The generation speed is 400x real-time (RTF=0.0025), which is practically deployable.
- Reflow significantly reduces the trajectory straightness metric $S(z)$, with the 2-rectified flow trajectory being almost entirely straight.
- Anchored optimization is most effective under large guidance scales, reducing FAD from 1.43 to 1.26.
- Initializing from a pre-trained CFM yields faster convergence and better performance than training from scratch.

## Highlights & Insights
- **Complementary Training Optimizations**: Bifocal samplers optimize timesteps, immiscible flow optimizes spatial pair matching, and anchored optimization corrects guidance errors. Together, they improve the quality of rectified flows from three independent dimensions.
- **Single-Step Outperforming 100-Step Diffusion**: This is a milestone result, demonstrating that straight trajectories combined with distillation can fully replace iterative sampling.
- **Anchored Optimization**: Addresses the practical incompatibility between CFG and rectified flows. While CFG alters the distribution, reflow assumes an invariant distribution; anchored optimization elegantly mitigates this conflict.
- The success of rectified flow in TTA validates the cross-modal generalization of this paradigm (Image $\rightarrow$ Speech $\rightarrow$ Audio).

## Limitations & Future Work
- Evaluation is only conducted on AudioCaps; other audio domains (such as music and environmental sound effects) remain unvalidated.
- Distillation increases training complexity, as it requires training a teacher model prior to distillation.
- There is a 10-second limit on audio length; the performance on longer audio remains unknown.
- Immiscible flow employs linear assignment with a computational complexity of $O(n^3)$, which may become a bottleneck for large batch sizes.

## Related Work & Insights
- **vs AudioLCM**: AudioLCM uses consistency distillation to perform few-step generation on curved trajectories, achieving a 2-step FAD of 1.67; FlashAudio achieves a superior 1-step FAD of 1.49.
- **vs ConsistencyTTA/SoundCTM**: These are also single-step generation methods but based on consistency models, having FAD values > 1.9. FlashAudio's rectified flow approach is superior.
- **vs InstaFlow (Image Domain)**: FlashAudio is a successful migration of the InstaFlow concept to the audio domain, confirming the cross-modal generality of rectified flows.

## Rating
- Novelty: ⭐⭐⭐⭐ First to apply rectified flows with three optimization techniques to TTA, where anchored optimization represents a new contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes objective and subjective metrics, ablation studies, and trajectory analysis, though limited to a single dataset.
- Writing Quality: ⭐⭐⭐⭐ Clear method motivation, though some technical details are quite dense.
- Value: ⭐⭐⭐⭐⭐ High-quality TTA generation with 400x real-time speed, ready for direct deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows](../../CVPR2025/image_generation/omniflow_any-to-any_generation_with_multi-modal_rectified_flows.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](../../NeurIPS2025/image_generation/on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[CVPR 2025\] GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing](../../CVPR2025/image_generation/glyphmastero_a_glyph_encoder_for_high-fidelity_scene_text_editing.md)
- [\[ICLR 2026\] Flow Straight and Fast in Hilbert Space: Functional Rectified Flow](../../ICLR2026/image_generation/flow_straight_and_fast_in_hilbert_space_functional_rectified_flow.md)
- [\[CVPR 2026\] Omni2Sound: Towards Unified Video-Text-to-Audio Generation](../../CVPR2026/image_generation/omni2sound_towards_unified_video-text-to-audio_generation.md)

</div>

<!-- RELATED:END -->
