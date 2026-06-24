---
title: >-
  [Paper Note] RIFLEx: A Free Lunch for Length Extrapolation in Video Diffusion Transformers
description: >-
  [ICML 2025][Video Generation][Video Diffusion Transformer] By systematically analyzing the roles of different frequency components in RoPE positional encoding, this paper identifies an "intrinsic frequency" that dominates temporal repetition during extrapolation. It proposes RIFLEx, a minimal intervention scheme that scales down only this frequency to keep it within a single period after extrapolation, achieving high-quality training-free 2× video extrapolation on CogVideoX-5…
tags:
  - "ICML 2025"
  - "Video Generation"
  - "Video Diffusion Transformer"
  - "RoPE"
  - "Frequency Analysis"
  - "Length Extrapolation"
  - "training-free"
date: 2026-05-08
content_hash: 1e5468422f45bc1d
---

# RIFLEx: A Free Lunch for Length Extrapolation in Video Diffusion Transformers

**Conference**: ICML 2025  
**arXiv**: [2502.15894](https://arxiv.org/abs/2502.15894)  
**Code**: [riflex-video.github.io](https://riflex-video.github.io/)  
**Area**: Video Generation / Length Extrapolation  
**Keywords**: Video Diffusion Transformer, RoPE, Frequency Analysis, Length Extrapolation, training-free  

## TL;DR
By systematically analyzing the roles of different frequency components in RoPE positional encoding, this paper identifies an "intrinsic frequency" that dominates temporal repetition during extrapolation. It proposes RIFLEx, a minimal intervention scheme that scales down only this frequency to keep it within a single period after extrapolation, achieving high-quality training-free 2× video extrapolation on CogVideoX-5B and HunyuanVideo.

## Background & Motivation
**Background**: Video diffusion Transformers (e.g., CogVideoX, HunyuanVideo) can generate high-quality minute-long videos but are limited by their training length limits, preventing them from directly generating longer videos.

**Limitations of Prior Work**: (1) Length extrapolation methods from text or image domains (PE, PI, NTK, YaRN) fail when directly applied to videos, leading to two typical failure modes: temporal repetition (video looping) or motion deceleration (stretched frames); (2) The goal of video extrapolation is fundamentally different from text/image tasks—it requires generating temporally coherent, evolving new content rather than expanding the context window or increasing resolution details.

**Key Challenge**: How to suppress temporal repetition while maintaining motion consistency without changing model weights?

**Key Insight**: Isolating each frequency component in RoPE one by one and analyzing their impact on video generation through zero-out and fine-tuning experiments.

**Core Idea**: Decreasing the "intrinsic frequency" to keep it within at most one full period within the extrapolation length, thereby eliminating repetition.

## Method

### Overall Architecture
RIFLEx consists of three steps: (1) Analyze the period $N_j = 2\pi/\theta_j$ of each frequency component $\theta_j$ in RoPE; (2) Identify the intrinsic frequency component $k = \arg\min_j |N_j - N|$, where $N$ is the position of the first observed repeating frame; (3) Modify only this component to $\theta_k' = 2\pi/(Ls)$, so that its period covers the entire extrapolation length. All other frequencies remain completely unchanged.

### Key Designs
1. **Frequency Component Role Analysis**:
    - **Function**: Reveal the impact of different frequency components in RoPE on video generation through isolation experiments.
    - **Mechanism**: Set all components in RoPE to zero except for a certain frequency $\theta_j$, fine-tune the model, and observe the generation behavior. It is found that high-frequency components ($r_j = L\theta_j/(2\pi) > 1$) capture short-term dependencies and fast motion, causing temporal repetition; low-frequency components ($r_j < 1$) encode long-term dependencies but cause motion deceleration.
    - **Design Motivation**: Existing methods (PE/PI/NTK/YaRN) treat all frequencies uniformly, lacking understanding of the role of each individual frequency.

2. **Intrinsic Frequency Identification**:
    - **Function**: Find the critical frequency component that dominates the repetitive behavior in video extrapolation.
    - **Mechanism**: Define the intrinsic frequency as the component whose period $N_j$ is closest to the first repeating frame $N$: $k = \arg\min_j |N_j - N|$. Experiments show that for the same model, the intrinsic frequency remains consistent across different videos (e.g., $k=2$ for CogVideoX-5B, and $k=4$ for HunyuanVideo).
    - **Design Motivation**: Not all frequencies lead to repetition—modifying just this one key frequency is sufficient.

3. **Minimal Frequency Modification**:
    - **Function**: Modify only the intrinsic frequency so that it does not exceed one period after extrapolation.
    - **Mechanism**: Reduce $\theta_k$ to $\theta_k' = 2\pi/(Ls)$, where $s$ is the extrapolation factor. This ensures that $N_k' = Ls \geq Ls$, keeping it within a single period after extrapolation. Ablation studies confirm that modifying higher-frequency components destroys fast motion, while modifying lower-frequency components is almost ineffective.
    - **Design Motivation**: Minimizing modification equals minimizing training-inference mismatch, enabling 2× extrapolation to work without any fine-tuning.

### Loss & Training
2× extrapolation is completely training-free. For 3× extrapolation or further quality improvement, fine-tuning is only required using 20k original-length videos (which is only 1/50000 of the pre-training computation cost), using the standard diffusion training loss.

## Key Experimental Results

### Main Results (CogVideoX-5B 2× Extrapolation, Training-free)

| Method | NoRepeat Score↑ | Dynamic Degree↑ | Imaging Quality↑ | Overall Consistency↑ | User-Overall Rank↓ |
|------|----------------|----------------|-------------------|---------------------|-------------------|
| PE | 46.6 | 58.6 | 55.0 | 22.9 | 2.4 |
| NTK | 43.4 | 58.3 | 55.3 | 22.9 | 2.1 |
| PI | 59.0 | **5.0** | 44.3 | 19.2 | 3.8 |
| YaRN | 59.4 | **5.6** | 44.6 | 19.3 | 3.7 |
| TASR | 10.8 | 26.9 | 50.5 | 21.5 | 3.6 |
| **RIFLEx** | 54.2 | 59.4 | **56.9** | **23.5** | **1.1** |

### Ablation Study

| Configuration | Description |
|------|------|
| 2× training-free | NoRepeat 54.2, Dynamic 59.4—high-quality extrapolation without training |
| 2× fine-tuned | NoRepeat 61.3, Dynamic 54.7—fine-tuning further improves results |
| HunyuanVideo 2× | User overall rank is 1.6, while NTK is 1.6 but with lower Dynamic |
| Spatial Extrapolation 480p→960p | Equally applicable to image spatial resolution extrapolation |
| Joint Spatiotemporal Extrapolation | Simultaneous 2× temporal + 2× spatial extrapolation is effective |

### Key Findings
- PI/YaRN completely eliminate repetition, but the Dynamic Degree is only 5-6 (nearly static scenes)—motion deceleration is a fatal issue.
- PE/NTK have a NoRepeat Score of only 43-47, showing severe temporal repetition.
- In user studies, RIFLEx outperforms original training-length videos by a margin of 61.6-70.2%.
- The intrinsic frequency remains consistent across different videos within the same model.

## Highlights & Insights
- The insight of "one frequency dictates everything" is extremely elegant—simplifying a complex extrapolation problem to modifying a single scalar value.
- A genuine "free lunch"—zero extra training cost and zero parameter changes for 2× extrapolation, modifying only a single RoPE frequency.
- Provides a unified, principled explanation for the failures of existing methods (PE/PI/NTK/YaRN/TASR).
- A unified framework for temporal, spatial, and joint extrapolation, demonstrating the generality of this insight.

## Limitations & Future Work
- Identifying the intrinsic frequency requires pre-generating videos to observe the repeating frame position, lacking an automated method.
- 3× extrapolation still requires fine-tuning, and the performance of larger extrapolation scales (4×+) remains unknown.
- Only validated on two models: CogVideoX-5B and HunyuanVideo.
- Special treatment is required for rare scenarios where the intrinsic frequency is inconsistent across different videos.

## Related Work & Insights
- **vs NTK-Aware RoPE**: Adjusting the base $b$ of all frequencies still leads to temporal repetition in videos (NoRepeat 43.4). RIFLEx proves that only one frequency needs to be modified.
- **vs PI**: Uniformly scaling down all frequencies by $\theta/s$ eliminates repetition but freezes motion entirely (Dynamic is only 5.0). RIFLEx preserves high-frequency motion information.
- **vs YaRN**: Grouping scaling strategies similarly lead to motion deceleration in videos (Dynamic 5.6).
- **vs TASR**: CogVideoX/HunyuanVideo adopt 3D RoPE, and TASR's temporal-step adaptive strategy shows limited efficacy on top of this.
- **Insight**: In the context of video extrapolation, the role analysis of frequency components provides a direction for designing better positional encodings in the future.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The insight is novel and elegant; the concept of "intrinsic frequency" is powerful, and the solution is extremely simple.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on two SOTA models, multiple extrapolation scales, joint spatiotemporal scenarios, and user studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Thorough problem analysis, progressing perfectly from failure modes -> frequency analysis -> intrinsic frequency -> solution.
- **Value**: ⭐⭐⭐⭐⭐ Paradigmatic contribution to the video generation community, with a highly practical training-free solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] UltraViCo: Breaking Extrapolation Limits in Video Diffusion Transformers](../../ICLR2026/video_generation/ultravico_breaking_extrapolation_limits_in_video_diffusion_transformers.md)
- [\[ICML 2025\] AsymRnR: Video Diffusion Transformers Acceleration with Asymmetric Reduction and Restoration](asymrnr_video_diffusion_transformers_acceleration_with_asymmetric_reduction_and_.md)
- [\[CVPR 2026\] Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction](../../CVPR2026/video_generation/free-lunch_long_video_generation_via_layer-adaptive_ood_correction.md)
- [\[CVPR 2025\] Towards Precise Scaling Laws for Video Diffusion Transformers](../../CVPR2025/video_generation/towards_precise_scaling_laws_for_video_diffusion_transformers.md)
- [\[CVPR 2025\] DiTFlow: Video Motion Transfer with Diffusion Transformers](../../CVPR2025/video_generation/video_motion_transfer_with_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
