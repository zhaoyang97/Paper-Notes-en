---
title: >-
  [Paper Note] Seeing Realism from Simulation: Efficient Video Transfer for Vision-Language-Action Data Augmentation
description: >-
  [ICML 2026][Robotics][Sim-to-Real] To address the issue of VLA (vision-language-action) models collapsing under minor perturbations…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Sim-to-Real"
  - "VLA"
  - "Video Diffusion"
  - "Velocity Caching"
  - "Coreset Sampling"
date: 2026-05-08
content_hash: 149da7ba3ee307a9
---

# Seeing Realism from Simulation: Efficient Video Transfer for Vision-Language-Action Data Augmentation

**Conference**: ICML 2026  
**arXiv**: [2605.02757](https://arxiv.org/abs/2605.02757)  
**Code**: Public (CODE link at end of paper)  
**Area**: Robotics / Embodied Intelligence / VLA / Video Generation & Data Augmentation  
**Keywords**: Sim-to-Real, VLA, Video Diffusion, Velocity Caching, Coreset Sampling

## TL;DR
To address the issue of VLA (vision-language-action) models collapsing under minor perturbations, this work proposes a video transfer pipeline—"extract semantic/geometric conditions → rewrite caption → conditional video diffusion re-rendering"—to inject visual and environmental diversity into simulation data. Additionally, a three-stage velocity caching reduces generation time by 61%, and a difficulty + diversity-driven coreset sampling selects only 10% of key trajectories. Ultimately, on Robotwin 2.0, LIBERO-Plus, and real robots, RDT-1B / $\pi_0$ achieve 5–15% improvement.

## Background & Motivation

**Background**: VLA models (RDT, $\pi_0$, $\pi_{0.5}$, OpenVLA, ACT) rely on large-scale real robot trajectories for end-to-end training. However, real data collection is expensive, slow, and hard to scale; simulation data is cheap and parallelizable, but exhibits significant visual/environmental gaps, causing trained policies to fail under lighting/background/viewpoint perturbations.

**Limitations of Prior Work**: LIBERO-Plus reports that policies with 95% original success rate drop below 30% under mild perturbations; LIBERO-PRO shows near 0% success with object position and instruction changes—indicating models memorize action sequences rather than truly understanding tasks. Simple random noise/color perturbations cannot capture the semantic complexity of real environments.

**Key Challenge**: To make simulation data "look real," the most direct approach is conditional video diffusion re-rendering. However, models like Cosmos-Transfer require 40 minutes on an A100 GPU to process a single 5-second 720p video, making scaling to millions of simulated trajectories infeasible.

**Goal**: (1) Design a pipeline that can transfer entire simulated videos to high-fidelity real styles while strictly preserving action trajectories; (2) Reduce generation cost to a scalable level; (3) Maximize the utility of each computation without full-scale augmentation.

**Key Insight**: The authors improve generation quality via "addition"—using caption rewriting + depth geometric control + conditional video diffusion to create visually diverse, realistic videos; and reduce generation cost via "subtraction"—observing that the velocity field in flow-based diffusion is nearly constant in the middle stage, enabling caching and reuse.

**Core Idea**: Decompose video augmentation into two orthogonal efficiency axes: "generation" (velocity caching to cut per-video cost) and "selection" (difficulty + diversity graph-based coreset to reduce the number of trajectories to generate).

## Method

### Overall Architecture
Given a batch of simulated training trajectories $\mathcal{S}=\{s_1,\dots,s_n\}$: (1) coreset sampling selects $\mathcal{S}'\subset\mathcal{S}$ for generation; (2) for each selected video, VideoChat2 extracts captions, Qwen3-8B rewrites captions to introduce environmental variables (background/object color), and depth is extracted for geometric control; (3) Cosmos-Transfer 2.5 performs conditional video diffusion with the new caption + depth, yielding "realistic" videos with altered visual style but unchanged action trajectory; (4) the generated videos are mixed with the remaining 90% simulation data for VLA training. The two key accelerators in the pipeline are velocity caching and coreset.

### Key Designs

1. **Conditional Video Transfer (Semantic + Geometric Dual Conditioning)**:

    - **Function**: Converts a simulated robot arm video into a real-style video with the same actions but in diverse environments.
    - **Mechanism**: VideoChat2 extracts temporal captions describing interactions, objects, and spatial relations; Qwen3-8B rewrites captions to diversify background, object color, etc., while preserving task intent; depth maps from the original video serve as stable geometric constraints (more geometry-preserving than edge/blur/segmentation); Cosmos-Transfer 2.5 then iteratively denoises with the new caption + depth.
    - **Design Motivation**: Caption-only changes cause object geometry drift and key pose distortion, losing the action; depth-only lacks semantic diversity. The two conditions are complementary: one ensures "looks different," the other "does the same thing."

2. **Three-Stage Velocity Caching**:

    - **Function**: Reuses velocity predictions in flow-based video diffusion, avoiding redundant transformer forward passes.
    - **Mechanism**: Empirical analysis of $\|v_{t+1}-v_t\|$ reveals three stages: rapid change at the start, near-constant in the middle, fine-tuning at the end. The $N$ denoising steps are divided into initial ($t<t_s$, compute every step), stable ($t_s\leq t< t_f$, compute every $\alpha$ steps, reuse otherwise), and final ($t\geq t_f$, compute every step). The stable phase starts when $\frac{\|v_t-v_{t+1}\|}{\|v_0-v_1\|} < k$ (with $k=0.4,\alpha=8, m=3$).
    - **Design Motivation**: Generic caching (e.g., DeepCache) assumes both ends are important, but does not match diffusion dynamics; the three-stage approach aligns with the "outline → refinement → finishing" denoising rhythm, enabling a 61.2% time reduction with minimal quality loss.

3. **Difficulty × Diversity Coreset Sampling**:

    - **Function**: Selects only the most valuable samples for augmentation from large-scale simulated trajectories.
    - **Mechanism**: Extends $\mathbb{D}^2$ Pruning to videos. Difficulty $x_i = \frac{1}{|\mathcal{T}_i|}\sum_{t}\mathcal{L}_{\text{policy}}(s_i^{(t)};\theta)$ is estimated using RDT-1B policy loss; diversity is measured by extracting 768-dim embeddings $\phi(s_i)$ with Cosmos-Embed1, then building a kNN graph with RBF kernel $e_{i,j}=\exp(-\gamma_f\|v_i-v_j\|^2)$. Forward message passing aggregates neighborhood difficulty $x_i' = x_i + \sum_{j\in\mathcal{N}(i)}e_{i,j}\cdot x_j$; greedy selection picks highest $x_i'$, and backward message passing suppresses scores of similar neighbors to avoid redundancy.
    - **Design Motivation**: Focusing only on difficulty risks clustering in hard regions; focusing only on diversity admits easy samples. Combining both prioritizes "hard and unique" trajectories, achieving near full-augmentation effect with only 10% budget.

### Loss & Training
The VLA model loss remains unchanged; only the training set is replaced with original simulation plus real-style videos augmented via coreset sampling. The paper compares two mixing strategies: mixture (retain all original data + add augmented) and replacement (replace selected coreset with augmented). $\pi_0$ benefits more from mixture, while the stronger $\pi_{0.5}$ prefers replacement—stronger models can handle larger distribution shifts.

## Key Experimental Results

### Main Results
Robotwin 2.0 single-task (RDT-1B) "Hard" scenario, original vs augmented:

| Task | Ori. (Hard) | Aug. (Hard) | $\Delta$ |
|------|-------------|-------------|----------|
| adjust_bottle | 72.0 | 82.0 | +10.0 |
| beat_block_hammer | 36.0 | 48.0 | +12.0 |
| place_burger_fries | 26.0 | 38.0 | +12.0 |
| open_laptop | 30.0 | 44.0 | +14.0 |
| **average** | **29.0** | **39.0** | **+10.0** |

LIBERO-Plus spatial suite, $\pi_0$ + 50% coreset augmentation:

| Perturbation Type | Ori. | Aug. | $\Delta$ |
|-------------------|------|------|----------|
| objects layout | 69.6 | 86.2 | +16.6 |
| language instructions | 37.9 | 55.9 | +22.0 |
| background textures | 81.1 | 87.6 | +6.5 |
| robot initial states | 10.3 | 6.3 | −4.0 |
| camera view points | 21.3 | 15.2 | −6.1 |
| **average** | **42.7** | **47.8** | **+5.1** |

Real robot AgileX Piper (two tasks, three scenarios, 10 trials each): $\pi_0$ average success rate increases from 60% → 75% (+15%), $\pi_{0.5}$ from 60% → 73% (+13%).

### Ablation Study

| Setting | Robotwin Hard Avg. | Note |
|---------|--------------------|------|
| Original Sim | 29.0 | baseline |
| Aug. w/ velocity cache | 26.5 | caching acceleration |
| Aug. w/o velocity cache | 27.0 | full-step computation |
| Aug. (no coreset, full augmentation) | 39.0 | upper bound |

Video generation quality vs RoboTransfer (adjust_bottle, lower is better RMSE/Abs.Rel/Sq.Rel):

| Method | RMSE | Abs.Rel | Sq.Rel | sim$\uparrow$ |
|--------|------|---------|--------|---------------|
| RoboTransfer | 0.46 | 0.37 | 0.39 | 21.5 |
| **Ours** | **0.28** | **0.16** | **0.07** | **26.3** |

### Key Findings
- Velocity caching yields almost no performance drop (26.5 vs 27.0) while reducing generation time by 61%, confirming that flow-based diffusion has substantial mid-stage computational redundancy exploitable in practice.
- 10% coreset raises RDT-1B average from 23% to 31% on Robotwin multi-task (300 trajectories/task), indicating that "selecting well" is far more cost-effective than "adding more" in highly redundant simulation data.
- In real robot experiments, augmentation yields the most gains for background and position perturbations ("Stack Tape" position 5/10 → 8/10), but performance drops for robot initial states and camera viewpoints—this method only augments appearance, not geometry/viewpoint, marking a clear limitation.
- On LIBERO (where evaluation and training distributions nearly match), there is a slight drop of 0.2–0.5 points, further confirming that "over-augmentation can pollute near-distribution scenarios."

## Highlights & Insights
- The dual-axis efficiency optimization (caching to cut per-sample cost + coreset to cut sample count) is highly pragmatic and applicable to any field relying on large-model-generated training data: first reduce per-sample generation cost, then see if some samples can be skipped entirely.
- Using caption rewriting as a "semantic abstraction layer" is clever—LLMs decide "what to change/what to keep," diffusion models focus on rendering, aligning with their respective strengths.
- The dual-signal coreset design is noteworthy: task-aware policy loss for difficulty + task-agnostic visual embedding for diversity, avoiding extremes of "all hard samples are similar failure modes" or "diverse but trivial tasks."

## Limitations & Future Work
- Only appearance and environment are augmented; geometric/viewpoint perturbations are not addressed. Solving this requires 3D scene or NeRF/3DGS-level camera reprojection.
- Even with caching, Cosmos-Transfer 2.5 remains heavy, with per-video generation still at the minute scale—far from RL online augmentation.
- Coreset difficulty estimation depends on a pretrained RDT-1B, introducing bias; may not generalize to task families without pretrained policies.
- Slight drop on LIBERO suggests the need for "task-aware augmentation strength adjustment," rather than a one-size-fits-all approach.

## Related Work & Insights
- **vs RoboTransfer**: Both perform sim-to-real video transfer; this work achieves 2–6× improvement on geometric metrics (RMSE / Abs.Rel / Sq.Rel) and reduces generation time from minutes to seconds.
- **vs Gigaworld / GigaBrain / Embodied Dreamer**: These use world-model-driven data generation, often requiring full-stack simulators; this work operates post-hoc at the video layer, without modifying the simulator, making deployment lightweight.
- **vs $\mathbb{D}^2$ Pruning**: The original method targets static data; this work extends it to trajectories (nodes as trajectory embeddings, difficulty as policy loss), making it "embodied."

## Rating
- Novelty: ⭐⭐⭐⭐ While individual engineering points are not brand new, the combination of "dual-axis efficiency + video transfer + coreset" is highly practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers simulation + real robot, two policy families, three benchmarks, plus generation quality comparison—very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear flowcharts, concise formulas; some tables are slightly messy.
- Value: ⭐⭐⭐⭐⭐ Provides the VLA community with a directly usable "low-cost sim-to-real" augmentation toolkit.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] InfoCLIP: Bridging Vision-Language Pretraining and Open-Vocabulary Semantic Segmentation via Information-Theoretic Alignment Transfer](../../AAAI2026/segmentation/infoclip_bridging_vision-language_pretraining_and_open-vocab.md)
- [\[CVPR 2026\] Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training](../../CVPR2026/segmentation/data_warmup_complexity-aware_curricula_for_efficient_diffusion_training.md)
- [\[CVPR 2026\] MPM: Mutual Pair Merging for Efficient Vision Transformers](../../CVPR2026/segmentation/mpm_mutual_pair_merging_for_efficient_vision_transformers.md)
- [\[CVPR 2026\] Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](../../CVPR2026/segmentation/seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)
- [\[ICCV 2025\] Joint Self-Supervised Video Alignment and Action Segmentation](../../ICCV2025/segmentation/joint_self-supervised_video_alignment_and_action_segmentation.md)

</div>

<!-- RELATED:END -->
