---
title: >-
  [Paper Note] Seeing Realism from Simulation: Efficient Video Transfer for Vision-Language-Action Data Augmentation
description: >-
  [ICML 2026][Robotics][Sim-to-Real] To address the performance collapse of Vision-Language-Action (VLA) models under simple perturbations…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Sim-to-Real"
  - "VLA"
  - "Video Diffusion"
  - "Velocity Caching"
  - "Coreset Sampling"
date: 2026-05-08
content_hash: c0d5b574695679ee
---

# Seeing Realism from Simulation: Efficient Video Transfer for Vision-Language-Action Data Augmentation

**Conference**: ICML 2026  
**arXiv**: [2605.02757](https://arxiv.org/abs/2605.02757)  
**Code**: Public (Link provided at the end of the paper)  
**Area**: Robotics / Embodied AI / VLA / Video Generation and Data Augmentation  
**Keywords**: Sim-to-Real, VLA, Video Diffusion, Velocity Caching, Coreset Sampling

## TL;DR
To address the performance collapse of Vision-Language-Action (VLA) models under simple perturbations, this paper proposes a video transfer pipeline consisting of "semantic/geometric condition extraction → caption rewriting → conditional video diffusion re-rendering" to enhance simulation data with visual and environmental diversity. Combined with a three-stage velocity caching mechanism that reduces generation time by 61% and a difficulty + diversity driven coreset sampling strategy that selects only 10% of critical trajectories, the method achieves a 5–15% performance gain for RDT-1B / $\pi_0$ on Robotwin 2.0, LIBERO-Plus, and real-world robots.

## Background & Motivation

**Background**: VLA models (e.g., RDT, $\pi_0$, $\pi_{0.5}$, OpenVLA, ACT) rely on large-scale real robot trajectories for end-to-end training. However, real-world data collection is expensive, slow, and difficult to scale. While simulation data is cheap and parallelizable, it suffers from significant visual and environmental gaps; policies trained on it often collapse when encountering fluctuations in lighting, background, or viewpoints.

**Limitations of Prior Work**: LIBERO-Plus reports that policies with a 95% success rate can drop below 30% under minor perturbations; LIBERO-PRO shows nearly 0% success under changes in object positions and instructions. This suggests models are memorizing action sequences rather than truly understanding tasks. Simple random noise or color jittering fails to capture the semantic complexity of real environments.

**Key Challenge**: To make simulation data "look real," the most direct approach is conditional video diffusion re-rendering. However, models like Cosmos-Transfer take approximately 40 minutes to process a 5-second 720p video on an A100, making it impossible to scale across millions of simulation trajectories.

**Goal**: (1) Design a pipeline to transfer entire simulation videos into high-fidelity real styles while strictly preserving action trajectories; (2) Reduce generation costs to a scalable level; (3) Ensure computational resources are focused on the most critical data through selective augmentation rather than full-scale processing.

**Key Insight**: The authors perform "addition" on generation quality—using caption rewriting + depth geometric control + conditional video diffusion to create realistic videos with diverse environments—and "subtraction" on generation overhead—observing that the velocity field in flow-based diffusion remains nearly constant during the middle stages, allowing for cache reuse.

**Core Idea**: Video augmentation is split into two orthogonal efficiency axes: "generation" and "selection." On the generation side, velocity caching reduces the cost per video. On the selection side, a graph-based coreset sampling driven by difficulty + diversity reduces the number of trajectories that need to be generated.

## Method

### Overall Architecture
Given a set of simulation training trajectories $\mathcal{S}=\{s_1,\dots,s_n\}$: (1) Coreset sampling selects a subset $\mathcal{S}'\subset\mathcal{S}$ for generation; (2) For each selected video, VideoChat2 extracts captions, and Qwen3-8B rewrites them to introduce environmental variables like background and object colors, while depth maps are extracted for geometric control; (3) Cosmos-Transfer 2.5 uses the new caption and depth as conditions for video diffusion, producing "realized" videos with altered visual styles but identical action trajectories; (4) The generated videos are mixed with the original 90% simulation data to train the VLA. The two most critical accelerators in this pipeline are velocity caching and coreset sampling.

### Key Designs

1.  **Conditional Video Transfer (Semantic + Geometric Dual Conditions)**:
    - **Function**: Transforms a simulation robot video into a real-style video with the same actions but in a diversified environment.
    - **Mechanism**: VideoChat2 extracts temporal captions describing interactions, objects, and spatial relationships; Qwen3-8B rewrites these captions to introduce variety in backgrounds and object colors while preserving task intent; depth maps are extracted from the original video as stable geometric constraints (superior to edge/blur/seg for geometry preservation); finally, Cosmos-Transfer 2.5 generates the video through iterative denoising conditioned on the new caption and depth.
    - **Design Motivation**: Changing only the caption causes geometric drift and distortion of robot poses, leading to action loss; providing only depth lacks semantic diversity. The two conditions are complementary, handling "looking different" and "doing the same thing," respectively.

2.  **Three-stage Velocity Caching**:
    - **Function**: Reuses velocity predictions in flow-based video diffusion to bypass numerous transformer forward passes.
    - **Mechanism**: After analyzing the temporal curve of $\|v_{t+1}-v_t\|$, the authors identified a three-stage dynamic: rapid change at the start, near-stability in the middle, and fine-tuning at the end. The $N$-step denoising is thus divided into an initial phase ($t<t_s$, computed every step), a stable phase ($t_s\leq t< t_f$, computed every $\alpha$ steps and reused elsewhere), and a final phase ($t\geq t_f$, computed every step). The start of the stable phase is detected via a threshold $\frac{\|v_t-v_{t+1}\|}{\|v_0-v_1\|} < k$, with parameters $k=0.4, \alpha=8, m=3$.
    - **Design Motivation**: General caching strategies (like DeepCache) assume both ends are equally important without considering diffusion dynamics. The three-stage approach aligns with the "outline → refinement → finalization" denoising rhythm, allowing for a 61.2% reduction in time with minimal quality loss.

3.  **Difficulty × Diversity Coreset Sampling**:
    - **Function**: Selects the few most valuable samples for augmentation from large-scale simulation trajectories.
    - **Mechanism**: Extends $\mathbb{D}^2$ Pruning to video. Difficulty $x_i = \frac{1}{|\mathcal{T}_i|}\sum_{t}\mathcal{L}_{\text{policy}}(s_i^{(t)};\theta)$ is estimated using the policy loss of RDT-1B. Diversity uses Cosmos-Embed1 to obtain 768-dimensional embeddings $\phi(s_i)$, building a kNN graph with an RBF kernel $e_{i,j}=\exp(-\gamma_f\|v_i-v_j\|^2)$. Forward message passing aggregates neighborhood difficulty $x_i' = x_i + \sum_{j\in\mathcal{N}(i)}e_{i,j}\cdot x_j$; high $x_i'$ are selected greedily, while backward message passing suppresses the scores of similar neighbors to avoid redundancy.
    - **Design Motivation**: Focusing solely on difficulty risks getting stuck in a "hard cluster," while focusing only on diversity might include trivial samples. Combining both ensures "hard and unique" trajectories are prioritized, approaching the effect of full augmentation with only a 10% budget.

### Loss & Training
The core VLA loss remains unchanged; the training set is replaced with original simulation data plus real-style videos augmented via coreset sampling. The paper compares two mixing strategies: *mixture* (retaining all original data + adding augmented data) and *replacement* (replacing selected coresets with augmented data). Results show that $\pi_0$ benefits more from mixture, while the stronger $\pi_{0.5}$ prefers replacement—stronger models can better handle larger distribution shifts.

## Key Experimental Results

### Main Results
Original vs. Augmented on Robotwin 2.0 Single-task (RDT-1B) in "Hard" scenarios:

| Task | Ori. (Hard) | Aug. (Hard) | $\Delta$ |
|------|-------------|-------------|----------|
| adjust_bottle | 72.0 | 82.0 | +10.0 |
| beat_block_hammer | 36.0 | 48.0 | +12.0 |
| place_burger_fries | 26.0 | 38.0 | +12.0 |
| open_laptop | 30.0 | 44.0 | +14.0 |
| **average** | **29.0** | **39.0** | **+10.0** |

LIBERO-Plus spatial suite, $\pi_0$ + 50% Coreset Augmentation:

| Perturbation Type | Ori. | Aug. | $\Delta$ |
|----------|------|------|----------|
| objects layout | 69.6 | 86.2 | +16.6 |
| language instructions | 37.9 | 55.9 | +22.0 |
| background textures | 81.1 | 87.6 | +6.5 |
| robot initial states | 10.3 | 6.3 | −4.0 |
| camera view points | 21.3 | 15.2 | −6.1 |
| **average** | **42.7** | **47.8** | **+5.1** |

Real-world AgileX Piper (Two tasks, three scenes, 10 trials each): $\pi_0$ average success rate increased from 60% → 75% (+15%), $\pi_{0.5}$ from 60% → 73% (+13%).

### Ablation Study

| Setting | Robotwin Hard Avg. | Description |
|------|--------------------|------|
| Original Sim | 29.0 | baseline |
| Aug. w/ velocity cache | 26.5 | Caching acceleration |
| Aug. w/o velocity cache | 27.0 | Full computation |
| Aug. (No coreset, full aug.) | 39.0 | Upper bound |

Video generation quality vs. RoboTransfer (adjust_bottle, lower is better for RMSE/Abs.Rel/Sq.Rel):

| Method | RMSE | Abs.Rel | Sq.Rel | sim$\uparrow$ |
|------|---------|---------|--------|---------------|
| RoboTransfer | 0.46 | 0.37 | 0.39 | 21.5 |
| **Ours** | **0.28** | **0.16** | **0.07** | **26.3** |

### Key Findings
- Velocity caching results in negligible performance loss (26.5 vs 27.0) while cutting generation time by 61%, proving that flow-based diffusion has significant computational redundancy in its middle stages that can be exploited.
- 10% coreset sampling improved RDT-1B's average from 23% to 31% on Robotwin multi-task (300 trajectories/task). This indicates that "choosing accurately" is far more cost-effective than "adding more" in repetitive simulation data.
- In real robot experiments, augmentation provided the greatest benefits for background and position perturbations ("Stack Tape" position 5/10 → 8/10). However, it caused drops in performance for robot initial states and camera viewpoints—this method only augments appearance and is ineffective against geometric or viewpoint changes.
- On LIBERO (where evaluation distribution is nearly identical to training), a slight drop of 0.2–0.5 points was observed, confirming that "over-augmentation can contaminate near-distribution scenarios."

## Highlights & Insights
- The dual-axis efficiency optimization (caching per-sample cost + coreset sample count) is highly pragmatic and applicable to any field relying on large-model data generation: first reduce the cost of generating each sample, then determine if specific samples need to be generated at all.
- Using caption rewriting as a "semantic abstraction layer" is a clever design—the LLM decides "what to change / what to keep," while the diffusion model handles rendering, respecting their respective capabilities.
- The dual-signal design of the coreset is worth noting: using task-aware policy loss for difficulty and task-agnostic visual embeddings for diversity avoids the extremes of "hard but identical failure modes" or "diverse but trivial tasks."

## Limitations & Future Work
- Augmentation is limited to appearance and environment; geometric and viewpoint perturbations are not covered. Addressing this requires 3D scenes or camera re-projection at the NeRF/3DGS level.
- Even with caching, Cosmos-Transfer 2.5 remains heavy, with generation times on the order of minutes, which is far from RL online augmentation requirements.
- Coreset difficulty estimation depends on a pre-trained RDT-1B, which introduces its own bias and may fail for task families without pre-trained policies.
- The slight decline on LIBERO suggests a need for "task-aware augmentation intensity adjustment" rather than a one-size-fits-all approach.

## Related Work & Insights
- **vs RoboTransfer**: While both perform sim-to-real video transfer, this work provides a 2–6× improvement in geometric metrics (RMSE / Abs.Rel / Sq.Rel) and reduces generation time from minutes to seconds.
- **vs Gigaworld / GigaBrain / Embodied Dreamer**: These world-model-driven generation methods often require full-stack simulators; this method operates post-hoc at the video level, making it easier to deploy.
- **vs $\mathbb{D}^2$ Pruning**: Whereas the original method targets static data, this work extends it for "embodied" use by replacing nodes with trajectory embeddings and classification loss with policy loss.

## Rating
- Novelty: ⭐⭐⭐⭐ Individual engineering components are not brand new, but the combination of "dual-axis efficiency + video transfer + coreset" is highly practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across simulation and real robots, two policy families, three benchmarks, and generation quality comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear flowcharts and concise formulas, though some table layouts are slightly cluttered.
- Value: ⭐⭐⭐⭐⭐ Provides a ready-to-use "low-cost sim-to-real" augmentation toolkit for the VLA community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] StableVLA: Towards Robust Vision-Language-Action Models without Extra Data](stablevla_towards_robust_vision-language-action_models_without_extra_data.md)
- [\[ICML 2026\] From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model](from_abstraction_to_instantiation_learning_behavioral_representation_for_vision-.md)
- [\[ICLR 2026\] TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models](../../ICLR2026/robotics/twinvla_data-efficient_bimanual_manipulation_with_twin_single-arm_vision-languag.md)
- [\[ICML 2026\] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)
- [\[ICLR 2026\] D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI](../../ICLR2026/robotics/d2e_scaling_vision-action_pretraining_on_desktop_data_for_transfer_to_embodied_a.md)

</div>

<!-- RELATED:END -->
