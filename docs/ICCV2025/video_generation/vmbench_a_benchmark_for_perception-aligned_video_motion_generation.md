---
title: >-
  [Paper Note] VMBench: A Benchmark for Perception-Aligned Video Motion Generation
description: >-
  [ICCV 2025][Video Generation][Video Motion Evaluation] This paper proposes VMBench — the first comprehensive benchmark for video motion quality evaluation, featuring five-dimensional perception-aligned motion metrics (PMM) and a meta-information-guided motion prompt generation framework (MMPG). VMBench covers 969 motion categories and achieves an average improvement of 35.3% in Spearman correlation over existing methods.
tags:
  - ICCV 2025
  - Video Generation
  - Video Motion Evaluation
  - Human Perception Alignment
  - text-to-video
  - benchmark
  - Motion Quality
date: 2026-05-08
content_hash: 0142ae4832915d1d
---

# VMBench: A Benchmark for Perception-Aligned Video Motion Generation

**Conference**: ICCV 2025  
**arXiv**: [2503.10076](https://arxiv.org/abs/2503.10076)  
**Code**: [https://github.com/GD-AIGC/VMBench](https://github.com/GD-AIGC/VMBench)  
**Area**: Video Generation  
**Keywords**: Video Motion Evaluation, Human Perception Alignment, text-to-video, benchmark, Motion Quality

## TL;DR

This paper proposes VMBench — the first comprehensive benchmark for video motion quality evaluation, featuring five-dimensional perception-aligned motion metrics (PMM) and a meta-information-guided motion prompt generation framework (MMPG). VMBench covers 969 motion categories and achieves an average improvement of 35.3% in Spearman correlation over existing methods.

## Background & Motivation

Text-to-video (T2V) generation models have advanced rapidly, yet **motion quality evaluation** remains a significant challenge. Existing evaluation methods suffer from two core problems:

**Problem 1: Motion metrics are misaligned with human perception.** Current motion evaluation is largely confined to motion smoothness (e.g., VBench uses frame interpolation models), failing to capture more complex motion defects such as spatiotemporal inconsistency, violations of physical laws, and object deformation or disappearance. Feature-based metrics (FID, FVD) neglect temporal coherence; rule-based metrics (VBench) are subjectively designed and one-sided; MLLM-based methods (VideoScore) provide coarse-grained scores and, due to training bias, tend to overlook subtle motion violations.

**Problem 2: Insufficient diversity of motion prompts.** Existing benchmarks feature limited and simplistic motion prompt types, failing to comprehensively explore a model's motion generation capabilities. VMBench covers 969 motion categories — far exceeding all other benchmarks.

The paper's starting point is to **simulate the hierarchical process by which humans perceive motion** — first constructing a holistic understanding of the scene (commonsense judgments, motion smoothness), then attending to motion details (object integrity, motion amplitude, temporal coherence) — thereby designing evaluation metrics that are genuinely aligned with human perception.

## Method

### Overall Architecture

VMBench consists of two core components: (1) **Perception-aligned Motion Metrics (PMM)** — fine-grained metrics across five dimensions; and (2) **Meta-information-guided Motion Prompt Generation (MMPG)** — a structured prompt library covering six major motion patterns. Together, they constitute a complete motion evaluation benchmark.

### Key Designs

1. **Commonsense Adherence Score (CAS)**:

    - Function: Evaluates whether videos conform to human commonsense and physical laws.
    - Mechanism: Collects 10k generated videos → establishes a perceptual baseline via systematic pairwise comparisons using the VideoReward model → discretizes preference scores into five-level labels (Bad/Poor/Fair/Good/Perfect) → trains VideoMAEv2 (ViT-Giant backbone) as a classifier. The final CAS is computed via Mean Opinion Score: $\text{CAS} = \sum_{i=1}^{5} p_i G(i)$, where $p_i$ is the probability of each category and $G(i)$ maps categories to quality weights.
    - Design Motivation: Existing methods lack judgment of overall scene plausibility. Removing CAS in the ablation study results in the largest accuracy drop (−6.5%), demonstrating its central role.

2. **Motion Smoothness Score (MSS)**:

    - Function: Detects temporal artifacts and motion blur.
    - Mechanism: Utilizes Q-Align aesthetic scoring to detect inter-frame quality degradation; frames where the score drop between consecutive frames exceeds an adaptive threshold are flagged as artifact frames. $\text{MSS} = 1 - \frac{1}{T}\sum_{t=2}^T \mathbb{I}(\Delta Q_t > \tau_s(t))$, where $\Delta Q_t = Q(f_{t-1}) - Q(f_t)$ and $\tau_s(t)$ is a scene-adaptive threshold.
    - Design Motivation: Prior metrics measure smoothness via optical flow deviation or simple motion models, which are misaligned with human perception. The adaptive threshold allows higher tolerance for quality degradation in high-motion scenes.

3. **Object Integrity Score (OIS)**:

    - Function: Detects unreasonable deformation of objects during motion.
    - Mechanism: Uses MMPose to detect subject keypoints, and analyzes inter-frame variations in skeleton length and joint angles to determine whether anatomical constraints are violated. $\text{OIS} = \frac{1}{F \cdot K}\sum_{f=1}^{F}\sum_{k=1}^{K}\mathbb{I}(\mathcal{D}_f^{(k)} \leq \tau^{(k)})$
    - Design Motivation: Existing methods (e.g., DINO semantic consistency) operate at the semantic level, overlooking shape deformations that are highly salient to the human eye.

4. **Perceptible Amplitude Score (PAS)**:

    - Function: Estimates subject motion amplitude after decoupling camera motion.
    - Mechanism: GroundingDINO localizes the subject → GroundedSAM tracks it stably → CoTracker traces keypoint displacements → perceptual thresholds are set according to scene type. $\text{PAS} = \frac{1}{T}\sum_{t=1}^T \min(\frac{\bar{D}_t}{\tau_s}, 1)$
    - Design Motivation: Traditional RAFT optical flow conflates camera motion with overall motion, leading to overestimation.

5. **Temporal Coherence Score (TCS)**:

    - Function: Detects abnormal object disappearance and reappearance.
    - Mechanism: GroundedSAM2 performs instance segmentation tracking → CoTracker provides secondary verification for objects with discontinuous presence → rule-based filtering removes legitimate occlusion and scene entry/exit cases. $\text{TCS} = 1 - \frac{1}{N}\sum_{i=1}^N \mathbb{I}(\mathcal{A}_i \wedge \neg \mathcal{R})$
    - Design Motivation: Existing CLIP/DINO inter-frame cosine similarity cannot distinguish natural motion from abrupt changes.

6. **Meta-information-guided Motion Prompt Generation (MMPG)**:

    - Function: Generates diverse prompts covering six major motion patterns.
    - Mechanism: A three-stage pipeline — (a) extracts subject/scene/action metadata from datasets such as VidProm and Place365; (b) GPT-4o randomly combines metadata to generate ~50k candidate prompts with self-validation; (c) DeepSeek-R1 combined with human expert review filters the final set to 1,050 high-quality prompts.
    - Design Motivation: Ensures physical plausibility and action diversity. The six motion patterns include fluid dynamics, biological motion, mechanical motion, weather phenomena, collective behavior, and energy transfer.

### Evaluation Setup

Six open-source T2V models are evaluated (OpenSora, CogVideoX, OpenSora-Plan, Mochi 1, HunyuanVideo, Wan2.1), each generating 1,050 videos. A random subset of 1,200 videos is sampled for human annotation validation.

## Key Experimental Results

### Main Results (Spearman Correlation ρ×100 with Human Perception)

| Method | Avg. | CAS | MSS | OIS | PAS | TCS |
|--------|------|-----|-----|-----|-----|-----|
| SSIM (Rule) | 1.6 | -0.9 | -12.1 | 8.3 | 17.8 | -4.8 |
| RAFT (Rule) | -1.7 | -0.7 | -17.0 | -16.6 | 47.7 | -21.9 |
| CLIP (Rule) | 15.0 | 21.5 | 36.5 | 31.7 | -42.7 | 28.0 |
| Dover Technical (Rule) | 20.6 | 40.2 | 32.6 | 34.5 | -6.2 | 2.2 |
| InternVideo2.5 (MLLM) | 26.9 | 22.7 | 21.9 | 29.6 | 44.3 | 15.8 |
| **PMM (Ours)** | **62.2** | **69.9** | **77.1** | **65.8** | **65.2** | **54.5** |

### Ablation Study (Effect of Removing Individual Metrics on Prediction Accuracy)

| Configuration | Accuracy (%) | Note |
|---------------|-------------|------|
| Full PMM (all 5 dimensions) | 70.6 | Baseline |
| w/o TCS | 66.9 | −3.7% |
| w/o PAS | 68.7 | −1.9% |
| w/o OIS | 65.2 | −5.4% |
| w/o MSS | 64.6 | −6.0% |
| w/o CAS | 64.1 | **−6.5%, largest drop** |
| CAS only | 58.9 | Starting point |
| CAS + MSS | 66.1 | +7.2% |
| CAS + MSS + OIS | 67.3 | +1.2% |

### Key Findings

- PMM substantially outperforms both rule-based and MLLM-based methods across all five dimensions. Average Spearman correlation: 62.2% vs. 26.9% for the best MLLM (InternVideo2.5).
- CAS (Commonsense Adherence) contributes most to the overall evaluation; its removal causes the largest accuracy drop.
- PAS (Perceptible Amplitude) is negatively correlated with other dimensions (ρ = −0.18 with OIS), revealing a trade-off between motion amplitude and structural integrity.
- Wan2.1 achieves the highest PMM composite score (78.4%), producing the most realistic motion.

## Highlights & Insights

- **First evaluation of motion quality from a human perception perspective**: The five-dimensional metric design strictly follows the hierarchical process of motion perception in cognitive science (global parsing → local detail).
- **Independence and complementarity of metrics**: The negative correlation between PAS and structural/temporal metrics challenges the assumptions of traditional optical flow evaluation frameworks and underscores the necessity of decoupling motion amplitude assessment.
- **Limitations of MLLMs in motion evaluation**: Even the strongest model, InternVideo2.5, achieves only 26.9% average correlation, demonstrating that general-purpose multimodal models cannot substitute dedicated motion evaluation tools.

## Limitations & Future Work

- The evaluation metrics are aligned with **general** human perception and cannot fully account for individual or cultural differences in perceptual preferences.
- OIS currently relies on keypoint detection (MMPose), limiting its applicability to integrity assessment of non-human and non-animal objects.
- The rule-based filtering in TCS may not cover all legitimate object disappearance scenarios.
- Although the 1,050 prompts span 969 motion categories, the number of samples per category remains small.
- Complex motion scenarios involving multi-object interaction are not yet covered.

## Related Work & Insights

- **VBench**: Incorporates motion metrics such as RAFT, CLIP, DINO, and AMT, but these rule-based metrics exhibit very low correlation with human perception (average −1.7% to 15.0%).
- **EvalCrafter**: Uses Dover Technical and Warping Error, also with limited alignment.
- **VideoScore / VideoPhy**: MLLM-based approaches with coarse granularity and amplified bias.
- **Insight**: Directly translating perceptual science theory into computational metrics is an underexplored research direction.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] WorldScore: A Unified Evaluation Benchmark for World Generation](worldscore_a_unified_evaluation_benchmark_for_world_generation.md)
- [\[ICCV 2025\] Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)
- [\[ICCV 2025\] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [\[ICCV 2025\] Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation](free-form_motion_control_controlling_the_6d_poses_of_camera_and_objects_in_video.md)
- [\[ICCV 2025\] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)

</div>

<!-- RELATED:END -->
