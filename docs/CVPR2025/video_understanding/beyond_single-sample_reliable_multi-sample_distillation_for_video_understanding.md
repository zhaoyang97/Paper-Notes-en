---
title: >-
  [Paper Note] Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding
description: >-
  [CVPR 2025][Video Understanding][Knowledge Distillation] Proposes R-MSD (Reliable Multi-Sample Distillation), which addresses the issue of unreliable single-sample teacher supervision in black-box distillation of video LVLMs by sampling multiple teacher responses for each input and incorporating task-adaptive quality matching. The 4B student model consistently improves performance on benchmarks such as VideoMME (+1.5%), Video-MMMU (+3.2%), and MathVerse (+3.6%).
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Knowledge Distillation"
  - "Multi-Sample Supervision"
  - "Adversarial Training"
  - "Large Vision-Language Models"
  - "Teacher Sampling Variance"
date: 2026-05-08
content_hash: eceba9e635893b8f
---

# Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding

**Conference**: CVPR 2025  
**arXiv**: [2603.11423](https://arxiv.org/abs/2603.11423)  
**Code**: To be confirmed  
**Area**: Video Understanding / Knowledge Distillation  
**Keywords**: Knowledge Distillation, Video Understanding, Multi-Sample Supervision, Adversarial Training, Large Vision-Language Models, Teacher Sampling Variance

## TL;DR

Proposes R-MSD (Reliable Multi-Sample Distillation), which addresses the issue of unreliable single-sample teacher supervision in black-box distillation of video LVLMs by sampling multiple teacher responses for each input and incorporating task-adaptive quality matching. The 4B student model consistently improves performance on benchmarks such as VideoMME (+1.5%), Video-MMMU (+3.2%), and MathVerse (+3.6%).

## Background & Motivation

### 1. Background
Large Vision-Language Models (LVLMs) have made significant progress in video understanding tasks, but their deployment is constrained by high computational costs. Knowledge distillation is an effective means to transfer knowledge from a powerful teacher model to a smaller student model.

### 2. Limitations of Prior Work
- **Unreliable Single-Sample Distillation**: Existing distillation methods assume that a single teacher sample per input provides reliable supervision, but this assumption fails in video understanding.
- **Large Cross-Question Variance**: The quality across 200 samples spans $[0.10, 1.0]$ ($\mu=0.75$, $\sigma=0.22$), with MCQs being stable ($\mu=0.96$) while visual QA exhibits high variation ($\mu=0.64$).
- **Intra-Question Sampling Uncertainty**: The sampling standard deviation ($\sigma_{\text{sampling}}$) of repeated samplings ranges from 0.07 (MCQs) to 0.15 (visual QA), with OCR quality ranging from $[0.50, 0.85]$.
- **Format Violations**: 1% overall, reaching up to 10% in temporal QA.
- **Ignored Task Heterogeneity**: Closed-ended tasks (verifiable outputs) and open-ended tasks (natural language descriptions) require different quality assessment strategies, yet existing pipelines treat them uniformly.

### 3. Key Challenge
How to effectively model teacher sampling variance under black-box distillation settings to provide reliable supervisory signals for closed-ended and open-ended tasks respectively?

### 4. Key Observation
Teacher sampling variance has two dimensions: (1) cross-question variance—the teacher quality varies across different questions; (2) intra-question variance—the quality of repeated samplings for the same question varies significantly. Additionally, RL methods are limited by the base model's distribution and cannot introduce entirely new reasoning modes like distillation does.

### 5. Core Idea
Sample $K$ teacher responses for each input to construct a teacher pool, and adaptively select the matching strategy based on the task type: use ground-truth (GT) quality-weighted matching for closed-ended tasks, and uniform matching for open-ended tasks to avoid fragile lexical metric bias.

### 6. Design Motivation
The supervision selection strategy is as important as the quantity of supervision—improving distillation quality by enhancing supervision reliability rather than simply increasing the sample size.

## Method

### Overall Architecture
R-MSD consists of three major components: (1) multi-sample teacher collection (sampling $K$ teacher outputs for each input), (2) task-adaptive quality assessment (using GT-based scoring for closed-ended tasks and uniform treatment for open-ended tasks), and (3) task-adaptive matching + online discriminator. Training is divided into two stages: Stage 1 SFT warmup, and Stage 2 RL-based adversarial distillation.

### Key Designs

#### Key Design 1: Task-Adaptive Quality Assessment and Matching
- **Function**: Determines how to evaluate the quality of teacher responses and match them to student rollouts based on task types.
- **Mechanism**: Since closed-ended tasks have objective correct answers, GT quality-weighted sampling is used (high-quality teacher responses are sampled more frequently). For open-ended tasks where reliable metrics are lacking, uniform matching $p_k = 1/K$ is applied.
- **Design Motivation**: In closed-ended tasks, high-quality teacher responses are objectively superior, justifying quality-biased matching. In open-ended tasks, lexical overlap metrics penalize semantically equivalent but differently worded answers, leading to false negatives.
- **Quality Score**: $q_k = \mathbb{I}(\text{valid}(T_k)) \cdot \text{Metric}(T_k, y^*)$, where $\text{Metric}$ is a task-specific metric (exact match for MCQs, temporal IoU for temporal localization, IoU for spatial localization, and $\epsilon$-accuracy for numerical tasks).

#### Key Design 2: Two-Stage Training Target
- **Function**: Performs SFT warmup to initialize the student first, followed by refinement via adversarial distillation.
- **Mechanism**: Stage 1 selects the best teacher response for standard autoregressive cross-entropy training. In Stage 2, the student online samples $N$ rollouts, each paired with a teacher response, and optimizes them through a composite reward.
- **Design Motivation**: SFT provides a stable initialization to prevent early collapse in adversarial training. The two-stage design progressively transitions from "learning from the best sample" to "learning from the distribution".
- **Stage 2 Composite Reward**: $R(S_i) = \alpha \cdot D_\phi(S_i) + \beta \cdot R_{\text{outer}}(S_i) + \eta \cdot R_{\text{task}}(S_i) + \delta \cdot R_{\text{content}}(S_i)$
    - $D_\phi$: Discriminator score (higher means closer to the teacher).
    - $R_{\text{outer}}$: Outer response format validation.
    - $R_{\text{task}}$: Task-specific format compliance check.
    - $R_{\text{content}}$: GT correctness score for closed-ended tasks.

#### Key Design 3: Discriminator and Adversarial Training
- **Function**: Trains an online discriminator to provide distribution-level supervision signals.
- **Mechanism**: Reuses the value head of the critic to score the last token of the response, and trains the discriminator using a quality-weighted GAD pairwise loss.
- **Design Motivation**: Static offline reward models can be exploited as the policy improves (reward hacking); an online co-evolving discriminator provides adaptive supervision.

### Loss & Training
- **Stage 1**: $\mathcal{L}_{\text{SFT}} = -\log \pi_S(T_{\text{best}} \mid V, Q)$ (standard cross-entropy)
- **Stage 2**: $\mathcal{L}_{\text{RL}} = -\mathbb{E}[R(S)] + \gamma \cdot D_{\text{KL}}(\pi_S \parallel \pi_{\text{ref}})$ (policy gradient + KL constraint)
- **Discriminator**: $\mathcal{L}_D = \mathbb{E}[q_{m(i)} \cdot -\log \sigma(D_\phi(T_{m(i)}) - D_\phi(S_i))]$ (quality-weighted pairwise loss)

## Key Experimental Results

### Main Results: Video and Image QA Benchmarks (4B Student Model, 64-frame Evaluation)

| Model | VideoMME | Video-MMMU | WorldSense | LongVideoBench | MLVU_MCQ | MathVista | MathVerse |
|------|----------|------------|------------|----------------|----------|-----------|-----------|
| Qwen3-VL-4B | 63.8 | 55.4 | 46.7 | 59.3 | 72.4 | 69.5 | 45.7 |
| Original SFT+RL (4B) | 64.0 | 55.9 | 46.3 | 57.2 | 73.1 | 71.2 | 46.8 |
| **R-MSD (4B)** | **65.3** | **58.6** | **49.2** | 58.8 | **73.2** | **72.1** | **49.3** |

- Improvements over the base model: VideoMME +1.5, Video-MMMU +3.2, WorldSense +2.5, MathVerse +3.6.
- The original SFT+RL baseline yielded only marginal improvements under the same budget, highlighting the advantage of task-adaptive multi-sample supervision.

### V-STaR Spatiotemporal Grounding Results (4B)

| Model | When Chain1 (tIoU) | When Chain2 | Where Chain1 (IoU) | Where Chain2 |
|------|-------------------|-------------|-------------------|-------------|
| Qwen3-VL-4B | 21.3 | 18.5 | 22.3 | 5.0 |
| **R-MSD (4B)** | **25.2** | **23.4** | **24.8** | **7.0** |

### Ablation Study: Core Components

| Setup | K | Filtering | Weighting | VideoMME | Video-MMMU |
|------|---|------|------|----------|------------|
| A (Single-Sample) | 1 | No | No | 63.8 | 54.4 |
| B (Multi-Sample) | 4 | No | No | 64.5 | 55.9 |
| C (+Filtering) | 4 | Yes | No | 65.0 | 57.2 |
| **D (Full)** | 4 | Yes | Yes | **65.3** | **58.6** |

### Sensitivity Analysis

| Teacher Samples K | VideoMME | Video-MMMU |
|-------------|----------|------------|
| 2 | 64.8 | 57.1 |
| 4 | 65.3 | 58.6 |
| 8 | 65.5 | 58.9 |

| Quality Threshold $\tau$ | Valid Sample Ratio | VideoMME | Video-MMMU |
|-----------|------------|----------|------------|
| 0.0 | 100% | 64.5 | 55.9 |
| 0.2 | 87% | 65.0 | 58.1 |
| 0.3 | 72% | 65.3 | 58.6 |
| 0.5 | 45% | 64.8 | 57.2 |

### Key Findings
1. Significant improvements are observed from $K=1$ to $K=4$ (VideoMME +0.7, Video-MMMU +1.5), with diminishing returns at $K=8$.
2. Quality filtering ($\tau=0.3$) substantially improves quality while retaining 72% of samples; an aggressive $\tau=0.5$ leads to insufficient samples.
3. High-variance tasks benefit the most: Video-MMMU (+3.2) > VideoMME (+1.5) > MLVU_MCQ (+0.8).
4. For closed-ended tasks, GT-based scoring surpasses uniform weighting (57.8 vs 56.2), whereas the opposite holds for open-ended tasks (59.1 vs 58.4).
5. Pass@k analysis indicates that R-MSD primarily boosts the correctness probability of a single sample (+3.2% Pass@1) rather than expanding the performance ceiling.

## Highlights & Insights

1. **Precise Problem Diagnostic**: Systematically quantifies the two dimensions of teacher sampling variance (cross-question + intra-question) in video LVLM distillation for the first time.
2. **Task-Adaptive Design**: Avoids a "one-size-fits-all" approach—using quality weighting for closed-ended tasks and uniform matching for open-ended tasks is simple yet effective.
3. **Supervision Quality > Supervision Quantity**: The core insight is that multi-sampling alone is insufficient (the B $\to$ C transition shows substantial gains from filtering alone); quality awareness is the key.
4. **Online Discriminator Prevents Reward Hacking**: Compared to static reward models, the online co-evolving critic is much more robust.
5. **Cross-Modal Generalization**: Demonstrates improvements on image QA (MathVista, MathVerse) as well, indicating the generality of the method.

## Limitations & Future Work

1. **Closed-Ended Quality Scoring Relies on GT Annotations**: Not directly applicable to weakly supervised scenarios.
2. **Conservative Strategy for Open-Ended Tasks**: Uniform weighting preserves semantic diversity but does not explicitly exploit semantic correctness.
3. **Linear Increase in Training Cost**: The multi-sample protocol increases the training time roughly by a factor of $K$.
4. **No Significant Improvement on LongVideoBench**: This may be due to training with 16 frames, whereas LongVideoBench requires longer context.
5. **Validation Limited to 4B and 2B Scales**: It remains unexplored whether larger student models would obtain similar benefits.
6. **Single Teacher**: Only Qwen3-VL-235B is used; the combination of multi-teacher aggregation and multi-sample distillation is not explored.

## Related Work & Insights

- **Comparison with GAD**: Building upon the online adversarial distillation of GAD, R-MSD incorporates a task-adaptive multi-sample mechanism specifically targeting video-inherent supervisory noise.
- **Comparison with RLVR**: RL improves sampling efficiency but does not expand the boundaries of reasoning capabilities, whereas distillation can transfer the teacher's unique reasoning patterns.
- **Comparison with OPD**: R-MSD unifies two major trends: supervised distillation (via quality-weighted selection) and RL/adversarial distillation (via distribution-level alignment).
- **Insights**: The methodology for analyzing teacher sampling variance can be transferred to other multimodal distillation scenarios. The design principles of task-adaptive strategies (using GT for verifiable tasks and uniform matching for non-verifiable ones) hold general applicability.

## Rating
- Novelty: ⭐⭐⭐⭐ (First to systematically quantify teacher sampling variance and propose task-adaptive multi-sample distillation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (8 benchmarks, complete ablation, sensitivity analysis, Pass@k, task-adaptive validation)
- Writing Quality: ⭐⭐⭐⭐ (Clearly defined problems, well-organized experiments, thoroughly articulated motivations)
- Value: ⭐⭐⭐⭐ (Practical improvements in video LVLM distillation, highly generalizable task-adaptive concepts)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ViTED: Video Temporal Evidence Distillation](vited_video_temporal_evidence_distillation.md)
- [\[ECCV 2024\] Boosting 3D Single Object Tracking with 2D Matching Distillation and 3D Pre-training](../../ECCV2024/video_understanding/boosting_3d_single_object_tracking_with_2d_matching_distillation_and_3d_pre-trai.md)
- [\[CVPR 2025\] MLVU: Benchmarking Multi-task Long Video Understanding](mlvu_benchmarking_multi-task_long_video_understanding.md)
- [\[CVPR 2025\] MMVU: Measuring Expert-Level Multi-Discipline Video Understanding](mmvu_measuring_expert-level_multi-discipline_video_understanding.md)
- [\[CVPR 2025\] MUST: The First Dataset and Unified Framework for Multispectral UAV Single Object Tracking](must_the_first_dataset_and_unified_framework_for_multispectral_uav_single_object.md)

</div>

<!-- RELATED:END -->
