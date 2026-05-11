---
title: >-
  [Paper Note] FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters
description: >-
  [CVPR 2026][Video Generation][Video generation acceleration] FastLightGen proposes a three-stage distillation algorithm that, for the first time, achieves joint distillation of sampling steps and model size. By identifying redundant layers, applying dynamic probabilistic pruning, and performing well-guided teacher guidance distribution matching, it compresses HunyuanVideo/WanX into a lightweight generator with 4 sampling steps and 30% parameter pruning, achieving approximately 35× speedup while surpassing the teacher model in performance.
tags:
  - CVPR 2026
  - Video Generation
  - Video generation acceleration
  - step distillation
  - model pruning
  - distribution matching
  - DiT compression
date: 2026-05-08
content_hash: 957ddc659b938408
---

# FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters

**Conference**: CVPR 2026
**arXiv**: [2603.01685](https://arxiv.org/abs/2603.01685)
**Code**: None
**Area**: Video Generation
**Keywords**: Video generation acceleration, step distillation, model pruning, distribution matching, DiT compression

## TL;DR

FastLightGen proposes a three-stage distillation algorithm that, for the first time, achieves joint distillation of sampling steps and model size. By identifying redundant layers, applying dynamic probabilistic pruning, and performing well-guided teacher guidance distribution matching, it compresses HunyuanVideo/WanX into a lightweight generator with 4 sampling steps and 30% parameter pruning, achieving approximately 35× speedup while surpassing the teacher model in performance.

## Background & Motivation

**Background**: Large-scale video generation models (HunyuanVideo, WanX) are based on DiT with 13B+ parameters and multi-step denoising. Generating a 5-second video on an H100 takes approximately 20 minutes.

**Core Problem**:
   - Existing acceleration methods either reduce steps (LCM/DMD) or reduce parameters (F3-Pruning/ICMD), with no joint optimization
   - Extreme step distillation (1–2 steps) leads to drastic performance degradation
   - Joint distillation achieves greater speedup at the same performance level (4 steps + 50% parameters = 50× vs. step-only 3 steps = 33.3×)

**Goal**: A three-stage pipeline — identifying redundant layers, dynamic probabilistic pruning, and well-guided teacher guidance distribution matching.

## Method

### Overall Architecture

A three-stage distillation pipeline: Stage I identifies unimportant blocks, Stage II applies dynamic probabilistic pruning training, and Stage III performs fine-grained distribution matching.

### Key Designs

#### 1. Stage I: Identifying Unimportant Model Blocks

Each DiT block is skipped individually, and importance is evaluated via ELBO estimation using the Tweedie formula. A **U-shaped pattern** is discovered: the initial and final layers are most critical, while the middle layers are redundant. In HunyuanVideo, double blocks are more critical than single blocks.

#### 2. Stage II: Dynamic Probabilistic Pruning Training

Unimportant layers are randomly skipped following a Bernoulli distribution ($p = 0.5$), constructing parameter-sharing unpruned/pruned models:

- Distillation loss: pruned outputs are aligned to unpruned outputs (stop gradient)
- Key finding: $\alpha = 1$ (completely removing GT supervision, using distillation only) yields the best results; "soft" supervision outperforms "hard" GT
- The output is a robust model that performs well across different depth configurations

#### 3. Stage III: Fine-Grained Distribution Matching

Based on the DMD2 framework, well-guided teacher guidance is introduced:

- The real DiT simultaneously uses pruned and unpruned outputs
- $\beta_1$ (inter CFG) controls text guidance strength; $\beta_2$ (intra CFG) controls unpruned-to-pruned guidance
- CFG is sampled from a uniform distribution to enhance robustness
- Two failure modes are avoided: an overly weak teacher (ineffective) and an overly strong teacher (student unable to follow)

### Loss & Training

- Stage II: 16× H100, lr = 1e-5, 4000 iterations, ~64 GPU days
- Stage III: lr = 5e-7, 1000 iterations, ~16 GPU days
- Optimal configuration: $(\alpha, \beta_1, \beta_2) = (1, 3.5, 0.25)$ for WanX
- Overly long training is inadvisable (excessive motion / oversaturated color)

## Key Experimental Results

### Main Results

**VBench-I2V Comparison (WanX-TI2V, Table 2)**:

| Method | motion smooth | dynamic deg | aesthetic | imaging | average | time |
|------|-------------|------------|-----------|---------|---------|------|
| Euler (teacher) | 0.982 | 0.461 | 0.653 | 0.711 | 0.790 | 885s |
| DMD2 | 0.977 | 0.160 | 0.583 | 0.683 | 0.716 | 35.4s |
| LCM | 0.979 | 0.003 | 0.570 | 0.665 | 0.684 | 35.4s |
| MagicDistillation | 0.980 | 0.561 | 0.634 | 0.701 | 0.798 | 35.4s |
| **FastLightGen** | **0.983** | 0.500 | **0.656** | **0.717** | 0.794 | **28.3s** |

**Comparison with Open-Source VDMs (Table 1)**:

| Method | average |
|------|---------|
| CogVideoX-I2V | 0.759 |
| SVD-XT-1.0 | 0.789 |
| WanX-TI2V (teacher) | 0.790 |
| **FastLightGen** | **0.794** |

### Ablation Study

**Distillation Weight Ablation (Table 4)**:

| distill weight alpha | average |
|---------------------|---------|
| 0.0 | 0.780 |
| 0.5 | 0.780 |
| 0.7 | 0.788 |
| **1.0** | **0.791** |

**Intra CFG Ablation (Table 5, $\beta_1 = 3.5$)**:

| beta_2 | dynamic deg | average |
|--------|------------|---------|
| 0.00 | 0.459 | 0.812 |
| **0.25** | 0.500 | **0.820** |
| 0.75 | 1.000 | 0.848 (with jitter) |

### Key Findings

- 4 steps + 30% pruning (retaining 70% of parameters) offers the optimal cost-effectiveness, achieving approximately 35.71× speedup
- Joint distillation achieves higher speedup at equivalent performance compared to single-dimension distillation (50× vs. 33.3×)
- $\alpha = 1$ pure distillation is optimal for Stage II
- Aesthetic quality and imaging quality surpass the teacher model

## Highlights & Insights

1. **Joint distillation paradigm**: First demonstration that joint step + size distillation outperforms single-dimension approaches
2. **Well-guided teacher**: Inter/intra CFG independently controls two orthogonal dimensions
3. **Dynamic probabilistic pruning**: A single model adapts to different depth configurations
4. **U-shaped importance**: A general finding that the initial and final layers of VDMs are most critical

## Limitations & Future Work

1. Validated only on the TI2V task
2. High training cost (~80 GPU days)
3. Motion artifacts appear when $\beta_2$ is large
4. Pruning operates only at the block level
5. Sensitivity to data quality

## Related Work & Insights

- **DMD2**: Foundation for distribution matching distillation
- **MagicDistillation**: Strong step distillation baseline
- **ICMD**: Pioneer in video size distillation
- **Insight**: "An overly strong teacher can be harmful" warrants further validation

## Rating

| Dimension | Score (1–5) | Notes |
|------|-----------|------|
| Novelty | 4 | Joint distillation + well-guided teacher |
| Technical Depth | 4 | Fine-grained three-stage design |
| Experimental Thoroughness | 4.5 | Multi-model, multi-metric, thorough ablation |
| Writing Quality | 4 | Clear figures and tables |
| Value | 4.5 | 35× speedup is highly significant |
| Overall | 4.2 | |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LightMover: Generative Light Movement with Color and Intensity Controls](lightmover_generative_light_movement_with_color_and_intensity_controls.md)
- [\[NeurIPS 2025\] MagCache: Fast Video Generation with Magnitude-Aware Cache](../../NeurIPS2025/video_generation/magcache_fast_video_generation_with_magnitudeaware_cache.md)
- [\[ICCV 2025\] Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks](../../ICCV2025/video_generation/generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)
- [\[CVPR 2026\] LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation](linvideo_a_post-training_framework_towards_on_attention_in_efficient_video_gener.md)
- [\[CVPR 2026\] DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching](disca_accelerating_video_diffusion_transformers_wi.md)

</div>

<!-- RELATED:END -->
