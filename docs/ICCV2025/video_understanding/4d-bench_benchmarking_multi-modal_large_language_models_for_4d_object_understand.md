---
title: >-
  [Paper Note] 4D-Bench: Benchmarking Multi-Modal Large Language Models for 4D Object Understanding
description: >-
  [ICCV 2025][Video Understanding][4D understanding] 4D-Bench is the first benchmark for evaluating multi-modal large language models (MLLMs) on 4D object understanding. It encompasses two tasks—4D object question answering and captioning—and reveals that even the state-of-the-art GPT-4o achieves only 63% accuracy against a human baseline of 91%, exposing significant deficiencies in multi-view temporal reasoning among current MLLMs.
tags:
  - ICCV 2025
  - Video Understanding
  - 4D understanding
  - MLLM evaluation
  - multi-view temporal reasoning
  - benchmark
  - visual question answering
date: 2026-05-08
content_hash: e1f0c682b72cdf37
---

# 4D-Bench: Benchmarking Multi-Modal Large Language Models for 4D Object Understanding

**Conference**: ICCV 2025
**arXiv**: [2503.17827](https://arxiv.org/abs/2503.17827)
**Code**: [https://4dbench.github.io/](https://4dbench.github.io/)
**Area**: Video Understanding / Multi-Modal
**Keywords**: 4D understanding, MLLM evaluation, multi-view temporal reasoning, benchmark, visual question answering

## TL;DR

4D-Bench is the first benchmark for evaluating multi-modal large language models (MLLMs) on 4D object understanding. It encompasses two tasks—4D object question answering and captioning—and reveals that even the state-of-the-art GPT-4o achieves only 63% accuracy against a human baseline of 91%, exposing significant deficiencies in multi-view temporal reasoning among current MLLMs.

## Background & Motivation

4D digital assets (dynamic 3D objects) are increasingly important in digital twins, augmented reality, gaming, and related domains. As MLLMs (e.g., GPT-4o, Qwen2-VL) have achieved substantial progress in 2D image/video understanding, a natural question arises: **can these models understand 4D objects?**

A critical gap currently exists:

**No public benchmark for 4D language understanding**: existing benchmarks either focus on 2D images/videos (ignoring multi-view understanding) or on static 3D scenes (ignoring temporal dynamics).

**Unique challenges of 4D understanding**:
   - **Multi-view ambiguity**: the same object presents different appearances from different viewpoints, requiring integration of multi-view information.
   - **Temporal evolution**: object parts move over time, necessitating tracking and reasoning.
   - **Joint reasoning across views and time**: as illustrated in Figure 1, a robot's right hand may be occluded from certain viewpoints and eventually disappear, so answering questions requires selecting the appropriate viewpoint, localizing the part, and tracking its changes.

**Counterfactual testing**: the synthetic objects in 4D-Bench can provide counterfactual data that violates physical laws or common sense (e.g., a spider with six legs, a ball rolling out of a hole), testing whether MLLMs genuinely understand the input rather than relying on memorized priors.

**Core Idea**: render 4D objects as multi-view videos and feed them directly into existing MLLMs for evaluation, without constructing new 4D understanding models. Carefully designed evaluation tasks expose specific shortcomings of MLLMs.

## Method

### Overall Architecture

4D-Bench consists of two tasks:
1. **4D Object Question Answering (QA)**: 751 four-choice questions covering 736 4D objects.
2. **4D Object Captioning**: 580 4D objects, each with 5 manually annotated captions.

### Key Designs

#### 1. Data Collection and Filtering

Data is sourced from dynamic 3D objects in Objaverse-XL and processed through a two-stage filtering pipeline:

- **Motion analysis**: motion boundaries are detected via pixel-change analysis to extract valid video segments, ensuring only dynamic objects are retained.
- **Visual quality assessment**: human annotators label thousands of images as high/low quality; a fine-tuned CLIP image encoder serves as a quality classifier, and multi-view voting filters out low-quality objects.

Each 4D object is rendered from 24 viewpoints.

#### 2. QA Task Design (5 Sub-tasks)

| Sub-task | Evaluation Target | Unique Challenge |
|----------|------------------|-----------------|
| **Appearance** | Visual attribute analysis | Synthetic/fictional objects deviate from real-world training distributions |
| **Action** | Fine-grained motion detection | Motion direction requires multi-view observation |
| **Object Counting** | Precise counting in dynamic scenes | Object appearance/disappearance + cross-view occlusion |
| **Spatial Relationship** | Cross-view spatial configuration understanding | Spatial relations differ across viewpoints |
| **Temporal Relationship** | Temporal evolution and order understanding | Joint reasoning across both temporal and viewpoint dimensions |

#### 3. Annotation Pipeline

**QA annotation**: a hybrid approach is employed.
- A professional annotation team performs initial labeling (retention rate drops from 92% to 62.5%, highlighting quality control challenges).
- Subsequently, GPT-4o/Qwen2-VL generate candidate QA pairs → Qwen2-VL 7B performs initial filtering → text-only blind testing (Qwen2.5 + Llama 3.1, discarding items both models answer correctly) → final human review.
- 751 high-quality QA pairs are retained.

**Captioning annotation**: fully manual; five professional annotators independently write a description for each object, and reviewers ensure that descriptions capture important details.

#### 4. Evaluation Setup

- $K=3$ views are uniformly sampled from 24 viewpoints.
- $N=6$ frames are sampled per view → input consists of $3 \times 6 = 18$ frames.
- The captioning task uses GPT-4o as the evaluator, producing separate GPT-Appearance and GPT-Action scores (0–5).

### Loss & Training

4D-Bench is an evaluation benchmark; no model training is involved.

## Key Experimental Results

### Main Results

**4D Object QA Accuracy (%)**:

| Model | Counting | Temporal Rel. | Action | Spatial Rel. | Appearance | Overall |
|-------|----------|--------------|--------|--------------|------------|---------|
| MiniGPT4-Video | 22.05 | 26.43 | 22.90 | 22.39 | 22.06 | 23.17 |
| Qwen2-VL 7B | 38.58 | 56.43 | 57.94 | 58.96 | 71.32 | 56.99 |
| LLaVA-Video 72B | 54.33 | 58.57 | 57.48 | 66.42 | 77.21 | 62.32 |
| GPT-4o | 44.09 | 59.29 | 63.55 | 69.40 | 77.21 | **62.98** |
| All Models Avg. | 37.29 | 49.29 | 49.37 | 53.57 | 63.92 | 50.69 |
| **Human** | **88.98** | **89.29** | **94.39** | **91.04** | **89.71** | **91.08** |

The gap between GPT-4o and the human baseline is nearly **28 percentage points**.

### Ablation Study (Effect of Number of Views and Sampling Rate)

| Configuration Change | Accuracy Change (Gemini 1.5 Flash) |
|---------------------|-----------------------------------|
| 1 view → 3 views (fixed 6 frames) | 41.3% → 53.7% (+12.4%) |
| 1 frame → 6 frames (fixed 3 views) | 46.3% → 53.7% (+7.4%) |
| 3 views → 6 views | 53.7% → decrease (information redundancy) |
| 6 frames → 9 frames | Negligible improvement |

Conclusion: the tasks genuinely require multi-view and temporal information, but exceeding 3 views or 6 frames introduces redundancy that interferes with model performance.

**Captioning Task GPT-Eval Scores**:

| Model | GPT-Appearance | GPT-Action | GPT-Eval |
|-------|---------------|-----------|---------|
| Qwen2-VL 72B | 3.324/5 | 2.791/5 | 3.057/5 |
| Gemini 1.5 Pro | 3.311/5 | 2.983/5 | 3.147/5 |
| GPT-4o | 3.507/5 | 3.258/5 | 3.382/5 |
| **Human** | **3.772/5** | **3.879/5** | **3.826/5** |

### Key Findings

1. **Counting is the most challenging sub-task**: the average across all models is only **37.29%** (near the random-guess baseline of 25%), requiring cross-view information integration to resolve occlusions.
2. **Appearance understanding >> Action understanding**: appearance averages 63.92% vs. action 49.37%, a gap of approximately 15%.
3. **The open-source vs. closed-source gap is larger on action understanding**: open-source models approach closed-source performance on appearance, but the gap on action understanding is substantial.
4. **Counterfactual data exposes "memory dependence"**: when presented with a six-legged spider or physically impossible scenarios, all MLLMs produce incorrect answers, indicating reliance on world-knowledge priors rather than genuine visual understanding.
5. **Good robustness**: changing frame ordering (view-first vs. time-first) or adding timestamps has minimal effect on results.

## Highlights & Insights

- **Filling the gap in 4D–language understanding evaluation**: a novel evaluation dimension is introduced between static 3D and single-view 2D video benchmarks.
- **Elegant counterfactual test design**: synthetic data naturally enables out-of-distribution evaluation beyond the real world, which is infeasible in 2D benchmarks.
- **Rigorous data quality control**: the hybrid annotation pipeline (human + MLLM + blind testing + final review) ensures that questions genuinely require multi-view temporal reasoning.
- **Actionable findings**: poor counting performance → better cross-view correspondence modeling is needed; weak action understanding → stronger temporal encoders are required.

## Limitations & Future Work

- The current approach uses concatenated multi-view videos as a proxy for 4D input rather than native 4D representations (e.g., point cloud sequences, 4D Gaussian Splatting), due to the input modality constraints of current MLLMs.
- The dataset scale is relatively limited (751 QA pairs + 580 captioning instances), which may be insufficient for comprehensive statistical conclusions.
- Objects are sourced from Objaverse-XL and are predominantly synthetic, introducing potential distribution gaps in appearance and motion relative to the real world.
- Only general-purpose 2D MLLMs are evaluated; dedicated 3D/4D understanding models (e.g., 3D-LLM) are not included.

## Related Work & Insights

- **MVBench** [Li et al., 2024]: a multi-task video understanding benchmark, but limited to single viewpoints.
- **ScanQA** [Azuma et al., 2022]: 3D scene question answering, but restricted to static scenes.
- **T3Bench** [He et al., 2023]: evaluates text-to-3D generation, focusing on generation quality rather than understanding.
- **4DGS** [Wu et al., 2024]: 4D Gaussian Splatting, providing a 4D representation but lacking language understanding evaluation.
- Implication: future MLLMs require native 4D input support (rather than multi-view video proxies) and stronger temporal modeling capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — the first benchmark for 4D object understanding, with a pioneering problem formulation.
- Technical Depth: ⭐⭐⭐ — primarily an evaluation work; methodological contributions are relatively limited.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 14 MLLMs, 5 sub-tasks, and multi-dimensional analyses (number of views, frames, ordering, counterfactuals).
- Practical Value: ⭐⭐⭐⭐ — provides clear directions for improving 4D understanding capabilities in MLLMs.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Aligning Effective Tokens with Video Anomaly in Large Language Models](aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)
- [\[ICCV 2025\] DisTime: Distribution-based Time Representation for Video Large Language Models](distime_distribution-based_time_representation_for_video_large_language_models.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[ICCV 2025\] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding](dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)
- [\[ICCV 2025\] Multi-modal Multi-platform Person Re-Identification: Benchmark and Method](multi-modal_multi-platform_person_re-identification_benchmark_and_method.md)

<!-- RELATED:END -->
