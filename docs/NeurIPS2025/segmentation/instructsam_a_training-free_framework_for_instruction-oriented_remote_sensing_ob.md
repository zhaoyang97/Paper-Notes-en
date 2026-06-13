---
title: >-
  [Paper Note] InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition
description: >-
  [NeurIPS 2025][Segmentation][Remote sensing] This paper introduces a new task — Instruction-oriented Counting, Detection, and Segmentation (InstructCDS) — along with the EarthInstruct remote sensing benchmark covering th…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Remote sensing"
  - "instruction-oriented"
  - "training-free"
  - "SAM2"
  - "binary integer programming"
  - "object counting"
  - "open-vocabulary detection/segmentation"
date: 2026-05-08
content_hash: 49f2ae6e8c20e7df
---

# InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition

**Conference**: NeurIPS 2025
**arXiv**: [2505.15818](https://arxiv.org/abs/2505.15818)  
**Code**: [https://VoyagerXvoyagerx.github.io/InstructSAM/](https://VoyagerXvoyagerx.github.io/InstructSAM/)  
**Area**: Remote Sensing Object Recognition / Open-Vocabulary Segmentation
**Keywords**: Remote sensing, instruction-oriented, training-free, SAM2, binary integer programming, object counting, open-vocabulary detection/segmentation

## TL;DR
This paper introduces a new task — Instruction-oriented Counting, Detection, and Segmentation (InstructCDS) — along with the EarthInstruct remote sensing benchmark covering three settings (open-vocabulary, open-ended, and open-subcategory). It proposes InstructSAM, a training-free framework that uses an LVLM to parse instructions and predict counts, SAM2 to generate mask proposals, and CLIP to compute similarities. A Binary Integer Programming (BIP) formulation then performs optimal mask-label assignment under counting constraints, achieving near-constant inference time while outperforming task-specific baselines.

## Background & Motivation

**Background**: Remote sensing object recognition plays an important role in sustainable development goals, including wildlife monitoring, poverty estimation, and disaster relief. CLIP-driven open-vocabulary detection and segmentation have seen growing adoption in the remote sensing domain.

**Limitations of Prior Work**:
- Existing open-vocabulary methods **rely on explicit category instructions** and cannot handle implicit reasoning (e.g., inferring which subcategories fall under "vehicles").
- Fixed category lists are inherently incomplete for remote sensing — aerial views expose a wide diversity of object types.
- Traditional detectors depend on **confidence thresholds** for filtering, which are unavailable in zero-shot scenarios.
- Directly prompting LVLMs to generate bounding boxes one by one leads to inference time that scales linearly with the number of objects.

**Core Idea**: Decompose the problem into three steps — LVLM counting ($O(1)$ inference time) + SAM2 mask proposals + BIP optimal matching — requiring no training, no thresholds, and near-constant inference time.

## InstructCDS Task & EarthInstruct Benchmark

### Three Settings

| Setting | Description | Example Instruction |
|---------|-------------|---------------------|
| **Open-Vocabulary** | User specifies target categories | "Detect football fields and parking lots" |
| **Open-Ended** | Detect all visible objects | "Detect all objects in the image" |
| **Open-Subcategory** | Detect all subcategories under a parent class | "Detect all sports venues" |

### EarthInstruct Benchmark
- Built upon two remote sensing datasets — NWPU-VHR-10 and DIOR — covering 20 categories.
- The two datasets differ in annotation conventions and spatial resolution, requiring models to interpret dataset-specific instructions.
- For example, vehicles are not annotated in low-resolution NWPU-VHR-10 images; in DIOR, airports are only annotated when fully visible.

### Evaluation Metric Innovations
- **Counting metrics**: Precision/Recall/F1 based on TP/FP/FN (replacing MAE/RMSE; normalized and capable of distinguishing over- vs. under-counting).
- **Confidence-free detection metrics**: mF1 + mAP$_{nc}$ (independent of confidence score ranking), suitable for generative detectors.
- **Semantic matching**: Cosine similarity > 0.95 via GeoRSCLIP text encoder is treated as category equivalence.

## Method

### InstructSAM Framework (Three-Step Pipeline)

#### Step 1: LVLM Instruction Parsing and Counting
GPT-4o or Qwen2.5-VL-7B serves as the counter. Given the image and a structured JSON prompt (including dataset-specific instructions), it outputs target categories $\{cat_j\}$ and counts $\{num_j\}$:

$$\{cat_j, num_j\}_{j=1}^M = \text{LVLM-Counter}(I, P)$$

#### Step 2: Class-Agnostic Mask Proposals via SAM2
SAM2-hiera-large automatically generates mask proposals $\{mask_i\}_{i=1}^N$ over a regular point grid. An additional mask generation pass is applied to cropped image regions to improve small-object recall.

#### Step 3: Counting-Constrained Mask-Label Matching (Core Contribution)
A semantic similarity matrix $S \in \mathbb{R}^{N \times M}$ is computed (image–text cosine similarity via GeoRSCLIP), and the following Binary Integer Programming problem is formulated:

$$\min_{\mathbf{X}} \sum_{i=1}^N \sum_{j=1}^M (1 - s_{ij}) \cdot x_{ij}$$

Subject to:
- Each mask is assigned to at most one category: $\sum_{j=1}^M x_{ij} \leq 1, \; \forall i$
- The number of masks assigned to each category equals its count: $\sum_{i=1}^N x_{ij} = num_j, \; \forall j$ (when proposals are sufficient)
- When proposals are insufficient, all are assigned: $\sum_{i=1}^N \sum_{j=1}^M x_{ij} = N$

The problem is solved efficiently via the PuLP solver. The BIP formulation elegantly integrates three sources of information: visual (CLIP mask embeddings), semantic (category text embeddings), and quantitative (LVLM counting constraints).

### Key Advantages
- **Training-free**: No task-specific training data for remote sensing is required.
- **Threshold-free**: Counting constraints replace confidence thresholds, avoiding the threshold selection problem in zero-shot scenarios.
- **Near-constant inference time**: The LVLM outputs only counts (few tokens) rather than generating bounding boxes one by one; SAM2 mask proposals are independent of the number of objects.

## Key Experimental Results

### Open-Vocabulary Setting (Zero-Shot)

| Method | NWPU Cnt-F1↑ | Box-F1↑ | Mask-F1↑ | DIOR Cnt-F1↑ | Box-F1↑ | Mask-F1↑ |
|--------|-------------|---------|----------|-------------|---------|----------|
| Grounding DINO | 14.9 | 14.0 | - | 10.7 | 6.0 | - |
| OWLv2 | 39.4 | 27.2 | - | 23.4 | 14.3 | - |
| Qwen2.5-VL | 68.0 | 36.4 | - | 52.0 | 27.8 | - |
| InstructSAM-Qwen | 73.2 | 38.9 | 23.7 | 59.3 | 24.7 | 24.0 |
| **InstructSAM-GPT4o** | **83.0** | **41.8** | **26.1** | **79.9** | **29.1** | **28.1** |

InstructSAM-GPT4o achieves a substantial lead in counting F1 (83.0 vs. 68.0) while simultaneously providing segmentation outputs.

### Open-Ended Setting

| Method | NWPU Cnt-F1↑ | Box-F1↑ | DIOR Cnt-F1↑ | Box-F1↑ |
|--------|-------------|---------|-------------|---------|
| Qwen2.5-VL | 48.6 | 32.0 | 36.6 | 21.7 |
| GeoPixel | 40.8 | 29.9 | 21.4 | 13.8 |
| LAE-Label | 46.2 | 27.3 | 23.3 | 11.5 |
| **InstructSAM-GPT4o** | **57.4** | **31.3** | **47.9** | **22.1** |

InstructSAM consistently outperforms remote sensing-specific models (e.g., SkySenseGPT, EarthDial, GeoPixel) in the open-ended setting as well.

### Open-Subcategory Setting

| Method | NWPU-S F1↑ | NWPU-T F1↑ | DIOR-S F1↑ | DIOR-T F1↑ |
|--------|-----------|-----------|-----------|-----------|
| Qwen2.5-VL | 32.4 | 42.2 | 34.0 | 39.2 |
| GPT4o+OWL | 19.8 | **65.9** | 27.6 | **70.9** |
| **InstructSAM-GPT4o** | **46.9** | 44.2 | **40.9** | 38.3 |

InstructSAM achieves a large margin on the "sports venues" (S) subcategory, while general-purpose detectors perform better on "vehicles" (T) — attributed to the richness of vehicle categories in OWLv2's training data.

### Efficiency Comparison
- InstructSAM reduces output token count by **89%** compared to Qwen2.5-VL.
- Total runtime is reduced by **>32%**.
- Inference time remains near-constant and does not scale with the number of objects.

## Highlights & Insights
1. **BIP matching replaces thresholds** — object recognition is formulated as a constrained optimization problem, eliminating the need for threshold tuning in zero-shot scenarios.
2. **Counting as a global constraint** — the LVLM's global perspective yields accurate counts that constrain mask assignment.
3. **Training-free generalization** — three foundation models (LVLM + SAM2 + CLIP) are composed without any remote sensing-specific fine-tuning.
4. **New task + new benchmark** — the InstructCDS three-setting formulation, EarthInstruct benchmark, and confidence-free evaluation metrics constitute substantial contributions.

## Limitations & Future Work
1. LVLM counting accuracy is the performance ceiling — GPT-4o achieves only 83% counting F1, and errors propagate to final detection results.
2. The framework relies on GeoRSCLIP's remote sensing-specific pretraining — transferring to natural image domains requires replacing the CLIP backbone.
3. The BIP formulation assumes each mask belongs to at most one category — heavily overlapping instances cannot be handled.
4. SAM2 may miss very small objects (<10 px); the cropping strategy partially alleviates but does not fully resolve this issue.
5. Inference depends on the GPT-4o API, introducing cost and network dependency.

## Related Work & Insights
- The BIP matching approach generalizes to any scenario where category counts are known and detection/segmentation proposals are available.
- The paradigm of replacing confidence thresholds with counting constraints is broadly applicable to all generative detection models.
- The dataset-specific instruction design in EarthInstruct is worth adopting in other domains, where annotation conventions vary across datasets.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ BIP matching with counting constraints as a threshold replacement represents a genuine paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across three settings with comparisons against diverse baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is precise, the framework is elegant, and the metric design is carefully motivated.
- Value: ⭐⭐⭐⭐⭐ The training-free, threshold-free paradigm has broad impact potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images](../../AAAI2026/segmentation/rsvg-zeroov_exploring_a_training-free_framework_for_zero-shot_open-vocabulary_vi.md)
- [\[NeurIPS 2025\] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing](roma_scaling_up_mamba-based_foundation_models_for_remote_sensing.md)
- [\[CVPR 2026\] Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation](../../CVPR2026/segmentation/task-oriented_data_synthesis_and_control-rectify_sampling_for_remote_sensing_sem.md)
- [\[ICCV 2025\] E-SAM: Training-Free Segment Every Entity Model](../../ICCV2025/segmentation/e-sam_training-free_segment_every_entity_model.md)
- [\[ICCV 2025\] SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation](../../ICCV2025/segmentation/score_scene_context_matters_in_openvocabulary_remote_sensing.md)

</div>

<!-- RELATED:END -->
