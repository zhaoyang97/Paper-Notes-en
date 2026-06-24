---
title: >-
  [Paper Note] RoMo: A Large-Scale, Richly Organized Dataset and Semantic Taxonomy for Human Motion Generation
description: >-
  [CVPR 2026][Human Understanding][Human Motion Dataset] Addressing the long-standing dilemma in 3D human motion generation between "small but clean mocap sets" and "large but noisy in-the-wild sets," RoMo employs a **taxonomy-aware adaptive filtering pipeline** to distill approximately 1% of high-quality motion from 125,000 hours of web videos. It constructs a large-scale dataset featuring 820,000 segments (approx. 1,238 hours), each with 5 rich text descriptions organized by…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "Human Motion Dataset"
  - "Semantic Taxonomy"
  - "Dynamic Score"
  - "Adaptive Filtering"
  - "Text-to-Motion Generation"
date: 2026-05-08
content_hash: 9fe4045e9f421990
---

# RoMo: A Large-Scale, Richly Organized Dataset and Semantic Taxonomy for Human Motion Generation

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_RoMo_A_Large-Scale_Richly_Organized_Dataset_and_Semantic_Taxonomy_for_CVPR_2026_paper.html)  
**Code**: https://davidzhang73.github.io/romo-website (Project Page)  
**Area**: Human Understanding / Human Motion Generation  
**Keywords**: Human Motion Dataset, Semantic Taxonomy, Dynamic Score, Adaptive Filtering, Text-to-Motion Generation

## TL;DR
Addressing the long-standing dilemma in 3D human motion generation between "small but clean mocap sets" and "large but noisy in-the-wild sets," RoMo employs a **taxonomy-aware adaptive filtering pipeline** to distill approximately 1% of high-quality motion from 125,000 hours of web videos. It constructs a large-scale dataset featuring 820,000 segments (approx. 1,238 hours), each with 5 rich text descriptions organized by a three-level taxonomy (Category → Subcategory → Atomic action). Accompanied by the Motion Toolbox for unified evaluation, models trained on RoMo achieve SOTA performance in fidelity, diversity, and fine-grained textual understanding.

## Background & Motivation
**Background**: The success of language, image, and video generation confirms that large-scale, high-quality data is the primary driver of capability. 3D human motion generation has progressed rapidly through diffusion models (e.g., MDM) and GPT-style tokenization (e.g., MoMask), but the underlying datasets have become a bottleneck.

**Limitations of Prior Work**: Motion data is trapped in a **dilemma**. On one hand, motion capture (mocap) data (e.g., AMASS/HumanML3D) offers high fidelity but is extremely small in scale; decades of academic effort have yielded fewer than 20,000 segments, failing to cover the "long tail" of human activities. On the other hand, large-scale sets extracted from in-the-wild videos (e.g., Motion-X, MotionMillion) offer scale but **lack rigorous cleaning**, being riddled with artifacts such as stillness, jitter, interpenetration, and floating, with many segments containing no meaningful motion. Consequently, researchers must choose between "small and clean data that no longer challenges modern models" and "large but unreliable data that biases models toward static, low-quality motions."

**Key Challenge**: There is a conflict between "scale" and "quality" in in-the-wild extraction, and both types of data **lack structured organization**. All motions are flattened into a single set, making it impossible to perform fine-grained evaluation or ensure balanced semantic coverage, which hides model blind spots in specific categories behind global metrics.

**Goal**: To simultaneously address three issues: scaling up, filtering for quality, and providing an analytical semantic structure for motion.

**Key Insight**: Quality and scale are not necessarily antithetical. The key is to **first establish a taxonomy covering the human motion manifold**, and then use it to **guide queries, segmentation, and filtering**. Because the inherent "dynamic intensity" varies significantly across categories (e.g., fishing is naturally quieter than acrobatics), filtering thresholds must be category-adaptive rather than a one-size-fits-all approach.

**Core Idea**: Use a three-level semantic taxonomy as the backbone to build an automated, taxonomy-aware data pipeline. This pipeline performs uncompromising large-scale filtering on massive web videos to distill high-quality, semantically aligned motion data.

## Method

### Overall Architecture
RoMo is built on two pillars: a **three-level taxonomy** covering the human motion space and a **multi-source video corpus** aggregated from large-scale web queries and public datasets. The automated pipeline consists of: taxonomy-driven **video querying** for candidates → **video processing** (metadata filtering, scene splitting, person detection, semantic temporal segmentation into atomic actions) → parallel **3D human and camera motion estimation** and **VLM motion description/classification** → **motion evaluation and filtering** (centered on the Dynamic Score) to remove static or corrupted reconstructions → output of high-quality motion sequences labeled by taxonomy. This pipeline processed approximately 125,000 hours of raw material, distilling only ~1% into the final dataset. Ideally, the "method" of this dataset/benchmark paper is the data pipeline itself.

### Key Designs

**1. Three-level Semantic Taxonomy: Providing a Drillable Backbone for Flattened Motion Sets**

The structural flaw of in-the-wild sets is their "one-pot" nature, which prevents fine-grained analysis. RoMo proposes a three-tier taxonomy: **Category** captures broad themes (Sports, Daily activities, Occupations, etc.); **Subcategory** consists of compact noun-phrase semantic groups (e.g., Table tennis, Cleaning activities); and **Atomic-action** comprises present-tense verb phrases lasting a few seconds (e.g., Swing racket, Climb stairs) serving as the basic unit for segmentation, retrieval, and description. This was built using a hybrid strategy: high-level categories were derived top-down from mainstream datasets, while subcategories and atomic actions were expanded via LLM and manually refined. The final structure includes **54 categories, 2,065 subcategories, and 28,874 atomic actions**. This allows evaluation to drill down from "a model is weak in gestures" to specific gesture types to identify failure modes, ensuring balanced coverage.

**2. Taxonomy-aware Adaptive Filtering Pipeline: Using Category Priors to Define "How Clean is Clean"**

Standard approaches use a uniform quality threshold, which may delete naturally quiet categories while retaining low-quality static segments. RoMo's filtering is **multi-stage and category-adaptive**. Initial stages include: metadata filtering (removing short segments or those <24 FPS); LLM-based scoring of labels/descriptions on four criteria (human motion, single person, full-body visibility, non-AI generated) with a [0,1] confidence score; scene detection (PySceneDetect for transitions and frame-difference based static removal); and single-person detection (YOLOv8 and ViTPose to ensure reasonable occupancy and 2D pose quality). **Semantic temporal segmentation** uses multi-modal VLMs (e.g., Qwen3-VL) fed with taxonomy nodes to return segments with precise timestamps, which outperforms fixed-length slicing. Finally, the **Dynamic Score** performs final filtering by taking the top-P percentile within each category, preserving subtle movements while removing excessive stillness. The pipeline distilled 125.3k hours down to ~1.3k hours (approx. 99% reduction).

**3. Dynamic Score: Quantifying "Action Intensity" as a Filterable Scalar**

The authors note that in-the-wild sets often ignore the distribution of "dynamic intensity," leading to datasets inflated by low-activity segments. The Dynamic Score quantifies activity using temporal and spatial components. Given joint positions $P\in\mathbb{R}^{F\times J\times 3}$ and velocities $V\in\mathbb{R}^{F\times J\times 3}$ ($F$ frames, $J$ joints), the temporal component measures instantaneous activity $S_{\text{temporal}}=\frac{1}{F\cdot J}\sum_{t=1}^{F}\sum_{j=1}^{J}\|v_{t,j}\|_2$, and the spatial component measures overall range $S_{\text{spatial}}=\frac{1}{J}\sum_{j=1}^{J}\big\|\max_t p_{t,j}-\min_t p_{t,j}\big\|_2$. The final weighted score is $S_{\text{Dynamic}}=w_v\cdot S_{\text{temporal}}+w_r\cdot S_{\text{spatial}}$, with $(w_v,w_r)=(0.7,0.3)$. This ensures both high-frequency motions (dancing) and large-amplitude motions (walking) receive appropriate scores. As a diagnostic tool, RoMo's mean score of 0.336 is 41.4% higher than MotionMillion's 0.222.

**4. Motion Toolbox: Standardizing Scattered Physical Metrics**

The field of motion evaluation has long suffered from "disparate, idiosyncratic metrics" (foot sliding, floating, jitter/jerk, penetration). The open-source Motion Toolbox provides unified support for MotionMillion/HumanML3D/SMPL representations with built-in converters, integrates validated physical plausibility metrics (foot sliding, penetration, jerk, floating), and offers a browser-based visualizer. Its significance lies in providing a single, open-source, verified implementation for critical metrics.

### Mechanism Example
Take a web video categorized as "Sports → Basketball → Dribbling": Diverse queries generate candidates → Metadata filtering confirms it is a full-body, single-person human motion → Scene detection removes transitions → YOLOv8/ViTPose verify pose estimability → Qwen3-VL segments atomic actions like "Executing a crossover dribble" → GVHMR estimates 3D motion in SMPL format (global translation, root orientation, 24 joints) resampled to 30FPS → Qwen3-VL generates 5 rich descriptions with taxonomy labels → If the Dynamic Score falls within the top-P for "Basketball," it is retained.

## Key Experimental Results

### Main Results
RoMo contains 813,938 segments, (~1,238 hours at 30FPS), with an average of 165 frames and 5 descriptions per segment. Comparison with existing 3D motion datasets (Core Set / Total Set):

| Dataset | 3-Level Taxonomy | Cat/Subcat | Core (Total) Segments | Duration (h) | Text Diversity | Source |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| HumanML3D | ✗ | — | 0 (15K) | 0 (28.6) | 1-3 | MoCap |
| Motion-X | ✗ | — | 48.6K (81.1K) | 86 (144.2) | 1 | Video+MoCap |
| MotionMillion | ✗ | — | 560K (2M) | 726.5 (2000) | >20 | Video |
| **RoMo (Ours)** | **✓** | **54 / 2065** | **820K (2.58M)** | **1237.8 (3023)** | **5** | Video |

RoMo is the first dataset to feature **both** large-scale taxonomy and massive scale. In terms of coverage, mapping MotionMillion to the same taxonomy covers only 1,277 subcategories; RoMo increases coverage by **61.7%**. In terms of dynamics, RoMo's mean Dynamic Score is 41.4% higher than MotionMillion. Evidence shows that filtering MotionMillion with thresholds of $\geq 0.05/0.10/0.15/0.50$ retains only 88.41%/78.46%/69.07%/31.55% of segments, indicating a heavy bias toward low-dynamic content.

### Ablation Study
Training the diffusion model MDM and autoregressive model MMGPT on RoMo:

| Model | Diversity ↑ | FID ↓ | Matching ↑ | Dynamic ↑ | Foot Slide ↓ ($ \times 10^{-3} $) | Floating ↓ ($ \times 10^{-2} $) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| MDM (Diffusion) | **27.67** | 20.63 | 12.06 | 0.2138 | **1.70** | 1.67 |
| MMGPT (GPT) | 16.68 | **12.80** | **22.08** | **0.3268** | 92.0 | **0.0311** |

> Note: Matching Score measures text-motion semantic alignment; physical metrics (Foot slide/Floating/Jerk) are from Motion Toolbox.

### Key Findings
- **Architecture Trade-offs**: Autoregressive MMGPT is stronger in FID and Matching Score (tokenization excels at semantic mapping and distribution fitting); diffusion-based MDM is superior in Diversity and physical plausibility (non-autoregressive refinement avoids error accumulation).
- **Taxonomy Reveals Blind Spots**: Per-category evaluation shows that SOTA models perform well on common actions but fail on fine-grained categories involving subtle interactions—a gap hidden by global metrics.
- **Filtering is Critical**: Reducing raw material by 99% is the root cause of RoMo's high dynamic scores and balanced coverage; skipping this step leads back to the "large but static" trap.

## Highlights & Insights
- **"Taxonomy-first" Design**: Starting with a taxonomy to drive query/segmentation/filtering solves the scale-quality-structure triad simultaneously.
- **Quantifying Dynamics**: The Dynamic Score uses temporal velocity and spatial range to expose "inflated counts" in web datasets, providing a reusable diagnostic metric.
- **Transferability**: This paradigm—semantic taxonomy + adaptive filtering + standardized toolbox—can be applied to other generation tasks such as audio, hand gestures, or human-object interaction.

## Limitations & Future Work
- **Dependence on Upstream Models**: Quality is capped by GVHMR (pose), Qwen3-VL (text/classification), and YOLOv8 (detection). Reconstructions may still exhibit floating or jitter.
- **Filtering Bias**: The top-P percentile approach protects quiet categories, but the weighting of $(w_v,w_r)=(0.7,0.3)$ is subjective and may not be optimal for all cases.
- **Metric Limitations**: FID/Matching depend on TMR-based neural evaluators, which carry their own biases.
- ⚠️ Discrepancies exist in the count of atomic actions (28,874 vs 2,897 subcategories) in different parts of the text; the final structure is cited as 54/2,065/28,874.

## Related Work & Insights
- **vs HumanML3D / AMASS (Mocap)**: RoMo scales motion to 820k core segments and covers in-the-wild/RGB scenes, addressing the long-tail diversity missing in indoor mocap.
- **vs Motion-X / MotionMillion (In-the-wild)**: Unlike previous sets that lack cleaning and structure, RoMo uses taxonomy-aware filtering and organization, solving both quality and utility issues.
- **vs BABEL**: While BABEL added labels to AMASS (8 categories / 260 subcategories), its scale is far smaller than RoMo’s 54 / 2,065.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First dataset with both massive taxonomy and scale; Dynamic Score and taxonomy-aware filtering are solid innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive dataset comparisons and dual-architecture training; however, generation experiments are limited to two models.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and pipeline; slight inconsistencies in statistical figures.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the data bottleneck in motion generation; the Dataset + Taxonomy + Toolbox provides a reproducible foundation for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Next-Scale Autoregressive Models for Text-to-Motion Generation](next-scale_autoregressive_models_for_text-to-motion_generation.md)
- [\[CVPR 2026\] LCA: Large-scale Codec Avatars - The Unreasonable Effectiveness of Large-scale Avatar Pretraining](lca_large-scale_codec_avatars_the_unreasonable_effectiveness_of_large-scale_avata.md)
- [\[CVPR 2026\] HUMAPS-4D: A Multimodal Dataset for HUman Motion Analysis with Physiological and Semantic informations](humaps-4d_a_multimodal_dataset_for_human_motion_analysis_with_physiological_and_.md)
- [\[CVPR 2026\] OpenT2M: No-frill Motion Generation with Open-source, Large-scale, High-quality Data](opent2m_no-frill_motion_generation_with_open-source_large-scale_high-quality_dat.md)
- [\[CVPR 2026\] OpenDance: Multimodal Controllable 3D Dance Generation with Large-scale Internet Data](opendance_multimodal_controllable_3d_dance_generation_with_large-scale_internet_.md)

</div>

<!-- RELATED:END -->
