---
title: >-
  [Paper Note] SITE: towards Spatial Intelligence Thorough Evaluation
description: >-
  [ICCV2025][Robotics][Spatial Intelligence] This paper presents SITE, a comprehensive spatial intelligence benchmark grounded in a tripartite cognitive-science taxonomy. It comprises 8…
tags:
  - "ICCV2025"
  - "Robotics"
  - "Spatial Intelligence"
  - "VLM Benchmark"
  - "Multi-View Reasoning"
  - "Cognitive Science"
  - "Embodied AI"
date: 2026-05-08
content_hash: 42407cbf273c147d
---

# SITE: towards Spatial Intelligence Thorough Evaluation

**Conference**: ICCV2025
**arXiv**: [2505.05456](https://arxiv.org/abs/2505.05456)  
**Code**: [https://github.com/SITE-project-page](https://github.com/SITE-project-page) (to be confirmed)  
**Area**: Robotics / Spatial Intelligence / VLM Evaluation
**Keywords**: Spatial Intelligence, VLM Benchmark, Multi-View Reasoning, Cognitive Science, Embodied AI

## TL;DR
This paper presents SITE, a comprehensive spatial intelligence benchmark grounded in a tripartite cognitive-science taxonomy. It comprises 8,068 multiple-choice VQA tasks spanning 31 datasets (images and videos). Evaluation results show that the strongest VLM (GPT-4o) still lags human experts by approximately 32% on overall spatial reasoning, and VLM spatial intelligence scores are highly correlated with robotic manipulation success rates (Pearson $r=0.902$).

## Background & Motivation

**Background**: Spatial Intelligence (SI) is a core cognitive ability encompassing the visualization, manipulation, and reasoning of spatial relationships. While large vision-language models (VLMs) have achieved notable progress on general VQA, their spatial reasoning capabilities are only sparsely incorporated into comprehensive benchmarks such as MME and MMBench, leaving systematic evaluation largely absent.

**Limitations of Prior Work**: Existing spatial reasoning benchmarks exhibit clear gaps — CVBench lacks perspective-taking tasks, 3DSRBench is limited to single images, and VSI-Bench covers only indoor video scenes. Each benchmark addresses only a subset of spatial intelligence dimensions, preventing a holistic characterization.

**Key Challenge**: Cognitive science identifies at least three major classification frameworks for SI (scale-based, visualization/orientation-based, and the 2×2 intrinsic/extrinsic × static/dynamic taxonomy), yet no existing benchmark simultaneously covers all three, with spatial orientation and dynamic-scene reasoning being particularly underrepresented.

**Goal**: To construct a comprehensive spatial intelligence benchmark that satisfies all three cognitive-science taxonomies, filling the gaps in perspective-taking and dynamic reasoning tasks, and systematically identifying the spatial reasoning blind spots of current VLMs.

**Key Insight**: A dual-path strategy is adopted — a bottom-up path selects spatially relevant tasks from 30 existing datasets, while a top-down path identifies uncovered dimensions from the cognitive-science taxonomies and designs new tasks accordingly.

**Core Idea**: Benchmark design is driven by three cognitive-science SI taxonomies, combining task selection from existing datasets with the creation of novel tasks (Ego-Exo view association and frame reordering), thereby enabling comprehensive and systematic evaluation of VLM spatial intelligence.

## Method

### Overall Architecture
SITE is constructed via two complementary paths. The **bottom-up** path selects spatially relevant tasks from 30 existing visual datasets, yielding 6,943 QA pairs. The **top-down** path identifies dimensions not covered by existing tasks based on cognitive-science taxonomies and introduces two new task types (1,125 QA pairs). The two sets are merged into 8,068 standardized multiple-choice VQA tasks.

### Key Designs

1. **Tripartite Cognitive-Science Taxonomy**

    - **Function**: Three complementary SI classification systems guide the coverage scope of benchmark design.
    - **Mechanism**: (a) **Scale-based**: figural (smaller than the body, perceivable from a single viewpoint), vista (room-scale), and environmental (requires navigation to perceive), each engaging distinct neural substrates; (b) **VZ/SO**: spatial visualization (mental rotation independent of one's own viewpoint) vs. spatial orientation (imagining observation from a different viewpoint); (c) **2×2**: intrinsic/extrinsic × static/dynamic.
    - **Design Motivation**: No single taxonomy can fully cover all facets of spatial intelligence; the three complementary frameworks together ensure no dimension is overlooked.

2. **Bottom-Up Data Collection and Filtering**

    - **Function**: Spatially relevant tasks are selected from 22 image datasets and 8 video datasets.
    - **Mechanism**: Dataset-provided labels are used for initial filtering, followed by a two-stage GPT-4o screening — a fast text-only pass and a subsequent multimodal pass. Tasks are organized into six coarse-grained spatial categories: Counting, Relationship Reasoning, Localization, 3D Information, Multi-View Reasoning, and Movement Prediction.
    - **Design Motivation**: Heterogeneous labeling systems across datasets necessitate a unified taxonomy; stratified sampling is applied after filtering the initial 223K samples to ensure category balance.

3. **Top-Down Novel Task Design (Ego-Exo4D)**

    - **Function**: Two new task types are designed based on the Ego-Exo4D dataset to fill identified gaps.
    - **Ego-Exo View Association**: Given an egocentric-view image, the model selects the matching exocentric-view image from candidates (and vice versa), probing extrinsic-static perspective-taking ability.
    - **Frame Reordering**: Start, end, and intermediate keyframes are extracted from a video, shuffled, and the model is required to infer the correct temporal order, probing extrinsic-dynamic spatiotemporal reasoning.
    - **Design Motivation**: Analysis reveals that existing tasks severely lack spatial orientation and dynamic-scene reasoning coverage, which are precisely the core components of spatial intelligence.

4. **Chance-Adjusted Accuracy (CAA)**

    - **Function**: An evaluation metric that eliminates bias introduced by random guessing.
    - **Mechanism**: $\mathcal{CAA} = (\sum X_i - \sum \frac{1}{n_i}) / (N - \sum \frac{1}{n_i})$, where $n_i$ is the number of options for question $i$. CAA = 1 indicates perfect performance, CAA = 0 is equivalent to random chance, and CAA < 0 indicates performance worse than random.
    - **Design Motivation**: The number of answer choices varies across questions (2–6 options), and raw accuracy introduces bias without accounting for this variation.

### Loss & Training
SITE is an evaluation benchmark rather than a training method; no loss function is involved. GPT-4o is used as an automated judge to parse VLM outputs.

## Key Experimental Results

### Main Results

| Model | Overall CAA | Counting | Localization | 3D Info | Multi-View | Relation | Movement |
|-------|-------------|----------|--------------|---------|------------|----------|----------|
| Human | 67.5 | 66.0 | 83.3 | 54.7 | 87.5 | 73.0 | 52.5 |
| GPT-4o | 37.8 | 44.6 | 56.0 | 26.9 | 22.0 | 54.6 | 18.4 |
| InternVL-2.5-8B | 32.8 | 47.1 | 37.0 | 23.2 | 9.05 | 47.6 | 28.7 |
| Qwen2.5-VL-7B | 31.4 | 52.6 | 44.1 | 9.42 | 1.08 | 51.5 | 18.9 |
| Gemini-1.5-Pro | 32.5 | 48.0 | 45.8 | 25.3 | 5.33 | 48.8 | 18.4 |
| LLaVA-OV-7B | 30.2 | 51.8 | 38.5 | 22.4 | 9.40 | 55.3 | 9.18 |

### Novel Tasks (View Association & Frames Reordering)

| Model | View Assoc. ego2exo | View Assoc. exo2ego | Reorder ego2exo | Reorder exo2ego |
|-------|---------------------|---------------------|-----------------|-----------------|
| Human | 100 | 100 | 98 | 96 |
| GPT-4o | 35.70 | 20.70 | -2.01 | -5.16 |
| Qwen2.5-VL-7B | 5.09 | -3.80 | 7.63 | 4.23 |
| InternVL-2.5-8B | -5.56 | 5.91 | 5.22 | -0.66 |

### Correlation Between Spatial Intelligence and Robotic Manipulation

| Model | SITE CAA | L2 Dist ↓ | Success Rate ↑ |
|-------|----------|-----------|----------------|
| LLaVA-OV-0.5B | 18.4 | 0.268 | 0.0% |
| LLaVA-OV-7B | 30.2 | 0.142 | 0.0% |
| Qwen2.5-VL-3B | 29.5 | 0.139 | 0.0% |
| Qwen2.5-VL-7B | 31.4 | 0.030 | 38.0% |

The Pearson correlation coefficient $r = 0.902$ indicates a strong positive correlation between spatial intelligence scores and robotic manipulation capability.

### Key Findings
- **Multi-View Reasoning is the largest weakness of VLMs**: All models achieve CAA below 10% in this category (GPT-4o reaches 22%), compared to 87.5% for humans, a gap exceeding 65 percentage points.
- **3D understanding remains a persistent challenge**: Most VLMs score below 15% on 3D Information Understanding.
- **Perspective-taking tasks are near-complete failures**: On frame reordering tasks, most VLMs yield negative CAA scores (worse than random guessing), indicating that current VLMs entirely lack cross-view temporal reasoning capability.
- **Model scale helps but is insufficient**: The 7B variants consistently outperform smaller counterparts, yet the strongest open-source models still lag far behind humans.

## Highlights & Insights
- **Cognitive-science-driven benchmark design paradigm**: Rather than simply aggregating datasets, the work systematically audits coverage using three cognitive-science taxonomies and supplements missing dimensions accordingly. This "taxonomy-first, gap-filling" methodology is transferable to the design of other capability evaluation benchmarks.
- **The strong correlation between SI and embodied AI** ($r=0.902$) is a significant finding: it suggests that spatial reasoning scores on VQA benchmarks can serve as proxy indicators for predicting robotic manipulation performance, providing a low-cost evaluation tool for practical VLM deployment.
- **The Ego-Exo view association task design is particularly elegant**: By exploiting the natural multi-view synchronized captures in the Ego-Exo4D dataset, it constructs tasks that humans solve effortlessly (100%) but that VLMs nearly entirely fail, precisely pinpointing a fundamental deficit in VLM spatial understanding.

## Limitations & Future Work
- **Absence of 3D input modalities**: All tasks use 2D image/video inputs; spatial reasoning over 3D raw inputs such as point clouds and depth maps is not evaluated.
- **Limited human annotation scale**: The human upper-bound evaluation involved only 7 participants on a small subset, which limits representativeness.
- **Dependence on GPT-4o as judge**: Using an LLM to parse VLM outputs may introduce additional noise.
- **Small scale of the SI–embodied AI correlation experiment**: Correlation is validated on only a single LIBERO-Spatial task with four models; larger-scale verification is needed.
- **Directions for improvement**: Interactive spatial reasoning tasks (e.g., questions requiring multi-step navigation) could be incorporated; 3D point clouds or depth maps could be introduced as input modalities; and reasoning-augmented VLMs (e.g., o1-style models) could be evaluated.

## Related Work & Insights
- **vs. CVBench**: CVBench focuses solely on vista-scale spatial relationships and lacks perspective-taking tasks. SITE covers all three scales as well as spatial orientation, yielding broader coverage.
- **vs. VSI-Bench**: VSI-Bench evaluates spatial reasoning via video but is limited to indoor scenes. SITE spans diverse indoor and outdoor scenarios and also includes the figural scale.
- **vs. 3DSRBench**: Limited to single images and unable to evaluate spatial reasoning in dynamic scenes. SITE encompasses both image and video inputs.
- The CAA metric is concise and effective, and is suitable for generalization to other multiple-choice benchmarks with varying numbers of answer options.

## Rating
- Novelty: ⭐⭐⭐⭐ The cognitive-science tripartite taxonomy-driven benchmark design is novel, though the innovation space for benchmark-type work is inherently limited.
- Experimental Thoroughness: ⭐⭐⭐⭐ Nine VLMs are evaluated, with human upper-bound assessment and embodied AI correlation analysis included.
- Writing Quality: ⭐⭐⭐⭐⭐ Structure is clear, narrative logic is coherent, and the cognitive science background is introduced in thorough detail.
- Value: ⭐⭐⭐⭐ The benchmark reveals clear blind spots in VLM spatial intelligence, though sustained value as a benchmark requires ongoing iteration and maintenance.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](../../NeurIPS2025/robotics/robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)
- [\[NeurIPS 2025\] Learning Spatial-Aware Manipulation Ordering](../../NeurIPS2025/robotics/learning_spatial-aware_manipulation_ordering.md)
- [\[NeurIPS 2025\] MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](../../NeurIPS2025/robotics/mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)
- [\[ICLR 2026\] From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors](../../ICLR2026/robotics/from_spatial_to_actions_grounding_vision-language-action_model_in_spatial_founda.md)
- [\[CVPR 2026\] ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation](../../CVPR2026/robotics/maniparena_comprehensive_real-world_evaluation_of_reasoning-oriented_generalist_.md)

</div>

<!-- RELATED:END -->
