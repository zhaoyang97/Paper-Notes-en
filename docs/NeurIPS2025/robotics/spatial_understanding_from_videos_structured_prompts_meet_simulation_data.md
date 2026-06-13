---
title: >-
  [Paper Note] Spatial Understanding from Videos: Structured Prompts Meet Simulation Data
description: >-
  [NeurIPS 2025][Robotics][Visual-spatial understanding] This paper proposes a two-pronged approach combining the SpatialMind structured prompting strategy and the ScanForgeQA synthetic QA dataset to substantially enhance…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Visual-spatial understanding"
  - "chain-of-thought prompting"
  - "synthetic data"
  - "vision-language models"
  - "3D reasoning"
date: 2026-05-08
content_hash: 78306eccede004e3
---

# Spatial Understanding from Videos: Structured Prompts Meet Simulation Data

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2506.03642](https://arxiv.org/abs/2506.03642)  
**Code**: [GitHub](https://github.com/Hyu-Zhang/SpatialMind)  
**Area**: Robotics
**Keywords**: Visual-spatial understanding, chain-of-thought prompting, synthetic data, vision-language models, 3D reasoning

## TL;DR

This paper proposes a two-pronged approach combining the SpatialMind structured prompting strategy and the ScanForgeQA synthetic QA dataset to substantially enhance VLMs' ability to perform 3D spatial reasoning from scanned videos, without modifying the underlying model architecture.

## Background & Motivation

- **Background**: Visual-spatial understanding — inferring spatial relationships and layout among objects from visual input — is a foundational capability for applications such as robotic navigation, autonomous driving, and augmented reality. Although point clouds are the dominant representation for 3D scene understanding, their acquisition requires expensive sensors and incurs significant computational overhead. Consequently, researchers have begun exploring purely visual approaches based solely on scanned video.

- **Limitations of Prior Work**: Two core challenges arise when performing 3D spatial reasoning from scanned video.

  **Spatial uncertainty**: In the absence of explicit depth information, models must infer 3D structure from inherently limited 2D observations; occlusion, perspective distortion, and texture ambiguity introduce substantial uncertainty, demanding multi-step logical reasoning across frames.

  **Data scarcity**: Existing datasets are small in scale, lack diversity, and are derived entirely from real-world scene scans that are difficult to scale, limiting VLMs' ability to acquire robust spatial knowledge.

- **Key Challenge**: Existing 2D spatial understanding methods (e.g., SpatialVLM, SpatialBot) suffer significant performance degradation in complex 3D environments, while most 3D methods rely on point clouds, restricting their practicality and scalability. A scalable, vision-only approach for enhancing VLMs' spatial reasoning is therefore needed.

## Method

### Overall Architecture

The framework consists of two main components: (1) **SpatialMind** — a structured chain-of-thought (CoT) prompting strategy that guides VLMs through step-by-step spatial reasoning; and (2) **ScanForgeQA** — a large-scale synthetic QA dataset automatically constructed from 3D simulated scenes for fine-tuning. The two components can be used independently or in combination, and neither requires modifications to the underlying VLM architecture.

### Key Designs

1. **Scene Decomposition**

   Scene decomposition proceeds in three steps:
   - **Local Modeling**: For each video frame, a VLM detects candidate target objects and estimates their local 3D coordinates $\mathbf{p}_{ij}^{\text{local}} \in \mathbb{R}^3$ relative to a reference object, constructing a local 3D map $\mathcal{L}_i$.
   - **Coordinate Mapping**: The VLM infers the relative rotation $\mathbf{R}_{k,k-1}$ and translation $\mathbf{t}_{k,k-1}$ between adjacent frames, accumulating the global transform for each frame as $\mathbf{T}_i = \prod_{k=1}^{i} \begin{bmatrix} \mathbf{R}_{k,k-1} & \mathbf{t}_{k,k-1} \\ \mathbf{0} & 1 \end{bmatrix}$, and converting local coordinates to global coordinates via homogeneous transformation. Repeated detections across frames are merged based on spatial proximity and semantic consistency to yield a global 3D map $\mathcal{G}$.
   - **Cognition Generation**: Three scene representations are explored: a 3D map, a 2D spatial grid (mapping objects to discrete cells $(i_k,j_k) = (\lfloor x_k/s \rfloor, \lfloor y_k/s \rfloor)$), and natural-language positional descriptions. Experiments show that VLMs comprehend textual descriptions most effectively.

2. **Question Decomposition**

   Spatial questions are categorized by type (e.g., object size, relative distance, relative direction), and a dedicated reasoning procedure is designed for each type. For instance, a "relative distance" question follows a four-step reasoning chain: identify objects → estimate coordinates → compute pairwise distances → select the minimum. During inference, the appropriate reasoning procedure is automatically selected based on question type.

3. **ScanForgeQA Dataset Construction**

   The dataset is built via a three-stage pipeline:
   - **Scene Construction**: (a) 34,116 single-room scenes are extracted from the 3D-FRONT dataset; (b) 160 additional scenes are synthesized using HoloDeck guided by an LLM.
   - **Scan Generation**: Scan videos are simulated in the Unity engine using two strategies: orbital scanning (circular trajectory at fixed height, one frame every 5°, 72 frames per revolution) and navigation scanning (paths planned in walkable areas with 360° rotations at start and end points, 72 frames per path).
   - **QA Generation**: Three question types are generated automatically — attribute estimation (object count, size, room area), spatial reasoning (relative distance, absolute distance, direction, contact relations), and hypothetical analysis (action feasibility). The final dataset comprises 34,276 scenes, 103K scan videos, and 925K QA pairs.

### Loss & Training

Standard supervised fine-tuning (SFT) is applied to train VLMs on ScanForgeQA. The SpatialMind prompting strategy is training-free. To mitigate any degradation of general-purpose capabilities, a small proportion of conventional video understanding data (e.g., 5%–10% ShareGPT4Video) can be mixed in during fine-tuning.

## Key Experimental Results

### Main Results

| Model | Method | VSI-Bench Avg | Gain |
|-------|--------|:---:|:---:|
| Qwen2.5-VL-7B | Baseline | 37.2 | - |
| Qwen2.5-VL-7B | +SpatialMind | 39.2 | ↑2.0% |
| Qwen2.5-VL-7B | +ScanForgeQA | 43.3 | ↑6.1% |
| Qwen2.5-VL-7B | +Both | 43.9 | ↑6.7% |
| InternVL2-40B | Baseline | 36.0 | - |
| InternVL2-40B | +Both | 44.5 | ↑8.5% |
| Qwen2.5-VL-72B | Baseline | 39.2 | - |
| Qwen2.5-VL-72B | +Both | 47.1 | ↑7.9% |
| GPT-4o | +SpatialMind | 40.8 | ↑6.8% |
| Gemini-1.5 Pro | +SpatialMind | 52.8 | ↑7.4% |

| Model | Method | OpenEQA Acc | ScanQA BLEU-1 | SQA3D EM-1 |
|-------|--------|:---:|:---:|:---:|
| Qwen2.5-VL-7B | Baseline | 50.1 | 32.5 | 17.2 |
| Qwen2.5-VL-7B | +Both | 58.6 | 37.9 | 24.5 |
| Qwen2.5-VL-72B | Baseline | 53.8 | 35.4 | 34.8 |
| Qwen2.5-VL-72B | +Both | 60.4 | 44.1 | 46.3 |

### Ablation Study

| Configuration | Room Size | VSI-Bench Avg | Notes |
|---------------|:---:|:---:|-------|
| Qwen2.5-VL-7B baseline | 38.9 | 37.2 | Baseline |
| +SQA3D fine-tuning | 38.8 | 38.9 | Limited gains from existing datasets |
| +ScanQA fine-tuning | 38.5 | 39.1 | Limited gains from existing datasets |
| +ScanForgeQA fine-tuning | 44.9 | 43.3 | Synthetic data yields substantially larger gains |
| CoT-Question only | 50.6 | 41.3 | Question decomposition alone |
| CoT-Scene only | 52.1 | 42.7 | Scene description contributes more |
| Full SpatialMind | 53.8 | 44.0 | The two components are complementary |

### Key Findings

- Textual descriptions are the scene representation format most readily understood by VLMs, outperforming 3D maps and 2D grids.
- Larger models benefit more from prompting strategies, while smaller models gain more from fine-tuning (7B: +6.1% from fine-tuning vs. +2.0% from prompting).
- Prompting and fine-tuning are complementary; combining both consistently yields additional gains.
- Fine-tuning on ScanForgeQA has only a minor effect on general video understanding capabilities (MVBench improves slightly; Video-MME declines slightly), and data mixing can mitigate this trade-off.

## Highlights & Insights

- The purely visual approach requires no architectural modification, making it broadly applicable across VLMs of various scales and types.
- The synthetic data pipeline is highly scalable, avoiding the high cost of real-world data collection.
- Humans and VLMs exhibit complementary strengths: humans excel at qualitative tasks (e.g., 100% accuracy on appearance ordering), while VLMs surpass humans at precise quantitative estimation.

## Limitations & Future Work

- Scene decomposition relies on the VLM's own pose estimation capability, and errors may accumulate under severe viewpoint changes.
- A domain gap between simulated and real-world data remains.
- Textual description formats may insufficiently compress information in scenes with dense objects.
- Integration with depth estimation models or SLAM techniques could be explored to improve coordinate accuracy.

## Related Work & Insights

- The paper contrasts its approach with 2D spatial understanding methods such as SpatialVLM and SpatialRGPT, highlighting their limitations in complex 3D scenes.
- The CoT prompting paradigm is generalizable to other visual tasks requiring multi-step reasoning.
- The combination of synthetic data and fine-tuning provides a replicable paradigm for data-scarce domains.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The dual strategy of structured prompting and synthetic data is a reasonable innovation, though individual components are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Experiments are highly comprehensive, spanning multiple benchmarks, model scales, and thorough ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rich figures and tables.
- **Value**: ⭐⭐⭐⭐ Provides a practical solution for enhancing spatial reasoning, with direct relevance to embodied intelligence research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking the Simulation vs. Rendering Dichotomy: No Free Lunch in Spatial World Modelling](rethinking_the_simulation_vs_rendering_dichotomy_no_free_lunch_in_spatial_world_.md)
- [\[ICLR 2026\] UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos](../../ICLR2026/robotics/urbanverse_scaling_urban_simulation_by_watching_city-tour_videos.md)
- [\[NeurIPS 2025\] Asymptotically Stable Quaternionic Hopfield Structured Neural Network with Supervised Projection-based Manifold Learning](asymptotically_stable_quaternion-valued_hopfield-structured_neural_network_with_.md)
- [\[NeurIPS 2025\] Learning Spatial-Aware Manipulation Ordering](learning_spatial-aware_manipulation_ordering.md)
- [\[NeurIPS 2025\] LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents](labutopia_high-fidelity_simulation_and_hierarchical_benchmark_for_scientific_emb.md)

</div>

<!-- RELATED:END -->
