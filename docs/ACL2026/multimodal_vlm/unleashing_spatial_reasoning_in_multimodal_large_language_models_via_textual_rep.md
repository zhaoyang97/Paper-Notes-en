---
title: >-
  [Paper Note] TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning
description: >-
  [ACL 2026][Multimodal VLM][Spatial Reasoning] This paper proposes TRACE (Textual Representation of Allocentric Context from Egocentric Video)…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Spatial Reasoning"
  - "Multimodal Large Language Models"
  - "Textual Representation"
  - "Egocentric Video"
  - "Prompt Engineering"
date: 2026-05-08
content_hash: 9bf3665b53965e3a
---

# TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning

**Conference**: ACL 2026  
**arXiv**: [2603.23404](https://arxiv.org/abs/2603.23404)  
**Code**: [https://trace-reasoning.github.io](https://trace-reasoning.github.io)  
**Area**: Multimodal VLM / Spatial Reasoning  
**Keywords**: Spatial Reasoning, Multimodal Large Language Models, Textual Representation, Egocentric Video, Prompt Engineering

## TL;DR

This paper proposes TRACE (Textual Representation of Allocentric Context from Egocentric Video), a prompting method that guides multimodal large language models to generate structured textual allocentric 3D environment representations—including meta-context, camera trajectories, and entity registries—as intermediate reasoning steps to enhance spatial question-answering capabilities. This approach consistently outperforms existing prompting strategies on VSI-Bench and OST-Bench.

## Background & Motivation

**Background**: Existing Multimodal Large Language Models (MLLMs) have made significant progress in tasks such as video understanding and image captioning, but perform poorly in 3D spatial reasoning. Cognitive science research suggests that humans perform 3D reasoning by constructing allocentric (environment-centered) spatial representations rather than directly operating at the pixel level.

**Limitations of Prior Work**: Current MLLMs rely excessively on 2D visual signals, learning spurious shortcut associations from implicit spatial cues and failing to establish hierarchical abstractions of 3D scenes. Prior works either fine-tune using large amounts of spatial reasoning data (poor scalability) or introduce additional geometric/stereo modalities (high system complexity), which are unsuitable for off-the-shelf MLLMs.

**Key Challenge**: Standard reasoning methods like Chain-of-Thought (CoT) are effective for arithmetic and symbolic tasks but are often ineffective or even harmful for complex spatial reasoning tasks. This is because these methods generate reasoning traces that fail to capture spatial geometric structure. Models need to reason explicitly based on global 3D representations.

**Goal**: To design a text-only spatial representation method as an intermediate reasoning step for MLLMs to enhance spatial reasoning capabilities without modifying model architecture or adding extra modalities.

**Key Insight**: Inspired by allocentric spatial reasoning in human cognition, where humans mentally place themselves within an environment to construct a global scene layout when answering spatial questions, the authors observe that such allocentric representations can be fully described with text.

**Core Idea**: To prompt MLLMs to first generate a structured textual 3D representation (including meta-context, camera trajectory, and entity registry) as a "spatial cache" loaded into the context window, and then reason based on this cache—thus transforming spatial reasoning into queries over structured text.

## Method

### Overall Architecture

TRACE employs a single-turn generation approach: given an egocentric video $V$ and a natural language question $Q$, the model first acts as a "spatial descriptor" to generate the TRACE representation $G$, and then as a "推理解析器" (reasoning parser) to generate the final answer $A$ based on $G$ and $V$. The reasoning process is formalized as $\hat{A}, \hat{G} = \arg\max P(A|G,V,Q) \cdot P(G|V,Q)$. The entire process is completed in one forward pass, where TRACE serves as a structured CoT.

### Key Designs

1.  **Meta Context**:
    - **Function**: Establish global coordinate systems and room layout information.
    - **Mechanism**: A "Room-Aligned Coordinate System" is proposed, with the observer's starting position as the origin $[0,0]$. The $y$-axis direction is determined by detecting the most prominent straight lines defined by large static objects (rather than the initial camera orientation). It also records room topology (e.g., "rectangular bedroom"), grid orientation, and the observer's initial orientation. This providing a unified reference frame for all subsequent spatial calculations.
    - **Design Motivation**: A common failure mode in spatial reasoning is the loss of camera initialization and coordinate system information. Defining coordinate axes using large static objects instead of camera orientation avoids reference frame instability caused by camera rotation.

2.  **Camera Trajectory**:
    - **Function**: Reconstruct the observer's movement path in 3D space.
    - **Mechanism**: The video is divided into discrete time steps, recording timestamps, estimated positions $[x, y]$, and camera headings for each step. Headings use 8 discrete directions (cardinal directions), as precise angle estimation is too difficult for models. Each step also includes an action attribute to encode camera movement context. Localization is performed relative to reference points of large static objects in the meta-context.
    - **Design Motivation**: Static maps cannot capture the dynamic nature of videos. Trajectory reconstruction enables models to answer navigation and path-planning questions by traversing the generated static map instead of relying on transient visual memory.

3.  **Entity Registry**:
    - **Function**: Maintain a structured record of attributes for all observed objects in the scene.
    - **Mechanism**: For each entity, the system records: timestamp (first appearance), visual signature (appearance description for disambiguation), metric estimation (2D coordinates $[x,y]$ relative to the origin in meters), and spatial relations (natural language relative relations with nearby entities). Entities must be listed individually (e.g., chair_01, chair_02) rather than grouped, ensuring precise counting and localization.
    - **Design Motivation**: Unlike Cognitive Map predictions that use loose grid cells, an entity registry with detailed attributes forces the model to parse spatial relations into geometric constraints. Timestamps and visual signatures provide deduplication and cross-time disambiguation capabilities.

### Loss & Training

TRACE is a pure prompting method that does not involve any training or fine-tuning. Inference is completed in a single forward pass: the model generates a schema-compatible TRACE representation, loads it as a "spatial cache" into the context window, and then derives the final answer by calculating Euclidean distances between entity coordinates or traversing trajectory nodes.

## Key Experimental Results

### Main Results

**Average performance of different prompting methods on VSI-Bench**

| Method | Gemini 3 Pro | Qwen2.5-VL-72B | MiMo-VL-7B |
|------|-------------|----------------|------------|
| Direct | 52.61 | 36.28 | 39.79 |
| CoT | 53.65 | 29.78 | 37.49 |
| ToT | 58.88 | 38.06 | 39.14 |
| LtM | 59.52 | 38.01 | 38.34 |
| CM (Cognitive Map) | 59.72 | 35.47 | 36.85 |
| **TRACE (Ours)** | **60.15** | **39.38** | **40.50** |

**Overall accuracy of different prompting methods on OST-Bench**

| Method | Gemini 3 Pro | Qwen2.5-VL-72B |
|------|-------------|----------------|
| Direct | 69.73 | 61.53 |
| CoT | 69.76 | 60.33 |
| CM | 68.47 | 57.45 |
| **TRACE (Ours)** | **70.36** | **62.68** |

### Ablation Study

| Configuration | VSI-Bench Avg | Description |
|------|--------------|------|
| Full TRACE | 60.15 | Complete model |
| w/o Meta Context | 58.27 | Decrease of 1.88 |
| w/o Trajectory | 58.92 | Decrease of 1.23 |
| w/o Entity Registry | 57.43 | Decrease of 2.72 |
| Grid only (no structured attributes) | 56.81 | Significantly lower performance |

### Key Findings

- CoT performed 6.5 points worse than Direct on Qwen2.5-VL-72B, confirming that standard reasoning prompts can be harmful to spatial tasks.
- TRACE achieved the best or near-best performance across all three base models, demonstrating consistency across models.
- The Entity Registry contributes the most—its removal leads to the largest performance drop, indicating that fine-grained object attributes and coordinate estimation are critical for spatial reasoning.
- TRACE is also effective in the multi-turn dialogue setting of OST-Bench, showing it is not limited to single-turn QA.
- Object counting and absolute distance estimation are the most difficult tasks; TRACE's improvements are particularly significant in these areas.

## Highlights & Insights

- The introduction of allocentric spatial cognition theory from cognitive science into MLLM prompt design to simulate human spatial mental representations via text is an elegant interdisciplinary approach.
- As a pure prompting method, TRACE does not require any training data or model modifications and can be directly applied to any off-the-shelf MLLM, making it highly practical.
- The "spatial cache" concept is clever—it transforms 3D spatial reasoning into queries over structured text, leveraging LLMs' strengths in textual reasoning to compensate for their weaknesses in 3D perception.

## Limitations & Future Work

- The quality of the generated TRACE representation depends entirely on the MLLM's visual understanding; if the model fails to accurately perceive object locations, subsequent reasoning will also be incorrect.
- Coordinate estimation is inherently approximate and may not be sufficiently accurate for tasks requiring precise metrics (e.g., absolute distance estimation).
- Validation was only conducted on indoor scenes (VSI-Bench and OST-Bench), and applicability to outdoor open scenes is unknown.
- Future work could consider introducing an iterative correction mechanism to allow the model to self-verify and refine the generated TRACE.

## Related Work & Insights

- **vs Cognitive Map (CM)**: CM uses loose grid cell predictions, whereas TRACE uses an entity registry with detailed attributes to provide finer-grained spatial information.
- **vs Thinking in Space**: The latter demonstrates the benefits of externalized spatial representations but requires specific training; TRACE achieves similar effects via pure prompting.
- **vs VideoTree/VideoAgent**: These methods optimize evidence retrieval for long videos, while TRACE focuses on enabling models to reason explicitly with 3D geometric cues.

## Rating

- Novelty: ⭐⭐⭐⭐ The translation of allocentric cognitive theory into structured prompts is novel, though the core remains a carefully designed CoT variant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Good coverage with two benchmarks, three models, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, intuitive method, and high-quality illustrations.
- Value: ⭐⭐⭐⭐ Provides a practical and general prompting strategy for spatial reasoning that is "plug-and-play."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)
- [\[NeurIPS 2025\] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs](../../NeurIPS2025/multimodal_vlm/struct2d_a_perception-guided_framework_for_spatial_reasoning_in_mllms.md)
- [\[ICLR 2026\] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](../../ICLR2026/multimodal_vlm/spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)
- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)

</div>

<!-- RELATED:END -->
