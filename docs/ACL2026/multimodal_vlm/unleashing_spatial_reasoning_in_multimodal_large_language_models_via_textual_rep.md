---
title: >-
  [Paper Note] TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning
description: >-
  [ACL 2026][Multimodal VLM][spatial reasoning] This paper proposes TRACE (Textual Representation of Allocentric Context from Egocentric Video), a prompting method that guides multimodal large language models to generate structured textual allocentric 3D environment representations from egocentric video—comprising meta context, camera trajectory, and an entity registry—as intermediate reasoning steps to enhance spatial question answering. TRACE consistently outperforms existing prompting strategies on both VSI-Bench and OST-Bench.
tags:
  - ACL 2026
  - Multimodal VLM
  - spatial reasoning
  - multimodal large language models
  - textual representation
  - egocentric video
  - prompt engineering
date: 2026-05-08
content_hash: 959b7aaeb1e8d792
---

# TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning

**Conference**: ACL 2026  
**arXiv**: [2603.23404](https://arxiv.org/abs/2603.23404)  
**Code**: [https://trace-reasoning.github.io](https://trace-reasoning.github.io)  
**Area**: Multimodal VLM / Spatial Reasoning  
**Keywords**: spatial reasoning, multimodal large language models, textual representation, egocentric video, prompt engineering

## TL;DR

This paper proposes TRACE (Textual Representation of Allocentric Context from Egocentric Video), a prompting method that guides multimodal large language models to generate structured textual allocentric 3D environment representations from egocentric video—comprising meta context, camera trajectory, and an entity registry—as intermediate reasoning steps to enhance spatial question answering. TRACE consistently outperforms existing prompting strategies on both VSI-Bench and OST-Bench.

## Background & Motivation

**Background**: Existing multimodal large language models have achieved notable progress on tasks such as video understanding and image captioning, yet perform poorly on 3D spatial reasoning. Cognitive science research indicates that humans construct allocentric (environment-centered) spatial representations to reason about 3D space, rather than operating directly at the pixel level.

**Limitations of Prior Work**: Current MLLMs over-rely on 2D visual signals, learning spurious shortcut associations from implicit spatial cues and failing to construct hierarchical abstractions of 3D scenes. Prior work either fine-tunes on large amounts of spatial reasoning data (poor scalability) or introduces additional geometric/depth modalities (high system complexity), neither of which is suitable for off-the-shelf MLLMs.

**Key Challenge**: Standard reasoning methods such as Chain-of-Thought are effective for arithmetic and symbolic tasks but often fail—or even degrade performance—on complex spatial reasoning tasks, because the reasoning traces they generate fail to capture spatial geometric structure. Models require explicit reasoning grounded in global 3D representations.

**Goal**: To design a purely textual spatial representation method that serves as an intermediate reasoning step for MLLMs, enhancing spatial reasoning capability without modifying model architecture or introducing additional modalities.

**Key Insight**: Inspired by allocentric spatial reasoning in human cognition—whereby humans mentally situate themselves within an environment and construct a global scene layout representation when answering spatial questions—the authors observe that such allocentric representations can be fully expressed in natural language text.

**Core Idea**: MLLMs first generate a structured textual 3D representation (comprising meta context, camera trajectory, and entity registry) as a "spatial cache" loaded into the context window, and then perform reasoning over this cache. This transforms spatial reasoning into a structured-text query task.

## Method

### Overall Architecture

TRACE adopts a single-turn generation paradigm: given an egocentric video $V$ and a natural language question $Q$, the model first acts as a "spatial describer" to generate the TRACE representation $G$, and then acts as a "reasoning parser" to generate the final answer $A$ conditioned on $G$ and $V$. The inference process is formalized as $\hat{A}, \hat{G} = \arg\max P(A|G,V,Q) \cdot P(G|V,Q)$. The entire process is completed in a single forward pass, with TRACE serving as a structured form of Chain-of-Thought.

### Key Designs

1. **Meta Context**:

    - Function: Establishes a global coordinate system and room layout information.
    - Mechanism: Proposes a "room-aligned coordinate system" with the observer's starting position as the origin $[0,0]$. The $y$-axis direction is determined by detecting the most salient straight line defined by large static objects, rather than by the camera's initial orientation. Room topology (e.g., "rectangular bedroom"), grid orientation, and the observer's initial heading are also recorded, providing a unified reference frame for all subsequent spatial computations.
    - Design Motivation: A common failure mode in spatial reasoning is loss of camera initialization and coordinate system information. Defining coordinate axes via large static objects rather than camera orientation avoids reference frame instability caused by camera rotation.

2. **Camera Trajectory**:

    - Function: Reconstructs the observer's motion path through 3D space.
    - Mechanism: The video is divided into discrete time steps; at each step, the timestamp, estimated position $[x, y]$, and camera heading are recorded. Heading is expressed using 8 discrete cardinal directions, as precise angle estimation is too difficult for the model. Each step also includes an action attribute encoding camera motion context. Large static objects from the meta context serve as reference points for localization.
    - Design Motivation: A static map cannot capture the dynamic nature of video. Trajectory reconstruction enables the model to answer navigation and path-planning questions by traversing the generated static map, rather than relying on instantaneous visual memory.

3. **Entity Registry**:

    - Function: Maintains structured attribute records for all observed objects in the scene.
    - Mechanism: Each entity is recorded with: timestamp (time of first appearance), visual signature (appearance description for disambiguation), metric estimates (2D coordinates $[x,y]$ in meters relative to the coordinate origin), and spatial relations (natural language relative relationships with nearby entities). Entities must be listed individually (e.g., chair_01, chair_02) and grouping is disallowed, ensuring precise counting and localization.
    - Design Motivation: Unlike Cognitive Map, which predicts loose grid cells, the detailed-attribute entity registry compels the model to resolve spatial relationships into geometric constraints. Timestamps and visual signatures further enable deduplication and cross-temporal disambiguation.

### Loss & Training

TRACE is a purely prompting-based method involving no training or fine-tuning. Inference is completed in a single forward pass: the model first generates a schema-compliant TRACE representation and loads it as a "spatial cache" into the context window, then derives the final answer by computing Euclidean distances between entity coordinates or traversing trajectory nodes.

## Key Experimental Results

### Main Results

**Average performance of different prompting methods on VSI-Bench**

| Method | Gemini 3 Pro | Qwen2.5-VL-72B | MiMo-VL-7B |
|--------|-------------|----------------|------------|
| Direct | 52.61 | 36.28 | 39.79 |
| CoT | 53.65 | 29.78 | 37.49 |
| ToT | 58.88 | 38.06 | 39.14 |
| LtM | 59.52 | 38.01 | 38.34 |
| CM (Cognitive Map) | 59.72 | 35.47 | 36.85 |
| **TRACE (Ours)** | **60.15** | **39.38** | **40.50** |

**Overall accuracy of different prompting methods on OST-Bench**

| Method | Gemini 3 Pro | Qwen2.5-VL-72B |
|--------|-------------|----------------|
| Direct | 69.73 | 61.53 |
| CoT | 69.76 | 60.33 |
| CM | 68.47 | 57.45 |
| **TRACE (Ours)** | **70.36** | **62.68** |

### Ablation Study

| Configuration | VSI-Bench Avg | Note |
|---------------|--------------|------|
| Full TRACE | 60.15 | Full model |
| w/o Meta Context | 58.27 | −1.88 |
| w/o Trajectory | 58.92 | −1.23 |
| w/o Entity Registry | 57.43 | −2.72 |
| Grid only (no structured attributes) | 56.81 | Substantial drop |

### Key Findings

- CoT underperforms Direct by 6.5 points on Qwen2.5-VL-72B, confirming that standard reasoning prompts can be detrimental for spatial tasks.
- TRACE achieves the best or near-best performance across all three base models, demonstrating consistent cross-model generalization.
- The Entity Registry contributes the most—its removal causes the largest performance drop—indicating that fine-grained object attributes and coordinate estimation are critical for spatial reasoning.
- TRACE remains effective in the multi-turn dialogue setting of OST-Bench, demonstrating that it is not limited to single-turn question answering.
- Object counting and absolute distance estimation are the most challenging task types, and TRACE yields the most substantial improvements on these tasks.

## Highlights & Insights

- Translating the allocentric spatial cognition theory from cognitive science into MLLM prompt design—using text to simulate human spatial mental representations—is an elegant cross-disciplinary contribution.
- As a purely prompting-based method requiring no training data or model modification, TRACE can be directly applied to any off-the-shelf MLLM, making it highly practical.
- The "spatial cache" concept is ingenious: transforming 3D spatial reasoning into structured-text querying leverages the capability at which LLMs excel (textual reasoning) to compensate for their greatest weakness (3D perception).

## Limitations & Future Work

- The quality of the single-pass TRACE generation is entirely dependent on the MLLM's visual understanding capability; if the model cannot accurately perceive object positions, downstream reasoning will also be erroneous.
- Coordinate estimation is inherently approximate, which may be insufficient for tasks requiring precise measurements (e.g., absolute distance estimation).
- Evaluation is limited to indoor scenes (VSI-Bench and OST-Bench); applicability to outdoor open-world scenes remains unknown.
- Future work could incorporate iterative correction mechanisms, enabling the model to self-verify and refine the generated TRACE representation.

## Related Work & Insights

- **vs. Cognitive Map (CM)**: CM predicts loose grid cells, whereas TRACE employs a detailed-attribute entity registry that provides more fine-grained spatial information.
- **vs. Thinking in Space**: That work demonstrates the benefits of externalizing spatial representations but requires task-specific training; TRACE achieves a similar effect through prompting alone.
- **vs. VideoTree/VideoAgent**: These methods optimize evidence retrieval for long videos, whereas TRACE focuses on enabling models to explicitly reason over 3D geometric cues.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of translating allocentric cognitive theory into structured prompts is original, though the core approach remains a carefully designed CoT variant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks, three models, and comprehensive ablations provide good coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, method intuition is well articulated, and figures are of high quality.
- Value: ⭐⭐⭐⭐ Provides a practical and generalizable plug-and-play prompting strategy for spatial reasoning.

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
