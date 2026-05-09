---
title: >-
  [Paper Note] Event-Driven Storytelling with Multiple Lifelike Humans in a 3D Scene
description: >-
  [ICCV 2025][3D Vision][Multi-Human Motion] This paper proposes an event-driven LLM framework that decomposes multi-character behavior planning in 3D scenes into two modules — a Narrator for event-by-event generation and an Event Parser for fine-grained spatial reasoning — achieving, for the first time, long-horizon natural interaction motion generation for 4–5+ characters in large-scale multi-room 3D scenes.
tags:
  - ICCV 2025
  - 3D Vision
  - Multi-Human Motion
  - LLM Planning
  - Event-Based
  - Scene-Aware
  - 3D Scene Graph
date: 2026-05-08
content_hash: fad8e73ec2483bfd
---

# Event-Driven Storytelling with Multiple Lifelike Humans in a 3D Scene

**Conference**: ICCV 2025
**arXiv**: [2507.19232](https://arxiv.org/abs/2507.19232)
**Code**: See project page
**Area**: 3D Vision / Multi-Agent Motion Generation / LLM Planning
**Keywords**: Multi-Human Motion, LLM Planning, Event-Based, Scene-Aware, 3D Scene Graph

## TL;DR
This paper proposes an event-driven LLM framework that decomposes multi-character behavior planning in 3D scenes into two modules — a Narrator for event-by-event generation and an Event Parser for fine-grained spatial reasoning — achieving, for the first time, long-horizon natural interaction motion generation for 4–5+ characters in large-scale multi-room 3D scenes.

## Background & Motivation

Generating realistic digital human motion in 3D scenes is critical for VR, gaming, and film production. However, existing motion synthesis work exhibits notable limitations:
- Most focus on **single-character** independent motion, or extend only to **a single interaction type** such as human-environment or human-human interaction.
- Lack of generalizability and scalability: unable to simultaneously generate plausible interactions among multiple characters in cluttered scenes.
- Correctly assigning each character's behavior at the right time requires **extensive dynamic context and spatial reasoning**.

**Potential and Challenges of LLMs**:
- LLMs excel at high-level planning and behavior modeling.
- However, directly generating commands suffers from hallucination or grounding errors (known limitations of foundation models).
- Carefully designed frameworks are needed to reduce reasoning complexity.

**Core Challenge**: How to achieve deep understanding of 3D space and coordinated planning of multi-character behavior while maintaining scalability?

## Method

### Overall Architecture
The system operates around an intermediate representation — the **event** $e$:
- **High-level behavior planning module** (LLM-driven): plans at the granularity of events.
- **Low-level motion synthesis module**: converts events into 3D character motions.

### High-Level Behavior Planning Module

Comprises three LLM sub-modules:

**1. Scene Describer**
- Automatically extracts a 3D scene graph (spatial relationships in JSON format) from the 3D scene $\mathcal{S}$.
- Applies DBSCAN to cluster objects into regions, discovering functional spaces (e.g., dining area, study area).
- Guides the LLM via in-context learning to generate a scene description $\mathcal{D}$ enriched with regional contextual information.

**2. Narrator**
- Generates one event at a time, conditioned on scene description $\mathcal{D}$, event history $\mathcal{H}$, and optional user instruction $\mathcal{T}$.
- Events are described in semi-narrative natural language, focusing on context rather than precise coordinates.
- Receives event status feedback ('ongoing'/'completed') to drive timeline progression.
- Employs chain-of-thought reasoning: first analyzes the current planning state and the states of other characters, then generates the event.

**3. Event Parser**
- Converts high-level event descriptions into low-level details: $e=(\mathcal{C}_e, \{p_i\}, \{d_i\}, \{a_i\})$
    - $\mathcal{C}_e$: set of involved characters
    - $p_i \in \mathbb{R}^2$: target position
    - $d_i \in \mathbb{R}^1$: target orientation
    - $a_i$: target action label
- **Python programming-style prompting**: provides spatial reasoning utility functions (e.g., `get_distance_between()`), enabling the LLM to reason about spatial relationships by writing code.
- **Region-conditioned position sampling**: first outputs a semantic description (e.g., "the chair farthest from the reception desk"), then samples coordinates within the corresponding region, compensating for the LLM's weakness in coordinate-level reasoning.

### Low-Level Motion Synthesis Module
- Plans collision-free multi-agent paths on a 2D grid map using a windowed cooperative A* algorithm.
- Motion matching rapidly generates actions toward target positions.
- For group events (e.g., chatting, handshaking): characters that arrive first remain idle until all participants are present, then execute actions synchronously.

## Key Experimental Results

### Benchmark Evaluation

| Model | Overall Success (Exec. Rate) | OA | RC | SS |
|-------|------------------------------|-----|-----|-----|
| **Ours (GPT-4o)** | **0.90 (0.98)** | **0.93 (0.99)** | **0.90 (0.98)** | **0.92 (0.98)** |
| w/o Event | 0.82 (0.92) | 0.88 (0.92) | 0.86 (0.93) | 0.77 (0.92) |
| Object List | 0.51 (0.85) | 0.61 (0.78) | 0.28 (0.88) | 0.65 (0.97) |
| Scene Graph | 0.82 (0.96) | 0.80 (0.96) | 0.82 (0.94) | 0.87 (0.95) |

OA = Object Arrangement Reasoning, RC = Regional Context Reasoning, SS = Scene State Reasoning

### Generalization Across LLM Engines

| Model | GPT-4o | GPT-4o mini | Llama-3.1-70B |
|-------|--------|-------------|---------------|
| Ours | 0.90 | 0.74 | 0.72 |
| w/o Event | 0.82 | 0.60 | 0.60 |
| Object List | 0.51 | 0.34 | 0.35 |

**Key Findings**:
- The event-driven design consistently outperforms the no-event variant across all LLM engines (+8%–14% success rate).
- The Scene Describer substantially outperforms a simple object list (+31%–39%), particularly on regional context reasoning (0.90 vs. 0.28).
- Performance degradation is moderate when using weaker LLMs (GPT-4o mini, Llama-3.1-70B).
- The Scene Graph variant performs comparably on the SS dimension (0.87 vs. 0.92) but shows a gap on the RC dimension (0.82 vs. 0.90).

## Highlights & Insights
1. **Event-driven decomposition**: Decomposes complex multi-character long-horizon planning into per-event decisions, significantly reducing LLM reasoning complexity.
2. **Modular decoupling**: The Narrator handles macro-level narrative while the Event Parser handles precise spatial grounding, with clearly separated responsibilities.
3. **Programmatic spatial reasoning**: Cleverly employs Python functions as spatial reasoning tools for the LLM, compensating for its weakness in coordinate-level reasoning.
4. **Functional region discovery**: DBSCAN clustering identifies functional regions within the scene, providing spatial understanding beyond simple object lists.

## Limitations & Future Work
- The motion synthesis module relies on existing frameworks (motion matching), and motion quality is constrained by the action library.
- 3D scenes require instance segmentation annotations as input.
- The evaluation benchmark is limited in scale (40 test cases); generalization to larger-scale and more complex scenes remains to be verified.
- LLM inference latency may hinder real-time interactive applications.
- Physical contact actions between characters (e.g., object handover, push/pull) are not addressed.

## Related Work & Insights
- Contextual motion synthesis: the Digital Life Project handles dyadic scenes but does not scale to more characters.
- LLM planning: SMART-LLM addresses multi-robot settings but with simplistic scene understanding (object list only).
- Scene understanding: 3D scene graphs provide spatial relationships but lack functional region reasoning.

## Rating
- **Novelty**: ★★★★☆ — Event-driven + modular LLM design is the first to address large-scale multi-character generation.
- **Practicality**: ★★★★☆ — Applicable to virtual scene population in VR/gaming/film.
- **Experimental Thoroughness**: ★★★☆☆ — Introduces a new benchmark but at a small scale; direct comparison with existing methods is lacking.
- **Writing Quality**: ★★★★☆ — Framework is clearly presented with rich illustrative examples.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [\[ICCV 2025\] ETCH: Generalizing Body Fitting to Clothed Humans via Equivariant Tightness](etch_generalizing_body_fitting_to_clothed_humans_via_equivariant_tightness.md)
- [\[ICCV 2025\] 3D Test-time Adaptation via Graph Spectral Driven Point Shift](3d_test-time_adaptation_via_graph_spectral_driven_point_shift.md)
- [\[ICCV 2025\] Event-based Tiny Object Detection: A Benchmark Dataset and Baseline](event-based_tiny_object_detection_a_benchmark_dataset_and_baseline.md)
- [\[ICCV 2025\] EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images](evagaussians_event_stream_assisted_gaussian_splatting_from_blurry_images.md)

<!-- RELATED:END -->
