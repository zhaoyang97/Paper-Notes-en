---
title: >-
  [Paper Note] UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction
description: >-
  [ICML 2025][Object Detection][GUI benchmark] UI-Vision is proposed – the first comprehensive offline evaluation benchmark for desktop environments, covering 83 software applications. It provides dense annotations of bounding boxes, UI labels, and action trajectories. It defines a three-level evaluation task from fine-grained to coarse-grained (Element Grounding $\rightarrow$ Layout Grounding $\rightarrow$ Action Prediction) to systematically evaluate and reveal key shortcomin…
tags:
  - "ICML 2025"
  - "Object Detection"
  - "GUI benchmark"
  - "desktop agent"
  - "element grounding"
  - "layout grounding"
  - "action prediction"
date: 2026-05-08
content_hash: 23bc88d08e7bcf8d
---

# UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction

**Conference**: ICML 2025  
**Authors**: Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Juan A. Rodriguez, Montek Kalsi, etc.  
**arXiv**: [2503.15661](https://arxiv.org/abs/2503.15661)  
**Code**: Open source (MIT License)  
**Area**: Object Detection  
**Keywords**: GUI benchmark, desktop agent, element grounding, layout grounding, action prediction  

## TL;DR

UI-Vision is proposed – the first comprehensive offline evaluation benchmark for desktop environments, covering 83 software applications. It provides dense annotations of bounding boxes, UI labels, and action trajectories. It defines a three-level evaluation task from fine-grained to coarse-grained (Element Grounding $\rightarrow$ Layout Grounding $\rightarrow$ Action Prediction) to systematically evaluate and reveal key shortcomings of SOTA models in professional software understanding, spatial reasoning, and complex actions.

## Background & Motivation

**Background**: Autonomous GUI agents automate tasks like document editing and file management by navigating graphical user interfaces, representing a major application area of LLM agents. Several Web and mobile GUI benchmarks (such as Mind2Web, ScreenSpot, OSWorld) have driven rapid progress in GUI understanding models.

**Limitations of Prior Work**: (1) **Severe lack of desktop environments**—the complexity of desktop native applications (IDEs, image editors, terminals, etc.) far exceeds Web pages, but almost no benchmarks exist due to data collection difficulties and software licensing issues; (2) **Unreproducible online evaluations**—existing benchmarks mostly rely on online interactive environments where execution results might vary per run, making standardized comparisons impossible; (3) **Sparse annotations**—existing desktop datasets lack dense annotations (precise bounding boxes, functional labels, complete action trajectories), resulting in coarse evaluation granularity; (4) **Lack of hierarchical evaluation**—it is impossible to pinpoint the specific bottlenecks of models along the visual perception pipeline.

**Key Challenge**: While the desktop environment is one of the most critical professional workspaces, the actual performance of current GUI agents' visual perception capabilities under desktop environments remains completely unknown.

**Key Insight**: Constructing a license-friendly (MIT License), offline static benchmark. Through a three-tiered task design, model capabilities are diagnosed layer by layer from "whether they can perceive elements" to "whether they can understand layouts" and then to "whether they can execute actions."

**Core Idea**: Utilizing densely annotated desktop GUI screenshots to build a hierarchical offline benchmark, thereby pinpointing the visual perception bottlenecks of GUI agents.

## Method

### Overall Architecture

UI-Vision consists of three core components: (1) a densely annotated dataset covering 83 desktop applications; (2) a three-level evaluation task hierarchy from fine to coarse; and (3) standardized evaluation metrics tailored for each task tier. All evaluations operate in an offline mode based on static screenshots rather than online interactions, ensuring full reproducibility.

### Key Designs

1. **Dataset and Annotation Schema**:

    - **Function**: Provide high-quality, high-density desktop GUI evaluation data.
    - **Mechanism**: Real-world operations from humans across 83 desktop software are collected, providing three types of dense annotations for each screenshot: (a) precise bounding boxes for UI elements; (b) functional description labels (UI labels) for each element; (c) complete action trajectories, including click coordinates, drag start/end points, and keyboard input sequences. Software categories cover creative tools (Photoshop, Blender), development environments (VS Code, Terminal), office suites, system tools, communication tools, etc.
    - **Design Motivation**: Dense annotations allow a single dataset to support multi-granularity evaluations, avoiding the need to construct separate datasets for different tasks.

2. **Three-Level Evaluation Tasks**:

    - **Function**: Diagnose model capability bottlenecks step-by-step along the visual perception pipeline, from fine-grained to coarse-grained.
    - **Mechanism**:
        - **Task 1: Element Grounding**—Given a screenshot and an element description, predict the bounding box of the target UI element. This is further subdivided into Basic (basic element identification), Functional (grounding based on functional description), and Spatial (involving spatial relationship reasoning) subsets. Evaluation Metric: IoU matching accuracy.
        - **Task 2: Layout Grounding**—Given a screenshot, identify and locate all UI elements and their hierarchical relationships to evaluate the model's understanding of the overall layout.
        - **Task 3: Action Prediction**—Given a screenshot and a task description, predict the next action (click coordinates, drag trajectory, or keyboard input). This is the most challenging end-to-end task, simultaneously testing visual understanding and operational reasoning.
    - **Design Motivation**: The hierarchical design enables researchers to precisely locate where failures occur—whether the model fails to see elements, fails to understand the layout, or does not know what action to perform.

3. **Offline Evaluation Design**:

    - **Function**: Ensure reproducibility, standardization, and scalability of evaluations.
    - **Mechanism**: All evaluations are based on the same set of static screenshots, requiring no interaction with the actual operating system. New models can be quickly evaluated by inferring on fixed inputs, avoiding the non-determinism of online environments.
    - **Design Motivation**: The execution results of online benchmarks are highly affected by environmental states (e.g., pop-ups, window position changes), making fair comparisons between different models challenging.

### Loss & Training

This work focuses on benchmark construction and does not involve model training.

## Experiments

### Main Results

| Model | Element (Basic) | Element (Spatial) | Layout Grounding | Action (Click) | Action (Drag) |
|------|----------------|-------------------|-----------------|----------------|---------------|
| UI-TARS-72B | Highest | Medium | Higher | Medium | Low |
| Claude 3.5 Sonnet | Medium | Lower | Medium | Lower | Low |
| GPT-4o | Medium | Lower | Lower | Lower | Low |
| Qwen2-VL | Medium | Lower | Medium | Lower | Low |

### Ablation Study

| Challenge Dimension | Performance Characteristics | Typical SOTA Errors |
|---------|---------|--------------|
| Professional software understanding | Significant degradation across all models | Inability to recognize special icons and controls in professional toolbars |
| Spatial reasoning | Accuracy of the Spatial subset is significantly lower than Basic | Incorrect judgments of positional descriptions like "the second button on the right" |
| Drag actions | Lowest score in Action Prediction | Large deviations in predicting start/end coordinates, weak path planning capability |
| Keyboard input | Worse than clicks but better than dragging | Difficulties in predicting shortcut key combinations |
| Cross-app generalization | Common apps >> Professional apps | Sharp drop in capability to understand niche/professional software interfaces |

### Key Findings

- Even the strongest UI-TARS-72B (72B parameters, specialized training) still performs poorly on Spatial element grounding and drag action prediction.
- Professional software (3D modeling, audio/video editing) is a blind spot for all models, as such interfaces are rarely included in training data.
- For Element Grounding, the difficulty escalates from Basic $\rightarrow$ Functional $\rightarrow$ Spatial, with Spatial reasoning representing the most common bottleneck.
- In action prediction, dragging actions are far more difficult than clicking; models must understand three dimensions: start point, end point, and path direction.
- The dataset covers 83 applications, spanning five major categories: creative, development, office, system, and communication.

## Highlights & Insights

- It fills a major gap in desktop GUI agent evaluation. Released under the MIT License, it is fully open-source and ready for community use.
- The three-tiered task design is elegant, offering step-by-step diagnosis from "seeing" to "understanding" to "acting," which provides more diagnostic value than purely end-to-end evaluations.
- The extensive coverage of 83 applications reveals the "long tail of applications" problem—the capabilities demonstrated on common applications do not generalize to professional domains.
- The offline static design addresses the long-standing reproducibility issues of online benchmarks and reduces evaluation costs.

## Limitations & Future Work

- The dataset size is limited (~1,464 annotated samples), which is relatively small compared to Web-based benchmarks.
- It mainly targets a specific desktop OS, with a lack of cross-OS (Windows/macOS/Linux) generalization evaluation.
- Offline evaluation cannot capture temporal dependencies and state feedback loops in dynamic interactions.
- Only single-step prediction is evaluated, without measuring the end-to-end success rate of multi-step tasks.
- Annotations in the spatial reasoning subset may contain ambiguities (e.g., the exact definition of "next to" varies from person to person).

## Related Work & Insights

- **Relationship with ScreenSpot**: ScreenSpot focuses on mobile devices, while UI-Vision focuses on desktops, complementing each other in the GUI agent evaluation ecosystem.
- **Differences from OSWorld**: OSWorld is an online interactive benchmark, whereas UI-Vision is an offline static benchmark; the latter sacrifices interactive realism in exchange for high reproducibility.
- **Insight**: Incorporating more screenshots of professional desktop software in training data is critical to improving the generalization capabilities of GUI agents.

## Rating

| Dimension | Score | Reason |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | First offline desktop GUI benchmark with an elegant three-level task design. |
| Technical Depth | ⭐⭐⭐ | Benchmark construction work, with moderate technical depth. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Covers multiple SOTA models with detailed multi-dimensional error analyses. |
| Writing Quality | ⭐⭐⭐⭐ | Clear structure, well-defined problems, and transparent annotation procedures. |
| Practicality | ⭐⭐⭐⭐⭐ | MIT open-source and released on Hugging Face, ready for immediate community deployment. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](../../CVPR2026/object_detection/mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)
- [\[ICCV 2025\] VisRL: Intention-Driven Visual Perception via Reinforced Reasoning](../../ICCV2025/object_detection/visrl_intention-driven_visual_perception_via_reinforced_reasoning.md)
- [\[ICML 2025\] FG-CLIP: Fine-Grained Visual and Textual Alignment](fg-clip_fine-grained_visual_and_textual_alignment.md)
- [\[ICCV 2025\] Visual Modality Prompt for Adapting Vision-Language Object Detectors](../../ICCV2025/object_detection/visual_modality_prompt_for_adapting_vision-language_object_detectors.md)
- [\[ICML 2025\] Self-Organizing Visual Prototypes for Non-Parametric Representation Learning](self-organizing_visual_prototypes_for_non-parametric_representation_learning.md)

</div>

<!-- RELATED:END -->
