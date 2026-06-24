---
title: >-
  [Paper Note] Embodied Scene Understanding for Vision Language Models via MetaVQA
description: >-
  [CVPR 2025][Multimodal VLM][Embodied Scene Understanding] A large-scale VQA benchmark (4.3 million questions) based on Set-of-Mark annotations and scene graphs is constructed to systematically evaluate the spatial reasoning and embodied understanding capabilities of VLMs. It demonstrates that fine-tuning on MetaVQA significantly improves spatial reasoning (+28 points), and the capabilities learned from simulator data successfully transfer zero-shot to real-world scenarios and…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Embodied Scene Understanding"
  - "VQA Benchmark"
  - "Set-of-Mark"
  - "Simulation-to-Real Transfer"
  - "Closed-Loop Driving"
date: 2026-05-08
content_hash: a8245097d9afbad0
---

# Embodied Scene Understanding for Vision Language Models via MetaVQA

**Conference**: CVPR 2025  
**arXiv**: [2501.09167](https://arxiv.org/abs/2501.09167)  
**Code**: [https://metadriverse.github.io/metavqa](https://metadriverse.github.io/metavqa)  
**Area**: Multimodal VLMs  
**Keywords**: Embodied Scene Understanding, VQA Benchmark, Set-of-Mark, Simulation-to-Real Transfer, Closed-Loop Driving

## TL;DR
A large-scale VQA benchmark (4.3 million questions) based on Set-of-Mark annotations and scene graphs is constructed to systematically evaluate the spatial reasoning and embodied understanding capabilities of VLMs. It demonstrates that fine-tuning on MetaVQA significantly improves spatial reasoning (+28 points), and the capabilities learned from simulator data successfully transfer zero-shot to real-world scenarios and unseen closed-loop driving tasks.

## Background & Motivation

**Background**: While VLMs excel at general visual question answering, they remain weak at driving-related scene understanding, such as spatial reasoning (relative positions of objects, distance estimation) and embodied understanding (predicting action consequences, situational awareness). Existing driving VLM evaluations are mostly small-scale or task-specific, lacking a systematic benchmark.

**Limitations of Prior Work**: (1) There is a lack of large-scale, multi-dimensional benchmarks for evaluating embodied scene understanding. (2) Manually annotating VQA data for driving scenes is extremely costly and struggles to cover all question types. (3) VLMs underperform random guessing on tasks requiring precise spatial reasoning (e.g., "is the vehicle ahead on the left or right") (LLaVA-NeXT: 29.5% vs. random 32.9%). (4) Whether simulation training can transfer to real scenarios remains under-explored.

**Key Challenge**: VLMs lack spatial reasoning capabilities, but acquiring large-scale training data for spatial reasoning requires accurate 3D annotations, which are extremely expensive. Simulators can generate annotations at zero cost, but suffer from domain gaps.

**Goal**: To build an automated QA generation pipeline and a systematic evaluation benchmark to evaluate and improve the embodied scene understanding capabilities of VLMs, while validating the feasibility of simulation-to-real transfer.

**Key Insight**: Automatically construct scene graphs using 3D annotations from nuScenes and Waymo, programmatically generate multiple-choice VQA based on these scene graphs, and pair them with Set-of-Mark (SoM) visual annotations to help VLMs accurately locate objects. Rebuild real scenarios in the MetaDrive simulator to generate simulated annotations, enabling joint simulation-real training.

**Core Idea**: Programmatically generate 4.3 million spatial and embodied VQA data points using scene graphs and SoM annotations, and validate that simulation training can transfer zero-shot to the real world and closed-loop driving.

## Method

### Overall Architecture
A three-step construction pipeline: (1) Scene aggregation—collecting real traffic data from Waymo/nuScenes and reconstructing simulated scenarios using MetaDrive; (2) SoM annotation—adding 2D bounding box labels to objects (using 3D-to-2D projection for real images, and shader-level instance segmentation for simulated images); (3) QA generation—randomly selecting target nodes from scene graphs, generating multiple-choice questions using templates, querying correct answers from the scene graph, and appending explanation fields to deepen understanding.

### Key Designs

1. **Programmatic QA Generation (30 question types)**:

    - **Function**: Automatically generate large-scale, multi-dimensional VQA data from scene graphs.
    - **Mechanism**: Define 30 question templates categorized into three classes: spatial (relative position, direction, and distance between objects), embodied (action consequences, situational awareness), and localization (association between SoM-labeled objects and textual descriptions). Fill templates using node and edge attributes from scene graphs, and query correct answers directly from the graph. Each QA is appended with an "explanation" field to train the model to not only select the correct answer but also explain why.
    - **Design Motivation**: Programmatic generation avoids the high cost and inconsistency of human annotation, and scene graphs ensure precise answers—making them far more reliable than LLM-generated QAs.

2. **Set-of-Mark (SoM) Visual Annotation**:

    - **Function**: Enable VLMs to accurately ground visual objects to textual references.
    - **Mechanism**: Draw 2D bounding boxes and add index labels for each target object. Real-world images use 3D bounding boxes projected onto the camera plane to obtain 2D boxes, while simulated images use shader-level instance segmentation directly. Zero-shot SoM localization accuracy averages 69.6% across mainstream VLMs (Qwen2 reaching up to 87.4%), which is close to human performance of 88%.
    - **Design Motivation**: Natural language descriptions of object locations are prone to ambiguity (e.g., there may be multiple "cars on the left"), whereas SoM provides unambiguous visual-to-textual grounding.

3. **Simulation-to-Real Transfer Validation**:

    - **Function**: Validate whether simulation training can improve real-world performance.
    - **Mechanism**: Reconstruct digital twins of nuScenes scenarios in MetaDrive, render images in simulation, and generate SoM annotations and QAs. The training set mixes three types of data: 50K Waymo + 50K nuScenes Real + 50K nuScenes Sim. Ablation studies independently evaluate the contribution of each data type.
    - **Design Motivation**: Simulation data can be generated infinitely with zero annotation cost. Experiments demonstrate that training purely on simulation data increases real-world accuracy from 63.2% to 81.9% (+18.7 points), and the sim+real mix reaches 88.4%.

### Loss & Training
Standard VLM instruction tuning (LoRA or full-parameter fine-tuning). The training set consists of 150K questions (50K for each of the three data types), and the test set contains 9,725 questions. Closed-loop driving evaluation is conducted in MetaDrive, where the VLM receives SoM-annotated images every 0.5 seconds and selects steer + throttle actions from a discrete action set.

## Key Experimental Results

### Main Results

| Model | Zero-shot | Fine-tuned | Gain |
|------|-----------|------------|------|
| LLaVA-NeXT | 0.295 | - | Below random (0.329) |
| GPT-4o | 0.628 | - | Best closed-source |
| Qwen2 | 0.539 | **0.844** | +0.305 |
| InternVL2-8B | 0.592 | **0.869** | +0.277 |
| Llama3.2 | 0.500 | **0.774** | +0.274 |

### Ablation Study

| Training Data | Overall | Sim | Real | Note |
|---------|------|------|------|------|
| Zero-shot | 0.592 | 0.552 | 0.632 | Baseline |
| Sim Only | 0.807 | 0.795 | **0.819** | Sim-to-real transfer is effective |
| Real Only | 0.825 | 0.792 | 0.858 | Real data is slightly better |
| Sim + Real | **0.869** | **0.853** | **0.884** | Mixed is optimal |
| 9,375 samples | 0.794 | 0.764 | 0.824 | Positively correlated with data volume |
| 150,000 samples | **0.869** | **0.853** | **0.884** | Large data yields continuous improvement |

### Key Findings
- **LLaVA-NeXT performs below random guessing**: A 27.5% parsing failure rate + frequent refusal to answer indicate that certain VLMs fail completely with SoM annotations.
- **Simulation-to-real transfer is incredibly effective**: Training solely on simulator data improves real-world accuracy from 63.2% to 81.9%, which is close to pure real-world training (85.8%).
- **VQA fine-tuning transfers to closed-loop driving**: VLMs fine-tuned on VQA tasks perform better even in unseen closed-loop driving tasks—Llama3.2's collision rate drops from 48.3% to 26.7%, proving the generalization capability of the learned embodied understanding.
- **Spatial reasoning shows the most significant improvement**: Accuracy on spatial questions increases the most after fine-tuning, demonstrating that VLMs have substantial untapped potential in spatial reasoning.

## Highlights & Insights
- **The fully automated QA generation pipeline** scales infinitely—massive volumes of VQA data can be generated at zero cost given 3D-annotated driving data, breaking the bottleneck of manual annotation.
- **The transfer from VQA to closed-loop driving** is a key finding: Spatial and embodied knowledge learned solely from Q&A tasks can improve autonomous driving behaviors, suggesting that VQA can serve as an efficient training proxy for embodied capability.
- **The effectiveness of SoM annotations** is validated through large-scale evaluation (69.6% zero-shot accuracy), providing a practical solution for object grounding in VLMs.

## Limitations & Future Work
- Only single-frame observations are used, lacking temporal context—video inputs could significantly improve accuracy on situational awareness questions.
- Only a fixed single-perspective camera is supported; multi-perspective (surround-view) and BEV inputs remain unexplored.
- Closed-loop evaluation uses a discrete action space (limited steer + throttle combinations), which has a gap with practical continuous control.
- The rendering quality of the MetaDrive simulator limits the upper bound of simulation-to-real transfer.

## Related Work & Insights
- **vs. DriveLM / DriveVLM**: These are domain-specific driving VLMs but lack systematic benchmarks and simulation-to-real transfer validation. MetaVQA provides a more comprehensive evaluation framework.
- **vs. NuScenes-QA**: NuScenes-QA has fewer question types and contains no simulated data. MetaVQA offers 30 question types, 4.3 million QAs, and joint simulation-real training.
- **vs. Set-of-Mark (SoM)**: SoM was originally designed for object grounding in general VLMs. MetaVQA is the first to systematically apply it to evaluating spatial reasoning in driving scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of programmatic QA generation and simulation-to-real verification is novel, and the discovery of VQA-to-closed-loop transfer is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive, covering 7 VLMs, open-loop + closed-loop evaluations, simulation/real ablations, and data volume ablations.
- Writing Quality: ⭐⭐⭐⭐ The benchmark design motivation is clear, and the evaluation dimensions are well-organized.
- Value: ⭐⭐⭐⭐⭐ The benchmark and dataset provide significant infrastructural value for the autonomous driving VLM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios](homesafe-bench_evaluating_vision-language_models_on_unsafe_action_detection_for_.md)
- [\[CVPR 2025\] From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons](from_multimodal_llms_to_generalist_embodied_agents_methods_and_lessons.md)
- [\[CVPR 2025\] Towards Understanding How Knowledge Evolves in Large Vision-Language Models](towards_understanding_how_knowledge_evolves_in_large_vision-language_models.md)
- [\[CVPR 2026\] Scene-VLM: Multimodal Video Scene Segmentation via Vision-Language Models](../../CVPR2026/multimodal_vlm/scene-vlm_multimodal_video_scene_segmentation_via_vision-language_models.md)
- [\[CVPR 2025\] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics](robospatial_teaching_spatial_understanding_to_2d_and_3d_vision-language_models_f.md)

</div>

<!-- RELATED:END -->
