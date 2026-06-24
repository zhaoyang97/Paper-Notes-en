---
title: >-
  [Paper Note] Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation
description: >-
  [ACL 2025][Multimodal VLM][World Models] Proposes a cognitive science-inspired two-stage framework (Perception + Prediction) and constructs WM-ABench, a large-scale benchmark (23 dimensions, 6 simulators, over 100k instances). Through 660 sets of experiments, it systematically reveals severe deficiencies in the foundational world modeling capabilities of 15 state-of-the-art VLMs.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "World Models"
  - "VLM Evaluation"
  - "Perception-Prediction Framework"
  - "Intuitive Physics"
  - "Atomic Benchmarks"
date: 2026-05-08
content_hash: d450bd8b43e0db9d
---

# Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation

**Conference**: ACL 2025  
**arXiv**: [2506.21876](https://arxiv.org/abs/2506.21876)  
**Code**: [Project Homepage](https://wm-abench.maitrix.org/)  
**Area**: Multimodal VLMs  
**Keywords**: World Models, VLM Evaluation, Perception-Prediction Framework, Intuitive Physics, Atomic Benchmarks  

## TL;DR

Proposes a cognitive science-inspired two-stage framework (Perception + Prediction) and constructs WM-ABench, a large-scale benchmark (23 dimensions, 6 simulators, over 100k instances). Through 660 sets of experiments, it systematically reveals severe deficiencies in the foundational world modeling capabilities of 15 state-of-the-art VLMs.

## Background & Motivation

**Background**: World models (WMs) enable agents to understand world states and predict transitions, serving as the foundation for advanced deliberative reasoning. Recent VLMs (such as OpenAI o3, GPT-4o, Gemini) encapsulate rich knowledge of world dynamics and are regarded as potential candidates for general world models.

**Limitations of Prior Work**: Existing benchmarks evaluate VLMs across single dimensions (e.g., visual perception, intuitive physics), but **lack systematic atomic evaluation**. A robust world model must integrate multiple fundamental capabilities in both perception and prediction; isolated evaluations provide only localized perspectives.

**Key Challenge**: While VLMs perform exceptionally well in certain visual and quantitative reasoning tasks, they may exhibit fatal flaws in fundamental dimensions like space, time, and motion—for instance, almost all models perform close to random when distinguishing motion trajectories. More critically, they may lack disentangled world representations (e.g., spuriously assuming that blue objects move faster than green ones).

**Goal**: To establish a systematic evaluation framework from first principles to comprehensively diagnose the fundamental capabilities of VLMs as world models and their interactions.

**Key Insight**: Drawing inspiration from comparative psychology and cognitive science research on animal/human cognition, the functions of a world model are decomposed into two stages: perception and prediction, with multiple orthogonal sub-dimensions under each stage.

**Core Idea**: A cognitive science-inspired two-stage framework (5-dimensional perception + 3-dimensional prediction = 23 fine-grained dimensions) is designed to atomically diagnose the world modeling capabilities of VLMs through controlled counterfactual experiments in simulation environments.

## Method

### Overall Architecture

**Two-stage Conceptual Framework**:

**Stage 1: Perception** — Extract and organize key information from multi-sensory cues
- **Space**: position, extension, relations
- **Time**: position, extension, relations
- **Motion**: direction, speed, trajectory
- **Quantity**: discrete, continuous, relations
- **Visual**: color, shape, material

**Stage 2: Prediction** — Predict future states based on perceptual representations
- **Mechanistic Simulation**: Causal understanding of intuitive physical dynamics and intentional behaviors
- **Transitive Inference**: Multi-step prediction, linking intermediate states
- **Compositional Inference**: Combining known mechanisms to predict the outcomes of novel combinations

### Key Designs

**6 Simulation Environments**:
1. **CARLA** (Autonomous Driving)
2. **Habitat** (Indoor Navigation/Manipulation)
3. **ManiSkill** (Robot Manipulation)
4. **ThreeDWorld (TDW)** (3D Physical World)
5. **PHYRE/Physion** (Intuitive Physics)

**Controlled Counterfactual Generation**: Generating difficult negative samples in two ways:
- **Counterfactual Action**: Fixing the correct pre-state, perturbing the action $\rightarrow (S_t^*, a', S_{t+1}')$
- **Counterfactual Pre-state**: Fixing the correct action, perturbing the pre-state $\rightarrow (S_t', a^*, S_{t+1}'')$

The generated incorrect options are visually highly similar to the correct state, requiring models to possess true understanding of world dynamics to distinguish them.

**Human Baseline Validation**: Measuring human performance to ensure the fairness and solvency of the questions.

### Evaluation Metrics

All tasks are multiple-choice questions, using accuracy as the evaluation metric.

## Key Experimental Results

### Main Results in Perception Stage (Accuracy %)

**Spatial and Temporal Perception**:

| Model | Spatial Relation (Mani) | Spatial Relation (TDW) | Temporal Position (Mani) | Temporal Position (TDW) |
|------|---------------|---------------|---------------|---------------|
| OpenAI o3 | **88.0** | **88.0** | **60.0** | **72.0** |
| Gemini-2.5-pro | 97.0 | 90.0 | 54.0 | 65.0 |
| GPT-4.5 | 72.0 | 79.0 | 47.0 | 66.0 |
| QWen2.5-VL-72b | 57.8 | 70.6 | 48.4 | 52.9 |
| Random Baseline | 25.0 | 25.0 | 33.3 | 33.3 |
| **Human** | 90.0 | 100.0 | 80.0 | 82.0 |

**Motion Perception** (Weakest Dimension):

| Model | Speed (Mani) | Speed (TDW) | Direction (Mani) | Direction (TDW) | Trajectory (Mani) | Trajectory (TDW) |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| OpenAI o3 | 27.0 | 63.0 | 89.0 | 100.0 | 83.0 | 89.0 |
| Best Open-Source | 47.6 | 74.7 | 85.4 | 85.4 | 71.1 | 84.4 |
| Random | 33.3 | 33.3 | 25.0 | 25.0 | 25.0 | 25.0 |
| **Human** | 84.0 | 90.0 | 100.0 | 98.0 | 96.0 | 98.0 |

**Key Findings: Almost all models perform close to random in distinguishing motion trajectories** (especially speed in the Mani environment, which is only ~25-48%).

### Prediction Stage

| Model | Intuitive Physics | Navigation | Manipulation | Transitive Inference | Compositional Inference |
|------|---------|------|------|---------|---------|
| OpenAI o3 | 48.0-70.0 | 85.0-90.0 | 83.0-89.0 | 85.0-96.0 | 93.0-92.0 |
| Gemini-2.5-pro | 36.0-51.0 | 81.0-92.0 | 46.0-60.0 | 81.0-89.0 | 92.0-93.0 |
| Most Models | 24-36 | 57-84 | 25-52 | 31-81 | 49-77 |
| Random | 25.0 | 25.0 | 25.0 | 25.0-50.0 | 33.3 |
| **Human** | 76.0-100.0 | 98.0-100.0 | 96.0-98.0 | 98.0 | 98.0-100.0 |

### Disentanglement Analysis — Key Insights

**VLMs Lack Independent World Representations**:
- Some models tend to predict that **blue objects move faster than green objects**.
- Changing the color of an object leads to incorrect model perception of its size.
- This reveals that VLMs learn spurious cross-dimensional correlations rather than disentangled world representations.

### Key Findings

1. **Visual and quantitative perception are strong** (color recognition ~95-100%), but **spatial, temporal, and motion perception are severely deficient**.
2. **Weak knowledge of physical causality**: Most intuitive physics tasks perform close to random.
3. **Limited transitive and compositional reasoning**: Performance in multi-step prediction drops sharply.
4. **Closed-source models (especially o3) lead significantly**, but remain far below human level.
5. **Spurious correlations are prevalent**: Color affects speed judgment, size judgment, etc.

## Highlights & Insights

- **Cognitive Science-Driven Framework Design**: Systematically decomposes world model functions starting from comparative psychology and cognitive science, with dimensions designed to ensure orthogonality and assessment feasibility.
- **Controlled Counterfactual Experiments**: Precise control of variables using simulators yields causal conclusions.
- **Insightful Finding of "Lack of Disentanglement"**: VLMs do not learn independent physical attributes, but rather spurious correlations between attributes—this has profound implications for understanding the internal representations of VLMs.
- **Unprecedented Scale**: 23 fine-grained dimensions, 6 simulators, 660 sets of experiments, 15 VLMs.

## Limitations & Future Work

1. Only covers limited perceptual dimensions (excluding temperature, sound, proprioception, etc.), which might miss important aspects.
2. All tasks are based on simulated environments, leaving a domain gap with the real world.
3. The multiple-choice format might underestimate the open-ended reasoning capabilities of the models.
4. Subset evaluation (marked with * models only used 100 instances) has weaker statistical significance.
5. Future work can longitudinally track the evolution of world modeling capabilities across model version updates.

## Related Work & Insights

- **Physion / PHYRE**: Focuses on intuitive physics, while WM-ABench expands to a complete 23-dimension framework.
- **Video-Generative World Models** (e.g., Sora) directly generate future frames, whereas VLMs predict via language-based reasoning—the two approaches are complementary.
- Insights: (a) Incorporate physical simulation data into VLM pre-training; (b) Introduce explicit learning objectives for disentangled representations; (c) Establish fine-grained world modeling capabilities as a new checklist for VLM development.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first framework to systematically and atomically evaluate VLM world modeling capabilities from a cognitive science perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 15 models, 6 simulators, 23 dimensions, 660 sets of experiments, combined with human baselines.
- **Writing Quality**: ⭐⭐⭐⭐ — The framework is logically greenhouse-rigorous, though the paper is relatively long and information-dense.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a systematic diagnostic tool for VLM world modeling research; findings such as "lack of disentanglement" have a high impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Words or Vision: Do Vision-Language Models Have Blind Faith in Text?](../../CVPR2025/multimodal_vlm/words_or_vision_do_vision-language_models_have_blind_faith_in_text.md)
- [\[CVPR 2025\] Taxonomy-Aware Evaluation of Vision-Language Models](../../CVPR2025/multimodal_vlm/taxonomy-aware_evaluation_of_vision-language_models.md)
- [\[ACL 2025\] SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation](sphere_unveiling_spatial_blind_spots_in.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[NeurIPS 2025\] JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models](../../NeurIPS2025/multimodal_vlm/jailbound_jailbreaking_internal_safety_boundaries_of_vision-language_models.md)

</div>

<!-- RELATED:END -->
