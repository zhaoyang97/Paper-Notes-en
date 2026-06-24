---
title: >-
  [Paper Note] Evaluating Vision-Language Models as Evaluators in Path Planning
description: >-
  [CVPR 2025][Multimodal VLM][Vision-Language Models] This paper introduces the PathEval benchmark to systematically evaluate the capability of Vision-Language Models (VLMs) serving as path planning evaluators. It is discovered that although VLMs can abstract features of the optimal path from scene descriptions, their visual components suffer from a severe bottleneck in perceiving low-level details of the path. End-to-end fine-tuning cannot effectively address this issue…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "Path Planning"
  - "Benchmarking"
  - "Plan Evaluation"
  - "Visual Perception Bottleneck"
date: 2026-05-08
content_hash: adc78b5654ff1c52
---

# Evaluating Vision-Language Models as Evaluators in Path Planning

**Conference**: CVPR 2025  
**arXiv**: [2411.18711](https://arxiv.org/abs/2411.18711)  
**Code**: [https://github.com/MohamedAghzal/PathEval](https://github.com/MohamedAghzal/PathEval)  
**Area**: Multimodal VLM  
**Keywords**: Vision-Language Models, Path Planning, Benchmarking, Plan Evaluation, Visual Perception Bottleneck

## TL;DR

This paper introduces the PathEval benchmark to systematically evaluate the capability of Vision-Language Models (VLMs) serving as path planning evaluators. It is discovered that although VLMs can abstract features of the optimal path from scene descriptions, their visual components suffer from a severe bottleneck in perceiving low-level details of the path. End-to-end fine-tuning cannot effectively address this issue, necessitating task-specific discriminative visual encoder adaptation.

## Background & Motivation

**Background**: Large Language Models (LLMs) have demonstrated great potential in complex reasoning but show limited performance in end-to-end planning tasks. The academic community has begun to explore an intriguing question: if these models cannot plan well, can they function effectively as evaluators within planning frameworks?

**Limitations of Prior Work**: Prior studies mainly focused on the evaluation capabilities of LLMs in text-only planning, lacking a systematic investigation into Vision-Language Models (VLMs) in planning evaluation scenarios that require visual perception. Path planning evaluation requires both high-level semantic reasoning (understanding what constitutes a good path) and low-level visual perception (precisely perceiving the geometric properties of the path), which poses a unique challenge for VLMs.

**Key Challenge**: VLMs perform well in high-level semantic understanding but exhibit limited capability in low-level visual perception (e.g., judging path length, smoothness, and distance to obstacles). This imbalance between high-level reasoning and low-level perception is the critical obstacle preventing VLMs from becoming effective planning evaluators.

**Goal**: (1) Construct PathEval, a benchmark for systematically evaluating VLMs as path evaluators; (2) Isolate and quantify the performance of VLMs across three capability dimensions: scene abstraction, path perception, and information integration; (3) Explore methods to improve the path evaluation capabilities of VLMs.

**Key Insight**: The authors decompose the path evaluation task into three sub-capabilities—abstracting optimal path features from scene descriptions, precisely perceiving low-level path attributes, and integrating information to make decisions—and design separate experiments to diagnose the performance of VLMs in each stage.

**Core Idea**: Through the PathEval benchmark, this work systematically reveals the performance bottlenecks of VLMs as planning evaluators, pointing out that visual perception is the core limiting factor and discovering that task-specific discriminative visual encoder adaptation is required for resolution.

## Method

### Overall Architecture

The core task of the PathEval benchmark is: given a side-by-side visualization of two paths P1 and P2 along with a scene description S, the VLM needs to judge which path better satisfies the optimization criteria of the scene. The scene description defines a set of path descriptors (e.g., length, smoothness, distance to obstacles), and the model must comprehensively consider these descriptors to make a decision. The dataset includes both 2D and 3D path visualization methods to probe the sensitivity of VLMs to different modalities of presentation.

### Key Designs

1. **Scene and Path Descriptor Design**:

    - **Function**: Defines a rich and diverse set of path evaluation scenarios.
    - **Mechanism**: Each scene S is a high-level description aimed at optimizing a set of path descriptors $\{m_1, m_2, ..., m_k\}$, where each descriptor evaluates a specific attribute of the path (e.g., shortest length, smoothest, furthest from obstacles). The complexity of the scene is modulated by the number of descriptors that need to be considered simultaneously. Programmatic generation of path pairs and corresponding ground truth ensures objective evaluation.
    - **Design Motivation**: Real-world path planning tasks usually involve trade-offs among multiple objectives; programmatic generation guarantees large-scale, unbiased evaluation data.

2. **Three-Tier Capability Diagnostic Framework**:

    - **Function**: Systematically isolates different capability dimensions of VLMs.
    - **Mechanism**: Three types of experiments are designed: (1) **Scene Abstraction Test**: Given a scene description, can the VLM correctly identify which path attributes to focus on? (2) **Path Perception Test**: Given raw path visualizations, can the VLM accurately judge low-level path attributes (e.g., which one is shorter, which one is smoother)? (3) **Information Integration Test**: Given the known values of path attributes, can the VLM correctly integrate them to make the final judgement?
    - **Design Motivation**: Only by diagnosing capabilities in a layered manner can one precisely pinpoint where the bottlenecks of VLMs lie, rather than making a general claim that "VLMs perform poorly."

3. **Discriminative Visual Encoder Adaptation Scheme**:

    - **Function**: Explores methods to enhance the path perception capabilities of VLMs.
    - **Mechanism**: Quantifies and compares two strategies: (1) end-to-end fine-tuning of the entire VLM; (2) task-specific discriminative visual encoder adaptation, which involves training a specialized visual feature extractor for path attribute perception and then integrating it into the inference pipeline of the VLM. Experimental results demonstrate that end-to-end fine-tuning yields limited improvement, whereas discriminative adaptation is significantly more effective.
    - **Design Motivation**: The visual encoders of VLMs (e.g., ViT) are not optimized for path geometric attributes during pre-training, and simple end-to-end fine-tuning cannot effectively bridge this capability gap.

### Loss & Training

PathEval itself is an evaluation benchmark, primarily utilizing classification accuracy as the evaluation metric. In the discriminative adaptation experiments, binary cross-entropy loss is employed to perform task-specific fine-tuning on the visual encoder, using training data from the PathEval training set.

## Key Experimental Results

### Main Results

| Model | Scene Abstraction Accuracy | Path Perception Accuracy | Integration Accuracy | Overall Accuracy |
|------|-------------|-------------|-----------|-----------|
| Random Baseline | 50.0% | 50.0% | 50.0% | 50.0% |
| GPT-4V | ~85% | ~55% | ~65% | ~58% |
| Gemini Pro Vision | ~80% | ~52% | ~60% | ~55% |
| LLaVA-1.5 | ~75% | ~51% | ~55% | ~52% |
| Claude 3 Opus | ~82% | ~54% | ~62% | ~56% |

### Ablation Study

| Method | Path Perception Gain | Overall Gain | Description |
|------|-----------|---------|------|
| Zero-shot (GPT-4V) | — | 58% | Baseline |
| + End-to-end Fine-tuning (LoRA) | +3-5% | ~62% | Limited gain |
| + Discriminative Visual Adaptation | +10-15% | ~70% | Significant improvement |
| Perfect Visual Oracle + VLM | — | ~88% | Vision as the core bottleneck |

### Key Findings

- **Strong Scene Abstraction Capability**: All VLMs perform well (80%+) in understanding scene descriptions and abstracting features of the optimal path, indicating that high-level reasoning capability is not the bottleneck.
- **Visual Perception is the Core Bottleneck**: Path perception accuracy is generally only slightly better than random guessing (~55%), showing that models struggle to precisely judge low-level geometric attributes such as path length and smoothness.
- **2D vs 3D Presentation**: 3D path visualization further degrades model performance, indicating that viewpoint perspective transformations increase the difficulty of visual perception.
- **Limited Effect of End-to-End Fine-Tuning**: Simple LoRA fine-tuning fails to effectively resolve visual perception issues, necessitating specialized discriminative adaptation.

## Highlights & Insights

- **Value of the Hierarchical Diagnostic Methodology**: Decoupling the failure of VLMs into three independent dimensions—scene understanding, visual perception, and information integration—precisely pinpoints the bottleneck. This diagnostic methodology can be transferred to evaluate VLM capabilities in other tasks requiring precise visual perception (e.g., engineering drawing review, medical image analysis, etc.).
- **The Insight of "VLMs Can Reason but Cannot See" is Highly Significant**: While VLMs excel at high-level semantics, they are severely deficient in tasks requiring precise spatial/geometric perception. This conclusion provides critical guidance for the practical application of VLMs.
- The effectiveness of discriminative adaptation suggests an important direction: general-purpose visual encoders in VLMs may need to be paired with task-specific visual modules to function effectively in professional domains.

## Limitations & Future Work

- The path planning scenarios are relatively simple (2D grid environments), whereas real-world planning evaluation involves more complex constraints.
- Only a limited number of VLM models were tested; newer models (e.g., GPT-4o) might exhibit different performance.
- The impact of visual presentation styles for paths (color coding, line width, etc.) on the results has not been thoroughly explored.
- Future work can extend to more complex scenarios such as path evaluation in 3D environments and multi-agent path coordination evaluation.
- The generalization capability of the discriminative adaptation scheme (across different scene types) warrants further study.

## Related Work & Insights

- **vs. LLM-as-a-Judge**: This paradigm primarily evaluates text quality, whereas PathEval extends it to planning domains that require visual perception, revealing vision as the key bottleneck.
- **vs. VisProg/ViperGPT**: These approaches combine VLMs with tools/programs to handle visual reasoning tasks. The findings of PathEval support this "VLM + specialized vision tool" design philosophy.
- **vs. Traditional Path Planning Evaluation**: Traditional methods employ precise mathematical metrics (e.g., path length, collision detection). PathEval explores the feasibility of using learned models to approximate these evaluations.

## Rating

- Novelty: ⭐⭐⭐⭐ Conceptually novel research problem (VLMs as planning evaluators); the hierarchical diagnostic approach is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple models, detailed error attribution analysis, and thorough comparison of improvement strategies.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, and the core problem is well-articulated.
- Value: ⭐⭐⭐⭐ Uncovers significant capability deficiencies in VLMs, providing guiding significance for their application in specialized domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Vision-Language Models Do Not Understand Negation](vision-language_models_do_not_understand_negation.md)
- [\[CVPR 2026\] SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models](../../CVPR2026/multimodal_vlm/simpact_simulation-enabled_action_planning_using_vision-language_models.md)
- [\[CVPR 2025\] HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios](homesafe-bench_evaluating_vision-language_models_on_unsafe_action_detection_for_.md)
- [\[NeurIPS 2025\] Adapting Vision-Language Models for Evaluating World Models](../../NeurIPS2025/multimodal_vlm/adapting_visionlanguage_models_for_evaluating_world_models.md)
- [\[CVPR 2025\] Realistic Test-Time Adaptation of Vision-Language Models](realistic_test-time_adaptation_of_vision-language_models.md)

</div>

<!-- RELATED:END -->
