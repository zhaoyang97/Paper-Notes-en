---
title: >-
  [Paper Note] HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][hand spatial reasoning] This paper introduces HandVQA — a large-scale diagnostic benchmark containing over 1.6 million multiple-choice questions…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "hand spatial reasoning"
  - "VQA benchmark"
  - "vision-language models"
  - "fine-grained understanding"
  - "zero-shot transfer"
date: 2026-05-08
content_hash: 98d8325bac262b87
---

# HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.26362](https://arxiv.org/abs/2603.26362)  
**Code**: [https://kcsayem.github.io/handvqa/](https://kcsayem.github.io/handvqa/)  
**Area**: Multimodal VLM
**Keywords**: hand spatial reasoning, VQA benchmark, vision-language models, fine-grained understanding, zero-shot transfer

## TL;DR
This paper introduces HandVQA — a large-scale diagnostic benchmark containing over 1.6 million multiple-choice questions, automatically generated from 3D hand joint annotations. The benchmark covers joint angles, distances, and relative positions, and systematically exposes severe deficiencies of current VLMs in fine-grained hand spatial reasoning. The paper further demonstrates that models fine-tuned on HandVQA can zero-shot transfer to downstream tasks such as gesture recognition (+10.33%) and hand-object interaction recognition (+2.63%).

## Background & Motivation

**Background**: Vision-language models (VLMs) have approached human-level performance on general VQA tasks (e.g., VQAv2), yet perform poorly on fine-grained spatial reasoning. Prior work has shown that VLMs achieve only ~56% accuracy on simple left/right discrimination (vs. 99% for humans), reflecting surface-level correlation rather than genuine geometric understanding.

**Limitations of Prior Work**: Hands are the primary medium through which humans convey actions, intentions, and control. Precise understanding of hand pose is critical in high-stakes scenarios such as robotic surgery, chip manufacturing, and AR/VR interaction. However, existing VLMs lack understanding of joint-level spatial relationships within hands (complex spatial configurations of 21 joints), and frequently exhibit "pose hallucinations" — misidentifying joint bending states or incorrectly estimating inter-finger distances.

**Key Challenge**: General VQA benchmarks cannot diagnose specific weaknesses of VLMs in fine-grained spatial reasoning. Existing spatial reasoning benchmarks (e.g., CLEVR, SPHERE) focus on inter-object relationships and do not evaluate part-level spatial structure within a single object (e.g., kinematic and geometric relationships among hand joints).

**Goal**: 1) How to systematically evaluate VLMs' understanding of joint-level spatial relationships in hands? 2) What are the specific failure modes of VLMs? 3) Can capabilities acquired through hand spatial reasoning training transfer to other tasks?

**Key Insight**: Leveraging precise 3D joint annotations from existing high-quality hand datasets (FreiHAND, InterHand2.6M, FPHA) to automatically generate diagnostic VQA questions, decomposing hand pose estimation into five independently evaluable subtasks.

**Core Idea**: Systematically converting 3D hand joint coordinates into structured natural-language multiple-choice questions to enable precise diagnosis and effective improvement of VLM hand spatial reasoning capabilities.

## Method

### Overall Architecture
The HandVQA automatic VQA generation pipeline consists of three deterministic stages: 1) Pose descriptor extraction $\mathcal{F}_{\text{pose}}$: computing continuous geometric quantities (angles, distances, relative positions) from normalized 3D joint coordinates and discretizing them into semantic category labels; 2) Sentence generation $\mathcal{F}_{\text{text}}$: filling category labels into natural-language sentences using deterministic templates, and selecting correct answers and distractors; 3) Multiple-choice question construction $\mathcal{F}_{\text{mcq}}$: pairing images with answer options to produce standardized multiple-choice questions. Up to 25 questions are generated per image (5 descriptor types × 5 sampled instances), yielding over 1.6 million questions in total.

### Key Designs

1. **Five-Dimensional Pose Descriptor System**:

    - Function: Discretizes continuous 3D hand geometry into linguistically expressible categories.
    - Mechanism: Three geometric measures are defined — angle $\theta_j$ (the angle formed by three consecutive joints of the same finger, classified into 4 categories: fully bent / bent / slightly bent / straight), distance $d_{(i,k)}$ (Euclidean distance between two joints, classified into 3 categories: close / spread / far apart), and relative position $\Delta_a(i,k)$ (signed offset along the X/Y/Z axis, classified into left–right / up–down / front–back). Category thresholds are fixed (e.g., angle $<105°$ is fully bent, $\geq 170°$ is straight), with ambiguous "aligned" cases excluded.
    - Design Motivation: Discretization ensures each descriptor has a unique semantic interpretation, eliminating ambiguity in continuous-value evaluation; the five-dimensional decomposition enables independent diagnosis of VLM capabilities across different spatial dimensions.

2. **Deterministic Template Sentence Generation and Distractor Selection**:

    - Function: Converts geometric category labels into natural-language multiple-choice questions.
    - Mechanism: Fixed syntactic templates are used for each descriptor type; for example, the distance template reads: "The {joint A} joint of the {finger A} is {category} the {joint B} joint of the {finger B}." For each joint or joint pair, the sentence corresponding to the ground-truth category serves as the correct answer, while sentences for the remaining categories serve as distractors, naturally forming a multiple-choice question.
    - Design Motivation: Templating ensures objectivity and reproducibility of evaluation; distractors are drawn directly from other categories of the same joint pair, guaranteeing discriminability and validity.

3. **Multi-Dataset Cross-Scenario Coverage**:

    - Function: Ensures comprehensiveness and generalizability of evaluation.
    - Mechanism: Three complementary hand datasets are used — FreiHAND (third-person view, single hand), InterHand2.6M (multi-view, two-hand), and FPHA (first-person/egocentric view) — covering diverse viewpoints and interaction modes. Each dataset is trained and evaluated independently, revealing performance differences across settings.
    - Design Motivation: The egocentric viewpoint of FPHA exposes VLM viewpoint bias (base model performance drops significantly on this dataset); the two-hand setting of InterHand2.6M increases the complexity of spatial reasoning.

### Loss & Training
VLMs are parameter-efficiently fine-tuned using LoRA (with the visual encoder frozen) on the HandVQA training set. Evaluation metrics include accuracy and MAE (for angle/distance tasks).

## Key Experimental Results

### Main Results

| Model | Fine-tuned | Angle Acc↑ | Angle MAE↓ | Distance Acc↑ | Distance MAE↓ |
|------|------|-----------|-----------|--------------|--------------|
| DeepSeek 7B (Base) | ✗ | 34.10 | 0.883 | 45.55 | 0.657 |
| LLaVA 7B (Base) | ✗ | 40.08 | 0.739 | 16.20 | 1.293 |
| Qwen 7B (Base) | ✗ | 37.92 | 0.779 | 19.58 | 1.247 |
| LLaVA 7B (Finetuned) | InterHand | **74.35** | **0.263** | **90.79** | **0.094** |
| DeepSeek 7B (Finetuned) | InterHand | 68.00 | 0.334 | 88.02 | 0.122 |

### Zero-Shot Transfer Results

| Model | Gesture Recognition↑ | Hand-Object Interaction↑ |
|------|----------|-----------|
| LLaVA 7B (Base) | 57.42% | - |
| LLaVA 7B (Finetuned) | **69.58%** (+12.16) | - |
| Qwen 7B (Base) | 71.86% | 80.26% |
| Qwen 7B (Finetuned) | **82.19%** (+10.33) | **82.89%** (+2.63) |

### Key Findings
- Base VLMs fail severely on distance judgment: LLaVA and Qwen achieve accuracies below random chance (33.3%), with Qwen incorrectly answering "close" 93% of the time when the correct answer is "spread."
- Angle tasks remain difficult even after fine-tuning: the highest accuracy is only 74.35% (vs. 90.79% for distance), suggesting that freezing the visual encoder may be a bottleneck.
- The egocentric viewpoint (FPHA) is particularly challenging for base models, indicating viewpoint bias in VLMs.
- Strength on individual tasks does not generalize across tasks: no single base model leads on all spatial dimensions.

## Highlights & Insights
- The pipeline design that converts 3D joint coordinates into multiple-choice VQA questions is highly elegant — fully automated, deterministic, and unambiguous — enabling large-scale diagnostic data generation at low cost. This paradigm can be extended to body pose, object 6DoF pose, and other domains.
- Zero-shot transfer experiments demonstrate that "3D spatial reasoning is a transferable skill" — joint-level spatial reasoning capabilities learned on HandVQA directly improve gesture recognition and video interaction recognition without task-specific training.
- The paper identifies a "pose hallucination" phenomenon in VLMs: models tend to respond to spatial reasoning questions with simplified answers (e.g., always answering "close" or "slightly bent"), which is distinct from object-level hallucination.

## Limitations & Future Work
- Only 7B models are evaluated; larger models may exhibit different behavior.
- LoRA fine-tuning freezes the visual encoder, potentially limiting the learning of fine-grained features such as joint angles.
- Discretization thresholds are fixed and may not fully reflect the continuity of human perception.
- The benchmark covers only static images and does not extend to dynamic hand reasoning in video.
- Templated language limits question diversity; more naturalistic formulations could be explored in future work.

## Related Work & Insights
- **vs. SPHERE**: SPHERE evaluates inter-object spatial relationships, whereas HandVQA focuses on part-level structure within a single object, making it more fine-grained and more challenging.
- **vs. SpatialVLM**: SpatialVLM injects spatial information via depth maps; HandVQA trains models to autonomously learn spatial reasoning capabilities through VQA.
- The automatic generation pipeline of HandVQA can inspire diagnostic benchmark construction in other domains.

## Rating
- Novelty: ⭐⭐⭐⭐ First large-scale benchmark to systematically evaluate VLM hand spatial reasoning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, three models, five subtasks, and zero-shot transfer — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and in-depth analysis.
- Value: ⭐⭐⭐⭐ Significant reference value for understanding and improving VLM spatial reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](cropvlm_learning_to_zoom_for_fine_grained_vision_language_perception.md)
- [\[CVPR 2026\] It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models](its_time_to_get_it_right_improving_analog_clock_reading_and_clock-hand_spatial_r.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)

</div>

<!-- RELATED:END -->
