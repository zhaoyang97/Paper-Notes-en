---
title: >-
  [Paper Note] OSCBench: Benchmarking Object State Change in Text-to-Video Generation
description: >-
  [ACL 2026][Video Generation][Text-to-Video] This paper proposes OSCBench, the first benchmark specifically designed to evaluate the Object State Change (OSC) capabilities of Text-to-Video (T2V) models. Based on cooking s…
tags:
  - "ACL 2026"
  - "Video Generation"
  - "Text-to-Video"
  - "Object State Change"
  - "Evaluation Benchmark"
  - "Cooking Scenarios"
  - "Multimodal Evaluation"
date: 2026-05-08
content_hash: 2fbfe533d355345e
---

# OSCBench: Benchmarking Object State Change in Text-to-Video Generation

**Conference**: ACL 2026  
**arXiv**: [2603.11698](https://arxiv.org/abs/2603.11698)  
**Code**: [Project Page](https://hanxjing.github.io/OSCBench)  
**Area**: Video Generation  
**Keywords**: Text-to-Video, Object State Change, Evaluation Benchmark, Cooking Scenarios, Multimodal Evaluation

## TL;DR
This paper proposes OSCBench, the first benchmark specifically designed to evaluate the Object State Change (OSC) capabilities of Text-to-Video (T2V) models. Based on cooking scenarios, it constructs 1,120 prompts across regular, novel, and compositional categories, revealing that even the strongest T2V models achieve only 0.786 in OSC accuracy.

## Background & Motivation

**Background**: T2V models have made significant progress in visual quality and temporal consistency. Existing benchmarks primarily evaluate perceptual quality, text-video alignment, or physical plausibility.

**Limitations of Prior Work**: Existing benchmarks neglect a critical dimension of action understanding—the Object State Changes (e.g., peeling a potato, slicing a lemon) explicitly specified by text prompts. T2V models may align well at high-level semantics but generate incorrect, incomplete, or inconsistent object state changes.

**Key Challenge**: High-quality visual appearance masks defects in modeling action consequences; the video appears realistic, but the object does not undergo the correct state change.

**Goal**: Construct a systematic OSC evaluation benchmark to diagnose specific deficiencies in T2V models regarding state change modeling.

**Key Insight**: Choosing cooking scenarios as the evaluation domain (where state changes are frequent, diverse, and well-defined) and designing three categories—regular, novel, and compositional—to test different levels of capability.

**Core Idea**: Divide OSC evaluation into two sub-dimensions: state change accuracy and state change consistency, supported by CoT-guided MLLM automatic evaluation.

## Method

### Overall Architecture
Starting from the HowToChange dataset, OSCBench utilizes a human-machine collaborative abstraction process to organize 20 actions and 134 objects into 9 action categories and 8 object categories (28 sub-categories). It constructs three types of OSC scenarios (108 Regular, 20 Novel, 12 Compositional), with 8 action-object combinations per scenario, totaling 1,120 prompts. Evaluation covers four dimensions: semantic following, OSC performance, scene alignment, and perceptual quality.

### Key Designs

1.  **Three Categories of OSC Scenarios**:

    - **Function**: Probing the model's OSC capabilities across different dimensions.
    - **Mechanism**: Regular scenarios cover common action-object combinations (e.g., slicing a lemon) to test fundamental capabilities; Novel scenarios use uncommon but physically plausible combinations (e.g., mashing a grapefruit) to test generalization; Compositional scenarios involve sequential multiple actions (e.g., peeling then slicing) to test temporal consistency.
    - **Design Motivation**: Distinguishing memorization from reasoning—regular scenarios can be solved by memorization, while novel scenarios require inferring state changes from action semantics.

2.  **CoT-guided MLLM Evaluation**:

    - **Function**: Automated, scalable, and fine-grained OSC evaluation.
    - **Mechanism**: Instead of using the MLLM as a black-box scorer, a chain-of-thought (CoT) strategy guides the MLLM through a reasoning process: standard grounding $\rightarrow$ evidence extraction $\rightarrow$ score justification, providing more reliable state change judgments.
    - **Design Motivation**: OSC evaluation requires multi-step reasoning (judging if the object reached the correct target state and if the transformation was smooth), which simple scoring cannot handle.

3.  **Multi-dimensional Evaluation System**:

    - **Function**: Comprehensive diagnosis of various T2V model capabilities.
    - **Mechanism**: Includes Semantic Following (subject/object/action alignment), OSC Performance (accuracy + consistency), Scene Alignment, and Perceptual Quality (realism + aesthetics). Each item uses a 1-5 Likert scale, with human evaluations based on the mean of three raters.
    - **Design Motivation**: OSC failure may stem from various stages, requiring dimension-wise diagnosis.

### Loss & Training
This is an evaluation study and does not involve model training.

## Key Experimental Results

### Main Results (Human Evaluation, Normalized 0-1)

| Model | Subject Alignment | Object Alignment | Action Alignment | OSC Accuracy | OSC Consistency | Realism |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Veo-3.1-Fast | 0.936 | 0.916 | **0.908** | **0.786** | **0.748** | Highest |
| Kling-2.5-Turbo | **0.938** | 0.900 | 0.826 | 0.726 | 0.726 | 0.732 |
| Wan-2.2 | 0.904 | 0.842 | 0.616 | 0.560 | 0.668 | 0.702 |
| HunyuanVideo-1.5 | 0.914 | 0.902 | 0.656 | 0.524 | 0.608 | 0.618 |
| Open-Sora-2.0 | 0.860 | 0.734 | 0.518 | 0.380 | 0.428 | 0.416 |

### Key Findings
- All models perform well in subject/object alignment (>0.73), but OSC accuracy and consistency are significantly lower.
- The strongest model, Veo-3.1, achieves an OSC accuracy of only 0.786, indicating that state change modeling is a key bottleneck for T2V.
- Performance in Novel and Compositional scenarios is worse than in Regular scenarios, revealing a lack of generalization capability.
- Closed-source models (Veo/Kling) significantly outperform open-source models, with the gap being particularly pronounced in the OSC dimension.

## Highlights & Insights
- The OSC perspective fills a critical gap in T2V evaluation—action is not just motion; it should produce the correct object state transformations.
- The design of the three scenario types cleverly distinguishes between memorization and reasoning capabilities.
- CoT-guided MLLM evaluation correlates highly with human assessment, providing a feasible path for large-scale automated OSC evaluation.

## Limitations & Future Work
- Focuses only on the cooking domain; OSC evaluation in other fields (crafting, chemical experiments) remains to be expanded.
- Currently evaluates single actions or two-step compositions; longer sequences of composite actions present greater challenges.
- While correlated with humans, MLLM evaluation is not a perfect substitute and may misjudge in cases of extreme failure.

## Related Work & Insights
- **vs VBench**: VBench focuses on overall video quality and lacks specific evaluation of object state changes.
- **vs PhyWorldBench**: Focuses on physical plausibility (gravity, collisions), whereas OSCBench focuses on action consequence modeling.
- **vs T2V-CompBench**: Evaluates compositional generation capabilities but does not address the accuracy and temporal consistency of state changes.

## Rating
- Novelty: ⭐⭐⭐⭐ First dedicated OSC benchmark, fills an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 models, dual human and automatic evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, comprehensive evaluation framework.
- Value: ⭐⭐⭐⭐ Points toward a critical improvement direction for T2V research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LocoT2V-Bench: Benchmarking Long-form and Complex Text-to-Video Generation](../../ICML2026/video_generation/locot2v-bench_benchmarking_long-form_and_complex_text-to-video_generation.md)
- [\[ACL 2026\] Self-Correcting Text-to-Video Generation with Misalignment Detection and Localized Refinement](self-correcting_text-to-video_generation_with_misalignment_detection_and_localiz.md)
- [\[CVPR 2026\] SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation](../../CVPR2026/video_generation/symphomotion_joint_control_of_camera_motion_and_object_dynamics_for_coherent_vid.md)
- [\[ICML 2026\] T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation](../../ICML2026/video_generation/t2av-compass_towards_unified_evaluation_for_text-to-audio-video_generation.md)
- [\[ICML 2026\] VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation](../../ICML2026/video_generation/vanim_rendering-aware_sparse_state_modeling_for_structure-preserving_vector_anim.md)

</div>

<!-- RELATED:END -->
