---
title: >-
  [Paper Note] OSCBench: Benchmarking Object State Change in Text-to-Video Generation
description: >-
  [ACL 2026][Video Generation][Text-to-Video] This paper proposes OSCBench — the first benchmark dedicated to evaluating object state change (OSC) capabilities in text-to-video (T2V) models. Built on cooking scenarios with…
tags:
  - "ACL 2026"
  - "Video Generation"
  - "Text-to-Video"
  - "Object State Change"
  - "Evaluation Benchmark"
  - "Cooking Scenarios"
  - "Multimodal Evaluation"
date: 2025-04-17
content_hash: 84568bb045a1a874
---

# OSCBench: Benchmarking Object State Change in Text-to-Video Generation

**Conference**: ACL 2026  
**arXiv**: [2603.11698](https://arxiv.org/abs/2603.11698)  
**Code**: [Project Page](https://hanxjing.github.io/OSCBench)  
**Area**: Video Generation  
**Keywords**: Text-to-Video, Object State Change, Evaluation Benchmark, Cooking Scenarios, Multimodal Evaluation

## TL;DR
This paper proposes OSCBench — the first benchmark dedicated to evaluating object state change (OSC) capabilities in text-to-video (T2V) models. Built on cooking scenarios with 1,120 prompts covering conventional/novel/compositional scenarios, it reveals that even the strongest T2V model achieves only 0.786 OSC accuracy.

## Background & Motivation

**Background**: T2V models have made significant progress in visual quality and temporal consistency. Existing benchmarks primarily evaluate perceptual quality, text-video alignment, or physical plausibility.

**Limitations of Prior Work**: Existing benchmarks overlook a critical dimension of action understanding — object state changes explicitly specified by text prompts (e.g., peeling potatoes, slicing lemons). T2V models may align well at the high-level semantic level but generate incorrect, incomplete, or inconsistent object state changes.

**Key Challenge**: High-quality visual appearance masks deficiencies in action consequence modeling — videos look realistic but objects do not correctly change state.

**Goal**: Construct a systematic OSC evaluation benchmark to diagnose specific deficiencies in T2V models' state change modeling.

**Key Insight**: Select cooking scenarios as the evaluation domain (frequent, diverse, well-defined state changes), and design conventional/novel/compositional scenarios to test different capability levels.

**Core Idea**: Decompose OSC evaluation into state change accuracy and state change consistency sub-dimensions, paired with CoT-guided MLLM automated evaluation.

## Method

### Overall Architecture
OSCBench starts from the HowToChange dataset, abstracting 20 actions and 134 objects into 9 action categories and 8 object categories (28 subcategories) through human-machine collaboration, constructing three types of OSC scenarios (conventional 108, novel 20, compositional 12), each with 8 action-object combinations, totaling 1,120 prompts. Evaluation covers four dimensions: semantic adherence, OSC performance, scene alignment, and perceptual quality.

### Key Designs

1. **Three Types of OSC Scenario Design**:

    - Function: Probe model OSC capabilities from different dimensions
    - Mechanism: Conventional scenarios cover common action-object combinations (e.g., slicing lemons) testing basic capabilities; novel scenarios use uncommon but physically plausible combinations (e.g., mashing grapefruit) testing generalization; compositional scenarios involve consecutive multi-action sequences (e.g., first peel then slice) testing temporal consistency
    - Design Motivation: Distinguish memorization from reasoning — conventional scenarios can be solved through memorization, while novel scenarios require inferring state changes from action semantics

2. **CoT-Guided MLLM Evaluation**:

    - Function: Automated, scalable fine-grained OSC evaluation
    - Mechanism: Rather than using MLLMs as black-box scorers, a CoT strategy guides MLLMs through standard grounding → evidence extraction → score argumentation reasoning processes, providing more reliable state change judgments
    - Design Motivation: OSC evaluation requires multi-step reasoning (determining whether objects reach the correct target state and whether the change process is smooth); simple scoring is insufficient

3. **Multi-Dimensional Evaluation System**:

    - Function: Comprehensively diagnose various T2V model capabilities
    - Mechanism: Semantic adherence (subject/object/action three-way alignment), OSC performance (state change accuracy + consistency), scene alignment, and perceptual quality (realism + aesthetics). Each item uses a 1-5 Likert scale; human evaluation takes the average of three annotators
    - Design Motivation: OSC failure may originate from multiple stages, requiring dimension-by-dimension diagnosis

### Loss & Training
This is an evaluation work; no model training is involved.

## Key Experimental Results

### Main Results (Human Evaluation, Normalized 0-1)

| Model | Subject Align. | Object Align. | Action Align. | OSC Accuracy | OSC Consistency | Realism |
|------|---------|---------|---------|----------|----------|--------|
| Veo-3.1-Fast | 0.936 | 0.916 | **0.908** | **0.786** | **0.748** | Highest |
| Kling-2.5-Turbo | **0.938** | 0.900 | 0.826 | 0.726 | 0.726 | 0.732 |
| Wan-2.2 | 0.904 | 0.842 | 0.616 | 0.560 | 0.668 | 0.702 |
| HunyuanVideo-1.5 | 0.914 | 0.902 | 0.656 | 0.524 | 0.608 | 0.618 |
| Open-Sora-2.0 | 0.860 | 0.734 | 0.518 | 0.380 | 0.428 | 0.416 |

### Key Findings
- All models perform well on subject/object alignment (>0.73) but show significantly lower OSC accuracy and consistency
- The strongest model Veo-3.1 achieves only 0.786 OSC accuracy, indicating state change modeling is a key bottleneck for T2V
- Novel and compositional scenarios perform worse than conventional scenarios, revealing insufficient generalization capability
- Closed-source models (Veo/Kling) significantly outperform open-source models, with the gap being especially pronounced in the OSC dimension

## Highlights & Insights
- The OSC perspective fills an important gap in T2V evaluation — actions should not only involve motion but should also produce correct object state changes
- The three-scenario design cleverly distinguishes memorization from reasoning capabilities
- CoT-guided MLLM evaluation shows high correlation with human evaluation, providing a feasible path for large-scale automated OSC evaluation

## Limitations & Future Work
- Focuses solely on the cooking domain; OSC evaluation in other domains (crafts, chemical experiments) awaits expansion
- Currently evaluates only single actions or two-step compositions; longer-sequence compositional actions pose greater challenges
- MLLM evaluation, while correlated with human judgment, is not a perfect substitute and may misjudge extreme failure cases

## Related Work & Insights
- **vs VBench**: Focuses on overall video quality, lacking dedicated evaluation of object state changes
- **vs PhyWorldBench**: Focuses on physical plausibility (gravity, collisions); OSCBench focuses on action consequence modeling
- **vs T2V-CompBench**: Evaluates compositional generation capability but does not address state change accuracy and temporal consistency

## Rating
- Novelty: ⭐⭐⭐⭐ First dedicated OSC benchmark, filling an important gap
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 models, dual human + automated evaluation
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is clear, evaluation system is complete
- Value: ⭐⭐⭐⭐ Points to a key improvement direction for T2V research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Self-Correcting Text-to-Video Generation with Misalignment Detection and Localized Refinement](self-correcting_text-to-video_generation_with_misalignment_detection_and_localiz.md)
- [\[CVPR 2026\] SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation](../../CVPR2026/video_generation/symphomotion_joint_control_of_camera_motion_and_object_dynamics_for_coherent_vid.md)
- [\[ICCV 2025\] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](../../ICCV2025/video_generation/decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)
- [\[CVPR 2026\] SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation](../../CVPR2026/video_generation/slvmeval_synthetic_meta_evaluation_benchmark_for_text-to-long_video_generation.md)
- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](../../ICCV2025/video_generation/stiv_scalable_text_and_image_conditioned_video_generation.md)

</div>

<!-- RELATED:END -->
