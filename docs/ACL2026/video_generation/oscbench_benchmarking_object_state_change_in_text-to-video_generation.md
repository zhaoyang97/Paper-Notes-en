---
title: >-
  [Paper Note] OSCBench: Benchmarking Object State Change in Text-to-Video Generation
description: >-
  [ACL 2026][Video Generation][Paper Note] The authors propose OSCBench—the first benchmark specifically designed to evaluate Object State Change (OSC) capabilities in text-to-video (T2V) models. Built on cooking scenarios with 1,120 prompts covering Regular, Novel, and Compositional scenarios, the benchmark reveals that even the strongest T2V models achieve an
tags:
  - ACL 2026
  - Video Generation
date: 2026-05-08
content_hash: e4ae1e9da8b973ec
---
# OSCBench: Benchmarking Object State Change in Text-to-Video Generation

**Conference**: ACL 2026  
**arXiv**: [2603.11698](https://arxiv.org/abs/2603.11698)  
**Code**: [Project Page](https://hanxjing.github.io/OSCBench)  
**Area**: Video Generation  
**Keywords**: Text-to-video, Object State Change, Evaluation Benchmark, Cooking Scenes, Multimodal Evaluation

## TL;DR
The authors propose OSCBench—the first benchmark specifically designed to evaluate Object State Change (OSC) capabilities in text-to-video (T2V) models. Built on cooking scenarios with 1,120 prompts covering Regular, Novel, and Compositional scenarios, the benchmark reveals that even the strongest T2V models achieve an OSC accuracy of only 0.786.

## Background & Motivation

**Background**: T2V models have made significant progress in visual quality and temporal consistency. Existing benchmarks primarily evaluate perceptual quality, text-video alignment, or physical plausibility.

**Limitations of Prior Work**: Current benchmarks ignore a critical dimension of action understanding—the Object State Changes (OSC) explicitly specified by text prompts (e.g., peeling a potato, slicing a lemon). T2V models may align well at a high-level semantic level but generate incorrect, incomplete, or inconsistent object state changes.

**Key Challenge**: High visual quality often masks deficiencies in modeling action consequences; videos may look realistic, but the objects fail to undergo the correct state transitions.

**Goal**: To construct a systematic OSC evaluation benchmark to diagnose the specific weaknesses of T2V models in state change modeling.

**Key Insight**: The authors select the cooking domain as the evaluation setting (due to frequent, diverse, and well-defined state changes) and design three types of scenarios (Regular, Novel, and Compositional) to test different levels of capability.

**Core Idea**: The evaluation of OSC is divided into two sub-dimensions: state change accuracy and state change consistency, supported by automated evaluation using CoT-guided MLLMs.

## Method

### Overall Architecture
OSCBench originates from the HowToChange dataset. Through human-machine collaboration, 20 actions and 134 objects were abstracted into 9 action categories and 8 object categories (28 sub-categories). Based on this, Regular, Novel, and Compositional OSC scenarios (108/20/12) were constructed, each containing 8 action-object combinations for a total of 1,120 prompts. Evaluation is conducted across four dimensions: semantic following, OSC performance, scene alignment, and perceptual quality.

### Key Designs

**1. Three OSC Scenario Types: Separating "Memory" from "Reasoning"**

If models are tested only on common action-object combinations, they might rely on memorized training data rather than true understanding. The authors design three levels: Regular scenarios use common combinations (e.g., slicing a lemon) to test basic capability; Novel scenarios use uncommon but physically plausible combinations (e.g., mashing a grapefruit), which cannot be solved by memory and require inferring transitions from action semantics, thus testing generalization; and Compositional scenarios involve sequential actions (e.g., peeling then slicing) to test temporal consistency. These levels distinguish between "solvable by memory" and "requires reasoning."

**2. CoT-guided MLLM Evaluation: Reasoning Before Scoring**

Determining if an object has reached the correct state and whether the process is smooth requires multi-step reasoning. Direct scoring by an MLLM as a black box is unreliable. This work uses a Chain-of-Thought (CoT) to guide the MLLM through a reasoning chain: "Standard Grounding → Evidence Extraction → Score Justification." The MLLM first aligns with evaluation criteria, then extracts evidence from the video, and finally provides a score based on that evidence. This ensures state change judgments are grounded and scalable for automated evaluation.

**3. Multi-dimensional Evaluation System: Pinpointing Specific OSC Failures**

OSC failures can occur at various stages, and a single total score cannot identify the root cause. This work decomposes the evaluation into four parts: Semantic Following (alignment of subject, object, and action), OSC Performance, Scene Alignment, and Perceptual Quality (realism and aesthetics). Each is measured on a 1-5 Likert scale, with human evaluations based on the mean of three raters. Within OSC Performance, two core metrics are defined: State Change Accuracy measures if the object reaches the correct target state, and State Change Consistency measures the smoothness and coherence of the transition process. Together, they reveal models that produce "pretty visuals but no state transition."

### Loss & Training
This work is an evaluation benchmark and does not involve model training.

## Key Experimental Results

### Main Results (Human Evaluation, Normalized 0-1)

| Model | Subject Alignment | Object Alignment | Action Alignment | OSC Accuracy | OSC Consistency | Realism |
|------|---------|---------|---------|----------|----------|--------|
| Veo-3.1-Fast | 0.936 | 0.916 | **0.908** | **0.786** | **0.748** | Highest |
| Kling-2.5-Turbo | **0.938** | 0.900 | 0.826 | 0.726 | 0.726 | 0.732 |
| Wan-2.2 | 0.904 | 0.842 | 0.616 | 0.560 | 0.668 | 0.702 |
| HunyuanVideo-1.5 | 0.914 | 0.902 | 0.656 | 0.524 | 0.608 | 0.618 |
| Open-Sora-2.0 | 0.860 | 0.734 | 0.518 | 0.380 | 0.428 | 0.416 |

### Key Findings
- All models perform well in subject and object alignment (>0.73), but OSC accuracy and consistency are significantly lower.
- The OSC accuracy of the strongest model, Veo-3.1, is only 0.786, indicating that state change modeling is a critical bottleneck for T2V.
- Performance in Novel and Compositional scenarios is worse than in Regular scenarios, revealing a lack of generalization.
- Closed-source models (Veo/Kling) clearly outperform open-source models, with the gap being most pronounced in OSC dimensions.

## Highlights & Insights
- The OSC perspective fills a significant gap in T2V evaluation—actions should be more than just motion; they must produce the correct state transitions.
- The design of the three scenario types effectively distinguishes between memorization and reasoning capabilities.
- CoT-guided MLLM evaluation correlates highly with human judgment, offering a feasible path for large-scale automated OSC evaluation.

## Limitations & Future Work
- The focus is limited to the cooking domain; OSC evaluation should be extended to other fields like crafts or chemical experiments.
- The current work evaluates only single actions or two-step combinations; longer sequential actions present a greater challenge.
- While MLLM evaluation correlates with human results, it is not a perfect replacement and may misjudge in cases of extreme failure.

## Related Work & Insights
- **vs VBench**: VBench focuses on general video quality but lacks specialized evaluation for object state changes.
- **vs PhyWorldBench**: This focuses on physical plausibility (gravity, collisions), whereas OSCBench focuses on action consequence modeling.
- **vs T2V-CompBench**: This evaluates compositional generation but does not address the accuracy or temporal consistency of state changes.

## Rating
- Novelty: ⭐⭐⭐⭐ First specialized benchmark for OSC, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 models evaluated via both human and automated methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition and a comprehensive evaluation system.
- Value: ⭐⭐⭐⭐ Identifies a key direction for future T2V research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Ego-InBetween: Generating Object State Transitions in Ego-Centric Videos](../../CVPR2026/video_generation/ego-inbetween_generating_object_state_transitions_in_ego-centric_videos.md)
- [\[ICML 2026\] LocoT2V-Bench: Benchmarking Long-form and Complex Text-to-Video Generation](../../ICML2026/video_generation/locot2v-bench_benchmarking_long-form_and_complex_text-to-video_generation.md)
- [\[CVPR 2026\] TiViBench: Benchmarking Think-in-Video Reasoning for Video Generation](../../CVPR2026/video_generation/tivibench_benchmarking_think-in-video_reasoning_for_video_generation.md)
- [\[CVPR 2026\] GenHOI: Towards Object-Consistent Hand-Object Interaction with Temporally Balanced and Spatially Selective Object Injection](../../CVPR2026/video_generation/genhoi_towards_object-consistent_hand-object_interaction_with_temporally_balance.md)
- [\[CVPR 2026\] M4V: Multimodal Mamba for Efficient Text-to-Video Generation](../../CVPR2026/video_generation/m4v_multimodal_mamba_for_efficient_text-to-video_generation.md)

</div>

<!-- RELATED:END -->
