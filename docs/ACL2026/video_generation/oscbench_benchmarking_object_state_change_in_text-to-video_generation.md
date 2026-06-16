---
title: >-
  [Paper Note] OSCBench: Benchmarking Object State Change in Text-to-Video Generation
description: >-
  [ACL 2026][Video Generation][Paper Note] This paper proposes OSCBench—the first benchmark specifically evaluating Object State Change (OSC) capabilities in text-to-video models. Built on cooking scenes with 1,120 prompts covering common, novel, and compositional scenarios, it reveals that even the strongest T2V models achieve an OSC accuracy of only 0.786.
tags:
  - ACL 2026
  - Video Generation
date: 2026-05-08
content_hash: 5ac019f12fb1a527
---
# OSCBench: Benchmarking Object State Change in Text-to-Video Generation

**Conference**: ACL 2026  
**arXiv**: [2603.11698](https://arxiv.org/abs/2603.11698)  
**Code**: [Project Page](https://hanxjing.github.io/OSCBench)  
**Area**: Video Generation  
**Keywords**: Text-to-Video, Object State Change, Evaluation Benchmark, Cooking Scenes, Multimodal Evaluation

## TL;DR
This paper proposes OSCBench—the first benchmark specifically evaluating Object State Change (OSC) capabilities in text-to-video models. Built on cooking scenes with 1,120 prompts covering common, novel, and compositional scenarios, it reveals that even the strongest T2V models achieve an OSC accuracy of only 0.786.

## Background & Motivation

**Background**: T2V models have made significant progress in visual quality and temporal consistency. Existing benchmarks primarily evaluate perceptual quality, text-video alignment, or physical plausibility.

**Limitations of Prior Work**: Existing benchmarks ignore a key dimension of action understanding—object state changes explicitly specified by text prompts (e.g., peeling a potato, slicing a lemon). T2V models may align well at a high semantic level but generate incorrect, incomplete, or inconsistent object state changes.

**Key Challenge**: High-quality visual appearance often masks defects in action consequence modeling—videos look realistic, but the objects do not correctly change state.

**Goal**: To construct a systematic OSC evaluation benchmark to diagnose the specific deficiencies of T2V models in state change modeling.

**Key Insight**: This work selects cooking scenes as the evaluation domain (where state changes are frequent, diverse, and well-defined) and designs common, novel, and compositional scenarios to test capabilities at different levels.

**Core Idea**: OSC evaluation is divided into two sub-dimensions: state change accuracy and state change consistency, paired with CoT-guided MLLM automatic evaluation.

## Method

### Overall Architecture
OSCBench starts from the HowToChange dataset. Through human-AI collaboration, 20 actions and 134 objects are abstracted into 9 action categories and 8 object categories (comprising 28 subcategories). Based on this, three types of OSC scenarios are constructed: Common, Novel, and Compositional (108/20/12 prompts respectively), with each scenario containing 8 action-object combinations, totaling 1,120 prompts. Evaluation is conducted across four dimensions: semantic following, OSC performance, scene alignment, and perceptual quality.

### Key Designs

**1. Three types of OSC scenarios: Separating "Memory" from "Reasoning"**

If only common action-object combinations are used for testing, models might rely on memorized training data, making it difficult to judge if they truly understand state changes. Consequently, three levels of scenarios are designed: Common scenarios use frequent combinations (e.g., cutting a lemon) to test basic capability; Novel scenarios use uncommon but physically plausible combinations (e.g., mashing a grapefruit), which cannot be solved by memory and must be inferred from action semantics, thus testing generalization; Compositional scenarios link multiple consecutive actions (e.g., peeling then slicing) to test temporal consistency. These levels distinguish "tasks solvable by memory" from those "requiring reasoning."

**2. CoT-guided MLLM evaluation: Reasoning before Scoring**

Judging whether an object reaches the correct state and whether the process is smooth requires multi-step reasoning. Using MLLMs as black-box scorers is unreliable. This work uses Chain-of-Thought (CoT) to guide MLLMs through a reasoning chain of "standard grounding → evidence extraction → score justification." The model first aligns with evaluation standards, extracts evidence from the video, and finally provides a score based on that evidence. This makes state change judgments more grounded and scalable for large-scale automated evaluation.

**3. Multi-dimensional evaluation system: Locating exactly where OSC fails**

OSC failures can occur at various stages, and a single total score cannot identify the root cause. The evaluation is decomposed into four parts: Semantic Following (subject/object/action alignment), OSC Performance, Scene Alignment, and Perceptual Quality (realism + aesthetics). Each item is evaluated on a 1-5 Likert scale, with human evaluations taking the mean of three annotators. OSC performance is characterized by two core metrics: State Change Accuracy (measuring if the object reaches the correct target state) and State Change Consistency (measuring if the transformation process is smooth and coherent). Together, these metrics answer whether the "action truly changed the object," exposing the "good aesthetics but no state change" defect.

### Loss & Training
This work is an evaluation study; no model training is involved.

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
- All models perform well on subject and object alignment (>0.73), but OSC accuracy and consistency are significantly lower.
- The strongest model, Veo-3.1, achieves an OSC accuracy of only 0.786, indicating that state change modeling is a critical bottleneck for T2V.
- Performance is worse in Novel and Compositional scenarios compared to Common scenarios, revealing a lack of generalization.
- Closed-source models (Veo, Kling) significantly outperform open-source models, with the gap being particularly pronounced in the OSC dimension.

## Highlights & Insights
- The OSC perspective fills a significant gap in T2V evaluation—actions should not just be motion but should produce correct object state changes.
- The three-tier scenario design cleverly distinguishes between memorization and reasoning.
- CoT-guided MLLM evaluation is highly correlated with human evaluation, providing a viable path for large-scale automated OSC assessment.

## Limitations & Future Work
- The current focus is limited to the cooking domain; OSC evaluation in other domains (e.g., crafting, chemistry) requires expansion.
- Currently, only single actions or two-step combinations are evaluated; longer sequences of composite actions present a greater challenge.
- While MLLM evaluation correlates with humans, it is not a perfect replacement and may misjudge extreme failure cases.

## Related Work & Insights
- **vs VBench**: VBench focuses on overall video quality but lacks specialized evaluation for object state changes.
- **vs PhyWorldBench**: Focuses on physical plausibility (gravity, collision), whereas OSCBench focuses on action consequence modeling.
- **vs T2V-CompBench**: Evaluates compositional generation but does not address the accuracy and temporal consistency of state changes.

## Rating
- Novelty: ⭐⭐⭐⭐ First specialized OSC benchmark, filling a critical gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 models evaluated with both human and automated metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is clear with a comprehensive evaluation system.
- Value: ⭐⭐⭐⭐ Points toward a key improvement direction for T2V research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LocoT2V-Bench: Benchmarking Long-form and Complex Text-to-Video Generation](../../ICML2026/video_generation/locot2v-bench_benchmarking_long-form_and_complex_text-to-video_generation.md)
- [\[CVPR 2026\] Ego-InBetween: Generating Object State Transitions in Ego-Centric Videos](../../CVPR2026/video_generation/ego-inbetween_generating_object_state_transitions_in_ego-centric_videos.md)
- [\[CVPR 2026\] M4V: Multimodal Mamba for Efficient Text-to-Video Generation](../../CVPR2026/video_generation/m4v_multimodal_mamba_for_efficient_text-to-video_generation.md)
- [\[CVPR 2026\] TGT: Text-Grounded Trajectories for Locally Controlled Video Generation](../../CVPR2026/video_generation/tgt_text-grounded_trajectories_for_locally_controlled_video_generation.md)
- [\[ICML 2026\] T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation](../../ICML2026/video_generation/t2av-compass_towards_unified_evaluation_for_text-to-audio-video_generation.md)

</div>

<!-- RELATED:END -->
