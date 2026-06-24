---
title: >-
  [Paper Note] HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios
description: >-
  [CVPR 2025][Multimodal VLM][Household safety] HomeSafe-Bench is the first benchmark to evaluate VLMs on unsafe action detection in household scenarios (438 cases covering 6 functional areas), and proposes HD-Guard, a hierarchical streaming architecture that coordinates a lightweight FastBrain and a large-scale SlowBrain to achieve real-time safety monitoring.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Household safety"
  - "unsafe action detection"
  - "VLM evaluation"
  - "embodied AI"
  - "dual-brain architecture"
date: 2026-05-08
content_hash: 99bee1dbc0b31e18
---

# HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios

**Conference**: CVPR 2025  
**arXiv**: [2603.11975](https://arxiv.org/abs/2603.11975)  
**Code**: Yes  
**Area**: Multimodal VLM / AI Safety  
**Keywords**: Household safety, unsafe action detection, VLM evaluation, embodied AI, dual-brain architecture

## TL;DR
HomeSafe-Bench is the first benchmark to evaluate VLMs on unsafe action detection in household scenarios (438 cases covering 6 functional areas), and proposes HD-Guard, a hierarchical streaming architecture that coordinates a lightweight FastBrain and a large-scale SlowBrain to achieve real-time safety monitoring.

## Background & Motivation

**Background**: Household robots are developing rapidly, but household environments introduce unpredictable safety risks (e.g., perception delays, lack of common-sense leading to dangerous operations). Existing safety evaluations are mostly limited to static images, text, or general harms.

**Limitations of Prior Work**: (1) Lack of a standardized benchmark for dynamic unsafe action detection; (2) household scenarios are more complex and variable than industrial environments, requiring contextual understanding to determine if an action is safe; (3) the capabilities and bottlenecks of VLMs in safety detection remain unclear.

**Key Challenge**: Real-time safety monitoring demands low latency, whereas accurate unsafe action detection requires deep multimodal reasoning—making it difficult to achieve both simultaneously.

**Goal**: To build an evaluation benchmark and design a real-time safety monitoring architecture.

**Key Insight**: (1) Construct a diverse unsafe action dataset through a hybrid pipeline of physical simulation and video generation; (2) utilize a dual-brain architecture to balance inference efficiency and detection accuracy.

**Core Idea**: FastBrain performs high-frequency lightweight screening, while SlowBrain conducts asynchronous deep reasoning, coordinating both to achieve real-time safety.

## Method

### Overall Architecture
HomeSafe-Bench contains 438 unsafe cases covering 6 functional areas such as kitchens and living rooms, with multi-dimensional fine-grained annotations. During inference, HD-Guard coordinates the fast and slow dual brains: FastBrain continuously screens video frames at a high frequency, triggering SlowBrain for deep multimodal analysis when suspicious actions are detected.

### Key Designs

1. **Hybrid Data Construction Pipeline**:

    - Function: Generate diverse and realistic videos of unsafe actions
    - Mechanism: A physical simulator generates basic scenes and actions, combined with advanced video generation models to enhance visual realism, followed by human annotation of unsafe categories, severity, and context
    - Design Motivation: Pure simulation lacks realism, while pure real-world data is difficult to scale to cover sufficient unsafe scenarios

2. **Hierarchical Dual-Brain Guard (HD-Guard)**:

    - Function: Real-time safety monitoring architecture
    - Mechanism: FastBrain is a lightweight model (e.g., a small ViT) that scans video frames at high frequency to output quick safety scores for each frame. When a score exceeds a threshold, it asynchronously triggers SlowBrain (a large VLM like GPT-4V) to conduct deep multimodal reasoning, combining vision, language, and common-sense knowledge to make the final decision
    - Design Motivation: Analogous to the human fast-and-slow systems (System 1/2)—fast intuitive judgment suffices most of the time, and deep reasoning is initiated only when necessary

3. **Multi-dimensional Fine-grained Annotation**:

    - Function: Support systematic evaluation
    - Mechanism: Each case is annotated with the type of unsafety (e.g., collision, fall, fire), severity, involved objects, and contextual dependency. The division into 6 functional areas ensures the evaluation covers various typical spaces in a home
    - Design Motivation: Coarse-grained "safe/unsafe" binary classification is insufficient to diagnose specific weaknesses of the models

### Loss & Training
FastBrain can be fine-tuned with a small amount of annotated data, while SlowBrain uses pretrained VLMs for zero/few-shot inference.

## Key Experimental Results

### Main Results

| Method | Detection Accuracy | Latency | Explanation |
|------|-----------|------|------|
| HD-Guard | Best trade-off | Low | Dual-brain coordination |
| Large VLM only | Highest accuracy | Very High | Unsuitable for real-time |
| Lightweight model only | Lower | Lowest | Severe missed detections |

### Ablation Study

| Configuration | Performance | Explanation |
|------|------|------|
| FastBrain + SlowBrain | Best | Complementary coordination |
| FastBrain only | High missed detections | Lacks deep reasoning |
| SlowBrain only | High latency | Unable to run in real-time |
| Different trigger thresholds | Trade-off exists | Lower threshold -> More SlowBrain calls |

### Key Findings
- Existing VLMs perform far from perfectly in unsafe action detection, especially in scenarios requiring common-sense reasoning (e.g., determining whether "a knife pointing towards a toddler" is hazardous).
- The fast-slow dual-brain coordination of HD-Guard significantly outperforms single-model strategies.
- Contextual dependency is a key bottleneck—the exact same action may be safe or unsafe depending on the context.

## Highlights & Insights
- **Critical Safety Evaluation Gap**: This is the first work to systematically evaluate the capability of VLMs in household unsafe action detection, which holds direct significance for embodied AI safety.
- **Practicality of the Dual-Brain Architecture**: The analogy to System 1/2 is intuitive and effective; this architecture can be transferred to other scenarios requiring real-time monitoring and deep analysis.
- **Hybrid Data Construction**: The simulation + generation pipeline presents a practical solution to address the scarcity of safety-related data.

## Limitations & Future Work
- The scale of 438 cases is relatively small and may not cover all unsafe scenarios.
- Unsafe actions generated via video models may possess a distribution shift compared to real-world actions.
- The optimal balance between the false positive rate of FastBrain and the invocation frequency of SlowBrain requires scenario-specific tuning.
- Safety concerns in multi-person interaction scenarios are not considered.

## Related Work & Insights
- **vs SafetyBench (Text Safety)**: Text-based safety evaluations do not involve visual and physical interactions, whereas HomeSafe-Bench is closer to embodied scenarios.
- **vs RoboCasa/Habitat**: Simulation platforms provide environments but do not focus on safety evaluations; HomeSafe-Bench fills this gap.
- **vs System 1/2 Architectures**: HD-Guard serves as a concrete implementation of dual-system theory in cognitive science for AI safety.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of a new benchmark and a dual-brain architecture is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes multi-VLM comparisons and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ The problem motivation and methodology are clearly described.
- Value: ⭐⭐⭐⭐⭐ Holds significant practical importance for embodied AI safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons](from_multimodal_llms_to_generalist_embodied_agents_methods_and_lessons.md)
- [\[CVPR 2025\] Embodied Scene Understanding for Vision Language Models via MetaVQA](embodied_scene_understanding_for_vision_language_models_via_metavqa.md)
- [\[CVPR 2025\] Evaluating Vision-Language Models as Evaluators in Path Planning](evaluating_vision-language_models_as_evaluators_in_path_planning.md)
- [\[ICML 2026\] VLA-Arena: An Open-Source Framework for Evaluating Vision-Language-Action Models](../../ICML2026/multimodal_vlm/vla-arena_an_open-source_framework_for_benchmarking_vision-language-action_model.md)
- [\[NeurIPS 2025\] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](../../NeurIPS2025/multimodal_vlm/mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)

</div>

<!-- RELATED:END -->
