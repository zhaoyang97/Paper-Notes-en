---
title: >-
  [Paper Note] A Thousand Words Paint a Picture: Multimodal Goal Tracking for Grounded Social Intelligence
description: >-
  [ACL 2025][Video Understanding][Multimodal Goal Tracking] This paper proposes a multimodal goal tracking framework that reasons about the implicit goals of participants in social situations by integrating visual and linguistic cues, thereby enhancing the model's understanding of social contexts (i.e., "grounded social intelligence").
tags:
  - "ACL 2025"
  - "Video Understanding"
  - "Multimodal Goal Tracking"
  - "Social Intelligence"
  - "Vision-Language Understanding"
  - "Dialogue System"
  - "Intent Reasoning"
date: 2026-05-08
content_hash: 06bddfb48e4d93a4
---

# A Thousand Words Paint a Picture: Multimodal Goal Tracking for Grounded Social Intelligence

**Conference**: ACL 2025  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Multimodal Goal Tracking, Social Intelligence, Vision-Language Understanding, Dialogue System, Intent Reasoning

## TL;DR
This paper proposes a multimodal goal tracking framework that reasons about the implicit goals of participants in social situations by integrating visual and linguistic cues, thereby enhancing the model's understanding of social contexts (i.e., "grounded social intelligence").

## Background & Motivation

**Background**: Social intelligence is an important research direction in artificial intelligence, requiring models to understand human intentions, goals, and behavioral motivations in social situations. Most current work focuses on single-modality textual dialogue understanding or pure visual action recognition, lacking a framework that effectively integrates the two.

**Limitations of Prior Work**: Existing research in social intelligence faces three core limitations: first, text-only models cannot capture non-verbal cues in visual scenes (such as expressions, gestures, and object interactions); second, vision-only models struggle to comprehend pragmatic intentions in dialogue contexts; third, there is a lack of standard benchmarks to evaluate models' capability to track participant goals in multimodal social contexts.

**Key Challenge**: Goals in social scenes are often implicit, dynamically changing, and require joint reasoning across multimodal information to be accurately recognized. Existing methods either only handle static object detection or perform reasoning in a single modality, failing to capture the dynamic evolution of goals.

**Goal**: To construct a multimodal goal tracking task and a corresponding evaluation framework, enabling models to continuously track the goal states of participants in vision-language-grounded social scenarios.

**Key Insight**: The authors observe that in social interactions, an image (a thousand words) often reveals goals that are not explicitly stated in the dialogue. Thus, goal tracking is modeled as a sequential multimodal reasoning problem.

**Core Idea**: To design the Grounded Social Intelligence task framework, combining visual scene grounding and linguistic goal reasoning, to achieve dynamic goal tracking of multi-party participants in social scenes.

## Method

### Overall Architecture
The entire method revolves around "goal tracking": given a social scenario containing a sequence of images and dialogue text, the model needs to identify the current goal state of each participant at each time step (e.g., whether completed, whether changed, newly added goals, etc.). The framework consists of three core stages: multimodal scene encoding, goal state reasoning, and dynamic goal tracking.

### Key Designs

1. **Multimodal Scene Encoder**:

    - **Function**: Encodes images and dialogue texts into a unified multimodal representation.
    - **Mechanism**: A pre-trained vision-language model is utilized as the backbone network, aligning visual regions (objects, humans, spatial relations) with referring expressions in the dialogue text through a cross-attention mechanism. Feature extraction is performed on images at each time step using a visual encoder, and semantic features are extracted from dialogue texts using a language encoder, which then interact through a cross-modality fusion layer.
    - **Design Motivation**: Understanding goals in social situations requires simultaneous consideration of "what they are looking at/doing" (vision) and "what they are saying" (language), which a single modality cannot accomplish alone.

2. **Goal State Reasoning Module**:

    - **Function**: Infers the implicit goals of participants at each time step.
    - **Mechanism**: Goals are modeled as a set of structured attributes (such as target object, target action, goal state), and predictions are made for each participant's goal at the current time step via conditional generation or classification. Social commonsense prior knowledge is introduced to help the model understand unexpressed social norms and implicit intentions. The reasoning process accounts for historical context to form a chain of goals.
    - **Design Motivation**: Many goals in social scenes are not explicitly expressed, necessitating inference from visual cues and dialogue hints.

3. **Dynamic Goal Tracker**:

    - **Function**: Tracks goal changes, completion, and addition across time steps.
    - **Mechanism**: A state machine-like mechanism is adopted to maintain a list of goal states for each participant, updating the goal state at each time step based on new multimodal inputs. This includes three state transitions: goal creation (emergence of new goals), goal update (modification of goals), and goal completion (achievement of goals).
    - **Design Motivation**: Social interactions are dynamic, with goals continuously evolving during the dialogue, requiring a mechanism capable of tracking such dynamic changes.

### Loss & Training
Training employs a multi-task learning strategy, jointly optimizing the goal state classification loss, goal content generation loss, and temporal consistency loss. Goal state classification utilizes the cross-entropy loss, goal content generation utilizes the negative log-likelihood loss for sequence generation, and the temporal consistency loss ensures that goal state changes across consecutive time steps are reasonable.

## Key Experimental Results

### Main Results

| Model | Goal F1 | State Acc | Grounding Acc | Overall Score |
|------|---------|-----------|---------------|----------|
| Ours | 68.5 | 74.2 | 71.3 | 71.3 |
| GPT-4V (zero-shot) | 52.3 | 61.7 | 58.4 | 57.5 |
| LLaVA-1.5 | 45.8 | 54.3 | 49.2 | 49.8 |
| Text-only baseline | 38.7 | 47.5 | - | 43.1 |

### Ablation Study

| Configuration | Goal F1 | Description |
|------|---------|------|
| Full model | 68.5 | Full model |
| w/o Visual Input | 51.2 | Removing vision drops performance by 17.3%, proving visual information is critical |
| w/o Dynamic Tracking | 59.8 | Removing temporal tracking drops performance by 8.7% |
| w/o Social Commonsense Prior | 63.1 | Removing commonsense drops performance by 5.4% |

### Key Findings
- The visual modality contributes the most to goal tracking; removing vision causes the largest performance drop (17.3%), indicating that goals in social scenes rely heavily on visual cues.
- The dynamic tracking module shows significant effectiveness for continuous scenarios, with the most pronounced improvements in multi-turn interaction scenes.
- Although zero-shot large models (such as GPT-4V) show some capability, they still fall significantly short in fine-grained goal tracking.

## Highlights & Insights
- Combining "social intelligence" with "multimodal grounding" is a novel perspective that fills a gap in current research. This integration enables the model to understand not only "what was seen" but also "why they did that."
- The state machine design for dynamic goal tracking is simple yet effective, and can be transferred to other tasks requiring state tracking, such as multi-turn dialogue state tracking.
- The proposed evaluation framework provides a standardized benchmark for future research in social intelligence.

## Limitations & Future Work
- The dataset scale may be limited by the cost of human annotation, especially since labeling implicit goals requires highly skilled annotators.
- The method is highly dependent on visual quality, and performance may degrade in blurry or occluded scenes.
- The current focus is primarily on two-person social scenarios; extending this to complex, multi-person social scenarios is a potential future direction.
- Integrating the reasoning capabilities of large language models with the structured tracking framework proposed in this paper is worth exploring.

## Related Work & Insights
- **vs Social-IQ**: Social-IQ mainly focuses on question-answering understanding in social scenarios, whereas this work further requires the model to track the dynamic changes of goals, providing a finer granularity.
- **vs VisDial**: VisDial focuses on visual dialogue but does not involve reasoning about social intelligence, whereas this paper extends to social goal inference.
- This work has a natural connection with Theory-of-Mind research and can serve as a complementary benchmark for evaluating the theory of mind capabilities of models.
- It structurally resembles the goal recognition task in embodied AI, but this paper places more emphasis on language-vision integrated social scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ Multimodal social intelligence goal tracking is a relatively novel task definition that fills the gap in existing benchmarks.
- Experimental Thoroughness: ⭐⭐⭐ The experimental design is reasonable, but limited by data scale and scene coverage.
- Writing Quality: ⭐⭐⭐⭐ The title is eye-catching, the paper structure is clear, and the problem definition is intuitive.
- Value: ⭐⭐⭐⭐ It opens a new evaluation dimension for social intelligence AI research and holds long-term research value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Disentangled Concepts Speak Louder Than Words: Explainable Video Action Recognition](../../NeurIPS2025/video_understanding/disentangled_concepts_speak_louder_than_words_explainable_video_action_recogniti.md)
- [\[ICML 2026\] Unified Multimodal Visual Tracking with Dual Mixture-of-Experts](../../ICML2026/video_understanding/unified_multimodal_visual_tracking_with_dual_mixture-of-experts.md)
- [\[CVPR 2025\] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](../../CVPR2025/video_understanding/mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)
- [\[ICCV 2025\] What You Have is What You Track: Adaptive and Robust Multimodal Tracking](../../ICCV2025/video_understanding/what_you_have_is_what_you_track_adaptive_and_robust_multimodal_tracking.md)
- [\[CVPR 2026\] Active Intelligence in Video Avatars via Closed-loop World Modeling](../../CVPR2026/video_understanding/active_intelligence_in_video_avatars_via_closed-loop_world_modeling.md)

</div>

<!-- RELATED:END -->
