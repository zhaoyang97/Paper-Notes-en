---
title: >-
  [Paper Note] Can Vision Language Models Understand Mimed Actions?
description: >-
  [Multimodal VLM] This paper proposes the Mime benchmark (86 mimed actions × 10 variations = 860 samples), constructing a controllable evaluation via motion capture + 3D rendering. It finds that while humans maintain near-100% accuracy under various perturbations, the strongest VLM achieves only 52.3% (multiple-choice) / 19.8% (free-form), revealing that VLMs heavily rely on scene context cues rather than the action itself.
tags:
  - "Multimodal VLM"
date: 2026-05-08
content_hash: 1a4f535a545226b6
---

# Can Vision Language Models Understand Mimed Actions?

| Metadata | Content |
|------|------|
| Conference | ACL 2025 |
| arXiv | [2506.21586](https://arxiv.org/abs/2506.21586) |
| Code | [justin-cho.com/mime](https://justin-cho.com/mime) |
| Area | Multimodal VLM |
| Keywords | Mime recognition, VLM evaluation, Action understanding, Non-verbal communication, Video QA |

## TL;DR

This paper proposes the Mime benchmark (86 mimed actions × 10 variations = 860 samples), constructing a controllable evaluation via motion capture + 3D rendering. It finds that while humans maintain near-100% accuracy under various perturbations, the strongest VLM achieves only 52.3% (multiple-choice) / 19.8% (free-form), revealing that VLMs heavily rely on scene context cues rather than the action itself.

## Background & Motivation

**Research Question:** Can Vision-Language Models (VLMs) reliably recognize mimed actions—a subset of non-verbal communication that conveys intent solely through body movements while removing crucial object contexts?

**Core Argument:** Mimed actions represent a unique subset of non-verbal communication (NVC). Unlike other gestures, mimed actions exhibit extremely high interpretative consistency among humans and are directly linked to physical actions. Therefore, understanding pantomimes is a **necessary foundational prerequisite** for VLMs to advance toward NVC understanding.

**Limitations of Prior Work:** While VLMs perform exceptionally on standard action recognition benchmarks, actions in these benchmarks are always accompanied by complete contextual cues (e.g., weightlifting in a gym with barbells, sportswear, etc.). When these cues are stripped away, the true understanding capabilities of VLMs are fully exposed.

## Method

### Overall Architecture

The benchmark is constructed using motion-capture (MoCap) and 3D computer graphics software (Blender), allowing flexible control over characters, backgrounds, and viewpoints to systematically evaluate the robustness of VLMs in mimed action recognition:

**Data Construction Pipeline:** (1) Vicon MoCap stage capture $\rightarrow$ (2) 3D character retargeting in Blender $\rightarrow$ (3) Transparent background frame rendering $\rightarrow$ (4) Overlaying onto specified backgrounds.

### Key Designs

**1. Motion Capture Data Collection:**
- Brainstormed 75 candidate mimed actions (actions lacking critical object context, such as playing violin without a violin, swimming without water).
- Captured by 2 actors (1 amateur male + 1 professional female) with 3 takes each.
- Retained only if at least 2 out of 3 authors could correctly identify the action.
- Finally retained 47 action types and 86 mime samples.

**2. 10 Variational Designs (Per Action):**

| Variation | Character | Background | Viewpoint |
|------|------|------|------|
| Base | Male human | Blank white | Frontal |
| Aligned Background | Male human | Matching action (e.g., basketball court) | Frontal |
| Adversarial Background | Male human | Mismatched (e.g., living room) | Frontal |
| Adversarial Character | Astronaut suit character | Blank white | Frontal |
| Female Character | Female human | Blank white | Frontal |
| 90°/180°/270° | Male human | Blank white | Rotated |

**3. Dual-Format Evaluation:**
- **Multiple-Choice (MC):** 4 options, with distractors excluding semantically similar actions.
- **Free-Form (FF):** No option prompts, using sentence embedding cosine similarity (threshold 0.5) to determine correctness.

## Experiments

### Main Results: Mime vs Real

| Model | Mime MC | Mime FF | Real MC | Real FF |
|------|--------|---------|---------|---------|
| Gemini 1.5 Flash | 52.3% | 19.8% | ~100% | ~95% |
| GPT-4o Mini | 41.9% | 11.6% | ~99% | ~92% |
| Qwen 2.5 VL (7B) | 39.5% | 5.8% | ~97% | ~85% |
| InternVL2.5 (8B) | 31.4% | 2.3% | ~96% | ~80% |
| **Human** | **99.6%** | **89.5%** | **~100%** | **~95%** |

### Ablation on Background Perturbations

| Model | Base (Blank) | Aligned Background | Adversarial Background |
|------|-----------|----------|----------|
| Gemini 1.5 Flash MC | 52.3% | **68.6%** | 37.2% |
| GPT-4o Mini MC | 41.9% | **66.3%** | 37.2% |
| Qwen 2.5 VL (7B) MC | 39.5% | **68.6%** | 32.6% |
| Human MC | 99.6% | 98.5% | 99.2% |

### Ablation on Viewpoint Perturbations

| Model | 0° | 90° | 180° | 270° | Std Dev ↓ |
|------|-----|------|-------|-------|---------|
| Gemini 1.5 Flash MC | 52.3% | 47.7% | 52.3% | 53.5% | 2.2 |
| GPT-4o Mini MC | 41.9% | 47.7% | 43.0% | 47.7% | 2.6 |
| Human MC | 99.6% | 98.8% | 98.8% | 98.7% | **0.4** |

### Key Findings

- **Massive gap between VLMs and humans:** Humans maintain ~99% MC accuracy across all variations, whereas the strongest VLM (Gemini 1.5 Flash) achieves only 52.3%, a gap of roughly 47 percentage points.
- **VLMs rely heavily on scene cues:** The aligned background improves Gemini's performance from 52.3% to 68.6% (+16.3%) while human performance remains virtually unchanged, indicating that VLMs guess action based on background rather than understanding the action itself.
- **Adversarial backgrounds seriously mislead VLMs:** Performance drops from 52.3% to 37.2%, while humans remain unaffected.
- **Humans are highly robust to viewpoint and character changes:** VLMs exhibit significant fluctuations across different viewpoints.
- **Chain of Thought offers no obvious help:** Manual inspection of Gemini's CoT outputs reveals that 80% of errors stem from incorrect motion observation, whereas only 15% arise from incorrect reasoning over correct descriptions.
- **Few-shot offers only marginal help for closed-source models:** However, performance remains far below that of humans.

## Highlights & Insights

- Elegant experimental design: Achieves complete decoupling of action, character, background, and viewpoint through MoCap and 3D rendering, enabling systematic ablation studies.
- Clearly exposes the fundamental flaw in VLM action understanding: It is not a matter of "poor performance" but rather a lack of "actual understanding of human actions."
- Constructs a "Real" control dataset to precisely quantify the impact of having versus lacking context cues on VLMs.
- Rigorous human evaluation design: Involves 60 participants spanning 8 nationalities and diverse backgrounds.

## Limitations & Future Work

- The scale of 86 mimed actions is relatively limited and may not fully cover all types of everyday actions.
- The 3D-rendered characters lack detailed facial expressions, which are critical components of non-verbal communication.
- The motion capture data of only 2 actors might not sufficiently capture individual performance variations.
- The multiple-choice distractors exclude semantically similar options, which might make the evaluation relatively easy.
- Factors such as video length and frame rate on VLM performance have not been fully explored.
- Only single-action recognition is considered, without addressing action sequence understanding.

## Related Work & Insights

- **Action Recognition Benchmarks:** Traditional benchmarks provide complete contextual cues, complementing Mime.
- **VLM Video Understanding:** Models like Qwen-VL, InternVL, Gemini, and GPT-4o Mini perform exceptionally well on standard video QA.
- **Non-verbal Communication Studies:** Mehrabian (1972) and Poyatos (1983) laid the foundations for NVC research.
- **Human Pantomime Cognition:** O'Reilly (1995) and Little & Firestone (2021) demonstrate that humans show extremely high consistency in recognizing mimed actions.
- **Contrastive Evaluation Paradigm:** Isolates specific capabilities by controlling variables, such as controlling context cues to test action understanding in this paper.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Score | 8.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](teaching_vlm_ask_ambiguity.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ACL 2025\] Redundancy Principles for MLLMs Benchmarks](redundancy_principles_for_mllms_benchmarks.md)
- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)

</div>

<!-- RELATED:END -->
