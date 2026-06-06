---
title: >-
  [Paper Note] GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks
description: >-
  [CVPR 2026][Multimodal VLM][GUI understanding] This paper proposes GUIDE, a benchmark comprising 67.5 hours of screen recordings and think-aloud annotations from 120 novice users across 10 software applications. It defin…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "GUI understanding"
  - "user behavior detection"
  - "intent prediction"
  - "assistance prediction"
  - "novice users"
date: 2026-05-08
content_hash: 63057183647b3d79
---

# GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks

**Conference**: CVPR 2026
**arXiv**: [2603.25864](https://arxiv.org/abs/2603.25864)  
**Code**: [https://guide-bench.github.io/](https://guide-bench.github.io/)  
**Area**: Multimodal VLM / Human-Computer Interaction / GUI Agents
**Keywords**: GUI understanding, user behavior detection, intent prediction, assistance prediction, novice users

## TL;DR
This paper proposes GUIDE, a benchmark comprising 67.5 hours of screen recordings and think-aloud annotations from 120 novice users across 10 software applications. It defines three hierarchical tasks—behavioral state detection, intent prediction, and assistance prediction—and finds that current state-of-the-art multimodal models show limited capability in understanding user behavior and judging assistance needs (behavioral detection accuracy of only 44.6%), while providing structured user context substantially improves performance (up to +50.2pp on assistance prediction).

## Background & Motivation

1. **Background**: Existing GUI agents primarily focus on full automation—given a goal, automatically executing clicks and keyboard actions to complete a task. Academic efforts such as VideoGUI and AssistGUI, as well as industry products like Microsoft Copilot and Figma Make, all follow this paradigm.
2. **Limitations of Prior Work**: Full automation overlooks how users actually work. In open-ended creative or analytical tasks, users need to explore, iterate, experiment, and revise their ideas. Repeated undo actions may not indicate redundant operations, but rather a process of preference formation—something automated agents treat as equivalent behavior and skip over.
3. **Key Challenge**: Existing benchmarks are built on expert demonstrations of closed tasks and evaluate whether a model can replicate the same sequence of actions. Yet the users who genuinely need assistance are novices engaged in open-ended exploration. Superficially similar action sequences can stem from entirely different intentions—repeated undos may signal confusion or deliberate refinement.
4. **Goal**: (1) Collect real behavioral data from novice users performing open-ended tasks; (2) Evaluate whether models can understand what users are doing, why they are doing it, and whether they need help.
5. **Key Insight**: Decompose GUI assistance into three progressive levels—understanding (behavior) → reasoning (intent) → acting (assistance)—aligned with Norman's Seven Stages of Action and Bloom's Taxonomy of cognitive levels.
6. **Core Idea**: Construct a three-tier progressive evaluation benchmark centered on novice users' open-ended GUI interactions, and reveal the critical role of structured user context in enabling effective assistance.

## Method

### Overall Architecture
GUIDE is constructed in three stages: (1) **Video collection**—54 novice users complete 40 open-ended tasks across 10 software applications, with 3 users per task, recording screen activity and think-aloud protocols; (2) **Behavioral taxonomy**—a hierarchical classification system with 9 behavioral state categories; (3) **Three evaluation tasks**—behavioral state detection, intent prediction, and assistance prediction, all supporting context-augmented evaluation.

### Key Designs

1. **Data Collection and Annotation Pipeline**:

    - **Function**: Construct a 67.5-hour dataset of novice user GUI interactions.
    - **Mechanism**: 54 self-reported novice users (skill rating 1–5, mean 2.8) are recruited from the Prolific platform. Each user completes at least 20 minutes of open-ended tasks on a designated application (e.g., "create a PowerPoint self-introduction"), with simultaneous screen recording and think-aloud. The annotation pipeline uses WhisperX for speech transcription, Gemini-2.5-Pro to generate initial annotations, and human review (96.1% agreement on behavioral labels; 88.68% retention for intent; 78.89% for assistance).
    - **Design Motivation**: The natural behaviors of novice users—confusion, exploration, trial and error—represent the scenarios most critical for assistive agents to understand, yet no prior dataset exists for this purpose. Think-aloud protocols provide ground truth for otherwise latent user intentions.

2. **Nine-Category Behavioral State Taxonomy**:

    - **Function**: Provide a structured vocabulary for understanding user behavior.
    - **Mechanism**: Four major categories with nine subcategories—Planning (goal setting, task planning), Execution (executing actions, exploring and deciding), Problem-solving (confusion/help-seeking, debugging/correcting, frustration), and Evaluation (checking progress, refining work). Constructed via a multi-stage human–AI collaborative process involving five iterative rounds among three authors, independent generation by Gemini, and merged validation.
    - **Design Motivation**: The taxonomy aligns with Norman's Seven Stages of Action (planning → execution → evaluation) and Bloom's cognitive hierarchy, grounding behavioral distinctions in cognitive theory and providing precise behavioral signals for downstream assistance.

3. **Three-Tier Progressive Evaluation Framework**:

    - **Function**: Comprehensively evaluate models' capacity for user understanding and assistance.
    - **Mechanism**: (1) *Behavioral state detection*: a 9-class classification task in which the model infers the user's current behavioral state from video alone; (2) *Intent prediction*: a 4-option MCQ in which the model infers what the user is currently trying to achieve; (3) *Assistance prediction*: first a binary classification of whether help is needed, then a 4-option classification of what type of help is needed. Each task supports incrementally added context—prior behavioral state, current behavioral state, and behavioral state plus intent—to quantify the value of contextual information.
    - **Design Motivation**: The progressive design enables precise localization of model capability bottlenecks—whether the model fails to understand the behavior, cannot infer the intent, or cannot judge the assistance need. The context-augmentation experiments reveal the critical value of structured user understanding.

### Loss & Training
This paper presents an evaluation benchmark rather than a training method; no training is involved. Evaluation is conducted in a zero-shot setting with 8 MLLMs (Gemini-2.5-Flash/Pro, GPT-4o/mini, Claude-4.5-Sonnet, Qwen3-VL-8B, InternVideo2.5-8B, InternVL3-8B). Each video clip is uniformly sampled at 32 frames; audio is excluded to simulate realistic deployment conditions.

## Key Experimental Results

### Main Results — Accuracy by Task

| Model | Behavior Detection | Intent Prediction | Assistance Need Detection | Assistance Type Prediction |
|---|---|---|---|---|
| Claude-4.5-Sonnet | **44.61%** | **71.39%** | 39.49% | **55.00%** |
| Gemini-2.5-Pro | 42.44% | 67.80% | **69.82%** | 52.74% |
| GPT-4o | 36.32% | 61.19% | 49.69% | 45.95% |
| Qwen3-VL-8B | 37.97% | 62.70% | 52.83% | 46.06% |
| InternVL3-8B | 22.57% | 46.11% | 34.94% | 27.03% |

### Effect of Context Augmentation (Assistance Need Detection Accuracy)

| Model | No Context | +Behavior State | +Behavior+Intent | Gain |
|---|---|---|---|---|
| GPT-4o-mini | 46.05% | 78.92% | **82.26%** | +36.21pp |
| GPT-4o | 49.69% | **87.79%** | 87.91% | +38.22pp |
| Gemini-2.5-Pro | 69.82% | 84.73% | 82.38% | +14.91pp |
| Claude-4.5-Sonnet | 39.49% | 58.56% | 59.43% | +19.94pp |
| InternVideo2.5-8B | 34.36% | 35.35% | 35.25% | +0.89pp |

### Key Findings
- **Behavioral detection is the hardest task**: Even the strongest model achieves only 44.6%; the most common error is misclassifying "frustration/debugging" as "executing actions"—i.e., missing subtle signals that a user is struggling.
- **Intent prediction is relatively the easiest**, though performance drops markedly under the stricter MBAcc metric, suggesting models often guess correctly but inconsistently.
- **Structured context yields dramatic gains**: Providing behavioral state raises GPT-4o's assistance need detection F1 from 47.73 to 90.19.
- **Open-source small models lag significantly**: InternVideo2.5 and InternVL3 achieve near-zero recall on assistance need detection, misclassifying almost all cases requiring help as not needing it.
- **Online incremental setting**: Performance improves steadily as more video is observed (25%→100%), indicating that temporal context is important for understanding user behavior.

## Highlights & Insights
- **A shift in evaluation perspective**: From "can the model complete the task for the user" to "can the model understand what the user needs." This represents an important directional shift in GUI agent research. Existing benchmarks assume fixed goals, whereas real users' goals evolve dynamically.
- **Layered context augmentation design**: By incrementally adding behavioral and intent context, the paper precisely quantifies each layer of user understanding's contribution to the final assistance decision. GPT-4o-mini's jump from 46% to 82% demonstrates that the bottleneck is not model capability but model awareness of the user.
- **The unique value of novice user data**: Expert demonstrations capture the "correct way to do things," while novice behaviors capture "real-world need scenarios." This insight transfers to domains such as educational AI and medical AI that must understand non-expert users.

## Limitations & Future Work
- The 9-category taxonomy may lack sufficient granularity; for example, "exploring and deciding" covers an overly broad range of behaviors.
- Evaluation relies on 32-frame sampling, which may miss rapid actions or subtle hesitation signals.
- Think-aloud annotation depends on users' ability to verbalize their thoughts; users who struggle with self-expression may introduce annotation noise.
- Only offline inference capability is evaluated; the feasibility of real-time, online assistance scenarios is not verified.
- The dataset size (120 clips) may be insufficient for model training, though it is appropriate as an evaluation benchmark.

## Related Work & Insights
- **vs. VideoGUI/AssistGUI**: These works use expert instructional videos for closed-task automation; GUIDE uses natural novice behavior for open-ended user understanding—different in both objective and data source.
- **vs. ProactiveVA**: ProactiveVA addresses proactive assistance in visual analytics; GUIDE covers 10 general-purpose applications and is more broadly applicable but does not perform actual intervention.
- **vs. OSWorld/AndroidWorld**: These are fully automated GUI manipulation benchmarks; GUIDE emphasizes human-in-the-loop understanding and assistance rather than replacement.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First GUI benchmark focused on novice user behavior understanding; the three-tier progressive framework is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 8 models, multiple context configurations, and both online/offline settings; training experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear logic, rich tables, and theoretical grounding in Norman/Bloom frameworks adds persuasiveness.
- **Value**: ⭐⭐⭐⭐⭐ — Reveals substantial gaps in current MLLMs' user understanding and charts a clear direction for assistive agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Do MLLMs Capture How Interfaces Guide User Behavior? A Benchmark for Multimodal UI/UX Design Understanding](../../ACL2026/multimodal_vlm/do_mllms_capture_how_interfaces_guide_user_behavior_a_benchmark_for_multimodal_u.md)
- [\[AAAI 2026\] SToLa: Self-Adaptive Touch-Language Framework with Tactile Commonsense Reasoning in Open-Ended Scenarios](../../AAAI2026/multimodal_vlm/stola_self-adaptive_touch-language_framework_with_tactile_commonsense_reasoning_.md)
- [\[CVPR 2026\] ENC-Bench: A Benchmark for Evaluating MLLMs in Electronic Navigational Chart Understanding](enc-bench_a_benchmark_for_evaluating_multimodal_large_language_models_in_electro.md)
- [\[CVPR 2026\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_highfidelity_incontext_learning_for_multimo.md)
- [\[NeurIPS 2025\] MM-OPERA: Benchmarking Open-ended Association Reasoning for Large Vision-Language Models](../../NeurIPS2025/multimodal_vlm/mm-opera_benchmarking_open-ended_association_reasoning_for_large_vision-language.md)

</div>

<!-- RELATED:END -->
