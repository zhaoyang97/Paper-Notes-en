---
title: >-
  [Paper Note] Vision-Language Models Mistake Head Orientation for Gaze Direction: Nonverbal Conversation Cues
description: >-
  [ACL 2026][Multimodal VLM][Paper Note] This paper uses 1,360 controlled real-world photos and pre-registered statistical tests to find that current VLMs perform much worse than humans in judging which object a person is looking at. They primarily mistake head orientation for gaze direction. Fine-tuning specialized gaze models can alleviate but cannot comple
tags:
  - ACL 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 5958152d3f2d9ee5
---
# Vision-Language Models Mistake Head Orientation for Gaze Direction: Nonverbal Conversation Cues

**Conference**: ACL2026  
**arXiv**: [2506.05412](https://arxiv.org/abs/2506.05412)  
**Code**: https://zoryzhang.github.io/gaze  
**Area**: Multimodal VLM  
**Keywords**: Gaze target inference, head orientation bias, nonverbal cues, VLM behavior evaluation, controlled experiments

## TL;DR
This paper uses 1,360 controlled real-world photos and pre-registered statistical tests to find that current VLMs perform much worse than humans in judging which object a person is looking at. They primarily mistake head orientation for gaze direction. Fine-tuning specialized gaze models can alleviate but cannot completely eliminate this bias.

## Background & Motivation
**Background**: Gaze is a critical nonverbal cue in human communication. Conversational robots, embodied intelligence, and multimodal assistants could better resolve references, understand intentions, and coordinate collaborative actions if they could determine which object a user is looking at. Specialized gaze estimation models have reached near-human levels on some internet image benchmarks, and VLMs possess cross-task universality, appearing to be an ideal vehicle for integrating gaze skills and linguistic reasoning.

**Limitations of Prior Work**: Existing VLM gaze evaluations mostly remain at the level of "can it answer correctly" and cannot explain why models make mistakes. Gaze images on the internet often contain shortcuts where head direction and eye direction are consistent; therefore, even if a model answers correctly, it might just be using head orientation or body direction without truly reading the eye's appearance.

**Key Challenge**: Applications require the model to understand "where the eye is actually looking," but training data and standard benchmarks may encourage models to learn the shortcut of "looking where the head points." Reporting accuracy alone cannot distinguish between true gaze inference and the head-orientation heuristic.

**Goal**: The authors aim to construct a controlled experiment to isolate the core capability of gaze target inference, compare the gap between VLMs and humans, determine if models exhibit head orientation bias, and judge through fine-tuning experiments whether this bias more likely stems from training data or model architecture.

**Key Insight**: The paper adopts a cognitive science-style experimental design: pre-registering the experimental protocol, controlling the number of objects, object distance, shooting perspective, gaze target, and head orientation, and using a set of statistical tests to classify model behavior into head-only, head-dominant, eye-head ambivalent, or more reliable gaze inference.

**Core Idea**: Instead of only using natural image benchmarks for VLMs, this study performs stress testing on whether models truly look at the eyes through systematic misalignments between eye direction and head direction.

## Method
The core contribution of this paper is the experimental design rather than the model algorithm. The authors first captured real-world photos: a person sitting at a table with 2 to 4 objects, looking at one of them. Crucially, the authors controlled the head orientation, making it sometimes aligned with the gaze direction, sometimes pointing at a distracting object, and sometimes naturally unconstrained. Consequently, if the model relies only on head orientation, it will systematically fail in the incongruent condition.

### Overall Architecture
The evaluation input consists of a real photo and a multiple-choice question asking, "Which object is this person looking at?", with options being the names of the objects on the table. The main experiment contains 1,360 photos, and each VLM views one of 16 prompt templates for each photo; thus, each VLM undergoes 21,760 trials. Main models tested include GPT-5.2, GPT-4o-2024-08-06, Qwen3-VL-30B-A3B-Instruct, InternLM-XComposer2-vl-7B, and GLM-4.6V, while adding Moondream2's specialized gaze function and the GazeLLE specialized vision model as controls.

Experimental variables include object quantity, object combination, shooting perspective, distance between objects, prompt template, stimulus ID, true GazeTarget, and HeadTarget. Dependent variables include the chosen object (Choice) and a more fine-grained "Wrongness." Wrongness is the relative distance between the chosen object and the true gaze target, ranging from 0 to 1, which reflects how far off the mistake is better than simple accuracy.

### Key Designs

**1. Controlled gaze/head separation in real photos: making the incongruent condition a causal probe for "whether the model truly looks at the eyes."**

Internet gaze images almost always have consistent head and eye directions; even if models answer correctly, it is difficult to distinguish whether they read the eyes or followed the head shortcut. This work uses real photography rather than synthetic images to create systematic misalignment: the Natural condition allows subjects to naturally turn their heads to the target, the Congruent condition requires both head and eyes to point at the same object, and the Incongruent condition fixes head orientation while only moving eyes to look elsewhere. In this way, when switching from congruent to incongruent, the only change in the scene is the eye appearance—if the model relies only on head orientation, it will inevitably fail systematically in the incongruent condition. The reason for choosing real photos over synthetic ones is that synthetic images might lack realistic lighting, shadows, and facial statistical patterns, while internet images cannot control head-eye misalignment; controlled photography satisfies both realism and causal interpretability.

**2. A test chain from overall gap to mechanism classification: using a series of statistical tests to translate "low accuracy" into specific bias types.**

Simply reporting accuracy cannot distinguish between "real gaze inference" and "head-only heuristics," and individual statistical results can easily be explained by other factors. Therefore, the paper designs a progressive chain of tests: Test 1 compares the overall gap between VLMs and humans; Head-Bias Test 2 includes four sub-tests checking if models more frequently choose the HeadTarget, whether they maintain their choice when only the eyes change, whether they perform better when head/gaze are consistent, and whether error distance increases as the distance between HeadTarget and GazeTarget increases; Test 3 checks if models respond to changes in eye targets; Test 4 further distinguishes whether the model oscillates randomly between gaze and head, or can only roughly lock onto the region between the two. Only when multiple independent tests point to head orientation bias is the conclusion sufficiently credible, allowing for specific improvement directions like "adding head-gaze inconsistent training samples" rather than vague statements about model failure.

**3. Strict response parsing and GLMM analysis: plugging leaks from prompts, option order, and individual stimuli.**

Behavioral evaluation is easily biased by prompt templates, option matching methods, and accidental factors of single images; concluding based on average accuracy is dangerous. For response parsing, models can either output option letters or give free-response answers; the system first uses rule matching, then Llama-3.1-70B for semantic matching, and finally manual review if still unparsed, ensuring that statistics reflect the model's actual choices rather than matching failures. For analysis, Generalized Linear Mixed Models (GLMM) are used, treating StimulusID, PromptID, and Objects as random effects to absorb individual differences in stimuli and templates, and treating View, Proximity, object quantity, and Condition as fixed effects or interaction terms to estimate the causal variables of interest. The resulting conclusion on head bias is thus valid after excluding these noise sources, rather than being a byproduct of a specific prompt style or option order.

### Loss & Training
VLMs in the main experiment are not trained, only performing zero-shot multiple-choice evaluation. The paper conducts an additional proof-of-concept fine-tuning experiment: dividing a total of 2,260 stimuli from the pilot and main study into training, validation, and test sets at a 7:1:2 ratio. GazeLLE is fine-tuned for 50 epochs, using the Euclidean distance between the predicted gaze point and the target object position as the training objective, with the learning rate decaying from $1e-3$ to $1e-6$ via a cosine schedule.

## Key Experimental Results

### Main Results
Main results show that humans significantly outperform all VLMs and specialized models in the same task. As the number of objects increases from 2 to 4, the accuracy of all models decreases, but most VLMs performance is close to the random baseline.

| Method | 2 Objects Acc | 3 Objects Acc | 4 Objects Acc | Note |
|--------|------|------|----------|------|
| Humans (n=59) | 94% | 88% | 76% | Humans still far exceed models even with low-resolution stimuli |
| GazeLLE-DinoV2-ViTL14 | 78% | 67% | 47% | Specialized gaze model superior to VLMs, but gap remains |
| Moondream2 (Hybrid) | 78% | 58% | 41% | Invoking specialized gaze function |
| GPT-5.2 | 64% | 46% | 31% | Significantly lower than humans, 4 objects near random |
| GPT-4o-20240806 | 65% | 41% | 30% | Close to GPT-5.2 |
| Qwen3-VL-30B | 59% | 39% | 28% | Significantly affected by object count |
| GLM-4.6V-Flash | 62% | 43% | 30% | Still lower than specialized model |
| InternLM-XComposer2-VL-7B | 64% | 43% | 29% | 4 objects near random |
| Guessing baseline | 50% | 33% | 25% | Multiple-choice random baseline |

Head bias tests further demonstrate that models are not simply "unable to understand objects" or "unable to answer multiple-choice questions," but systematically use head orientation as a gaze cue. The paper also ruled out several alternative explanations: increasing resolution to 896 or 1024 did not improve GPT-5.2; providing object names from left to right only brought small improvements of 2% and 1.5% for GPT-5.2 and GPT-4o; GPT-5.2 outputted letters perfectly, showing results were not caused by the option parsing process.

### Ablation Study

| Method | Test 2.1 | Test 3 | Test 4 | Behavior Class | Note |
|------|---------|------|------|------|------|
| GazeLLE | - | + | - | Head-only | Specialized model also heavily dependent on head orientation |
| Moondream2 | - | + | + | Head-dominant | Has gaze response, but head cues dominate |
| GPT-5.2 | + | + | - | Head-only | Strongest head bias, almost ignoring eye changes |
| GPT-4o | - | - | + | Head-dominant | More like oscillating between head and eyes, but head leads |
| Qwen3-VL-30B | - | + | - | Head-only | Insufficient response to eye changes |
| GLM-4.6V-Flash | - | - | + | Unclear | Bias exists but classification is unstable |
| InternLM | - | - | + | Head-dominant | Head cues dominate |

| GazeLLE Settings | Pilot Acc | Congruent Acc | Incongruent Acc | Natural Acc | Note |
|------|---------|------|------|------|------|
| Original GazeLLE (ViT-B/14) | 63.80 | 62.82 | 15.65 | 65.22 | Incongruent lower than chance, showing strong head shortcut |
| After fine-tuning | 85.77 ± 1.40 | 70.00 ± 5.01 | 34.15 ± 1.76 | 76.81 ± 2.90 | Significant improvement on anti-shortcut samples, but still lower than other conditions |

### Key Findings
- 94/111 pre-pilot VLMs did not significantly exceed random choice, indicating that gaze target inference is not a stable emergent capability of current VLMs.
- All primary tested VLMs show a Proximity effect: the closer the objects, the higher the Wrongness, indicating that the models are indeed using visual information rather than complete randomness or just linguistic priors.
- Except for GPT-5.2, most primary tested VLMs did not have stable View effects like humans, indicating they are not sensitive enough to eye appearance changes in profile views.
- Fine-tuning GazeLLE improved Incongruent accuracy from 15.65% to 34.15%, supporting "lack of head-gaze inconsistent cases in training data" as one of the primary reasons.

## Highlights & Insights
- The most valuable part of this paper is moving from benchmarking to mechanism diagnosis. It is not satisfied with showing that VLM gaze is poor but locates the error to head orientation bias through a series of experiments.
- The choice of controlled real photos is clever. Synthetic images might introduce generation artifacts, and internet images cannot control head-eye misalignment; controlled photography keeps the conclusion close to the visual system issue itself.
- Wrongness is more suitable for this task than accuracy. Choosing an adjacent object and choosing the furthest object are not the same kind of error; Wrongness can reflect the spatial precision of gaze inference.
- Fine-tuning experiments provided very practical improvement directions: collecting or constructing enough head, body, and eye inconsistent training samples may be more effective than simply increasing model scale.

## Limitations & Future Work
- The Incongruent condition is a necessary causal probe, but the frequency of strong head-eye misalignment in reality is limited; slight misalignment is more natural, while extreme misalignment is more like a stress test.
- Character diversity in the main experiment is limited (two females in the pilot, one male in the main study). While beneficial for controlling variables, it cannot cover different face shapes, eye makeup, skin tones, glasses, etc.
- The task is isolated multiple-choice gaze target inference, not including context, dialogue history, and joint attention in real interactions.
- The scale of data used for specialized model fine-tuning was small, which only shows that data can alleviate bias but doesn't prove all architectures can completely solve the problem with this type of data.
- Future work should test more open environments, more characters, multi-round interactions, and downstream reference resolution to see if gaze capability can migrate to real human-computer collaboration.

## Related Work & Insights
- **vs GazeFollow / in-the-wild gaze benchmarks**: Head direction in internet images is often sufficient to predict the answer; models might achieve high scores via shortcuts. This paper exposes real eye-reading capabilities through controlled incongruent cases.
- **vs specialized gaze estimation models**: Models like GazeLLE undergo supervised training but also show head bias, indicating the problem is not just VLM architecture but related to training distributions.
- **vs Conventional VLM capability evaluation**: Conventional evaluations mostly look at average accuracy; this paper shows that hypothesis-driven behavioral probing is more helpful for understanding model mechanisms.
- **Inspiration for follow-up work**: Multimodal evaluation can draw on cognitive science experiments, systematically decoupling key visual cues rather than just stacking larger natural image datasets.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Using pre-registered controlled experiments to locate VLM gaze error mechanisms is highly distinct.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes pre-pilot, human control, main experiment, multiple models, multiple statistical tests, and fine-tuning verification.
- Writing Quality: ⭐⭐⭐⭐ The experimental logic is clear, but the statistical test chain is long and requires patience to follow.
- Value: ⭐⭐⭐⭐⭐ Very valuable for VLM visual understanding evaluation, robotic reference resolution, and nonverbal social cue modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

<!-- RELATED:END -->

## Related Papers

- [\[CVPR 2026\] StructXLIP: Enhancing Vision-Language Models with Multimodal Structural Cues](../../CVPR2026/multimodal_vlm/structxlip_enhancing_vision-language_models_with_multimodal_structural_cues.md)
- [\[ACL 2025\] Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues](../../ACL2025/multimodal_vlm/speaking_beyond_language.md)
- [\[CVPR 2026\] Direction-aware 3D Large Multimodal Models](../../CVPR2026/multimodal_vlm/direction-aware_3d_large_multimodal_models.md)
- [\[ICCV 2025\] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](../../ICCV2025/multimodal_vlm/multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)
- [\[ICLR 2026\] Procedural Mistake Detection via Action Effect Modeling](../../ICLR2026/multimodal_vlm/procedural_mistake_detection_via_action_effect_modeling.md)

</div>

<!-- RELATED:END -->
