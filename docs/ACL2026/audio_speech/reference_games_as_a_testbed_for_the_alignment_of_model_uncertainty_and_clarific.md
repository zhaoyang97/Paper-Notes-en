---
title: >-
  [Paper Note] Reference Games as a Testbed for the Alignment of Model Uncertainty and Clarification Requests
description: >-
  [ACL2026][Audio & Speech][reference game] This paper employs color-grid reference games to examine whether VLMs can translate internal uncertainty into appropriate clarification requests. It finds that even in highly con…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "reference game"
  - "clarification request"
  - "VLM calibration"
  - "pragmatic competence"
  - "uncertainty alignment"
date: 2026-05-08
content_hash: cab6cca782df2c78
---

# Reference Games as a Testbed for the Alignment of Model Uncertainty and Clarification Requests

**Conference**: ACL2026  
**arXiv**: [2601.07820](https://arxiv.org/abs/2601.07820)  
**Code**: https://github.com/Manarali-bit/reference-games-clarification  
**Area**: Multimodal Interaction / VLM Uncertainty / Clarification Requests  
**Keywords**: reference game, clarification request, VLM calibration, pragmatic competence, uncertainty alignment  

## TL;DR
This paper employs color-grid reference games to examine whether VLMs can translate internal uncertainty into appropriate clarification requests. It finds that even in highly controlled tasks, Qwen2.5-VL and GPT-5-mini still exhibit interaction gaps such as overconfidence, unstable clarification behavior, and low-quality clarification questions.

## Background & Motivation
**Background**: In human dialogue, the listener is not a passive recipient. When references are ambiguous, information is insufficient, or multiple candidates exist, the listener initiates repairs—such as asking clarification questions—to maintain mutual understanding.

**Limitations of Prior Work**: Large Language Models (LLMs) and Vision-Language Models (VLMs) can answer fluently but do not necessarily know when they are uncertain. Fluent output can mask understanding failures, leading users to believe the model is certain; meanwhile, the linguistic confidence expressed by models often fails to match their actual accuracy.

**Key Challenge**: In open dialogue, it is difficult to define "when clarification is mandatory" because the user's intent and the space of possible interpretations are not closed. Without a clear ground truth, it is hard to judge whether a model should have asked a question.

**Goal**: To identify a controllable, closed, and quantifiable test scenario to evaluate whether a VLM, acting as a listener, can recognize internal uncertainty and express it through appropriate clarification requests.

**Key Insight**: The authors choose reference games. This task features fixed candidate referents, clear targets, and difficulty conditions, making it straightforward to determine if a model selects the correct object and whether it should have queried difficult samples.

**Core Idea**: Extend the reference game from a "test of referential resolution capability" to a "test of the alignment between model uncertainty and clarification behavior," evaluated through baseline confidence, CR-rate, relaxed accuracy, and human-AI interaction experiments.

## Method
The study utilizes a color-grid reference game: each round presents three $3\times3$ color-block grids, where one is the target and two are distractors. A human speaker provides a description of the target, and the model, as the listener, must identify it. The authors design three experiments: a baseline requiring only target selection; a clarification experiment explicitly allowing questions when uncertain; and an interaction experiment where humans answer the model's questions to test if those clarifications are actually helpful.

### Overall Architecture
Data is sourced from the color-grid reference game by McDowell and Goodman (2019), comprising 197 games with 60 rounds each. Samples are categorized into far, split, and close difficulty levels based on the color similarity between targets and distractors. The authors test Qwen2.5-VL-7B, Qwen2.5-VL-72B, and GPT-5-mini. The Qwen models were run on the full dataset (with a 500-sample subset reported), while GPT-5-mini was evaluated only on the 500-round subset due to API costs (excluding 19 null answers).

### Key Designs
1. **Baseline Multi-sampling for Uncertainty Estimation**:
    - **Function**: Measures the model's referential resolution accuracy and internal consistency when no clarification opportunity is provided.
    - **Mechanism**: Each round is sampled 5 times, using majority voting as the predicted answer; baseline confidence is defined as the proportion of the majority answer across the 5 samples, resulting in possible values of $0.4, 0.6, 0.8, 1.0$.
    - **Design Motivation**: A single output cannot reflect model uncertainty; diverse sampling provides a simple yet interpretable consistency signal.

2. **Clarification Experiment**:
    - **Function**: Examines whether the model proactively issues clarification requests when uncertain.
    - **Mechanism**: The prompt is modified to explicitly instruct the model that it can ask questions if uncertain. Three metrics are evaluated: CR-Rate (proportion of clarification questions produced), Accuracy (considering only non-clarification responses), and Relaxed Accuracy (treating both correct answers and clarification requests as acceptable behaviors).
    - **Design Motivation**: If a model uses clarification reasonably, it should query more frequently on difficult, low-confidence samples and answer directly on high-confidence samples.

3. **Human-in-the-loop Interaction**:
    - **Function**: Determines whether the questions asked by the model actually assist in resolving references.
    - **Mechanism**: The authors manually inspect 116 clarification requests generated by Qwen2.5-VL-72B, labeling them as task-relevant or not. For relevant questions, humans provide a direct answer; for irrelevant ones, humans rewrite the original description. The model then re-answers based on the clarified dialogue, comparing accuracy and confidence before and after.
    - **Design Motivation**: A high CR-rate does not equate to strong interaction capability; a model might merely utter generic phrases like "please clarify" without identifying the specific ambiguity.

### Loss & Training
This study does not involve model training and belongs to evaluation research. The key "strategy" is the experimental protocol: confidence estimation via majority voting (5 samples) for the baseline, single-sample parsing for the clarification phase, and expert human response intervention for the evaluation of Qwen2.5-VL-72B during interaction.

## Key Experimental Results

### Main Results
| Model | Baseline Accuracy | Baseline Confidence | CR-Rate | Clarification Accuracy | Relaxed Accuracy |
|-------|-------------------|---------------------|---------|-------------------------|------------------|
| Qwen2.5-VL-7B | 0.53 (0.52 full) | 0.88 (0.87 full) | 0.0 (0.002 full) | 0.46 (0.42 full) | 0.46 (0.42 full) |
| Qwen2.5-VL-72B | 0.77 (0.71 full) | 0.91 (0.92 full) | 0.24 (0.24 full) | 0.73 (0.71 full) | 0.80 (0.78 full) |
| GPT-5-mini | 0.91 | 0.99 | 0.13 | 0.94 | 0.94 |
| Human | 0.92 full | N/A | N/A | N/A | N/A |

### Key Findings by Difficulty
| Model / Condition | Baseline Accuracy | CR-Rate | Relaxed Accuracy | Phenomenon |
|-------------------|-------------------|---------|------------------|------------|
| GPT-5-mini close | 0.87 | 0.17 | 0.92 | Queries more often in difficult conditions |
| GPT-5-mini far | 0.98 | 0.06 | 0.99 | Queries less often in simple conditions |
| Qwen2.5-VL-72B close | 0.68 (0.65 full) | 0.24 (0.23 full) | 0.71 (0.73 full) | CR-rate shows no significant change with difficulty |
| Qwen2.5-VL-7B ALL | 0.53 (0.52 full) | 0.0 (0.002 full) | 0.46 (0.42 full) | Almost never queries |

### Interaction Experiment
| Setting | Before Accuracy | After Accuracy | Before Confidence | After Confidence | Conclusion |
|---------|-----------------|----------------|-------------------|------------------|------------|
| Qwen-72B CR-only ALL | 0.776 | 0.741 | 0.871 | 0.902 | Accuracy dropped by 0.035 after human response |
| Qwen-72B full ALL | 0.767 | 0.759 | 0.914 | 0.921 | Accuracy dropped slightly (0.008), confidence rose slightly |
| Qwen-72B CR quality | 42% task-relevant | 58% not task-relevant | - | - | Most clarification questions failed to capture task-relevant ambiguity |

### Key Findings
- GPT-5-mini’s clarification behavior most closely resembles a rational pattern: higher CR-rate in difficult conditions, with non-clarification accuracy improving from 0.91 to 0.94.
- Qwen2.5-VL-72B asks questions, but its frequency of querying is inconsistently aligned with task difficulty and uncertainty.
- Qwen2.5-VL-7B rarely asks questions, indicating that "allowing questions" is insufficient to induce clarification behavior in smaller models.
- Interaction experiments demonstrate that clarification requests must be specific and task-relevant to be valuable; generalized template-style questions may increase confidence without improving accuracy.

## Highlights & Insights
- The contribution lies not in a new model but in the evaluation scenario design. Reference games transform the question of "whether to clarify" from a vague open dialogue into a measurable problem, making them highly suitable for evaluating pragmatic competence.
- Relaxed accuracy is a useful metric: it recognizes "querying when uncertain" as a successful interactive behavior rather than only rewarding direct answers.
- The manual analysis showing only 42% task-relevance is crucial, indicating that simply counting CR-rate overestimates model capability. The real challenge is asking questions that effectively narrow down the candidate space.

## Limitations & Future Work
- Human data comes from 60-round interaction dyads where participants build common ground; VLMs only see the initial description, which might put them at a comparative disadvantage.
- Baseline confidence is based on diversity sampling, which does not necessarily equal the model's internal probabilistic uncertainty. The authors note that while information-theoretic uncertainty estimates for Qwen-72B are more granular, they still fail to better align clarification behavior.
- Some human speaker descriptions in the data may be flawed; thus, baseline errors are not exclusively the fault of the listener model.
- Future work could extend to multi-round reference games, allowing the model to establish mutual understanding through its own clarifications rather than just evaluating the first round.

## Related Work & Insights
- **vs. Open-domain Clarification Studies**: Open dialogue makes it difficult to judge when clarification is needed; the closed candidate set in reference games defines clarification needs more cleanly.
- **vs. Uncertainty Quantification**: Entropy or sampling consistency can measure uncertainty, but this paper further asks whether the model can translate that uncertainty into interactive behavior.
- **vs. Referential Resolution Benchmarks**: Traditional reference games focus on selecting the correct target; this paper expands it to an evaluation of interaction quality involving "selection or questioning."
- **Insight**: When evaluating conversational models, the ability to "timely refuse to answer, counter-question, or clarify" should be included in task success metrics, rather than focusing solely on one-shot answer accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Using reference games as an uncertainty-clarification alignment test is insightful; the method is simple, but the problem definition is excellent.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers three models, three difficulty levels, and human-AI interaction; the model scale and task types could still be expanded.
- Writing Quality: ⭐⭐⭐⭐☆ The argumentation is clear, experimental metrics are easy to understand, and the discussion is balanced.
- Value: ⭐⭐⭐⭐☆ Highly valuable for building more reliable interactive VLMs, particularly highlighting that "knowing how to ask" is not the same as "knowing how to ask useful questions."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation](../../NeurIPS2025/audio_speech/model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)
- [\[ICLR 2026\] AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer](../../ICLR2026/audio_speech/ac-foley_reference-audio-guided_video-to-audio_synthesis_with_acoustic_transfer.md)
- [\[ACL 2026\] UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions](unisonate_a_unified_model_for_speech_music_and_sound_effect_generation_with_text.md)
- [\[ACL 2026\] ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment](immersivetts_environment-aware_text-to-speech_with_multimodal_diffusion_transfor.md)
- [\[ACL 2026\] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation](unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)

</div>

<!-- RELATED:END -->
