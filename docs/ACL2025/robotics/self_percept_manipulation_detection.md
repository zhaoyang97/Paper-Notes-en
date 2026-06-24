---
title: >-
  [Paper Note] SELF-PERCEPT: Introspection Improves LLMs' Detection of Multi-Person Mental Manipulation in Conversations
description: >-
  [ACL 2025][Robotics][mental manipulation detection] This paper proposes the SELF-PERCEPT two-stage prompting framework, drawing on psychological Self-Perception Theory. It guides LLMs to first observe the behavioral cues of conversational participants before inferring their internal attitudes, significantly improving the detection of mental manipulation in multi-person, multi-turn dialogues.
tags:
  - "ACL 2025"
  - "Robotics"
  - "mental manipulation detection"
  - "multi-turn dialogue"
  - "multi-person conversation"
  - "prompting framework"
  - "Self-Perception Theory"
date: 2026-05-08
content_hash: d495c9c4a7386c19
---

# SELF-PERCEPT: Introspection Improves LLMs' Detection of Multi-Person Mental Manipulation in Conversations

**Conference**: ACL 2025  
**arXiv**: [2505.20679](https://arxiv.org/abs/2505.20679)  
**Code**: [https://github.com/danushkhanna/self-percept](https://github.com/danushkhanna/self-percept)  
**Area**: Robotics  
**Keywords**: mental manipulation detection, multi-turn dialogue, multi-person conversation, prompting framework, Self-Perception Theory

## TL;DR

This paper proposes the SELF-PERCEPT two-stage prompting framework, drawing on psychological Self-Perception Theory. It guides LLMs to first observe the behavioral cues of conversational participants before inferring their internal attitudes, significantly improving the detection of mental manipulation in multi-person, multi-turn dialogues.

## Background & Motivation

**Importance of Mental Manipulation Detection**: Mental manipulation is a covert abusive behavior in interpersonal communication that controls others' thoughts and emotions through deceptive means for personal gain, posing a severe threat to the victims' mental health.

**Limitations of Prior Work**: The representative existing dataset, MentalManip, is based on movie scripts (Cornell Movie Corpus) and only covers **two-person dialogues** with unbalanced distribution, making it difficult to reflect complex manipulation scenarios of multi-person games in the real world.

**Key Challenge**: In reality, manipulation often occurs in group consultations (e.g., team meetings, social situations), involving multiple participants and multi-turn interactions. LLM performance in detecting manipulation in such scenarios remains severely inadequate.

**Defects of Traditional Prompting Methods**: Methods like Zero-shot, Few-shot, and CoT focus on step-by-step reasoning but struggle to capture implicit manipulation signals such as inconsistency between words and actions. They perform poorly, especially when distinguishing between benign persuasion and malicious manipulation.

**Key Insight**: Self-Perception Theory (SPT) suggests that individuals infer their internal attitudes by observing their own behavior. This cognitive mechanism can be transferred to LLMs, equipping them with the analytical capability to "first observe behavior, then infer intent."

**Goal**: Build a multi-person manipulation detection dataset closer to real-world scenarios and design a psychologically-inspired prompting framework to improve the detection accuracy of LLMs on manipulation behaviors in complex dialogues.

## Method

### Overall Architecture

SELF-PERCEPT is a two-stage prompting framework that simulates the human cognitive process of "behavioral observation $\rightarrow$ self-inference." In contrast to CoT, which focuses on step-by-step logical reasoning, SELF-PERCEPT explicitly decouples behavioral cue extraction from attitude inference. This allows LLMs to better handle complex social dynamics in multi-person interactions.

### Module 1: MultiManip Dataset Construction

- **Data Source**: Multi-person, multi-turn dialogues are extracted from the public transcripts of the Fandom for the reality show *Survivor*, where the competitive nature guarantees rich samples of manipulative behaviors.
- **Scale and Distribution**: A total of 220 dialogues with a **balanced distribution of manipulative/non-manipulative samples**, covering 11 manipulation techniques (e.g., accusation, shaming, denial, feigning innocence, etc.).
- **Annotation Process**: Five annotators answered two questions: $\mathcal{Q}_1$ (presence of manipulation, binary classification) and $\mathcal{Q}_2$ (manipulation types, multi-label). Using majority voting aggregation, Fleiss' Kappa = 0.429 (moderate agreement), reflecting the inherent challenge of the task.
- **Preprocessing**: Llama-3.1-70B was used for initial filtering, followed by cross-validation with GPT-4o/Llama-3.1-8B, and final manual verification. This multi-model strategy helps mitigate LLM biases.

### Module 2: Stage 1 — Self-Percept (Behavioral Observation)

- Given the complete multi-person dialogue as input, the LLM is required to comprehensively observe and analyze the **verbal cues** and **non-verbal cues** of each participant.
- Focusing on identifying **inconsistencies between words and actions**: For instance, a speaker vocally agreeing but using a sighing tone, which may imply passive-aggressive intent.
- The output is a structured behavioral observation list that records potential contradictions and suspicious manipulation signals, serving as the foundation for the next stage of inference.

### Module 3: Stage 2 — Self-Inference (Attitude Inference)

- Based on the behavioral observations from Stage 1, the LLM infers the **internal attitudes and beliefs** of each participant.
- Paying special attention to whether manipulative behaviors exist, and classifying them according to the 11 predefined manipulation types.
- The output is a concise inferential conclusion aimed at capturing the essence of interpersonal dynamics.

### Evaluation Strategy

- Evaluated on MultiManip (this paper's dataset) and MentalManip (existing dataset).
- Models: GPT-4o, Llama-3.1-8B.
- Baselines: Zero-Shot, Few-Shot, Chain-of-Thought.
- Metrics: Accuracy, Precision ($P$), Recall ($R$), Macro $F_1$.

## Key Experimental Results

### Table 1: Multi-Label Manipulation Detection on the MultiManip Dataset

| Model | Prompting | Acc. | $P$ | $R$ | $F_1$ |
|------|-----------|------|-----|-----|-------|
| GPT-4o | Zero-Shot | 0.27 | 0.20 | 0.31 | 0.16 |
| GPT-4o | Few-Shot | 0.39 | 0.19 | 0.21 | 0.22 |
| GPT-4o | CoT | 0.34 | 0.21 | 0.32 | 0.34 |
| GPT-4o | **SELF-PERCEPT** | **0.42** | **0.31** | 0.20 | **0.37** |
| Llama-3.1-8B | Zero-Shot | 0.11 | 0.09 | 0.37 | 0.29 |
| Llama-3.1-8B | Few-Shot | 0.22 | 0.17 | 0.36 | 0.13 |
| Llama-3.1-8B | CoT | 0.28 | 0.23 | 0.26 | 0.10 |
| Llama-3.1-8B | **SELF-PERCEPT** | **0.30** | 0.17 | 0.26 | **0.34** |

### Table 2: Multi-Label Manipulation Detection on the MentalManip Dataset

| Model | Prompting | Acc. | $P$ | $R$ | $F_1$ |
|------|-----------|------|-----|-----|-------|
| GPT-4o | Zero-Shot | 0.11 | 0.30 | 0.62 | 0.38 |
| GPT-4o | Few-Shot | 0.22 | 0.39 | 0.53 | 0.39 |
| GPT-4o | CoT | 0.35 | 0.37 | 0.56 | 0.43 |
| GPT-4o | **SELF-PERCEPT** | **0.45** | 0.34 | 0.55 | **0.47** |
| Llama-3.1-8B | Zero-Shot | 0.02 | 0.11 | 0.56 | 0.17 |
| Llama-3.1-8B | Few-Shot | 0.04 | 0.07 | 0.35 | 0.11 |
| Llama-3.1-8B | CoT | 0.19 | 0.14 | 0.38 | 0.18 |
| Llama-3.1-8B | **SELF-PERCEPT** | **0.23** | **0.21** | 0.32 | **0.19** |

### Key Findings

1. **Consistent Advantage of SELF-PERCEPT**: Across both datasets and models, SELF-PERCEPT achieves the best performance in Accuracy and $F_1$, demonstrating robust generalization across datasets and models.
2. **Precision-Recall Trade-off**: The Recall of SELF-PERCEPT is slightly lower than that of Zero-Shot and CoT, but its Precision is significantly improved (e.g., GPT-4o on MultiManip achieves $P$ = 0.31 compared to CoT's 0.21). This indicates that the behavioral observation stage effectively reduces false positives.
3. **SHAP Interpretability Analysis**: Stage 1 correctly captures words relating to psychological pressure and persuasive intent such as "anxious," "situation," and "teamwork" (negative SHAP values $\rightarrow$ classified as manipulation), whereas CoT relies excessively on neutral words like "game" and "desire," leading to false negatives (misclassification as non-manipulative).
4. **Limited Absolute Performance**: The highest $F_1$ score is only 0.47 (GPT-4o + SELF-PERCEPT on MentalManip), indicating that multi-person multi-turn manipulation detection remains an extremely challenging task.

## Highlights & Insights

- **Psychology-driven prompting design**: Translates Self-Perception Theory into an operational two-stage prompting process, offering an elegant transfer paradigm from "domain theory $\rightarrow$ NLP methods."
- **Decoupling of behavioral observation and intent inference**: Resolves the "one-step" reasoning bottleneck of CoT when processing implicit social signals, improving interpretability through explicit intermediate representations.
- **Practical value of the MultiManip dataset**: Built on reality shows rather than fictional scripts, the multi-person, multi-turn design is closer to real manipulation scenarios, filling a crucial data gap in this field.
- **SHAP visualization enhances credibility**: Visually demonstrates the attention differences between SELF-PERCEPT and CoT through word-level attribution comparisons, intuitively proving the method's effectiveness.

## Limitations & Future Work

- **Extremely small dataset scale**: With only 220 samples, it lacks sufficient statistical power, making it difficult to rigorously verify the significance of performance gains (e.g., $+5\%$ $F_1$).
- **Single domain**: Data is sourced solely from the reality show *Survivor*. The manipulation patterns in competitive dialogues might not represent daily scenarios (e.g., workplace gaslighting, relationship manipulation).
- **Lack of fine-tuning experiments**: The framework is only validated at the inference-time prompting level, without exploring the feasibility of distilling SELF-PERCEPT's behavioral analysis capabilities into smaller models.
- **Limited evaluation metrics**: Only standard classification metrics are reported, without evaluating fine-grained performance at the manipulation *type* level (e.g., which manipulation techniques are easier or harder to detect).
- **Recall loss**: The improvement in Precision comes at the cost of a decrease in Recall, which is suboptimal for security scenarios where false negatives are highly critical.

## Related Work & Insights

- **Mental Manipulation Detection**: MentalManip (Wan et al., 2024) is the first mental manipulation dataset but is limited to two-person dialogues; Ma et al. (2024) used intent-aware prompting for binary classification but ignored multi-person scenarios.
- **Toxic Language Detection**: DeTexD (Yavnyi et al., 2023) and troll tweets detection (Miao et al., 2020) focus on explicit toxicity, showing insufficient coverage of implicit manipulation.
- **Multi-Turn Dialogue Understanding**: Li et al. (2022) and Yang et al. (2022) advanced multi-turn dialogue modeling but did not specifically target manipulative behaviors.
- **LLM prompting**: Chain-of-Thought (Wei et al., 2022) is a milestone work; SELF-PERCEPT builds on this by introducing psychological priors to achieve task specialization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The approach of transferring Self-Perception Theory to prompting is novel, and the two-stage behavior-inference decoupling design is original.
- **Technical Quality**: ⭐⭐⭐ — The methodology is clear, but essentially boils down to prompt engineering with no model-level innovation; the dataset scale is relatively small.
- **Value**: ⭐⭐⭐⭐ — Mental manipulation detection is of high social security significance, and the framework can be directly applied to content moderation on online platforms.
- **Writing Quality**: ⭐⭐⭐⭐ — Compelling motivation with a natural introduction of psychological theory; the experimental analysis features SHAP visualization to enhance persuasion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Vulnerability of LLMs to Vertically Aligned Text Manipulations](vulnerability_of_llms_to_vertically_aligned_text_manipulations.md)
- [\[ACL 2025\] Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context](dice_idiomaticity.md)
- [\[ICLR 2026\] Self-Refining Vision Language Model for Robotic Failure Detection and Reasoning](../../ICLR2026/robotics/self-refining_vision_language_model_for_robotic_failure_detection_and_reasoning.md)
- [\[ACL 2026\] Cultivating Forensic Reasoning for Generalizable Multimodal Manipulation Detection](../../ACL2026/robotics/cultivating_forensic_reasoning_for_generalizable_multimodal_manipulation_detecti.md)
- [\[NeurIPS 2025\] AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling](../../NeurIPS2025/robotics/autotom_scaling_model-based_mental_inference_via_automated_agent_modeling.md)

</div>

<!-- RELATED:END -->
