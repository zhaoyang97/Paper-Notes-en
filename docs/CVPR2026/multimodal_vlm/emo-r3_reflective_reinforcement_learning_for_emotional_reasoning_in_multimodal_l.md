---
title: >-
  [Paper Note] EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning in Multimodal Large Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Emotional Reasoning] This paper proposes EMO-R3, which guides MLLMs to perform step-by-step emotional reasoning via Structured Emotional Thinking (SET)…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Emotional Reasoning"
  - "GRPO"
  - "Structured Thinking"
  - "Reflective Reward"
  - "Multimodal Affective Understanding"
date: 2026-05-08
content_hash: 597e6284f488c633
---

# EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning in Multimodal Large Language Models

**Conference**: CVPR 2026
**arXiv**: [2602.23802](https://arxiv.org/abs/2602.23802)  
**Code**: [GitHub](https://github.com/xiaomi-research/emo-r3)  
**Area**: Multimodal VLM
**Keywords**: Emotional Reasoning, GRPO, Structured Thinking, Reflective Reward, Multimodal Affective Understanding

## TL;DR

This paper proposes EMO-R3, which guides MLLMs to perform step-by-step emotional reasoning via Structured Emotional Thinking (SET), and introduces a Reflective Emotional Reward (RER) that prompts the model to re-evaluate the visual-textual consistency and emotional coherence of its reasoning, substantially improving both interpretability and accuracy in multimodal affective understanding.

## Background & Motivation

**Limitations of MLLMs in affective understanding**: Although MLLMs demonstrate strong performance in visual reasoning, they remain inadequate at capturing the complexity and subjectivity inherent in human emotion.

**Limitations of SFT-based approaches**: Supervised fine-tuning methods for emotion recognition (e.g., EmoVIT, EmotionLLaMA) are constrained by fixed label taxonomies and limited category sets, leading to poor generalization and overfitting to the training distribution.

**Mismatch of generic GRPO**: While GRPO improves generalization, its thinking process is not tailored for emotional reasoning — reasoning trajectories lack tight correspondence with final answers (unlike mathematical reasoning, where a single error propagates to an incorrect result).

**Unique characteristics of emotional understanding**: Emotional reasoning is highly subjective and context-dependent; reasoning paths may diverge from final answers due to individual differences, and constraining only the answer is insufficient to guide the reasoning process.

**Think-answer decoupling phenomenon**: Experiments reveal that re-inferring the think text from GRPO rollout samples frequently yields emotional conclusions inconsistent with the final answers.

**Necessity of Cold Start**: The emotional priors of pretrained MLLMs may be misaligned with downstream label systems, necessitating lightweight alignment.

## Method

### Overall Architecture

EMO-R3 = Structured Emotional Thinking (SET) prompt + Reflective Emotional Reward (RER) + GRPO optimization. An optional Cold-Start-Emo stage provides lightweight SFT pre-alignment.

### Key Designs

#### Structured Emotional Thinking (SET)

Reasoning is constrained into three explicit stages, simulating the human affective cognition process:

1. **Emotional Trigger Identification** (Step 1): Detect elements in the scene that may elicit emotional responses (objects, actions, environments, facial expressions).
2. **Human Emotional Reflection** (Step 2): Describe the emotional reactions of a human observer to those identified elements.
3. **Emotional Conclusion** (Step 3): Determine the overall emotional polarity (positive/negative) and arousal level.

Output format: $o = \{s_1, s_2, s_3, \hat{\mathcal{E}}\}$, with the final answer $\hat{\mathcal{E}}$ enclosed in `\boxed{}`.

#### Reflective Emotional Reward (RER)

Two reflective rewards constrain reasoning quality:

- **Visual-Textual Consistency Reward** $\mathcal{R}_{\text{cons}}$: The output $s_1$ from Step 1 is fed back to the model together with the image, with the query "Can the following text describe this image?" Yes = 1, No = 0.
- **Emotional Coherence Reward** $\mathcal{R}_{\text{coh}}$: The outputs from Steps 1 and 2 are fed back to the model, with the query "Which emotion best describes the above text?" A match with the ground-truth label yields a score of 1.

$$\mathcal{R}_{\text{RER}} = \frac{\mathcal{R}_{\text{cons}} + \mathcal{R}_{\text{coh}}}{2}$$

### Loss & Training

Overall reward: $\mathcal{R}_{\text{overall}} = (1-\lambda_1-\lambda_2)\mathcal{R}_{\text{acc}} + \lambda_1 \mathcal{R}_{\text{RER}} + \lambda_2 \mathcal{R}_{\text{format}}$

Optimization is performed within the GRPO framework using intra-group relative advantage normalization.

## Key Experimental Results

### Main Results: Qwen2.5-VL-3B Emotional Reasoning (In-Domain / Out-of-Domain)

| Method | EmoSet (in-domain) | Emotion6 (out-of-domain) | WebEmo (out-of-domain) | Overall $\mathcal{A}$ |
|---|---|---|---|---|
| Vanilla* | 51.55 | 50.00 | 40.65 | 47.40 |
| SFT | 77.15 | 34.51 | 17.75 | 43.84 |
| GRPO (G=4) | 74.60 | 60.10 | 49.50 | 59.97 |
| DAPO (G=4) | 68.99 | 56.90 | 49.80 | 58.28 |
| **EMO-R3 (G=4)** | **75.50** | **60.44** | **50.45** | **60.50** |
| **EMO-R3 (G=8)** | **76.40** | **59.26** | **49.70** | **60.42** |

### Ablation Study

| Component | Effect |
|---|---|
| SFT (Cold-Start-Emo) | High in-domain performance but severe out-of-domain degradation |
| GRPO only | Good generalization but reasoning quality not guaranteed |
| + SET | Structured reasoning with improved emotional coherence |
| + RER | Significant improvement in visual and emotional consistency of reasoning |
| + Cold-Start-Emo | Alleviates reward sparsity and stabilizes training |

### Key Findings

- SFT achieves high in-domain performance (77.15) but suffers severe out-of-domain degradation (17.75), confirming overfitting.
- EMO-R3 outperforms both GRPO and DAPO across all settings.
- The reflective reward effectively constrains the reasoning process rather than merely the final answer.
- Cold-Start-Emo requires neither chain-of-thought annotations nor large data volumes.

## Highlights & Insights

- **First systematic attempt to adapt GRPO for affective understanding**, revealing the insufficiency of generic RL on subjective tasks.
- The SET design elegantly simulates the human cognitive chain of "perception → reaction → judgment."
- RER cleverly leverages the model's own capabilities to assess reasoning quality without external annotation.
- Analysis of the think-answer decoupling phenomenon opens a new research perspective for affective AI.
- The design motivation of Cold-Start-Emo is clearly articulated — aligning emotional priors rather than enhancing reasoning capacity.

## Limitations & Future Work

- RER requires additional forward passes through the model, increasing training overhead.
- Emotion labels remain discrete categories, leaving the continuous and multi-dimensional nature of emotion unmodeled.
- Validation is limited to Qwen2.5-VL-3B; performance on larger models remains to be confirmed.
- The three-step structured thinking may not cover all emotional reasoning scenarios.

## Related Work & Insights

- Both R1-Omni and EMO-R3 apply GRPO to the affective domain, but EMO-R3 more deeply adapts the reasoning process itself.
- Relation to DeepSeek-R1: EMO-R3 inherits the GRPO framework while introducing essential modifications for subjective tasks.
- The design philosophy of RER is generalizable to other subjective evaluation tasks (e.g., aesthetic assessment, subjective quality estimation).

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)
- [\[CVPR 2026\] Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory](evolving_contextual_safety_in_multi-modal_large_language_models_via_inference-ti.md)
- [\[ICCV 2025\] DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding](../../ICCV2025/multimodal_vlm/docthinker_explainable_multimodal_large_language_models_with.md)
- [\[CVPR 2026\] Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)
- [\[CVPR 2026\] Nano-EmoX: Unifying Multimodal Emotional Intelligence from Perception to Empathy](nano-emox_unifying_multimodal_emotional_intelligence_from_perception_to_empathy.md)

</div>

<!-- RELATED:END -->
