---
title: >-
  [Paper Note] CHEER-Ekman: Fine-grained Embodied Emotion Classification
description: >-
  [ACL 2025][Robotics][Embodied Emotion Classification] This paper proposes the CHEER-Ekman dataset, extending the binary embodied emotion annotations of the CHEER dataset into Ekman's six basic emotions. It employs an LLM-based automatic Best-Worst Scaling (BWS) technique to achieve fine-grained emotion classification without task-specific training, outperforming supervised BERT.
tags:
  - "ACL 2025"
  - "Robotics"
  - "Embodied Emotion Classification"
  - "Ekman Emotions"
  - "Best-Worst Scaling"
  - "Large Language Models"
  - "Prompt Engineering"
date: 2026-05-08
content_hash: bd2eef0a28e89c7c
---

# CHEER-Ekman: Fine-grained Embodied Emotion Classification

**Conference**: ACL 2025  
**arXiv**: [2506.01047](https://arxiv.org/abs/2506.01047)  
**Code**: [https://github.com/menamerai/cheer-ekman](https://github.com/menamerai/cheer-ekman)  
**Area**: Robotics  
**Keywords**: Embodied Emotion Classification, Ekman Emotions, Best-Worst Scaling, Large Language Models, Prompt Engineering

## TL;DR

This paper proposes the CHEER-Ekman dataset, extending the binary embodied emotion annotations of the CHEER dataset into Ekman's six basic emotions. It employs an LLM-based automatic Best-Worst Scaling (BWS) technique to achieve fine-grained emotion classification without task-specific training, outperforming supervised BERT.

## Background & Motivation

**Background**: Emotions are not only abstract psychological states but are also deeply intertwined with bodily experiences—such as smiling when happy or experiencing an accelerated heart rate when afraid. This "embodied emotion" is understudied in NLP, with existing work primarily focusing on explicit emotion classification and sentiment analysis.

**Limitations of Prior Work**: Although the CHEER dataset (Zhuang et al., 2024) provides 7,300 human-annotated sentences expressing emotions through body parts, it only supports binary classification (whether a sentence contains embodied emotion) and cannot distinguish specific emotion types—for instance, whether a rapid heart rate represents fear or excitement.

**Key Challenge**: Fine-grained emotion annotation incurs high human costs, while directly using LLMs for zero-shot classification underperforms in instruction following, often leading to unstable and incorrect outputs.

**Goal**: (1) To construct a fine-grained embodied emotion classification dataset; (2) To find an effective classification method without requiring task-specific training.

**Key Insight**: Introduce Ekman's six basic emotions (Joy, Sadness, Anger, Disgust, Fear, Surprise) into embodied emotion recognition, and leverage automatic BWS technology from emotion intensity annotation to replace direct classification.

**Core Idea**: Use the comparative judgment mechanism of Best-Worst Scaling to enable LLMs to achieve unsupervised fine-grained embodied emotion classification that outperforms supervised BERT.

## Method

### Overall Architecture

This paper comprises three core components: (1) enhancing LLM's embodied emotion detection capability through prompt simplification and CoT; (2) constructing the CHEER-Ekman dataset; (3) utilizing the BWS framework for emotion classification.

### Key Designs

#### 1. Prompt Simplification

- **Function**: Rewriting the original technical prompts into simple, everyday language.
- **Mechanism**: Comparing the effects of the base prompt (original prompt from Zhuang et al., 2024) and the simplified prompt (reduced syntactic and lexical complexity) on Llama-3.1 and DeepSeek-R1. Deterministic outputs are obtained by comparing the logit probabilities of "True" and "False".
- **Design Motivation**: Technical definitions can create understanding barriers for LLMs; simplifying language reduces potential confusion.

#### 2. Chain-of-Thought (CoT) Prompting

- **Function**: Guiding models to understand mind-body relationships through multi-step reasoning.
- **Mechanism**: Designing three CoT variants: 2-step (assessing emotional causality and subconscious expression), 3-step (adding body part identification), and simplified 2-step (linguistically simplified 2-step).
- **Design Motivation**: Explicit causal reasoning helps smaller 8B models achieve performance close to that of 70B models.

#### 3. CHEER-Ekman Dataset Construction

- **Function**: Labeling the 1,350 positive samples of the CHEER dataset with Ekman's six basic emotions.
- **Mechanism**: Recruiting two annotators who are presented with sentences, associated body parts, and context (up to 3 preceding sentences) to choose the best-matching emotion. Cohen's Kappa agreement is 0.64.
- **Emotion Distribution**: Fear 24.7%, Joy 21.2%, Sadness 19.3%, Surprise 13.3%, Disgust 12.5%, Anger 9.0%.

#### 4. Best-Worst Scaling (BWS) Emotion Classification

- **Function**: Predicting emotions through comparative judgment instead of direct classification.
- **Mechanism**: Presenting 4-sentence tuples to the LLM and asking it to identify the instances that best and worst represent a given Ekman emotion. The score of each sentence for each emotion is calculated using the formula $\frac{\#Best - \#Worst}{\#Total}$, and the emotion with the highest score is selected as the prediction.
- **Design Motivation**: Comparative judgment is more stable than direct classification and bypasses LLM instruction-following issues. Instantiations from $2N$ to $72N$ tuples were tested.

### Loss & Training

This study uses a training-free method and does not involve a loss function. The key hyperparameter in BWS is the number of tuples; experiments show that $36N$ achieves optimal performance, after which plateauing occurs.

## Key Experimental Results

### Main Results

**Embodied Emotion Detection Task** (Binary classification, CHEER dataset, 7,300 sentences):

| Model | Macro F1 | EE F1 | Neutral F1 |
|------|----------|-------|------------|
| Llama-70B (base prompt) | 37.2 | 35.3 | 39.0 |
| Llama-70B (simple prompt) | 66.7 | 52.8 | 80.6 |
| DeepSeek-70B (base prompt) | 32.6 | 33.7 | 31.5 |
| DeepSeek-70B (simple prompt) | 74.2 | 58.9 | 89.5 |
| GPT-3.5 (base prompt) | 70.2 | 53.5 | 86.9 |
| BERT (fine-tuned) | 83.5 | 72.6 | 94.4 |

**Embodied Emotion Classification Task** (6-way classification, CHEER-Ekman dataset, 1,350 sentences):

| Model | Macro F1 | Joy | Sadness | Fear | Anger | Disgust | Surprise |
|------|----------|-----|---------|------|-------|---------|----------|
| Llama-8B (zero-shot) | 31.6 | 39.4 | 43.6 | 26.6 | 32.2 | 19.1 | 28.5 |
| DeepSeek-8B (zero-shot) | 28.4 | 43.3 | 35.7 | 33.1 | 23.1 | 14.8 | 20.2 |
| BWS 36N (Llama-8B) | **50.6** | 66.7 | 64.7 | 48.0 | 53.2 | 22.0 | 48.9 |
| BERT (supervised) | 49.6 | 68.2 | 57.5 | 50.1 | 30.2 | 56.1 | 35.7 |

### Ablation Study

**CoT's improvement effect on 8B models:**

| Model | Macro F1 |
|------|----------|
| DeepSeek-8B (2-step) | 52.2 |
| DeepSeek-8B (3-step) | 57.4 |
| DeepSeek-8B (2-step-simple) | 67.5 |
| Llama-8B (2-step) | 53.4 |
| Llama-8B (3-step) | 54.8 |
| Llama-8B (2-step-simple) | 60.1 |

**Ablation on BWS Tuple Count:**

| Tuple Count | Macro F1 |
|--------|----------|
| 4N | 41.8 |
| 12N | 44.6 |
| 36N | 50.6 |
| 48N | 49.8 |
| 72N | 49.5 |

### Key Findings

1. Simplifying prompts improved the F1 of Llama-70B by 29.5 points and DeepSeek-70B by 41.6 points.
2. CoT closed the gap between the 8B and 70B models to 6.7 F1 points.
3. BWS with 36N tuples achieved 50.6 F1, outperforming supervised BERT (49.6).
4. Error analysis indicates that 93.3% of errors are false positives, primarily arising from metaphorical expressions (41%), functional actions (42%), and non-action body part references (17%).

## Highlights & Insights

1. **Counter-intuitive Finding**: Prompts simplified into everyday language perform better than technically defined prompts (improving F1 by nearly 30 points), indicating that LLMs face significant difficulties in understanding domain terminology.
2. **Comparison Over Classification**: BWS bypasses the instability of direct LLM classification through a comparative mechanism, presenting an elegant engineering solution.
3. **Small Model Potential**: CoT combined with simplified prompts enables the 8B model to approach 70B-level performance, demonstrating the great value of prompt engineering.
4. **Dataset Value**: CHEER-Ekman fills the gap in fine-grained embodied emotion classification.

## Limitations & Future Work

1. The dataset contains only 1,350 sentences; this small scale may introduce positive sample bias.
2. Simplifying prompts may lead to overfitting on specific phrasing while neglecting more subtle metaphorical expressions.
3. The high tuple count in BWS brings significant computational overhead, limiting its scalability.
4. Due to context window constraints, the few-shot setting was not implemented.
5. Future work could explore finer-grained emotion classification taxonomies (e.g., 27 emotion categories).

## Related Work & Insights

- **Bagdon et al. (2024)**: Pioneering work on automatic BWS for emotion intensity annotation; this paper extends it to classification tasks.
- **CHEER (Zhuang et al., 2024)**: The original embodied emotion detection dataset and the basis of this work.
- **Ekman (1992)**: The six basic emotions framework, establishing the taxonomic foundation.
- **Insight**: The comparative judgment approach of BWS can be generalized to other tasks where LLMs struggle with direct classification.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Phoenix: A Motion-based Self-Reflection Framework for Fine-grained Robotic Action Correction](../../CVPR2025/robotics/phoenix_a_motion-based_self-reflection_framework_for_fine-grained_robotic_action.md)
- [\[AAAI 2026\] Cross Modal Fine-Grained Alignment via Granularity-Aware and Region-Uncertain Modeling](../../AAAI2026/robotics/cross_modal_fine-grained_alignment_via_granularity-aware_and_region-uncertain_mo.md)
- [\[CVPR 2025\] SortScrews: A Dataset and Baseline for Real-time Screw Classification](../../CVPR2025/robotics/sortscrews_a_dataset_and_baseline_for_real-time_screw_classification.md)
- [\[ECCV 2024\] LLM as Copilot for Coarse-Grained Vision-and-Language Navigation](../../ECCV2024/robotics/llm_as_copilot_for_coarse-grained_vision-and-language_navigation.md)
- [\[NeurIPS 2025\] PROFIT: A Specialized Optimizer for Deep Fine Tuning](../../NeurIPS2025/robotics/profit_a_specialized_optimizer_for_deep_fine_tuning.md)

</div>

<!-- RELATED:END -->
