---
title: >-
  [Paper Note] BQA: Body Language Question Answering Dataset for Video Large Language Models
description: >-
  [ACL 2025][NLP Understanding][Body language] Based on the BoLD dataset, BQA is constructed via a four-step semi-automatic pipeline. BQA is a body language emotion recognition multiple-choice QA benchmark containing 7,632 short videos. Evaluation reveals that the strongest VideoLLMs (GPT-4o/Gemini) achieve an accuracy of only about 60%, which is far below human performance (85%). Furthermore, it exposes the models' over-reliance on facial expressions and significant biases tow…
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "Body language"
  - "VideoLLM"
  - "Emotion recognition"
  - "Multiple-choice QA"
  - "Social bias analysis"
  - "Multimodal CoT"
date: 2026-05-08
content_hash: 55b1f572342ee1c0
---

# BQA: Body Language Question Answering Dataset for Video Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2410.13206](https://arxiv.org/abs/2410.13206)  
**Code/Data**: [HuggingFace](https://huggingface.co/datasets/naist-nlp/BQA)
**Institution**: Nara Institute of Science and Technology (NAIST)
**Area**: NLP Understanding  
**Keywords**: Body language, VideoLLM, Emotion recognition, Multiple-choice QA, Social bias analysis, Multimodal CoT

## TL;DR

Based on the BoLD dataset, BQA is constructed via a four-step semi-automatic pipeline. BQA is a body language emotion recognition multiple-choice QA benchmark containing 7,632 short videos. Evaluation reveals that the strongest VideoLLMs (GPT-4o/Gemini) achieve an accuracy of only about 60%, which is far below human performance (85%). Furthermore, it exposes the models' over-reliance on facial expressions and significant biases towards specific racial groups.

## Background & Motivation

**Background**: Video Large Language Models (VideoLLMs) have made significant progress in tasks like video summarization and video question answering, demonstrating their ability to integrate multimodal inputs to understand video content. In human communication, a vast amount of information relies on non-verbal cues—facial expressions, eye contact, and body language. These non-verbal communications lack formal rules and require complex reasoning based on common sense.

**Limitations of Prior Work**: Prior work (e.g., SMILE) only investigated the detection of a single emotion (smiling), which fails to evaluate the models' generalized understanding of the full spectrum of human emotions. Although the BoLD dataset annotates 26 emotional categories, its goal was to directly predict emotion intensity values, which is unsuitable for evaluating VideoLLMs via natural language prompting.

**Key Challenge**: If VideoLLMs cannot accurately interpret human emotions from body language, they will find it difficult to be applied in future scenarios requiring emotional perception, such as dialogue systems and AI robotics. A model's misunderstanding of humans' unconscious movements can lead to severe interaction failures.

**Goal**: To construct a systematic evaluation benchmark to comprehensively test the capability of VideoLLMs in understanding emotions conveyed through body language, and to deeply analyze the models' biases across different demographic dimensions.

**Key Insight**: Converting the existing BoLD pose estimation dataset into a multiple-choice QA format suitable for VideoLLM evaluation, leveraging Gemini for semi-automatic dataset construction, and revealing the models' bias patterns through multi-dimensional (gender/age/race) error analysis.

**Core Idea**: Transforming body language videos annotated with 26 emotion labels into a high-quality 4-choice QA benchmark via a four-step semi-automatic pipeline, demonstrating that body language understanding remains a major challenge for current VideoLLMs.

## Method

### Overall Architecture

The construction of the BQA dataset is based on BoLD (Body Language Dataset), which contains 9,876 video clips (each about 5 seconds, 25fps, ~125 frames) clipped from 150 movies, annotated with 26 emotion labels on a 10-point scale by crowd-sourced workers. BQA converts BoLD into a multiple-choice QA format via a four-step pipeline:

1. **STEP1 Option Extraction**: The 26 emotion labels are categorized into four major groups based on psychological theory (James 1890): Happiness, Anger, Sadness, and Pleasure. The correct answer is selected as the label with the highest emotion score. The three distractors are chosen from each of the other three groups to ensure maximum distinguishability among the options.
2. **STEP2 Question Generation**: The four candidate options and the corresponding video are input into Gemini-1.5-pro to generate natural language questions such as "What emotion does the person in the video exhibit?", drawing inspiration from the prompt design of mCSQA.
3. **STEP3 Quality Filtering**: Gemini is utilized to inspect whether the generated questions are sufficiently objective and whether they contain leak cues pointing to the correct answer (such as "He looks shocked"), while filtering out questions with harmful content.
4. **STEP4 Difficulty Annotation**: Gemini is prompted to answer the constructed QAs. Correctly answered ones are labeled as "Easy", while incorrect ones are labeled as "Hard", facilitating subsequent fine-grained analysis.

The final dataset is split into training (4,651), validation (1,538), and testing (1,443) sets with a 6:2:2 ratio. All audio is removed to ensure a pure evaluation of visual understanding capabilities.

### Key Designs 1: Cross-Group Stratified Sampling Strategy for Emotions

Traditional multiple-choice QA benchmarks often select distractors randomly. However, in emotion recognition, the variances between subcategories within the same major emotional group (e.g., Confidence and Affection under Happiness) are extremely small, rendering random selection ineffective for robust evaluation. BQA adopts cross-group stratified sampling: after clustering the 26 emotions into 4 major psychological groups, the correct answer and the 3 distractors are selected from the 4 different groups respectively. For instance, if the correct answer is Surprise (Pleasure group), the distractors are selected from the Happiness group (e.g., Confidence), Anger group (e.g., Anger), and Sadness group (e.g., Embarrassment). This design guarantees sufficient semantic distance among candidates, allowing the QA task to evaluate the models' discriminative capability across core emotion categories rather than fine-grained subcategories.

### Key Designs 2: Gemini-Driven Self-Calibration of Difficulty and Quality Control

Gemini plays multiple roles during dataset construction—not only generating questions but also participating in quality filtering and difficulty calibration. This yields an intriguing property: if Gemini itself cannot correctly answer the questions it generated (labeled as Hard), it indicates that the question is sufficiently challenging. Experiments confirm that Gemini achieves only 8% accuracy on Hard samples compared to 91% on Easy samples, validating the effectiveness of this labeling mechanism. Further analysis reveals the shared features of Hard samples: (1) the person in the video has a neutral expression and lacks salient emotional cues; (2) the face is blocked by glasses, hats, or sunglasses. This highlights the tendency of VideoLLMs to rely excessively on facial expressions rather than holistic body movements when understanding body language.

### Key Designs 3: Multi-Dimensional Bias Analysis Framework

BQA inherits demographic metadata (gender, age, race) from BoLD, enabling a systematic analysis of individual models' error rate distribution across different groups on top of standard performance evaluations. Specifically, for each model, the error ratios are compiled across gender (male/female), age groups (child/teen/adult/senior), and racial groups (White/Black/Asian/Latino/Indigenous, etc.). Visual comparisons are then leveraged to discover bias patterns. This analysis framework elevates BQA from a pure performance benchmark to a fairness evaluation tool.

## Key Experimental Results

### Main Results: Test Set Performance Comparison (Table 1)

| Model | Frames | Easy | Hard | Total | Easy(CoT) | Hard(CoT) | Total(CoT) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Human (100 samples) | - | 0.96 | 0.77 | **0.85** | - | - | - |
| Gemini-1.5-pro | 1fps | 0.91 | 0.08 | 0.61 | 0.94 | 0.90 | 0.92 |
| GPT-4o | 1fps | 0.78 | 0.38 | 0.60 | 0.97 | 0.95 | 0.96 |
| Phi-3.5 | 16 | 0.77 | 0.41 | 0.58 | 1.00 | 0.96 | 0.98 |
| Qwen2-VL | 16 | 0.68 | 0.27 | 0.47 | 0.98 | 0.95 | 0.97 |
| LLaVA-NeXT | 16 | 0.66 | 0.30 | 0.47 | 1.00 | 0.96 | 0.98 |
| VideoLLaMA2 | 16 | 0.15 | 0.01 | 0.08 | 0.15 | 0.01 | 0.08 |
| VideoLLaMA2 (FT) | 16 | 0.98 | 0.91 | **0.94** | 0.98 | 0.90 | 0.94 |

### Validation Set Performance & Dataset Statistics (Table 3 & 4)

| Model | Easy | Hard | Total |
|------|:---:|:---:|:---:|
| Phi-3.5 | 0.76 | 0.38 | 0.56 |
| Qwen2-VL | 0.69 | 0.32 | 0.50 |
| LLaVA-NeXT | 0.65 | 0.31 | 0.47 |
| VideoLLaMA2 | 0.40 | 0.09 | 0.24 |
| VideoLLaMA2 (FT) | 0.89 | 0.68 | 0.78 |

Dataset scale: Train 4,651 (Easy 2,192 / Hard 2,459), Valid 1,538 (746 / 792), Test 1,443 (707 / 736). Self-BLEU (4-gram) across the three sets is 0.87, 0.86, and 0.85 respectively, indicating that question diversity is well-balanced across splits.

### Key Findings

1. **Body language understanding is highly challenging**: The strongest models, GPT-4o and Gemini, achieve an accuracy of only about 60%, far below human performance (85%), with a larger gap on the Hard subset (Gemini scores only 8%).
2. **CoT improvements are driven by answer leakage**: Multimodal CoT improves GPT-4o's performance from 60% to 96%. However, analysis reveals that the generated rationales frequently contain the correct answer explicitly—which is answer leakage rather than a genuine improvement in reasoning ability.
3. **Models rely heavily on facial expressions**: The characteristics of Hard samples are neutral expressions or facial occlusions. This suggests that VideoLLMs do not genuinely understand body language, but instead rely primarily on facial cues for emotion judgment.
4. **Significant racial bias is observed**: LLaVA-NeXT's accuracy on videos related to Native Hawaiians is only about 25%. Gemini also exhibits a clear performance bias towards this group. No significant bias is observed in the gender dimension.
5. **Surprisingly effective fine-tuning**: VideoLLaMA2's performance increases from 8% to 94% after LoRA fine-tuning (LoRA $r=128$, $\alpha=256$, 1 epoch), even outperforming Gemini. This indicates that while pre-training data lacks body language knowledge, the model possesses strong learning capability.
6. **Emotion prediction bias**: Error analysis shows that no models mistakenly predict emotions as "Pleasure". However, when the correct answer is "Happiness", they frequently misclassify it as opposing emotions such as "Sadness" or "Anger".

## Highlights & Insights

- **Filling the evaluation gap**: This is the first benchmark dataset to systematically evaluate VideoLLMs' capability in understanding body language emotions, containing 7,632 videos covering 26 emotions and diverse demographic dimensions.
- **Exposing the pitfalls of Multimodal CoT**: The leakage of answers within the generated rationales causes "spurious" performance increases. This finding serves as a cautionary tale for the VLM evaluation community—CoT scores should not be equated with real understanding capabilities.
- **Opening a new dimension in bias analysis**: Combining performance evaluation with fairness analysis reveals that VideoLLMs exhibit severe performance discrepancies on minority groups, which holds significant social value.
- **Exquisitely designed self-calibration for difficulty**: Leveraging Gemini's own answering capability to calibrate question difficulty avoids additional human annotation costs, and the validation experiments confirm the effectiveness of the Easy/Hard labels.
- **Thorough argument against data contamination**: The absence of data contamination is argued from two perspectives: question originality and Gemini's limited accuracy, while self-preference bias is also ruled out due to the external source of the ground truth (BoLD).

## Limitations & Future Work

- **Video Quality**: BoLD is derived from old movies, and some videos suffer from poor image quality, potentially limiting the evaluation ceiling of the models' capabilities.
- **Limited Scale of Human Evaluation**: Only three annotators evaluated 100 random samples (agreement of 0.79), lacking the diverse perspectives of cross-cultural annotators.
- **Frame Limitations**: Evaluating with a fixed 16-frame input does not explore the impact of denser frames. Moreover, some models (e.g., Gemini/GPT-4o) use 1fps rather than a fixed frame count, leading to a somewhat unfair comparison.
- **Loss of Emotional Granularity**: Although simplifying 26 emotions into 4 groups guarantees the feasibility of QA, it sacrifices the ability to evaluate fine-grained emotional distinctions.
- **Discarding Audio Information**: While removing audio ensures purely visual evaluation, it neglects crucial vocal cues like intonation, failing to reflect real-world multimodal scenarios.
- **Disregard for Cultural Differences**: Emotional expressions in body language can vary significantly across different cultures; however, the dataset does not control for or analyze this variable.

## Related Work & Insights

- **vs SMILE (Hyun et al., NAACL 2024)**: SMILE only analyzes the multimodal understanding of a single emotion (smiling), whereas BQA scales up the evaluation scope to cover 26 emotional labels.
- **vs Video-MME (Fu et al., 2024)**: Video-MME is a comprehensive video understanding benchmark, while BQA focuses on the specific capability dimension of understanding emotions in body language, offering complementary value.
- **vs mCSQA (Sakai et al., ACL 2024)**: BQA borrows the semi-automatic QA generation methodology of mCSQA, migrating it from the textual common-sense reasoning domain to video emotion understanding.
- **Insight**: Understanding "what action a person is performing" and "what emotion a person is feeling" are two entirely different ontological levels of tasks. Although current VideoLLMs perform well on action recognition, a huge chasm remains in emotional perception. Superficial performance gains achieved through reasoning-enhancement methods like CoT may mask a lack of genuine capability, urging more caution in evaluation design.

## Rating

- Novelty: ⭐⭐⭐⭐ Novel topic and unique entry angle, establishing the first systematic benchmark for body language understanding.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 6 VideoLLMs + human baseline, with exhaustive analyses of bias and CoT answer leakage.
- Writing Quality: ⭐⭐⭐ Well-structured, although some portions of the analysis (especially the bias aspect) could benefit from deeper quantitative insights.
- Value: ⭐⭐⭐⭐ Dataset open-sourced on HuggingFace, directly ready for assessing VideoLLMs' fairness and comprehension capability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Generating Diverse Training Samples for Relation Extraction with Large Language Models](generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)
- [\[ACL 2025\] Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints](rethinking_semantic_parsing_for_large_language_models_enhancing_llm_performance_.md)
- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](../../ACL2026/nlp_understanding/table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[ACL 2026\] The Imperfective Paradox in Large Language Models](../../ACL2026/nlp_understanding/the_imperfective_paradox_in_large_language_models.md)
- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)

</div>

<!-- RELATED:END -->
