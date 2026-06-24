---
title: >-
  [Paper Note] Can MLLMs Understand the Deep Implication Behind Chinese Images?
description: >-
  [ACL 2025][Multimodal VLM][MLLM Evaluation] This paper proposes CII-Bench (Chinese Image Implication Understanding Benchmark), which contains 698 Chinese internet/traditional culture images and 800 multiple-choice questions. It systematically evaluates MLLMs' high-level understanding of the deep implications within Chinese images. The study reveals that the best-performing model achieves an accuracy of only 64.4%, significantly lower than the human average of 78.2%…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "MLLM Evaluation"
  - "Chinese Image Implication"
  - "Traditional Chinese Culture"
  - "CII-Bench"
  - "High-Level Perception"
date: 2026-05-08
content_hash: c1ac5bcd9223fe84
---

# Can MLLMs Understand the Deep Implication Behind Chinese Images?

**Conference**: ACL 2025  
**arXiv**: [2410.13854](https://arxiv.org/abs/2410.13854)  
**Code**: [https://cii-bench.github.io/](https://cii-bench.github.io/)  
**Authors**: Chenhao Zhang, Xi Feng, Yuelin Bai, Xinrun Du, et al.  
**Institutions**: Huazhong University of Science and Technology, Shenzhen Institute of Advanced Technology (CAS), University of Science and Technology of China, M-A-P, 01.ai, et al.  
**Area**: Multimodal VLM  
**Keywords**: MLLM Evaluation, Chinese Image Implication, Traditional Chinese Culture, CII-Bench, High-Level Perception  

## TL;DR

This paper proposes CII-Bench (Chinese Image Implication Understanding Benchmark), which contains 698 Chinese internet/traditional culture images and 800 multiple-choice questions. It systematically evaluates MLLMs' high-level understanding of the deep implications within Chinese images. The study reveals that the best-performing model achieves an accuracy of only 64.4%, significantly lower than the human average of 78.2%, with models performing worst in the domain of traditional Chinese culture.

## Background & Motivation

**Background**: As the capabilities of MLLMs continue to advance, there is a growing need to evaluate their high-level perceptual abilities. While the previous English image implication understanding benchmark, II-Bench, exists, Claude-3.5-Sonnet has already achieved an accuracy of 80.9% on it, approaching the human level of 90.3%. This indicates a pressing need for more challenging benchmarks.

**Chinese-English Differences**: Chinese and English images exhibit significant cultural differences in expression. Traditional Chinese landscape paintings not only depict natural scenery but also convey philosophical concepts such as harmony between humanity and nature through artistic techniques like the interplay of virtual and real (empty and solid spaces), blank space (liubai), and brushwork. Su Shi's famous saying, "poetry and painting share the same origin" (Shi Hua Tong Yuan), precisely summarizes the deep connotation of Chinese imagery. Similarly, New Year paintings and cartoons utilize symbolism and metaphors to convey meanings of happiness and prosperity.

**Core Motivation**: Currently, there is a lack of work evaluating MLLMs' high-level perception and understanding of Chinese visual content. CII-Bench fills this vacancy by comprehensively testing models' capabilities in perception, reasoning, and comprehension within the Chinese cultural context.

## Method

### 3.1 Benchmark Overview

CII-Bench contains **698 images** and **800 multiple-choice questions**, covering six domains:
- **Life**: 216 questions
- **Art**: 123 questions
- **Society**: 157 questions
- **Politics**: 21 questions
- **Environment**: 51 questions
- **Traditional Chinese Culture (CTC)**: 130 questions

Image types include illustrations, memes, posters, single-panel cartoons, multi-panel cartoons, and paintings. Each question provides 6 options with only one correct answer.

### 3.2 Data Construction Pipeline

**Data Collection**: 17,695 raw images were collected from several well-known illustration websites, ensuring compliance with copyright and licensing regulations.

**Three-Stage Data Filtering**:
1. **Image Deduplication**: Image similarity algorithms were used for pixel-level comparison to eliminate duplicates.
2. **Text Area Control**: OCR technology was employed to identify text areas, excluding images with excessive text ratios to maintain a vision-centric focus.
3. **Visual Review**: Strict reviews were conducted to remove images lacking metaphoric depth. This process filtered out over 95% of the initial images, resulting in fewer than 1,000 high-quality images.

**Data Annotation**: Annotation was completed by 30 undergraduate students from various disciplines and institutions. The annotation pipeline included:
- Pre-annotation consistency check
- Multi-round annotation and cross-validation (each image was annotated by two annotators, with disagreements resolved by a third-party reviewer)
- Annotation content refinement (difficulty, type, sentiment labels, domains, rhetorical devices, etc.)
- Contextual analysis and post-annotation review

### 3.3 Dataset Statistics

- Each question contains an average of approximately 11 Chinese characters.
- Each option averages 28 characters.
- Each image is paired with a detailed human-written description.
- Difficulty is classified into three levels: Easy / Medium / Hard.
- Sentiment is categorized into three classes: Positive / Neutral / Negative.
- Rhetorical devices annotation: Metaphor, Exaggeration, Symbolism, Visual Misplacement, Contrast, Analogy, Personification, Comparison.

### 3.4 Evaluation Setup

Eight configurations were utilized to evaluate each model:
- **None** (zero-shot standard prompt)
- **1/2/3-shot**
- **CoT** (Chain of Thought)
- **Domain** (domain information provided)
- **Emotion** (sentiment polarity information provided)
- **Rhetoric** (rhetorical device information provided)

In addition, a subset of text-only LLMs (without image inputs) was selected to complete the task to verify the necessity of visual inputs for answering questions.

## Experiments

### Main Results

| Model | Overall | Life | Art | Society | CTC | Positive | Negative |
|------|---------|------|-----|---------|-----|----------|----------|
| **Human Avg.** | **78.2** | 81.0 | 67.7 | 82.7 | 65.9 | 77.9 | 75.2 |
| **Human Best** | **81.0** | 83.2 | 73.6 | 87.2 | 66.7 | 78.2 | 78.8 |
| Qwen2-VL-72B | **64.4** | 61.7 | 61.2 | 68.0 | 59.9 | 62.7 | 63.8 |
| GLM-4V | 60.9 | 55.0 | 59.9 | 66.5 | 55.5 | 58.5 | 64.5 |
| Gemini-1.5 Pro | 60.1 | 60.0 | 63.3 | 62.4 | 51.1 | 54.8 | 65.6 |
| InternVL2-40B | 57.9 | 55.8 | 55.1 | 61.9 | 52.6 | 54.4 | 58.0 |
| GPT-4o | 54.1 | 54.1 | 55.8 | 52.1 | 51.8 | 51.9 | 56.2 |
| Text-only DeepSeek-67B | 27.1 | 26.6 | 32.7 | 30.9 | 18.2 | 25.7 | 22.2 |

**Key Findings**:
1. **Significant Human-Machine Gap**: The best model, Qwen2-VL-72B, achieves only 64.4%, whereas the average human accuracy is 78.2% and the best human performance is 81.0%.
2. **Open-Source Outperforms Closed-Source**: The best open-source model (Qwen2-VL-72B, 64.4%) outperforms the best closed-source model (GLM-4V, 60.9%) by a margin of over 3%.
3. **Traditional Chinese Culture is the Most Challenging**: All models score lowest in the CTC domain, far below other domains. GPT-4o only observes surface-level information and struggles to deeply interpret the complex cultural elements in traditional Chinese paintings.

### Ablation Study

**Comparison of Prompting Strategies**:
- **Sentiment prompts** are the most effective: after providing the models with sentiment polarity (positive/negative) information, the accuracy of most models improves significantly. This aligns with intuition—sentiment information helps models eliminate irrelevant options.
- **Domain and rhetoric prompts** show limited effectiveness: because this information generally does not effectively help eliminate options.
- **CoT is not always effective**: MiniCPM-v2.6 drops from 45.0% to 38.9%, and LLaVA-1.6-72B drops from 48.0% to 45.3%. This is because CoT tends to lead to over-interpretation.

**Few-shot Effectiveness**:
- As the number of few-shot exemplars increases, accuracy **decreases instead**.
- Reasons: (1) Insufficient multi-image processing capability, (2) Poor performance in processing long contexts as the input length increases.
- For instance, InternVL2-40B drops from None 57.9% → 1-shot 53.0% → 3-shot 41.9%.

**Text-Only Experiments**:
- DeepSeek-67B-Chat achieves only 27.1% accuracy, proving that CII-Bench is highly dependent on visual content.

### Deep Evaluation of Traditional Chinese Culture

A five-dimensional evaluation metric was designed to evaluate models' understanding of traditional Chinese paintings:
1. Surface-level Information
2. Aesthetic Characteristics
3. Brush and Ink Skills
4. Culture and History
5. Deep Implications

GPT-4o achieves an overall score of 2.71 (out of 5), indicating that the model only observes surface-level painting information, showing a massive gap compared to humans in deeply interpreting complex cultural elements of traditional Chinese art.

## Highlights & Insights

1. **First Benchmark for Understanding Chinese Image Implication**: Fills the gap in evaluating MLLMs' high-level comprehension of Chinese visual content.
2. **Quantitative Presentation of Cultural Differences**: The implicit expressive style of Chinese images presents a greater challenge to MLLMs—even advanced models struggle to understand Chinese aesthetic concepts like the "interplay of the virtual and the real" (xu shi xiang sheng).
3. **Sentiment Comprehension Discrepancies**: Models perform better on negative-sentiment images, whereas humans are more sensitive to positive-sentiment images. This finding is contrary to the conclusion of the English II-Bench, reflecting the difference in emotional expression between Chinese and English cultures.
4. **Open-Source Outperforming Closed-Source**: As an open-source model, Qwen2-VL-72B outperforms all closed-source models by a margin of over 3%.
5. **GPT-4o's Cultural Blind Spots**: On understanding traditional Chinese culture, GPT-4o only reaches a surface level and is virtually incapable of deep cultural interpretation.

## Limitations & Future Work

1. The dataset size is relatively limited (698 images / 800 questions), which may not comprehensively represent the diversity of Chinese images.
2. Annotations carry a degree of subjectivity, particularly for the understanding of deep implications which varies depending on individual cultural backgrounds.
3. The capability to understand video modalities was not evaluated.
4. With the rapid progress of MLLM capabilities, the benchmark may face the risk of becoming obsolete.

## Related Work & Insights

- **MLLM Benchmarks**: Comprehensive evaluation frameworks such as MMBench, SEED, MMMU, and CMMMU.
- **Image Implication Understanding**: II-Bench (English) is the first benchmark specifically designed to evaluate the implicit understanding of images by MLLMs.
- **MLLM Development**: Representative multimodal models such as BLIP-2, LLaVA, mPLUG-Owl2, and InternVL.

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ — The first benchmark for Chinese image implication understanding, offering a novel perspective.
- **Value**: ⭐⭐⭐⭐ — Reveals significant deficiencies of MLLMs in cross-cultural understanding, yielding guidance value for model improvement.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Highly comprehensive evaluation across multiple perspectives and dimensions.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and in-depth analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relations?](cordial_can_multimodal_large_language_models_effectively_understand_coherence_re.md)
- [\[ACL 2025\] NegVQA: Can Vision Language Models Understand Negation?](negvqa_can_vision_language_models_understand_negation.md)
- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](can_vision_language_models_understand_mimed_actions.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] Finding Needles in Images: Can Multi-modal LLMs Locate Fine Details?](finding_needles_in_images_can_multi-modal_llms_locate_fine_details.md)

</div>

<!-- RELATED:END -->
