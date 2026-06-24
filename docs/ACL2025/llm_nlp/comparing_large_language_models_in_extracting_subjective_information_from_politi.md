---
title: >-
  [Paper Note] Comparing Large Language Models in Extracting Subjective Information from Political News
description: >-
  [ACL 2025][LLM (Other)][Subjective Information Extraction] This paper systematically compares the capabilities of various large language models in extracting subjective information (sentiment inclination, stance, bias, framing effects, etc.) from political news, finding that different LLMs exhibit significant performance variations across different dimensions of subjective information extraction, while revealing the impact of the LLMs' inherent political biases on the extract…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Subjective Information Extraction"
  - "Political News Analysis"
  - "LLM Evaluation"
  - "Sentiment Analysis"
  - "Stance Detection"
date: 2026-05-08
content_hash: c20c494bf59c6994
---

# Comparing Large Language Models in Extracting Subjective Information from Political News

**Conference**: ACL 2025  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Subjective Information Extraction, Political News Analysis, LLM Evaluation, Sentiment Analysis, Stance Detection

## TL;DR
This paper systematically compares the capabilities of various large language models in extracting subjective information (sentiment inclination, stance, bias, framing effects, etc.) from political news, finding that different LLMs exhibit significant performance variations across different dimensions of subjective information extraction, while revealing the impact of the LLMs' inherent political biases on the extraction results.

## Background & Motivation

**Background**: Political news contains a substantial amount of subjective information, such as the author's sentiment inclination, the political stance of the media, and the choice of narrative framing. Traditional NLP methods (e.g., sentiment analysis tools) underperform on political texts because political subjectivity is more implicit and complex. The emergence of LLMs offers new possibilities for automating the extraction of this subjective information.

**Limitations of Prior Work**: (1) Political subjective information differs from general sentiment analysis, requiring an understanding of the political context, ideological spectrum, and policy stances; (2) LLMs themselves may possess political biases (due to training data bias), which compromises their reliability in political text analysis; (3) There is a lack of standardized evaluation benchmarks for extracting subjective information from political news.

**Key Challenge**: Although LLMs possess strong language understanding capabilities, their inherent political biases can systematically distort the results of subjective information extraction. A method is needed to evaluate the accuracy of LLMs in extracting subjective information while quantifying their degree of bias.

**Goal**: (1) Construct a comprehensive evaluation framework for extracting subjective information from political news; (2) Compare the performance of mainstream LLMs such as GPT-4, Claude, and Llama; (3) Quantify and analyze the impact of LLMs' political biases on the results.

**Key Insight**: The authors collected articles from news sources spanning different political spectrums (left, center, right), annotated multi-dimensional subjective information labels by political science experts, and then systematically evaluated LLM performance based on this benchmark.

**Core Idea**: Through a multi-dimensional, multi-source evaluation of subjective information in political news, this study reveals the capability boundaries and bias patterns of LLMs in political text analysis, providing guidance for the reliable application of LLMs in political text analysis.

## Method

### Overall Architecture
The overall evaluation framework consists of four layers: (1) Data layer—news articles from diverse political stances annotated by experts; (2) Task layer—defining five subjective information extraction tasks (sentiment polarity, political stance, media bias, narrative framing, implicit opinions); (3) Model layer—various LLMs executing extraction tasks under zero-shot and few-shot settings; (4) Analysis layer—evaluating the results in terms of accuracy, consistency, and bias.

### Key Designs

1. **Multi-dimensional Subjective Information Annotation System**:

    - **Function**: Establish fine-grained subjective annotation standards for political news.
    - **Mechanism**: Define five subjective information dimensions: (a) Sentiment polarity—positive/negative/neutral attitude towards reported events/figures; (b) Political stance—left/center/right ideological orientation; (c) Media bias—the degree of objectivity in reporting, whether facts are selectively presented; (d) Narrative framing—the adopted narrative perspective (economic, moral, conflict framing, etc.); (e) Implicit opinion—opinions implicitly expressed through word choices, cited sources, etc. Each dimension is independently annotated by 3 political science experts, with majority voting taken as the gold standard.
    - **Design Motivation**: A single sentiment analysis dimension fails to capture the rich layers of subjective information in political text. Multi-dimensional annotation can evaluate LLMs' depth of understanding more comprehensively.

2. **Bias Detection and Quantification Method**:

    - **Function**: Measure systematic biases of LLMs in political text analysis.
    - **Mechanism**: Design paired experiments—selecting left-wing and right-wing media reports on the same political event, prompting LLMs to extract subjective information, and then comparing the offset direction and magnitude of LLM judgments against gold annotations. For instance, if an LLM systematically provides worse ratings on right-wing media articles (compared to human annotation), it indicates a left-leaning bias. Use the mean and variance of the offset to quantify the direction and consistency of the bias.
    - **Design Motivation**: Using LLMs to analyze political texts without detecting bias is hazardous; bias quantification helps users calibrate results or select appropriate models.

3. **Prompt Engineering and Calibration Strategy**:

    - **Function**: Mitigate LLM bias and improve accuracy through optimized prompt design.
    - **Mechanism**: Design three types of prompt strategies: (a) Neutral prompts—requiring LLMs to remain objective and explicitly stating not to inject personal viewpoints; (b) Role-playing prompts—instructing LLMs to analyze as a journalism professor; (c) Contrastive prompts—asking LLMs to list evidence from both positive and negative perspectives before making a judgment. Contrast the effects of these three prompt strategies on bias reduction and accuracy.
    - **Design Motivation**: Prompt engineering is the most direct way to steer LLM behavior. Proper prompt design can alleviate bias without sacrificing performance.

### Loss & Training
Since this paper primarily focuses on evaluation and analysis, no model training is involved. LLM evaluation is conducted under zero-shot and 5-shot settings, with each experiment repeated 5 times to calculate the standard deviation for evaluating result stability.

## Key Experimental Results

### Main Results

| Model | Sentiment Polarity F1 | Political Stance F1 | Media Bias F1 | Narrative Framing F1 | Implicit Opinion F1 | Average |
|------|-----------|-----------|-----------|-----------|-----------|------|
| GPT-4 (0-shot) | 74.2 | 62.5 | 55.3 | 48.7 | 41.2 | 56.4 |
| GPT-4 (5-shot) | 78.6 | 68.3 | 61.7 | 54.2 | 47.8 | 62.1 |
| Claude-3 (0-shot) | 72.8 | 60.1 | 57.8 | 46.3 | 43.5 | 56.1 |
| Llama-3-70B (0-shot) | 69.5 | 56.8 | 51.2 | 43.1 | 38.6 | 51.8 |
| XLM-R (Fine-tuned) | 76.3 | 64.7 | 58.9 | 51.5 | 44.3 | 59.1 |

### Ablation Study (Impact of Prompt Strategies on GPT-4 Bias)

| Prompt Strategy | Accuracy F1 | Left-leaning Offset | Bias Reduction |
|---------|---------|-----------|------------|
| Baseline Prompt | 62.1 | +0.18 | Baseline |
| Neutral Prompt | 61.8 | +0.12 | -33% |
| Role-play Prompt | 63.4 | +0.09 | -50% |
| Contrastive Prompt | 64.7 | +0.06 | -67% |

### Key Findings
- All LLMs perform best on the simplest dimension of sentiment polarity, but their F1 on implicit opinion detection is below 50%, indicating that underlying subjective information understanding remains a substantial challenge.
- Both GPT-4 and Claude exhibit a slight left-leaning bias (with an offset of +0.15~0.18), while Llama-3's bias patterns are less consistent.
- The contrastive prompt strategy is the most effective in reducing bias (by 67%) while simultaneously improving accuracy by 2.6 F1 points.
- The fine-tuned medium model (XLM-R) performs on par with or slightly better than zero-shot LLMs on average performance, indicating that fine-tuning still possesses irreplaceable value.

## Highlights & Insights
- The design of the LLM bias quantification method is ingenious. By utilizing paired experiments to control variables, the bias measurement leans towards causal inference rather than mere correlation.
- The design of the five-level subjective information dimensions progresses from shallow to deep, serving as a graduated benchmark to evaluate the depth of LLMs' political text understanding.
- The finding that contrastive prompts (forcing the listing of evidence from both sides) reduce bias and boost accuracy is insightful, suggesting that compulsory reasoning/deep thinking helps overcome heuristic judgment.

## Limitations & Future Work
- The study primarily focuses on English political news. Extracting subjective information from non-English political texts may present additional cultural and linguistic challenges.
- The annotated dataset size is limited (hundreds of articles), which may not sufficiently cover the diversity of political topics.
- LLMs' political bias might evolve with version updates, making the conclusions of this paper a snapshot at a specific point in time.
- Multimodal political information (such as the combination of text and images in political advertisements) is not covered.

## Related Work & Insights
- **vs Media Bias Detection**: Traditional media bias detection typically only examines source-level bias, whereas this work delves into multi-dimensional subjective information at the article level.
- **vs Political Stance Detection**: Stance detection usually processes short text (tweets), whereas this work processes long-form news articles, necessitating stronger long-text understanding capabilities.
- **vs LLM Bias Studies**: Prior studies on LLM bias mostly used questionnaires or multiple-choice tests. This work evaluates bias on real-world political texts, aligning closer with actual application scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-dimensional political subjective information evaluation framework is innovative, and the bias quantification method is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ It covers multiple models, dimensions, and prompt strategies, presenting deep analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation of the problem, with logical experimental design.
- Value: ⭐⭐⭐⭐ Provides significant guidance on the reliable usage of LLMs in political text analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Perspective Transition of Large Language Models for Solving Subjective Tasks](perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)
- [\[ACL 2025\] Comparing Linguistic Acceptability Judgments of Autoregressive Language Models](comparing_linguistic_acceptability_judgments_of_autoregressive_language_models.md)
- [\[ACL 2025\] Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions](beyond_demographics_fine-tuning_large_language_models_to_predict_individuals_sub.md)
- [\[ACL 2025\] Behavioral Analysis of Information Salience in Large Language Models](behavioral_analysis_of_information_salience_in_large_language_models.md)
- [\[ACL 2025\] Information Locality as an Inductive Bias for Neural Language Models](information_locality_as_an_inductive_bias_for_neural_language_models.md)

</div>

<!-- RELATED:END -->
