---
title: >-
  [Paper Note] Mind the Gap! Static and Interactive Evaluations of Large Audio Models
description: >-
  [ACL2025][Audio & Speech][Large Audio Models] By collecting 7,500 interaction evaluation data points from 484 participants, this paper systematically compares the static benchmarks and interactive evaluation performance of Large Audio Models (LAMs) for the first time. It reveals a significant gap between the two ($R^2=0.30$) and uncovers the real-world usage scenarios and user preferences of LAMs.
tags:
  - "ACL2025"
  - "Audio & Speech"
  - "Large Audio Models"
  - "Interactive Evaluation"
  - "User Preferences"
  - "Speech Benchmarks"
  - "LAM"
date: 2026-05-08
content_hash: c39c031bcb642c7e
---

# Mind the Gap! Static and Interactive Evaluations of Large Audio Models

**Conference**: ACL2025  
**arXiv**: [2502.15919](https://arxiv.org/abs/2502.15919)  
**Code**: [TalkArena.org](https://talkarena.org/)  
**Area**: Audio & Speech  
**Keywords**: Large Audio Models, Interactive Evaluation, User Preferences, Speech Benchmarks, LAM  

## TL;DR

By collecting 7,500 interaction evaluation data points from 484 participants, this paper systematically compares the static benchmarks and interactive evaluation performance of Large Audio Models (LAMs) for the first time. It reveals a significant gap between the two ($R^2=0.30$) and uncovers the real-world usage scenarios and user preferences of LAMs.

## Background & Motivation

Compared to text, speech interaction offers faster communication speed and the ability to convey paralinguistic information (such as tone of voice and emotion). This has driven the development of Large Audio Models (LAMs), such as Qwen-Audio and GPT-4o.

However, critical issues exist in modern LAM evaluations:

**Limitations of Static Benchmarks**: Current evaluation frameworks (e.g., AIRBench, AudioBench) are extended from traditional ASR tasks. They use static metrics (WER, accuracy) based on reference answers, which fail to reflect real-world user demands.

**Insights from the Text Domain**: In the text LLM domain, benchmarks such as MMLU and AlpacaEval correlate highly with interactive evaluations from Chatbot Arena ($\rho > 0.8$). However, whether this holds true for the speech domain remains entirely unknown.

**Lack of Interactive Data**: No prior work has collected user preference data specifically for LAMs.

Three core research questions:
- What tasks do users expect LAMs to perform?
- Which models perform best on these tasks, and why?
- Which static benchmarks best predict user preferences?

## Method

### Interactive Evaluation Platform

A web platform built on Gradio (TalkArena.org) with the following core designs:

- **Free Interaction**: Users are not provided with specific task examples; they are only prompted to "interact with the voice AI assistant in your desired way" to maximize the capture of real-world usage scenarios.
- **Pairwise Comparison**: Upon submitting a speech query, users receive text responses from two anonymous models and indicate their preference (A is better, B is better, or tie).
- **Streaming Output**: Token-by-token streaming of outputs is used to prevent users from identifying models based on tokenization patterns.
- **Optional Feedback**: Users could provide justifications for their preferences via text or voice (with 44.9% of users choosing to do so).

### Data Collection

- **Participants**: 484 participants, recruited via the Prolific platform.
- **Screening Criteria**: Prior experience with LLM chat products + possession of a microphone.
- **Gender Balance**: Ensured fair representation.
- **Scale**: 50 participants per model pair $\times$ 10 votes = 500 votes per pair.
- **Total**: 7,500 votes.
- **Compensation**: \$2.50 per 10 votes, ensuring a minimum of \$15/hour.

### Model Ranking

The **Bradley-Terry model** is used to convert pairwise preferences into model rankings:

$$Pr(i > j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}}$$

The coefficients $\beta$ are estimated by maximizing the log-likelihood of the observed preference data.

### Evaluated Models

**Interactive Evaluation** (6 models):
- GPT-4o
- Gemini-1.5-pro
- Qwen2-Audio
- Typhoon-1.5
- DiVA-8B
- ASR Pipeline (Whisper-large-v2 + Llama3-8B-Instruct)

**Static Evaluation** (3 additional models): NExTGPT, PandaGPT, Qwen-Audio

### Static Benchmark Evaluation

A superset of 20 datasets was constructed, covering three major dimensions:

1. **Speaker Cognitive State**: Intent detection, humor/sarcasm recognition, emotion recognition.
2. **Speaker Identity**: Language identification, accent classification, gender/age classification, relationship classification.
3. **Speech Content Understanding**: ASR, speech grounding, entity recognition, instruction following, question answering.

## Experiments

### User Case Scenario Analysis

Through topic modeling (K-Means + BERT embeddings) on 1,000 random samples, four major categories were identified:

| Category | Proportion | Example |
|------|------|------|
| **Knowledge Queries** | 50% | "What is the Milky Way?" |
| **Seeking Advice** | 17% | "What should I pay attention to when keeping shrimp?" |
| **Chitchat** | 16% | "Good morning, how are you?" |
| **Task Execution** | 10% | "Summarize the first volume of Lord of the Mysteries" |

**Key Findings**: In 77% of usage scenarios, speech primarily serves efficiency purposes (such as task execution) rather than conveying information unique to the audio modality. 7% of recordings contain background noise. Compared to interactions with text LLMs, users rarely ask math and programming questions.

### User Preference Ranking

**Surprising Result**: The ASR Pipeline (Whisper + Llama3-8B) is the most preferred!

Analysis of reasons:
1. Most user queries heavily rely on textual semantics.
2. Three out of the top five categories of user feedback focus on **text output style**.

### Reasons for User Preferences (Analysis of 100 Samples)

| Reason | Proportion |
|------|------|
| 1. Detail level of response | **31%** |
| 2. Helpfulness | 24% |
| 3. Appropriateness of language | 12% |
| 4. Accuracy | 11% |
| 5. Human-likeness | 11% |

Interestingly, there is a divergence regarding "human-likeness": some users prefer the AI to admit its inability to have opinions, while others prefer a friendly and inquisitive AI.

### Static Benchmark Performance

GPT-4o ranks first in 6 out of 14 tasks and enters the top three in 11. Among open-source models, Qwen2-Audio (8/14 top three) and Typhoon (7/14 top three) are the strongest.

 | Model | Static Benchmark Ranking | Interactive Ranking |
|------|-------------|------------|
| GPT-4o | 1 | Non-optimal |
| ASR Pipeline | Medium | **1** |
| DiVA | Lower-medium | 2 |

### Predictive Power of Static Benchmarks

**Key Findings**:

- **Weak Correlation of Single Benchmarks**: All individual benchmarks correlate poorly with interactive evaluations, with correlation coefficients $\tau \leq 0.33$.
- **Limited Predictive Power of Aggregated Benchmarks**: The marginal $R^2$ of the mixed-effects regression model is $0.30$.
- 5 principal components explain 95% of the variance across the 20 benchmarks, indicating that despite the large number of benchmarks, very few core capability axes are actually evaluated.
- **Only Two Datasets Exhibit Significant Positive Correlation**:
    - CommonVoice-Age ($\beta = 0.314$): However, all models perform below the random baseline on this task.
    - Public-SG-Speech ($\beta = 0.167$): A speech QA task that can be completed solely using text transcriptions.

**Stark Contrast with Text LLMs**: In the text domain, static and interactive evaluations are highly correlated, which is clearly not the case in the speech domain.

## Highlights & Insights

1. **Revealing the "Gap" in LAM Evaluation for the First Time**: Static benchmarks can hardly predict user preferences, which serves as a major warning to the entire LAM evaluation community.
2. **Implications of the Pipeline Model's Success**: In current usage scenarios (which predominantly rely on contractual/textual semantics), the advantages of end-to-end audio models are not realized. This suggests that the most effective way to enhance the interactive capabilities of LAMs is to improve the interaction capabilities of their text-based LLM components.
3. **Discovery of User Interaction Patterns**: The main value of speech interaction lies in communication efficiency rather than leveraging audio-specific information, contrasting with the current development focus of LAMs (e.g., paralinguistic feature recognition).
4. **Contribution to Evaluation Methodology**: Detailed user feedback analysis identifies five critical dimensions affecting preferences, pointing out directions for future benchmark design.

## Limitations & Future Work

1. **Single-turn Interaction Only**: The evaluation does not support multi-turn dialogue, which may underestimate the value of LAMs in long-term interactions.
2. **Paid Participants**: Since this is not a natural daily-use scenario, user behavior might be influenced by the task-oriented nature of the assignment.
3. **English Only**: All participants are US residents, which potentially penalizes multilingual models like Typhoon (Thai) and Qwen (Chinese) unfairly.
4. **Speech-to-Text Modality**: Speech output was not evaluated (which is only natively supported by GPT-4o), potentially affecting model rankings.
5. **Minimally Constrained User Tasks**: The evaluation only captures the scenarios that users first think of; user preferences may evolve after long-term usage.

## Related Work & Insights

- **Large Audio Models**: SpeechGPT, LTU, Qwen-Audio series, DiVA, etc., which integrate audio encoders with text LLMs.
- **LAM Evaluation**: Aggregated static benchmarks like AIRBench, AudioBench, VoiceBench, etc., which still depend on reference metrics.
- **Interactive Evaluation**: Systems like Chatbot Arena (text LLMs) and WildVision Arena (vision-language models). This work is the first of its kind in the speech/audio domain.

## Rating

⭐⭐⭐⭐⭐ — This work fills an important gap in the interactive evaluation of LAMs. Its core finding (that static benchmarks can hardly predict user preferences) has profound implications for the entire field. The data collection scale is substantial (7,500+ interactions), the analytical dimensions are rich (usage scenarios, reasons for preference, benchmark predictive power), and the experimental design is rigorous. The counterintuitive finding of the Pipeline model outperforming other models is highly insightful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VGGSounder: Audio-Visual Evaluations for Foundation Models](../../ICCV2025/audio_speech/vggsounder_audio-visual_evaluations_for_foundation_models.md)
- [\[ACL 2025\] Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models](who_can_withstand_chat-audio_attacks_an_evaluation_benchmark_for_large_audio-lan.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](../../ACL2026/audio_speech/closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ACL 2025\] Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models](audio_dialogue_benchmark.md)
- [\[ACL 2025\] Investigating and Enhancing Vision-Audio Capability in Omnimodal Large Language Models](investigating_and_enhancing_vision-audio_capability_in_omnimodal_large_language_.md)

</div>

<!-- RELATED:END -->
