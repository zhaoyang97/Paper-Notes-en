---
title: >-
  [Paper Note] On the Risk of Evidence Pollution for Malicious Social Text Detection in the Era of LLMs
description: >-
  [ACL2025][LLM (Other)][Evidence Pollution] This paper systematically investigates the risk of "evidence pollution" in malicious social text detection in the LLM era. It proposes 13 pollution methods and 3 defense strategies, finding that LLM-generated fake evidence can cause detector performance degradation by up to 14.4%, and existing defense strategies face practical deployment challenges.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Evidence Pollution"
  - "Malicious Social Text"
  - "LLM Misuse"
  - "Defense Strategy"
  - "Fake News Detection"
date: 2026-05-08
content_hash: 1566eb59a8afa547
---

# On the Risk of Evidence Pollution for Malicious Social Text Detection in the Era of LLMs

**Conference**: ACL2025  
**arXiv**: [2410.12600](https://arxiv.org/abs/2410.12600)  
**Code**: [GitHub](https://github.com/whr000001/EvidencePollution)  
**Area**: LLM/NLP  
**Keywords**: Evidence Pollution, Malicious Social Text, LLM Misuse, Defense Strategy, Fake News Detection  

## TL;DR

This paper systematically investigates the risk of "evidence pollution" in malicious social text detection in the LLM era. It proposes 13 pollution methods and 3 defense strategies, finding that LLM-generated fake evidence can cause detector performance degradation by up to 14.4%, and existing defense strategies face practical deployment challenges.

## Background & Motivation

### Background
Malicious social text detection (including fake news, hate speech, rumor, and sarcasm detection) is a core issue in the NLP security domain. Existing high-performance detectors rely heavily on "evidence" (such as user comments and external knowledge) to enhance their judgment. However, the rise of LLMs introduces new security risks—malicious actors can exploit LLMs to manipulate evidence associated with social texts to obfuscate evidence-based detectors.

### Core Motivation
- Traditional detection methods rely on evidence (comments, external knowledge, etc.), but this evidence itself can be manipulated.
- The powerful generation capabilities of LLMs make evidence manipulation more efficient and stealthy.
- Existing studies show that LLMs can generate hard-to-detect malicious content, but there is still no systematic risk assessment of evidence pollution.
- Two key questions need to be answered: (1) To what extent can LLMs manipulate evidence to confuse detectors? (2) What mitigation strategies are available?

### Research Significance
This is the first work to systematically study the impact of LLM-driven evidence pollution on malicious social text detectors, which is of great significance for understanding AI security risks and building more robust detection systems.

## Method

### Overall Architecture
The paper designs a total of 13 evidence pollution methods across three categories, along with three defense strategies. Given a social text $s$ and its corresponding $m$ pieces of evidence (comments) $\{c_i\}_{i=1}^m$, the evidence-enhanced detector $f$ learns the distribution $p(y|s, \{c_i\}, f, \theta)$. The objective of the evidence pollution strategy $\mathcal{G}$ is to manipulate the evidence to disturb this distribution.

### Key Designs

#### 1. Basic Pollution — Without using LLMs
- **Remove**: Randomly removes half of the comments, simulating scenarios where comments are unavailable in the early stages of propagation.
- **Repeat**: Repeats the same comment 5 times, simulating the bandwagon effect.

#### 2. Rephrase Evidence — LLM rephrasing of existing comments
- **Rephrase**: Directly prompts the LLM to rephrase existing comments.
- **Rewrite**: Infuses malicious intent and rewrites comments to make malicious text appear normal.
- **Reverse**: Reverses the stance of the comments.
- **Modify**: Injects non-factual information with minimal edits.

#### 3. Generate Evidence — LLM directly generating comments
- **Vanilla**: Simply generates comments relevant to the text.
- **Stance**: Generates comments with a preset stance (supporting/opposing).
- **Publisher**: Simulates the publisher releasing comments that enhance credibility.
- **Echo**: Simulates the echo chamber effect, generating comments that repeatedly reinforce beliefs.
- **Makeup**: Generates comments that dilute refutations to evade detection.
- **Amplify**: Generates initial comments to accelerate dissemination.

All LLM-based methods employ zero-shot prompting, which consists of two parts: $p_{input}$ (input text) and $p_{inst}$ (strategy-specific instructions).

### Defense Strategies

#### 1. Machine-Generated Text Detection (Data side)
- Fine-tunes DeBERTa-v3 to detect generated text (requires labeled data).
- Uses metric-based detectors such as Fast-DetectGPT and Binocular (training-free, black-box setting).

#### 2. Mixture of Experts (Model side, parameter-free)
- Divides evidence into $k$ groups, with each group predicting independently.
- Obtains the final prediction via majority voting: $y = \arg\max_{y_j} \sum_{i=1}^k \mathbf{I}(y_i = y_j)$.
- Reduces the impact of a single polluted evidence.

#### 3. Parameter Updating (Model side)
- Assumes that some incorrect judgments can be corrected by experts.
- Uses feedback as labels to update detector parameters.

## Key Experimental Results

### Experimental Setup
- **4 tasks**: Fake news detection, hate speech detection, rumor detection, sarcasm detection.
- **10 datasets**: Politifact, Gossipcop, ANTiVax, HASOC, Pheme, Twitter15, Twitter16, RumorEval, Twitter, Reddit.
- **7 detectors**: dEFEND, Hyphen, GET, BERT, DeBERTa, Mistral, ChatGPT.
- **2 LLM generators**: Mistral-7B (open-source), ChatGPT (closed-source).
- **Evaluation metrics**: Accuracy, Macro F1, ARacc, ARF1, AUC.

### Main Results

| Detector | Baseline Performance | Post-Generate Pollution | Performance Drop |
|--------|------------|--------------|---------|
| DeBERTa | 96.9 (Politifact) | ~82.5 | Up to 14.4% |
| BERT+evidence | 94.7 | ~80.3 | Significant drop |
| dEFEND | 84.3 | ~75.0 | ~10% |

Key Findings:
1. **Generation strategies are the most threatening**: Generate-type pollution causes the largest performance degradation (up to 14.4%) because it fully replaces the original evidence.
2. **Encoder-based LMs are the most vulnerable**: Encoder models are more sensitive to evidence pollution, with accuracy dropping by up to 21.8%.
3. **Evidence enhancement is a double-edged sword**: Evidence aids detection but simultaneously increases the attack surface for pollution.

### Defense Experimental Results
- **Parameter Updating**: The most effective defense strategy, but requires labeled data and continuous updates.
- **Mixture of Experts**: Can partially mitigate the impact, but at a high computational cost.
- **Machine Text Detection**: Has limited effectiveness against phrasing-based pollution due to the high quality of the generated text.

### Ablation Study

| Analysis Dimension | Key Findings |
|---------|---------|
| Pollution Quality | LLM-generated polluted evidence exhibits high quality in both metrics and human evaluation. |
| Model Calibration | Pollution significantly harms model calibration, increasing ECE by up to 21.6%. |
| Combined Attack | Combining multiple pollution strategies can amplify the negative impact. |
| LLM Detector | LLM-based detectors (Mistral, ChatGPT) show unstable reliance on evidence. |

## Highlights & Insights

1. **Systematic Evaluation Framework**: Establishes the first comprehensive evidence pollution-defense evaluation framework, covering 4 tasks, 10 datasets, 13 attacks, and 3 defenses.
2. **Revelation of Practical Threats**: Quantitatively proves the adversarial risks of LLMs in social media security, where the generated fake comments are of extremely high quality.
3. **Discovery of Calibration Impact**: Not only affects accuracy but also significantly harms the probability calibration of the models, increasing the confidence of incorrect judgments.
4. **Defense Dilemma**: All three defense strategies have limitations—requiring labeled data, high computational costs, or unclear stopping times.

## Limitations & Future Work

1. Only considers comments as evidence sources, without covering other evidence types such as metadata and propagation patterns.
2. Defense strategies require their own assumptions (labeled data, multiple experts, feedback mechanisms), making practical deployment constrained.
3. Uses only two LLMs (Mistral-7B and ChatGPT); stronger models may pose greater threats.
4. Does not consider the dynamic game between adversarial attacks and defenses (multi-turn attack and defense).

## Related Work

- **Malicious Text Detection**: From content analysis to evidence-enhanced detection (dEFEND, Hyphen, GET).
- **LLM Security Risks**: LLM generation of malicious content, bias, and adversarial attacks.
- **Machine-Generated Text Detection**: Watermarking methods, fine-tuned detectors, and metric-based methods.
- **Social Media Security**: Fake news detection, hate speech detection, and rumor propagation analysis.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Evaluation | ⭐⭐⭐⭐ |

> This is a very comprehensive security study paper with extensive experimental coverage (13 attacks × 10 datasets × 7 detectors) and a strong systematic approach. The core contribution lies in quantifying the risk of LLM-driven evidence pollution for the first time and revealing that existing defense strategies have clear shortcomings. It has significant reference value for AI security and content moderation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Explicit and Implicit Data Augmentation for Social Event Detection](explicit_and_implicit_data_augmentation_for_social_event_detection.md)
- [\[ACL 2025\] Stress-testing Machine Generated Text Detection: Shifting Language Models Writing Style to Fool Detectors](stress-testing_machine_generated_text_detection_shifting_language_models_writing.md)
- [\[ACL 2025\] Synergizing Unsupervised Episode Detection with LLMs for Large-Scale News Events](synergizing_unsupervised_episode_detection_with_llms_for_large-scale_news_events.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] SDD: Self-Degraded Defense against Malicious Fine-tuning](sdd_self-degraded_defense_against_malicious_fine-tuning.md)

</div>

<!-- RELATED:END -->
