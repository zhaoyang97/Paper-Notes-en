---
title: >-
  [Paper Note] Catching Shortcuts: A Framework for Evaluating Shortcuts in Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Shortcut learning] This paper proposes a systematic framework to detect and evaluate the phenomenon of shortcut learning in large language models (LLMs). By constructing contrastive test sets and diagnostic metrics, it reveals that LLMs rely on spurious correlations rather than true semantic understanding across multiple NLP tasks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Shortcut learning"
  - "spurious correlation"
  - "evaluation framework"
  - "LLM robustness"
  - "bias detection"
date: 2026-05-08
content_hash: 59e24db9910dfb42
---

# Catching Shortcuts: A Framework for Evaluating Shortcuts in Large Language Models

**Conference**: ACL 2025  
**Area**: LLM/NLP  
**Keywords**: Shortcut learning, spurious correlation, evaluation framework, LLM robustness, bias detection

## TL;DR
This paper proposes a systematic framework to detect and evaluate the phenomenon of shortcut learning in large language models (LLMs). By constructing contrastive test sets and diagnostic metrics, it reveals that LLMs rely on spurious correlations rather than true semantic understanding across multiple NLP tasks.

## Background & Motivation

**Background**: Large language models have achieved impressive scores on various NLP benchmarks. However, a growing body of research has found that these high scores may stem from the models exploiting statistical shortcuts in the training data rather than truly understanding the task. Shortcut learning refers to the model relying on spurious correlations in the data to make predictions, such as the association of negation words with contradiction labels, or the co-occurrence of specific entities with answers.

**Limitations of Prior Work**: Although some work has studied shortcut phenomena in specific tasks (such as NLI and QA), there lacks a unified and scalable framework to systematically detect and quantify shortcut dependency in LLMs. Current methods are typically tailored to a single model or task, making cross-model and cross-task comparative analysis difficult. Additionally, whether shortcut behaviors diminish as the scale of LLMs increases remains an open question.

**Key Challenge**: The high accuracy of models on standard test sets may mask their reliance on shortcuts, and existing evaluation systems cannot effectively distinguish between "true capability" and "shortcut exploitation."

**Goal**: To design a general shortcut evaluation framework that can (1) systematically identify different types of shortcut patterns, (2) construct targeted contrastive test sets, and (3) provide quantifiable diagnostic metrics to measure the degree of the model's shortcut dependency.

**Key Insight**: The authors observe that shortcuts can be formalized as spurious statistical associations between input features and labels. By controlling the presence or absence of these features, the model's shortcut dependency can be precisely measured.

**Core Idea**: Construct a "shortcut-sensitive evaluation framework" that quantifies the dependency of LLMs on various shortcuts by comparing original samples with those where shortcut features have been removed or flipped.

## Method

### Overall Architecture
The framework consists of three core phases: (1) Shortcut pattern identification—mining potential shortcut patterns from training data via statistical analysis and heuristic rules; (2) Contrastive test set construction—generating paired samples (keeping shortcuts vs. removing/flipping shortcuts) for each shortcut pattern; (3) Diagnostic evaluation—comprehensively assessing the model's shortcut dependency using multi-dimensional metrics.

### Key Designs

1. **Shortcut Taxonomy**:

    - **Function**: Systematically classify shortcuts that LLMs might exploit.
    - **Mechanism**: Categorize shortcuts into three levels: lexical-level (e.g., associations between specific keywords and labels), syntactic-level (e.g., sentence length, associations between specific sentence structures and labels), and semantic-level (e.g., sentiment polarity, associations between the presence of negation words and labels). Define formalized detection criteria and trigger conditions for each shortcut type.
    - **Design Motivation**: Shortcuts at different levels affect model behavior through different mechanisms; classifying them allows for more precise problem localization.

2. **Contrastive Sample Generator**:

    - **Function**: Automatically generate paired test samples for each shortcut pattern.
    - **Mechanism**: Given an original sample $x$ and a shortcut feature $s$, generate a minimally edited contrastive sample $x'$ such that $x'$ no longer contains the shortcut feature $s$ while maintaining the same semantic label. Utilize rule-based replacement and LLM-assisted rewriting strategies to ensure the naturalness and label consistency of the contrastive samples.
    - **Design Motivation**: Minimal editing ensures that performance differences are primarily driven by the presence or absence of shortcut features rather than other semantic changes.

3. **Shortcut Dependency Score (SDS)**:

    - **Function**: Quantify the level of model dependency on specific shortcuts.
    - **Mechanism**: Define $SDS = \frac{Acc_{with} - Acc_{without}}{Acc_{with}}$, where $Acc_{with}$ is the accuracy on samples containing shortcuts, and $Acc_{without}$ is the accuracy on samples with shortcuts removed. A higher SDS indicates a greater dependency of the model on that shortcut. Additionally, introduce an aggregated metric, Overall SDS, to measure the model's comprehensive performance across all shortcut types.
    - **Design Motivation**: Normalized metrics allow for fair comparison across different models and tasks.

### Loss & Training
This paper presents an evaluation framework rather than a training method, so it does not involve a specific loss function. However, the authors propose debiasing training recommendations based on SDS feedback: oversampling or adversarially augmenting samples of high-SDS shortcut types during the fine-tuning phase.

## Key Experimental Results

### Main Results

| Model | NLI-SDS↓ | QA-SDS↓ | Sentiment-SDS↓ | Avg-SDS↓ |
|------|----------|---------|----------------|----------|
| GPT-4 | 8.2 | 5.1 | 3.7 | 5.7 |
| LLaMA-2-70B | 15.6 | 12.3 | 9.8 | 12.6 |
| LLaMA-2-13B | 22.4 | 18.7 | 14.2 | 18.4 |
| LLaMA-2-7B | 28.1 | 24.5 | 19.6 | 24.1 |
| Mistral-7B | 19.8 | 16.2 | 12.1 | 16.0 |

### Ablation Study

| Shortcut Type | LLaMA-2-7B SDS | LLaMA-2-70B SDS | Description |
|----------|----------------|-----------------|------|
| Lexical-level Shortcuts | 31.2 | 17.8 | Small models are most heavily dependent on lexical shortcuts |
| Syntactic-level Shortcuts | 24.5 | 13.2 | Sentence length bias is relatively common |
| Semantic-level Shortcuts | 22.7 | 11.4 | The negation word shortcut has a significant impact |
| Mixed Shortcuts | 34.8 | 20.1 | Superposition effect of multiple shortcuts |

### Key Findings
- As model scale increases, shortcut dependency generally decreases; however, even GPT-4-level models still exhibit significant shortcut dependency on specific tasks.
- Lexical-level shortcuts (such as negation words and specific entity names) have the largest and most pervasive impact, followed by syntactic-level shortcuts.
- RLHF-trained models (such as ChatGPT) have lower SDS than base models, suggesting that alignment training mitigates the shortcut problem to some extent.
- Shortcut patterns vary greatly across different tasks, with the NLI task suffering from the most severe shortcut issues.

## Highlights & Insights
- A unified and scalable shortcut evaluation framework is proposed, achieving systematic cross-model and cross-task comparisons for the first time, which is far more comprehensive than previous fragmented analyses targeted at single tasks.
- The "minimal edit" principle of the contrastive sample generation strategy is clever, ensuring the causal validity of the evaluation.
- The SDS metric is simple yet effective, and can be directly used for shortcut diagnosis in any new model, showcasing high practical value.

## Limitations & Future Work
- The framework currently focuses primarily on shortcuts in text inputs, without addressing visual shortcuts in multimodal scenarios.
- The automatic generation quality of contrastive samples depends on the rewriting capability of the assistant model, which may introduce additional noise.
- The causal mechanisms of shortcuts—why models learn specific shortcuts and how to eliminate them from an architectural level—are not thoroughly explored.
- The framework can be extended to more tasks like code generation and mathematical reasoning to detect LLM shortcut behavior in these areas.
- The scalability of the framework warrants further validation—specifically, how maintenance costs scale when task and shortcut types grow substantially.

## Related Work & Insights
- **vs McCoy et al. (2019) HANS**: HANS focuses on syntactic heuristic shortcuts in NLI tasks, whereas this framework covers more tasks and shortcut types.
- **vs Geirhos et al. (2020)**: Geirhos focuses on texture bias in CV, while this paper systematically introduces similar ideas into NLP/LLM evaluation.
- **vs Ribeiro et al. (2020) CheckList**: CheckList is a general NLP testing framework, whereas this work focuses specifically on the issue of shortcuts, providing deeper diagnosis.

## Rating
- Novelty: ⭐⭐⭐⭐ The framework's idea is clear, but research on shortcuts itself is not a brand new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ It covers multiple models and tasks, but lacks evaluation of even larger-scale LLMs.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-defined motivation.
- Value: ⭐⭐⭐⭐ Possesses practical reference value for the field of LLM evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?](do_large_language_models_perform_latent_multi-hop_reasoning_without_exploiting_s.md)
- [\[ACL 2025\] SocialEval: Evaluating Social Intelligence of Large Language Models](socialeval_evaluating_social_intelligence_of_large_language_models.md)
- [\[ACL 2025\] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective](evaluating_implicit_bias_in_large_language_models_by_attacking_from_a_psychometr.md)
- [\[ACL 2025\] SCoP: Evaluating the Comprehension Process of Large Language Models from a Cognitive View](scop_evaluating_the_comprehension_process_of_large_language_models_from_a_cognit.md)

</div>

<!-- RELATED:END -->
