---
title: >-
  [Paper Note] Sentiment Reasoning for Healthcare
description: >-
  [ACL2025][NLP Understanding][Sentiment Reasoning] This work introduces a new task termed "Sentiment Reasoning," which requires models to generate explanatory rationales while predicting sentiment labels for healthcare conversations. A multimodal sentiment analysis dataset comprising 30K samples across five languages is constructed. Rationale-augmented training improves classification accuracy and macro-F1 by approximately 2%.
tags:
  - "ACL2025"
  - "NLP Understanding"
  - "Sentiment Reasoning"
  - "Healthcare Conversations"
  - "Multimodal Sentiment Analysis"
  - "Rationale Generation"
  - "Explainable AI"
date: 2026-05-08
content_hash: 05b213623b56bb9b
---

# Sentiment Reasoning for Healthcare

**Conference**: ACL2025  
**arXiv**: [2407.21054](https://arxiv.org/abs/2407.21054)  
**Code**: [leduckhai/Sentiment-Reasoning](https://github.com/leduckhai/Sentiment-Reasoning)  
**Area**: NLP Understanding  
**Keywords**: Sentiment Reasoning, Healthcare Conversations, Multimodal Sentiment Analysis, Rationale Generation, Explainable AI

## TL;DR
This work introduces a new task termed "Sentiment Reasoning," which requires models to generate explanatory rationales while predicting sentiment labels for healthcare conversations. A multimodal sentiment analysis dataset comprising 30K samples across five languages is constructed. Rationale-augmented training improves classification accuracy and macro-F1 by approximately 2%.

## Background & Motivation
Transparency in AI decision-making is critical in the healthcare sector. Traditional sentiment analysis only outputs labels (positive/neutral/negative) without explaining the underlying judgment, which limits healthcare professionals' trust in and adoption of model predictions.

Existing problems:
- **Subjectivity and Complexity of Speech Emotion**: Emotional expression in speech is highly ambiguous due to variations in speaking styles and intonation, leading to low inter-annotator agreement even among humans (Cohen's kappa < 0.5).
- **High-Risk Healthcare Scenarios**: Misclassifications can lead to severe consequences, requiring models to not only predict accurately but also provide reasons.
- **Limitations of Existing Datasets**: Current multimodal sentiment datasets (MOSI 3K, CMU-MOSEI 23K, MELD 13K) are monolingual and non-medical, lacking large-scale multilingual resources specifically for healthcare dialogues.
- **Lack of Explainable Task Definitions**: CoT distillation research has shown that rationale-augmented training improves small model performance, but it has not been systematically applied to medical sentiment analysis.

Core Motivation: To define a new task—Sentiment Reasoning—unifying sentiment classification and rationale generation into a multi-task framework to provide both explainability and performance improvements.

## Method

### Overall Architecture
Given an input transcript (manual or ASR-generated), the model simultaneously performs two sub-tasks: (1) Sentiment Classification—outputting POSITIVE/NEUTRAL/NEGATIVE labels; and (2) Rationale Generation—outputting natural language explanations. Both end-to-end speech and cascaded (ASR $\rightarrow$ LM) pipelines are supported.

### Key Designs

#### Key Design 1: Multi-Architecture Multi-Task Training Strategy

Different rationale enrichment schemes are adopted for different model architectures:

- **Encoder-Decoder (ViT5, BARTpho)**: Utilizes the "Distilling Step-by-Step" multi-task training strategy. By prepending specific prefixes to the input, the model is guided to switch between "label output" and "rationale generation" tasks. Both tasks share encoder representations but trigger separate decoders via different prefixes.
- **Decoder-Only (Vistral-7B, vmlu-llm)**: Employs a "Post-thinking" strategy, where the rationale is appended after the label as the training target (`<LABEL> <RATIONALE>`). Compared to "Pre-thinking" (reasoning before label output), Post-thinking is more stable, reduces hallucination, and allows users to obtain the classification label starting from the very first generated token.
- **Encoder-Only (PhoBERT, ViHealthBERT)**: Used only as classification baselines without the capability to generate rationales.

#### Key Design 2: Systematic Study of Rationale Formats

Based on human-annotated rationales, GPT-3.5-turbo is utilized to generate two additional formats to study the impact of rationale formats on performance:

1. **Elaborated Rationales**: Expands human rationales into detailed 1-2 sentence versions, maintaining semantic consistency while increasing explanatory depth.
2. **Chain-of-Thought (CoT) Rationales**: A step-by-step reasoning format—(a) identify medical entities, (b) extract progression details of the entity from the transcript, and (c) derive the sentiment judgment based on the entity information and human rationales. Inspired by aspect-based sentiment analysis.

#### Key Design 3: Data Construction and Quality Control

Constructed based on the VietMed real-world doctor-patient dialogue dataset:
- Three annotators independently annotated the data. Due to the ambiguity of medical sentiments, an "all-member consensus" strategy was adopted instead of majority voting to merge labels.
- Five participants (three annotators + one linguist + one biomedical expert) discussed and determined the final labels and rationales.
- Manually translated into four languages (English, Chinese, German, French), totaling 30K samples.
- A TESOL-certified professional linguist formulated and continuously revised the annotation guidelines.

## Key Experimental Results

### Table 3: Baseline performance on manual transcripts (Vietnamese)

| Model | Type | Training Method | Acc. | Macro F1 | BERTScore |
|---|---|---|---|---|---|
| ViHealthBERT | Encoder | Label-only | 0.6752 | 0.6741 | — |
| PhoBERT | Encoder | Label-only | 0.6674 | 0.6651 | — |
| ViT5 | Encoder-Decoder | Label-only | 0.6628 | 0.6545 | — |
| ViT5 | Encoder-Decoder | Label + Rationale | 0.6633 | 0.6615 | 0.8093 |
| BARTpho | Encoder-Decoder | Label + Rationale | 0.6619 | 0.6585 | 0.8077 |
| Vistral-7B | Decoder | Label-only | 0.6716 | 0.6676 | — |
| **Vistral-7B** | **Decoder** | **Label + Rationale** | **0.6812** | **0.6781** | **0.8101** |
| vmlu-llm | Decoder | Label + Rationale | 0.6729 | 0.6687 | 0.8086 |

### Table 5: Impact of different rationale formats on performance

| Model | Rationale Format | Acc. | Macro F1 |
|---|---|---|---|
| Vistral-7B | Human Rationale | **0.6812** | **0.6781** |
| Vistral-7B | Elaborated Rationale | 0.6688 | 0.6685 |
| Vistral-7B | CoT Rationale | 0.6706 | 0.6670 |
| vmlu-llm | Human Rationale | 0.6729 | 0.6687 |
| vmlu-llm | Elaborated Rationale | **0.6867** | **0.6808** |
| vmlu-llm | CoT Rationale | 0.6821 | 0.6819 |

### Table 6: End-to-end audio-language models

| Model | Training Method | Acc. | Macro F1 |
|---|---|---|---|
| PhoWhisper | Label-only | 0.4651 | 0.4333 |
| Qwen2-Audio | Label-only | 0.5815 | 0.5688 |
| Qwen2-Audio | Label + Rationale | 0.5884 | 0.5781 |

## Key Findings

1. **Rationale-augmented training consistently improves performance**: On both manual and ASR transcripts, rationale augmentation yields an approximate +2% increase in accuracy and macro-F1 (statistically significant at $\alpha = 0.1$ via Student's t-test).
2. **ASR errors have limited impact**: Despite an ASR WER of 29.6%, the macro-F1 drops by only about 5 percentage points; furthermore, gains for the rationale-augmented models are more pronounced on ASR transcripts (average +0.85% Acc, +1.4% Macro F1).
3. **Semantics of generated rationales are close to human quality**: BERTScore remains stable around ~0.8 (with no significant difference between manual and ASR transcripts), showing that models capture similar semantics despite using different wordings.
4. **Rationale formats have negligible impact on performance**: There is no clear performance superiority among human, elaborated, or CoT rationales, which aligns with existing studies on CoT formats.
5. **Neutral category is the primary source of misclassifications**: The confusion matrix reveals that 23.43% of negative and 27.08% of positive samples are misclassified as neutral, reflecting the ambiguous emotional boundaries in clinical dialogues.
6. **Domain-specific pre-training is effective**: ViHealthBERT outperforms the general-purpose PhoBERT in both accuracy (+0.8%) and F1 (+0.9%), demonstrating the value of domain-specific pre-training in healthcare.

## Highlights & Insights

- **Contribution of Task Definition**: Generates a shift from pure classification to a "classification-plus-explanation" reasoning task, offering a new paradigm for explainability in healthcare AI.
- **Dataset Scale and Multilingual Coverage**: With 30K samples across 5 languages, it constitutes the largest multimodal sentiment analysis dataset to date, built on authentic doctor-patient consultations rather than laboratory settings.
- **Practical Quality Control Scheme**: In response to low inter-annotator agreement, a consensus-of-all merging strategy (involving linguists and biomedical experts) was adopted, which is more reliable than simple majority voting.
- **Practical Advantages of Post-thinking**: Placing the label before the rationale in generation allows rapid retrieval of predictions during inference without waiting for the full reasoning chain, balancing efficiency and explainability.

## Limitations & Future Work

- **Limitations of Cascaded Architecture**: The main experiments use an ASR $\rightarrow$ LM cascaded pipeline that only leverages semantic features. Acoustic features such as intonation and prosody are neglected, which might lead to a loss of sentiment-relevant signals.
- **High Complexity of Hybrid ASR**: The wav2vec 2.0 hybrid ASR system utilized requires a multi-step GMM-HMM $\rightarrow$ DNN-HMM pipeline, which is difficult for non-experts to replicate.
- **Inherent Gap in End-to-End Model Performance**: The end-to-end Qwen2-Audio approach achieves a macro-F1 of only ~0.578, lagging significantly behind the cascaded pipeline's ~0.678. End-to-end multimodal reasoning still has substantial room for improvement.
- **Dataset Imbalance**: Positive samples account for only ~20%, causing the positive class F1 score to perform the poorest across all models.
- **Single Data Source**: All samples originate from the single VietMed dataset, limiting scenario diversity; translation-generated multilingual versions may introduce translation bias.

## Related Work & Insights

- **CoT Distillation**: This work directly benefits from the findings of Distilling Step-by-Step (Hsieh 2023) and Post-thinking (Chen 2024), which show that rationale enrichment improves the performance of smaller models. Crucially, this study employs human-annotated rationales instead of LLM-generated CoTs, verifying that human rationales are equally effective.
- **Aspect-Based Sentiment Analysis**: The multi-step design of "identifying medical entities $\rightarrow$ tracking progression $\rightarrow$ determining sentiment" in the CoT rationale format is inspired by aspect-based sentiment instruction-tuning (Varia 2022).
- **Healthcare NLP Sentiment Analysis**: Earlier medical sentiment analysis mostly focused on text-only modalities like forums or patient reviews (Ali 2013, Biyani 2013). This work extends it to multimodal (audio + text) real-world conversations.
- **Insights**: Rationale-augmented training represents a low-cost strategy to enhance interpretability, which can be extended to other clinical AI tasks requiring transparency, such as medical QA and diagnostic assistance.

## Rating

- Novelty: ⭐⭐⭐ — Defining sentiment reasoning as a new task is meaningful, but it is essentially a direct application of multi-task learning and CoT distillation, offering limited methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — The experiments comprehensively cover three model architectures, various rationale formats, manual/ASR transcriptions, and end-to-end models. However, evaluations against larger LLMs (such as LLaMA or GPT series) are lacking.
- Writing Quality: ⭐⭐⭐⭐ — The paper is well-structured, the task definition is formally rigorous, and the details of the data construction workflow are exhaustively documented.
- Value: ⭐⭐⭐⭐ — The dataset contribution is notable (the largest multimodal, multilingual medical sentiment dataset to date), and the task definition holds valuable reference value for explainability in healthcare AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis](dot_absa_template.md)
- [\[ACL 2025\] Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification](analyzing_political_bias_in_llms_via_target-oriented_sentiment_classification.md)
- [\[ACL 2025\] Multi-Hop Reasoning for Question Answering with Hyperbolic Representations](multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)
- [\[ACL 2025\] Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering](self-critique_guided_iterative_reasoning_for_multi-hop_question_answering.md)
- [\[ACL 2025\] SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Analysis](syngraph_a_dynamic_graph-llm_synthesis_framework_for_sparse_streaming_user_senti.md)

</div>

<!-- RELATED:END -->
