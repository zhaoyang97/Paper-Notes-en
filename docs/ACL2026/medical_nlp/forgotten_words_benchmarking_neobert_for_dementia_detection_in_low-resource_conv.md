---
title: >-
  [Paper Note] Forgotten Words: Benchmarking NeoBERT for Dementia Detection in Low-Resource Conversational Filipino and English Speech
description: >-
  [ACL2026][Medical NLP][Dementia Detection] This paper evaluates TF-IDF, BERT, NeoBERT, XLM-R, and RoBERTa-Tagalog using a system of 4…
tags:
  - "ACL2026"
  - "Medical NLP"
  - "Dementia Detection"
  - "Filipino-English"
  - "NeoBERT"
  - "Cross-lingual Transfer"
  - "Bilingual Fine-tuning"
date: 2026-05-08
content_hash: 4120765d25bda6a4
---

# Forgotten Words: Benchmarking NeoBERT for Dementia Detection in Low-Resource Conversational Filipino and English Speech

**Conference**: ACL2026  
**arXiv**: [2605.26007](https://arxiv.org/abs/2605.26007)  
**Code**: https://github.com/rezsam09/Filipino-English-Dementia-Classification  
**Area**: Clinical NLP / Low-Resource Languages  
**Keywords**: Dementia Detection, Filipino-English, NeoBERT, Cross-lingual Transfer, Bilingual Fine-tuning  

## TL;DR
This paper evaluates TF-IDF, BERT, NeoBERT, XLM-R, and RoBERTa-Tagalog using a system of 4,000 English-Filipino parallel DementiaBank dialogue transcripts. It finds that cross-lingual robustness in dementia detection stems primarily from language coverage during training rather than modern encoder architectures.

## Background & Motivation
**Background**: Detecting dementia from spontaneous speech or dialogue transcripts is a significant direction in clinical NLP, as cognitive decline is reflected in linguistic phenomena such as reduced lexical diversity, repetition, pauses, syntactic simplification, and decreased referential coherence. The Cookie Theft picture description task from DementiaBank is one of the most commonly used data sources in this field.

**Limitations of Prior Work**: The vast majority of dementia NLP systems are English-centric. There is a lack of systematic evaluation for low-resource languages, Southeast Asian languages, and real-world bilingual code-switching scenarios. While Filipino-English code-switching is prevalent in daily and clinical communication in the Philippines, no prior NLP-based dementia detection work has specifically studied Filipino speech.

**Key Challenge**: High scores for a model on the English DementiaBank do not necessarily mean it has learned language-agnostic signals of cognitive decline. Performance degradation may result from language transfer failure, pre-training corpus bias, clinical domain shifts, or differences in data acquisition. To isolate language shift, English and Filipino samples must be as parallel as possible in terms of clinical content, task structure, and class proportions.

**Goal**: The authors construct a controlled bilingual dataset to compare English-only, multilingual, language-matched, and modern encoders under three training settings: English monolingual, Filipino monolingual, and English-Filipino bilingual. Particular focus is placed on whether modern encoder architectures like NeoBERT are truly more robust under clinical and cross-lingual conditions.

**Key Insight**: Instead of adding manual linguistic features or complex acoustic features, the paper fixes the same preprocessing and fine-tuning pipeline to ensure that performance differences primarily reflect representation and pre-training language exposure. This allows for a clearer distinction between "superior architecture" and "sufficient language coverage."

**Core Idea**: On parallel bilingual data where both the domain and task are controlled, if monolingual training still fails cross-linguistically, it indicates that the bottleneck is language alignment in the representation space; if bilingual fine-tuning eliminates the gap, then training language coverage is more critical than model architecture updates.

## Method
The paper constructs a balanced bilingual binary classification setup. There are 2,000 samples per language, with 1,000 dementia-positive and 1,000 healthy control samples each, totaling 4,000. English samples are sourced from DementiaBank, while Filipino samples are manually translated from the full 2,000 English transcripts. Translations are required to preserve discourse-level cognitive decline markers such as repetitions, hesitations, false starts, and syntactic degradation.

### Overall Architecture
All transcripts undergo the same preprocessing: Unicode normalization, whitespace normalization, and lowercasing. Filled pauses, repetitions, and hesitation markers are preserved. Stemming, lemmatization, or parsing are avoided to prevent erasing diagnostic signals. The maximum input length is truncated to 128 tokens.

The models cover five categories: TF-IDF + Logistic Regression as an interpretable lexical baseline; BERT-base-uncased and NeoBERT representing English pre-training (with NeoBERT including modern designs like RoPE, Pre-LayerNorm, RMSNorm, and SwiGLU); XLM-RoBERTa as a 100-language multilingual baseline; and RoBERTa-Tagalog representing Filipino language-matched pre-training.

Three training settings are employed: English-only, Filipino-only, and English+Filipino bilingual. Evaluations include in-domain same-language, zero-shot cross-lingual, and bilingual mixed-language tests. Primary metrics are Accuracy and Macro-F1, while class-wise F1 and dementia recall are also reported to prevent average metrics from masking clinical sensitivity issues.

### Key Designs
1. **Construction of Parallel Bilingual Data**:
	- **Function**: Treats language transfer as the primary variable, reducing interference from task content and clinical domain differences.
	- **Mechanism**: Filipino data is not sourced from a different corpus but via manual translation of the full DementiaBank transcripts, maintaining class distribution, discourse structure, elicitation task, and clinical content identical to the English version.
	- **Design Motivation**: If independent clinical corpora from different languages were compared directly, performance differences might stem from patient populations, task designs, or recording protocols. Parallel translation allows the paper to state more definitively that failures arise from language shift.

2. **Unified Preprocessing Retaining Disfluency**:
	- **Function**: Preserves linguistic degradation signals upon which dementia detection heavily relies.
	- **Mechanism**: Retains repetitions, hesitations, pause markers, and syntactic fragments. Machine translation is avoided because it might "polish" disfluent speech into fluent text.
	- **Design Motivation**: Dementia-related linguistic signals are often hidden in disfluency and organizational failure. Over-normalization would turn the task into standard text classification, weakening clinical significance.

3. **Comparative Experiments on Architecture vs. Language Coverage**:
	- **Function**: Determines whether performance stems from model modernization or language coverage in pre-training and fine-tuning.
	- **Mechanism**: BERT and NeoBERT are both English-only but have different architectures; XLM-R has explicit multilingual pre-training; RoBERTa-Tagalog has language-matched pre-training. All transformers use masked mean pooling, dropout of 0.1, and a linear head.
	- **Design Motivation**: If NeoBERT outperforms BERT but remains cross-linguistically unstable, it suggests that modern architectures improving monolingual fit do not equate to cross-lingual clinical robustness. If bilingual training causes all models to converge, then supervised coverage is the core factor.

### Loss & Training
TF-IDF uses unigram + bigram, sublinear TF scaling, a minimum document frequency of 2, a maximum document frequency of 0.95, and a maximum vocabulary of 20,000. It utilizes $l_2$ regularized Logistic Regression with a liblinear solver for up to 2,000 iterations.

Transformer models are fine-tuned end-to-end using attention-masked mean pooling: $h=\sum_i m_iH_i/\sum_i m_i$, followed by dropout and a linear classification head. The optimization objective is standard cross-entropy: $L(\theta)=-E_{(x,y)}\log p_\theta(y|x)$, using the AdamW optimizer.

Hyperparameters are selected via grid search, with learning rates including $5e^{-6}, 6e^{-6}, 1e^{-5}, 2e^{-5}, 3e^{-5}$, weight decay of $1e^{-2}$ and $1e^{-5}$, and batch sizes of 4 and 8. Training lasts up to 10 epochs. Validation Macro-F1 early stopping is used during the search phase, and stratified 10-fold cross-validation is used for final reporting to measure stability.

## Key Experimental Results

### Main Results
| Model | Training Language | EN Macro-F1 | TL Macro-F1 | Combined Macro-F1 | Gap |
|--------|------|------|----------|------|------|
| TF-IDF + LR | EN | 0.930±0.013 | 0.649±0.008 | 0.836±0.005 | 0.281 |
| BERT | EN | 0.952±0.014 | 0.455±0.012 | 0.744±0.008 | 0.497 |
| NeoBERT | EN | 0.952±0.013 | 0.617±0.109 | 0.802±0.045 | 0.335 |
| XLM-RoBERTa | EN | 0.948±0.017 | 0.936±0.018 | 0.942±0.016 | 0.013 |
| RoBERTa-Tagalog | EN | 0.951±0.014 | 0.934±0.005 | 0.942±0.015 | 0.017 |
| BERT | EN+TL | 0.954±0.009 | 0.984±0.009 | 0.969±0.007 | 0.030 |
| XLM-RoBERTa | EN+TL | 0.953±0.010 | 0.990±0.007 | 0.972±0.006 | 0.037 |
| RoBERTa-Tagalog | EN+TL | 0.958±0.010 | 0.988±0.007 | 0.973±0.006 | 0.030 |
| NeoBERT | EN+TL | 0.956±0.015 | 0.983±0.009 | 0.970±0.007 | 0.027 |

### Clinical Sensitivity and Class Performance
| Model / Setting | Healthy F1 (TL) | Dementia F1 (TL) | Dementia Recall (TL) | Interpretation |
|------|---------|------|------|------|
| BERT EN→TL | 0.216 | 0.695 | 0.931 | High surface recall, but classifies almost all Filipino samples as dementia; Healthy class collapses |
| NeoBERT EN→TL | 0.504 | 0.729 | 0.939 | Better than BERT, but with high variance and an unstable decision boundary |
| XLM-RoBERTa EN→TL | 0.950 | 0.947 | 0.920 | Better class balance, suitable for screening scenarios |
| RoBERTa-Tagalog EN→TL | 0.952 | 0.950 | 0.928 | Language-matched pre-training also supports English-to-Filipino transfer |
| NeoBERT EN+TL | 0.956 | 0.955 | 0.938 | Regains stability after bilingual training |

### Key Findings
- English-trained BERT achieves a Macro-F1 of 0.952 on English but only 0.455 on Filipino, indicating that strong in-domain performance does not guarantee cross-lingual clinical robustness.
- NeoBERT's architectural modernization does not solve cross-lingual issues; English-trained NeoBERT on Filipino yields 0.617±0.109. High variance suggests it may form tighter boundaries on the English side.
- XLM-RoBERTa and RoBERTa-Tagalog show minimal gaps in English-to-Filipino transfer (0.013 and 0.017, respectively).
- Bilingual fine-tuning causes the combined Macro-F1 of all transformers to converge to 0.969-0.973, with differences nearly vanishing. This implies that the bottleneck is primarily language coverage and representation alignment.
- TF-IDF with bilingual training also reaches 0.954, though it remains lower than transformers, showing that surface vocabulary retains some dementia signals, but stable transfer still requires contextual representations.

## Highlights & Insights
- **Clean Experimental Control**: Manual translation maintains clinical content and discourse structure, making the conclusion regarding "language transfer failure" more convincing.
- **Valuable Negative Results for NeoBERT**: Modern encoder architectures can improve general benchmarks but do not automatically provide robustness in low-resource clinical cross-lingual tasks.
- **Class-Level Analysis Prevents Misinterpretation**: While BERT EN→TL shows high dementia recall, its Healthy F1 is only 0.216, indicating over-prediction of the positive class rather than understanding of Filipino dementia cues.
- **Interesting Results for RoBERTa-Tagalog**: Even though it was pre-trained only on Filipino corpora, it can handle English-to-Filipino transfer, likely because Filipino itself contains many English loanwords and code-switching structures.

## Limitations & Future Work
- Filipino data is based on human-translated DementiaBank, not natural clinical speech from local Filipino patients. Even if disfluency is preserved, semantic content and conversational structure are derived from English source data.
- The data scale is only 4,000 samples. Although 10-fold CV was performed, generalization for cross-lingual and clinical deployment still requires verification with larger local cohorts.
- The paper only studies text transcripts and does not incorporate acoustic and non-linguistic cues such as pitch variance, pause duration, or phonation rate; real-world dementia screening likely requires multimodal speech.
- Insufficient model explanation mechanisms. Feature attribution, error case audits, and linguistic marker analysis are needed before clinical deployment to confirm the model does not rely on translation artifacts or data construction artifacts.
- The current task is binary classification and does not cover MCI, different dementia stages, longitudinal changes, or realistic variations in code-switching proportions.

## Related Work & Insights
- **vs ADReSS-M / English-Greek transfer**: These works mostly transfer between Indo-European languages; Filipino-English better tests typological distance and code-switching scenarios.
- **vs Nepali / Amis low-resource dementia detection**: These studies use translation or augmentation to improve low-resource performance, but this paper more systematically compares zero-shot cross-lingual, language-matched pre-training, and bilingual fine-tuning.
- **vs ClinicalBERT / AD-BERT**: Domain adaptation or disease-specific pre-training is not equivalent to low-resource language robustness; this paper suggests clinical NLP cannot expand only within the English domain.
- **Insight**: In low-resource clinical NLP, priorities should likely be the collection of local language supervision and the retention of real disfluency and code-switching, rather than blindly switching to larger English models.

## Rating
- Novelty: ⭐⭐⭐⭐☆ First evaluation of a Filipino dementia NLP system and places NeoBERT in a clinical cross-lingual context; the method itself is a benchmark evaluation rather than a new model.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Models, training settings, 10-fold CV, and class-level metrics are all comprehensive; the main drawback is that Filipino data comes from translation rather than native clinical collection.
- Writing Quality: ⭐⭐⭐⭐☆ Problem definition is clear, and the interpretation of results is disciplined, particularly the helpful breakdown of Macro-F1 vs. clinical recall.
- Value: ⭐⭐⭐⭐☆ Highly insightful for low-resource clinical NLP and multilingual medical screening, clearly demonstrating that language coverage is more critical than architectural upgrades.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)
- [\[ACL 2026\] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)
- [\[ACL 2026\] MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts](medfact_benchmarking_the_fact-checking_capabilities_of_large_language_models_on_.md)
- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)
- [\[ICLR 2026\] From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents](../../ICLR2026/medical_nlp/from_conversation_to_query_execution_benchmarking_user_and_tool_interactions_for.md)

</div>

<!-- RELATED:END -->
