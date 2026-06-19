---
title: >-
  [Paper Note] Forgotten Words: Benchmarking NeoBERT for Dementia Detection in Low-Resource Conversational Filipino and English Speech
description: >-
  [ACL 2026][Medical NLP][Filipino-English] This paper evaluates TF-IDF, BERT, NeoBERT, XLM-R, and RoBERTa-Tagalog using 4,000 parallel English-Filipino DementiaBank dialogue transcriptions, finding that cross-lingual robustness in dementia detection primarily stems from language coverage during the training stage rather than modern encoder architectures.
tags:
  - ACL 2026
  - Medical NLP
  - Filipino-English
  - NeoBERT
date: 2026-05-08
content_hash: 551a5ec31cc02ab2
---
# Forgotten Words: Benchmarking NeoBERT for Dementia Detection in Low-Resource Conversational Filipino and English Speech

**Conference**: ACL2026  
**arXiv**: [2605.26007](https://arxiv.org/abs/2605.26007)  
**Code**: https://github.com/rezsam09/Filipino-English-Dementia-Classification  
**Area**: Clinical NLP / Low-Resource Languages  
**Keywords**: Dementia Detection, Filipino-English, NeoBERT, Cross-lingual Transfer, Bilingual Fine-tuning  

## TL;DR
This paper evaluates TF-IDF, BERT, NeoBERT, XLM-R, and RoBERTa-Tagalog using 4,000 parallel English-Filipino DementiaBank dialogue transcriptions, finding that cross-lingual robustness in dementia detection primarily stems from language coverage during the training stage rather than modern encoder architectures.

## Background & Motivation
**Background**: Detecting dementia from spontaneous speech or dialogue transcriptions is a critical direction in clinical NLP, as cognitive decline is reflected in linguistic phenomena such as reduced lexical diversity, repetitions, pauses, syntactic simplification, and decreased referential coherence. The Cookie Theft picture description task from DementiaBank is one of the most commonly used data sources in this field.

**Limitations of Prior Work**: The vast majority of dementia NLP systems are English-centric. There is a lack of systematic evaluation for low-resource languages, Southeast Asian languages, and real-world bilingual code-switching scenarios. While Filipino-English code-switching is prevalent in daily and clinical communication in the Philippines, no prior NLP-based dementia detection work has specifically investigated Filipino speech.

**Key Challenge**: High scores on the English DementiaBank do not necessarily mean the model has learned language-agnostic signals of cognitive decline. Performance drops may result from failed language transfer, pre-training corpora bias, clinical domain shifts, or differences in data collection. To isolate the language shift, English and Filipino samples must be as parallel as possible regarding clinical content, task structure, and class proportions.

**Goal**: The authors construct a controlled bilingual dataset to compare English-only, multilingual, language-matched, and modern encoders under three training settings: English monolingual, Filipino monolingual, and English-Filipino bilingual. A particular focus is placed on whether modern encoder architectures like NeoBERT are truly more robust under clinical and cross-lingual conditions.

**Key Insight**: Rather than adding handcrafted linguistic features or complex acoustic features, the paper fixes the same preprocessing and fine-tuning pipeline. This allows performance differences to primarily reflect representation and pre-training language exposure, clearly distinguishing "stronger architecture" from "sufficient language coverage."

**Core Idea**: On parallel bilingual data where both domain and task are controlled, if monolingual training still fails cross-lingually, the bottleneck is language alignment in the representation space. If bilingual fine-tuning eliminates the gap, then training language coverage is more critical than model architecture updates.

## Method

### Overall Architecture
The paper does not propose a new model but builds a controlled bilingual evaluation platform to answer a specific question: does cross-lingual robustness in dementia detection come from modern encoder architectures or language coverage during training? To this end, the authors construct a balanced bilingual binary classification setup—2,000 samples per language, with 1,000 dementia-positive and 1,000 healthy control samples each, totaling 4,000. English samples are taken from DementiaBank, while Filipino samples are manually translated from the 2,000 English transcriptions, intentionally preserving discourse-level markers of cognitive decline such as repetitions, hesitations, false starts, and syntactic degradation.

All transcriptions undergo the same preprocessing pipeline: Unicode normalization, whitespace normalization, and lowercasing, while retaining filled pauses, repetitions, and hesitation markers without performing stemming, lemmatization, or parsing. Inputs are truncated to 128 tokens. Five categories of models are deployed: TF-IDF + Logistic Regression as an interpretable lexical baseline, BERT-base and NeoBERT representing English pre-training (NeoBERT incorporates modern designs like RoPE, Pre-LayerNorm, RMSNorm, and SwiGLU), XLM-RoBERTa as a multilingual baseline across 100 languages, and RoBERTa-Tagalog for language-matched pre-training in Filipino. Each model is evaluated across three training settings: English-only, Filipino-only, and English+Filipino bilingual, tested on in-domain, zero-shot cross-lingual, and bilingual mixed-language scenarios. Primary metrics are Accuracy and Macro-F1, with additional reporting of class-wise F1 and dementia recall to prevent average metrics from masking clinical sensitivity.

### Key Designs

**1. Parallel Bilingual Data Construction: Isolating "Language Transfer" as the Sole Variable**

Comparing independent clinical corpora from two languages would contaminate performance differences with confounding factors like patient populations, task designs, and recording protocols. This paper's approach is to avoid separate corpora and instead use manual translation of full DementiaBank transcriptions, forcing the Filipino side to align with the English side in class distribution, discourse structure, elicitation task, and clinical content. This ensures that the only systematic change is the language itself, allowing cross-lingual performance drops to be cleanly attributed to language shift rather than data collection discrepancies.

**2. Uniform Preprocessing Retaining Disfluency: Avoiding "Polishing Away" Diagnostic Signals**

Linguistic signals of dementia are often hidden in disfluencies and organizational failures—repetitions, hesitations, pause markers, and syntactic fragments are themselves evidence of cognitive decline. Preprocessing deliberately retains these markers and avoids machine translation (which tends to smooth disfluent speech into fluent text, potentially erasing diagnostic clues). Over-normalization, common in general text classification, would degrade the task into identifying semantic content and weaken clinical significance; retaining disfluency allows models to learn actual degradation patterns.

**3. Controlled Experiments on Architecture vs. Language Coverage**

To determine if robustness comes from architecture or language coverage, these factors must be separable in model selection. BERT and NeoBERT are both English-only but represent old vs. new architectures; comparing them isolates the contribution of "architectural modernization." XLM-R (explicit multilingual pre-training) vs. RoBERTa-Tagalog (language-matched pre-training) reveals the contribution of "language coverage." To eliminate other interference, all transformers use masked mean pooling, a dropout of 0.1, and a linear head. The logic is clean: if NeoBERT outperforms BERT but remains unstable cross-lingually, modern architecture merely fits the English side tighter without providing cross-lingual clinical robustness.

### Loss & Training
TF-IDF utilizes unigrams and bigrams with sublinear TF scaling, a minimum document frequency of 2, a maximum document frequency of 0.95, and a maximum vocabulary of 20,000. It is paired with $l_2$-regularized Logistic Regression using a liblinear solver for up to 2,000 iterations.

Transformer models are fine-tuned end-to-end using attention-masked mean pooling: $h=\sum_i m_iH_i/\sum_i m_i$, followed by dropout and a linear classification head. The optimization objective is standard cross-entropy: $L(\theta)=-E_{(x,y)}\log p_\theta(y|x)$ using the AdamW optimizer.

Hyperparameters were selected via grid search, including learning rates ($5e^{-6}$ to $3e^{-5}$), weight decay ($1e^{-2}$ and $1e^{-5}$), and batch sizes (4 and 8). Training lasted up to 10 epochs, using validation Macro-F1 for early stopping, and final results were reported using stratified 10-fold cross-validation for stability.

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
| BERT EN→TL | 0.216 | 0.695 | 0.931 | High surface recall, but predicts almost all Filipino samples as dementia; Healthy class collapses. |
| NeoBERT EN→TL | 0.504 | 0.729 | 0.939 | Better than BERT, but high variance; unstable decision boundaries. |
| XLM-RoBERTa EN→TL | 0.950 | 0.947 | 0.920 | Better class balance, suitable for screening scenarios. |
| RoBERTa-Tagalog EN→TL | 0.952 | 0.950 | 0.928 | Language-matched pre-training supports English-to-Filipino transfer. |
| NeoBERT EN+TL | 0.956 | 0.955 | 0.938 | Stability restored after bilingual training. |

### Key Findings
- English-trained BERT achieves a Macro-F1 of 0.952 on English but only 0.455 on Filipino, indicating that strong in-domain performance does not guarantee cross-lingual clinical robustness.
- NeoBERT's architectural modernization does not solve cross-lingual issues; English-trained NeoBERT reaches 0.617±0.109 on Filipino with the highest variance, suggesting it may form tighter English-side boundaries.
- XLM-RoBERTa and RoBERTa-Tagalog show minimal gaps in English-to-Filipino transfer (0.013 and 0.017, respectively).
- Bilingual fine-tuning causes all transformers to converge to a combined Macro-F1 of 0.969-0.973, nearly eliminating differences and proving the bottleneck is language coverage and representation alignment.
- TF-IDF with bilingual training also reaches 0.954, but remains lower than transformers, indicating that surface vocabulary captures partial signals but stable transfer requires contextual representations.

## Highlights & Insights
- **Clean Experimental Control**: Using manual translation to maintain clinical content and discourse structure makes the "failed language transfer" conclusion more convincing.
- **Valuable Negative Results for NeoBERT**: Modern encoder architectures improve general benchmarks but do not automatically provide robustness in low-resource clinical cross-lingual tasks.
- **Class-level Analysis Prevents Misinterpretation**: BERT's high dementia recall in the EN→TL setting is misleading, as the Healthy F1 of 0.216 reveals the model is over-predicting positive cases rather than understanding Filipino dementia cues.
- **Intriguing Results for RoBERTa-Tagalog**: Even though pre-trained only on Filipino, it successfully handles English-to-Filipino transfer, likely due to the high volume of English loanwords and code-switching structures in Filipino.

## Limitations & Future Work
- Filipino data is based on translated DementiaBank rather than natural clinical speech from local patients; semantic content and session structure remain derived from English sources.
- Data size is limited to 4,000 samples; although 10-fold CV was used, generalization to real clinical deployment requires larger local cohorts.
- The study focuses on text transcriptions and excludes acoustic/non-linguistic cues like pitch variance, pause duration, and phonation rate; actual screening likely requires multimodal speech.
- Insufficient model interpretability. Clinical deployment requires feature attribution and linguistic marker analysis to ensure models do NOT rely on translation artifacts or data construction artifacts.
- The current task is binary classification and does not cover MCI, different dementia stages, longitudinal changes, or real-world code-switching ratios.

## Related Work & Insights
- **vs. ADReSS-M / English-Greek transfer**: Most prior work focuses on Indo-European language transfer; Filipino-English better tests typological distance and code-switching.
- **vs. Nepali / Amis Low-resource Detection**: Those studies use translation or augmentation, whereas this paper systematically compares zero-shot, language-matched, and bilingual fine-tuning.
- **vs. ClinicalBERT / AD-BERT**: Domain adaptation or disease-specific pre-training is not equivalent to low-resource robustness; this paper suggests clinical NLP cannot expand solely within the English domain.
- **Insight**: In low-resource clinical NLP, the priority should be collecting local language supervision and preserving authentic disfluency/code-switching rather than blindly adopting larger English models.

## Rating
- Novelty: ⭐⭐⭐⭐☆ First evaluation of a Filipino dementia NLP system; assesses NeoBERT in a clinical cross-lingual context.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Comprehensive models and training settings; limited by the use of translated rather than natively collected Filipino data.
- Writing Quality: ⭐⭐⭐⭐☆ Clear problem definition and cautious interpretation, especially regarding Macro-F1 vs. clinical recall.
- Value: ⭐⭐⭐⭐☆ Highly insightful for low-resource clinical NLP; clarifies that language coverage is more critical than architecture upgrades.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)
- [\[ACL 2026\] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)
- [\[ACL 2026\] MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts](medfact_benchmarking_the_fact-checking_capabilities_of_large_language_models_on_.md)
- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)
- [\[ICLR 2026\] From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents](../../ICLR2026/medical_nlp/from_conversation_to_query_execution_benchmarking_user_and_tool_interactions_for.md)

</div>

<!-- RELATED:END -->
