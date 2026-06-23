---
title: >-
  [Paper Note] Forgotten Words: Benchmarking NeoBERT for Dementia Detection in Low-Resource Conversational Filipino and English Speech
description: >-
  [ACL 2026][Medical NLP][Filipino-English] This paper evaluates TF-IDF, BERT, NeoBERT, XLM-R, and RoBERTa-Tagalog using 4,000 parallel English-Filipino DementiaBank dialogue transcripts. It finds that cross-lingual robustness in dementia detection primarily stems from language coverage during training rather than modern encoder architectures.
tags:
  - ACL 2026
  - Medical NLP
  - Filipino-English
  - NeoBERT
date: 2026-05-08
content_hash: 3f6e702b0e51534c
---
# Forgotten Words: Benchmarking NeoBERT for Dementia Detection in Low-Resource Conversational Filipino and English Speech

**Conference**: ACL2026  
**arXiv**: [2605.26007](https://arxiv.org/abs/2605.26007)  
**Code**: https://github.com/rezsam09/Filipino-English-Dementia-Classification  
**Area**: Clinical NLP / Low-resource Languages  
**Keywords**: Dementia Detection, Filipino-English, NeoBERT, Cross-lingual Transfer, Bilingual Fine-tuning  

## TL;DR
This paper evaluates TF-IDF, BERT, NeoBERT, XLM-R, and RoBERTa-Tagalog using 4,000 parallel English-Filipino DementiaBank dialogue transcripts. It finds that cross-lingual robustness in dementia detection primarily stems from language coverage during training rather than modern encoder architectures.

## Background & Motivation
**Background**: Detecting dementia from spontaneous speech or dialogue transcripts is a critical direction in clinical NLP, as cognitive decline manifests in linguistic phenomena such as reduced lexical diversity, repetitions, pauses, syntactic simplification, and decreased referential coherence. The Cookie Theft picture description task from DementiaBank is one of the most widely used data sources in this field.

**Limitations of Prior Work**: The vast majority of dementia NLP systems are English-centric, with a lack of systematic evaluation for low-resource languages, Southeast Asian languages, and real-world bilingual code-switching scenarios. While Filipino-English code-switching is prevalent in everyday and clinical communication in the Philippines, no prior NLP-based dementia detection work has specifically studied Filipino speech.

**Key Challenge**: High performance on the English DementiaBank dataset does not guarantee that a model has learned language-agnostic signals of cognitive decline. Performance degradation may result from failed language transfer, pre-training corpus bias, clinical domain shifts, or differences in data acquisition. To isolate the language shift, it is necessary to ensure that English and Filipino samples are as parallel as possible regarding clinical content, task structure, and class distribution.

**Goal**: The authors construct a controlled bilingual dataset to compare English-only, multilingual, language-matched, and modernized encoders across three training settings: English monolingual, Filipino monolingual, and English-Filipino bilingual. The focus is specifically on whether modern encoder architectures like NeoBERT are truly more robust under clinical and cross-lingual conditions.

**Key Insight**: Rather than incorporating handcrafted linguistic features or complex acoustic features, this paper fixes the same preprocessing and fine-tuning pipeline so that performance differences primarily reflect representation and pre-training language exposure. This allows for a clearer distinction between "superior architecture" and "sufficient language coverage."

**Core Idea**: If monolingual training still fails cross-lingually on domain- and task-controlled parallel bilingual data, the bottleneck is language alignment in the representation space; if bilingual fine-tuning eliminates the gap, then training language coverage is more critical than the novelty of the model architecture.

## Method

### Overall Architecture
Instead of proposing a new model, this paper builds a controlled bilingual evaluation framework to answer a specific question: Does cross-lingual robustness in dementia detection come from modernized encoder architectures or language coverage during training? To achieve this, the authors construct a balanced bilingual binary classification setup—2,000 samples per language, with 1,000 dementia-positive and 1,000 healthy control samples each, totaling 4,000. English samples are sourced from DementiaBank, while Filipino samples are manually translated from the full 2,000 English transcripts, intentionally preserving discourse-level markers of cognitive decline such as repetitions, hesitations, false starts, and syntactic degradation.

All transcripts follow a unified preprocessing pipeline: Unicode normalization, whitespace normalization, and lowercasing, while retaining filled pauses, repetitions, and hesitation markers without performing stemming, lemmatization, or parsing. Inputs are truncated to 128 tokens. Five types of models are evaluated: TF-IDF + Logistic Regression as an interpretable lexical baseline; BERT-base and NeoBERT representing English pre-training (with NeoBERT featuring modern designs like RoPE, Pre-LayerNorm, RMSNorm, and SwiGLU); XLM-RoBERTa as a multilingual baseline covering 100 languages; and RoBERTa-Tagalog for language-matched pre-training. Each model is evaluated under three training settings: English-only, Filipino-only, and English+Filipino bilingual, tested across in-domain, zero-shot cross-lingual, and bilingual mixed-language scenarios. The primary metrics are Accuracy and Macro-F1, with additional reporting of class-wise F1 and dementia recall to ensure clinical sensitivity is not obscured by average metrics.

### Key Designs

**1. Parallel Bilingual Data Construction: Isolating "Language Transfer" as the Sole Variable**

If clinical corpora from two different languages were compared directly, performance gaps would be contaminated by confounding factors like patient cohorts, task designs, and recording protocols. This paper avoids this by manually translating the complete DementiaBank transcripts, forcing the Filipino side to align perfectly with the English side in class distribution, discourse structure, elicitation task, and clinical content. Consequently, the only systematic variation between the two sides is the language itself, allowing cross-lingual performance drops to be cleanly attributed to language shift.

**2. Disfluency-Preserving Preprocessing: Avoiding "Polishing Away" Diagnostic Signals**

Linguistic signals of dementia are often hidden in disfluencies and organizational failures—repetitions, hesitations, pause markers, and syntactic fragments are themselves evidence of cognitive decline. Preprocessing deliberately retains these markers and avoids machine translation (MT) for the Filipino side, as MT tends to smooth disfluent speech into fluent text, effectively erasing diagnostic clues. Excessive normalization would degrade the task into simple semantic recognition, weakening clinical utility; retaining disfluency allows models to learn actual degradation patterns.

**3. Controlled Experiment on Architecture vs. Language Coverage: Separating Hypotheses with a Model Matrix**

To determine if robustness stems from architecture or language coverage, these factors must be separable in model selection. BERT and NeoBERT are both English-only but represent old vs. new architectures; comparing them isolates the contribution of "architectural modernization." XLM-R (explicit multilingual pre-training) vs. RoBERTa-Tagalog (language-matched pre-training) captures the contribution of "language coverage." To eliminate other interference, all transformers utilize masked mean pooling, a 0.1 dropout rate, and a linear head. The logic is straightforward: if NeoBERT outperforms BERT but remains unstable cross-lingually, modern architectures simply fit the English side better rather than providing cross-lingual clinical robustness.

### Loss & Training
TF-IDF uses unigrams + bigrams, sublinear TF scaling, a minimum document frequency of 2, a maximum document frequency of 0.95, and a maximum vocabulary of 20,000. It is paired with an $l_2$ regularized Logistic Regression using the liblinear solver for up to 2,000 iterations.

Transformer models are fine-tuned end-to-end using attention-masked mean pooling: $h=\sum_i m_iH_i/\sum_i m_i$, followed by dropout and a linear classification head. The optimization objective is standard cross-entropy: $L(\theta)=-E_{(x,y)}\log p_\theta(y|x)$, using the AdamW optimizer.

Hyperparameters were selected via grid search, including learning rates of $5e^{-6}$, $6e^{-6}$, $1e^{-5}$, $2e^{-5}$, and $3e^{-5}$; weight decay of $1e^{-2}$ and $1e^{-5}$; and batch sizes of 4 and 8. Training lasted up to 10 epochs, with early stopping based on validation Macro-F1. Final results are reported using stratified 10-fold cross-validation to measure stability.

## Key Experimental Results

### Main Results

| Model | Training Lang | EN Macro-F1 | TL Macro-F1 | Combined Macro-F1 | Gap |
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

### Key Findings
- English-trained BERT achieves a Macro-F1 of 0.952 on English but drops to 0.455 on Filipino, indicating that strong in-domain performance does not guarantee cross-lingual clinical robustness.
- NeoBERT's architectural modernization does not resolve cross-lingual issues; English-trained NeoBERT reaches only 0.617±0.109 on Filipino with the highest variance, suggesting it may form a tighter boundary around English-side features.
- XLM-RoBERTa and RoBERTa-Tagalog show minimal gaps (0.013 and 0.017) in English-to-Filipino transfer.
- Bilingual fine-tuning causes all transformers to converge to a combined Macro-F1 of 0.969-0.973, with performance differences nearly disappearing. This suggests the bottleneck is language coverage and representation alignment rather than architecture.
- TF-IDF also achieves 0.954 under bilingual training but remains lower than transformers, indicating that while surface vocabulary retains some dementia signals, stable transfer requires contextual representations.

## Highlights & Insights
- **Clean Experimental Control**: Human translation maintains clinical content and discourse structure, making the conclusion regarding "language transfer failure" much more convincing.
- **Value of NeoBERT's Negative Results**: Modernized encoder architectures can improve general benchmarks but do not automatically provide robustness in low-resource clinical cross-lingual tasks.
- **Class-wise Analysis Avoids Misinterpretation**: While BERT EN→TL shows high dementia recall, its Healthy F1 is only 0.216, indicating the model is over-predicting positives rather than understanding Filipino dementia cues.
- **Interesting Findings for RoBERTa-Tagalog**: Even when pre-trained only on Filipino, it successfully handles English-to-Filipino transfer, likely because Filipino contains many English loanwords and code-switching structures.

## Limitations & Future Work
- Filipino data is based on translated DementiaBank rather than natural clinical speech from local Filipino patients. Even with preserved disfluency, semantic content reflects the English source.
- The data scale is 4,000 samples; while 10-fold CV was used, generalization to cross-lingual clinical deployment needs validation with larger local cohorts.
- The study focuses on text transcripts and currently excludes acoustic or non-verbal cues like pitch variance, pause duration, and phonation rate, which are likely necessary for real-world screening.
- Lack of model interpretability. Clinical deployment requires feature attribution and error analysis to ensure models do not rely on translation artifacts.
- The task is binary classification and does not cover MCI, different stages of dementia, or longitudinal changes.

## Related Work & Insights
- **vs. ADReSS-M / English-Greek transfer**: These works primarily transfer between Indo-European languages; Filipino-English better tests typological distance and code-switching scenarios.
- **vs. Nepali / Amis low-resource dementia detection**: While those studies use translation or augmentation, this paper more systematically compares zero-shot cross-lingual, language-matched pre-training, and bilingual fine-tuning.
- **vs. ClinicalBERT / AD-BERT**: Domain adaptation or disease-specific pre-training is not equivalent to low-resource robustness; this paper suggests clinical NLP cannot expand solely within the English domain.
- **Insight**: In low-resource clinical NLP, priority should perhaps be given to collecting local language supervision and preserving authentic disfluency and code-switching rather than blindly adopting larger English-centric models.

## Rating
- Novelty: ⭐⭐⭐⭐☆ First evaluation of a Filipino dementia NLP system; puts NeoBERT into a clinical cross-lingual context.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Comprehensive models, settings, and metrics; the main drawback is the use of translated rather than native clinical data.
- Writing Quality: ⭐⭐⭐⭐☆ Clear problem definition and disciplined interpretation of results.
- Value: ⭐⭐⭐⭐☆ Highly insightful for low-resource clinical NLP and multilingual medical screening; clearly demonstrates that language coverage outweighs architectural upgrades.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

## Related Papers

- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)
- [\[ACL 2026\] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)
- [\[ACL 2026\] MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts](medfact_benchmarking_the_fact-checking_capabilities_of_large_language_models_on_.md)
- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)
- [\[ICML 2026\] MedCase-Structured: A Text-to-FHIR Dataset for Benchmarking Diagnostic Reasoning in Clinically Realistic EHR Settings](../../ICML2026/medical_nlp/medcase-structured_a_text-to-fhir_dataset_for_benchmarking_diagnostic_reasoning_.md)

</div>

<!-- RELATED:END -->
