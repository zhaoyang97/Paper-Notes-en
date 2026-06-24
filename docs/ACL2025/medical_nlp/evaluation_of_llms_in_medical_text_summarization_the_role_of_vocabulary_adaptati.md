---
title: >-
  [Paper Note] Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings
description: >-
  [ACL 2025 (Findings)][Medical LLM][vocabulary adaptation] A systematic benchmark study reveals that LLM performance significantly degrades under high OOV (out-of-vocabulary) and high-novelty medical text summarization scenarios. Through various vocabulary adaptation strategies (MEDVOC, MEDVOC-LLM, ScafFix), it demonstrates that even Llama-3.1 (128K vocabulary size) still suffers from over-fragmentation, and vocabulary adaptation yields remarkable improvements.
tags:
  - "ACL 2025 (Findings)"
  - "Medical LLM"
  - "vocabulary adaptation"
  - "medical summarization"
  - "OOV"
  - "LLM"
  - "tokenization"
  - "continual pretraining"
date: 2026-05-08
content_hash: 3da15db71398a285
---

# Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2505.21242](https://arxiv.org/abs/2505.21242)  
**Code**: [GitHub](https://github.com/gb-kgp/LLM-MedicalSummarization-Benchmark)  
**Area**: Medical NLP  
**Keywords**: vocabulary adaptation, medical summarization, OOV, LLM, tokenization, continual pretraining

## TL;DR

A systematic benchmark study reveals that LLM performance significantly degrades under high OOV (out-of-vocabulary) and high-novelty medical text summarization scenarios. Through various vocabulary adaptation strategies (MEDVOC, MEDVOC-LLM, ScafFix), it demonstrates that even Llama-3.1 (128K vocabulary size) still suffers from over-fragmentation, and vocabulary adaptation yields remarkable improvements.

## Background & Motivation

**Background**: LLMs have achieved success in medical text summarization, primarily relying on in-context learning (ICL) and parameter-efficient fine-tuning (QLoRA). Recent works like ClinSumm explored various LLM adaptation strategies, but these studies only report aggregated performance scores on the full test set.

**Limitations of Prior Work**: Existing research lacks fine-grained evaluation in challenging scenarios. Tokenizers for LLMs suffer from severe **over-fragmentation** on medical terminology—for instance, "cardiomyopathy" is split into 6 tokens ('_card', 'iom', 'y', 'op', 'ath', 'y') by Llama-2, leading to a loss of semantic information.

**Key Challenge**: Even for Llama-3.1 with a 128K vocabulary size, its fragment score in the medical domain is still **13.08%** higher than that in the generic domain, with over 33% of medical terms being split more than 3 times. Such vocabulary mismatch causes semantic loss in the encoding stage and increases token counts during generation.

**Goal**: (1) Systematically evaluate LLM performance in scenarios with high OOV and high novelty (domain words appearing in reference summaries but not in source documents); (2) Verify whether vocabulary adaptation strategies can effectively alleviate this bottleneck.

**Key Insight**: Design 360 evaluation configurations (4 models × 3 vocabulary strategies × 3 datasets × 2 pretraining strategies × 5 fine-grained scenarios) to comprehensively benchmark the impact of vocabulary adaptation on LLM medical summarization.

**Core Idea**: Demonstrate through fine-grained benchmarking that LLM performance significantly drops in high-OOV medical summarization scenarios, and propose ScafFix (vocabulary adaptation without scaffolding tokens) for effective improvement.

## Method

### Overall Architecture

A three-step vocabulary adaptation pipeline: (1) Generate candidate vocabulary tokens from the target domain dataset; (2) Select important terms using utility functions (such as fragment score); (3) Learn embeddings for the newly added tokens and integrate them into the LLM.

### Key Designs

1. **MEDVOC**: The baseline SOTA vocabulary adaptation strategy—constructed candidate vocabularies $V_{PAC}$ from a medical corpus (PubMed Abstract Collection, PAC) and $V_{TGT}$ from target task datasets, and identified the optimal vocabulary set in $V_{PAC} \cap V_{TGT}$ via hyperparameter search. The utility function is fragment score $= \frac{1}{|\mathcal{C}|}\sum_{w \in \mathcal{C}} \text{subwords}(w, \mathcal{V})$, which measures the average number of subwords a word is split into.

2. **MEDVOC-LLM**: Tailors MEDVOC results for LLM tokenizers—removes terms that never appear in the reference summaries of the target task training set, as well as invalid tokens combining numbers and punctuation (e.g., "-9,"), making vocabulary adaptation better aligned with the LLM's tokenization scheme.

3. **ScafFix (Core Contribution)**: Resolves the **scaffolding token overhead** in existing vocabulary adaptation. Taking "cholesterol" as an example, adding this word requires adding intermediate "choke" tokens (due to the pairwise operations of BPE merge rules), which are rarely used and under-trained after the entire word is added. ScafFix directly selects the top $x$ medical terms by frequency ($x=500$, steps of 50), bypasses the subword construction phase of tokenization, and then employs the **AdaptBPE** tokenization scheme—first checking if the longest prefix of the input token exists in the added vocabulary to keep it unsplit, and running standard BPE on the remainder. Approximately **20%** of the added tokens in Llama-3.1 belong to such redundant scaffolding tokens.

4. **Continual Pre-training Strategy**: Newly added token embeddings are initialized with the average of existing subword embeddings and then trained on 20K PubMed documents using LoRA (rank=32, alpha=64). Two strategies are utilized:

    - **End-to-End**: Freezes the base layers, unfreezes input/output embedding layers and LoRA adapters, and trains end-to-end for 5 epochs.
    - **Two-Stage**: First freezes LoRA to train only the embedding layer for 2 epochs/10K samples, and then unfreezes LoRA to joint-train for 3 epochs/20K samples. This is more stable and prevents embedding space overfitting.

### Fine-Grained Evaluation Setup

Five difficult scenarios (taking the top 10% percentile): DifficultRS (high OOV concentration in reference summaries), DifficultSD (high OOV concentration in source documents), NovelRS (high novel term concentration in reference summaries), and AllSD/AllRS (all OOV concentrations).

## Key Experimental Results

### Main Results (Rouge-L, Selected Key Results)

| Dataset | Model | BASE | CPT-only | MEDVOC-LLM | ScafFix |
|--------|------|------|----------|------------|---------|
| PubMedQA | Llama-2 | 26.33 | 27.12 | 26.90 | **27.61** |
| PubMedQA | Llama-3.1 | 28.10 | 26.62 | 27.69 | 27.67 |
| EBM | Llama-2 | 18.56 | 19.13 | 19.27 | 18.65 |
| EBM | Llama-3.1 | 20.04 | 20.13 | **20.75** | 20.79 |
| BioASQ-S | Llama-2 | 32.12 | 33.30 | 32.40 | **32.88** |
| BioASQ-S | Llama-3.1 | 35.25 | 36.01 | **37.15** | 36.70 |
| BioASQ-M | Llama-2 | 28.50 | 27.22 | 24.50 | **26.16** |
| BioASQ-M | Llama-3.1 | 29.28 | 27.56 | 27.45 | **28.91** |

### Performance under Difficult Scenarios (Overall = Average of Four Scenarios)

| Dataset | Llama-2 BASE → Best Adaptation | Gain | Llama-3.1 BASE → Best Adaptation | Gain |
|--------|------------------------|------|--------------------------|------|
| PubMedQA | 23.51 → 26.18 | +11.4% | 24.58 → 25.65 | +4.4% |
| EBM | 15.77 → 16.46 (CPT) | +4.4% | 15.44 → 17.15 | +11.1% |
| BioASQ-S | 28.62 → 30.21 | +5.6% | 27.52 → 32.04 | +16.4% |
| BioASQ-M | 26.33 → 26.78 | +1.7% | 26.51 → 26.94 | +1.6% |

### Key Research Question Findings

- **RQ1**: Vocabulary adaptation outperforms BASE in 5 out of 8 on full test set configurations, with an average improvement of 3.68% for Llama-2 and 4.57% for Llama-3.1.
- **RQ2**: CPT-Only (without vocabulary adaptation) shows improvements in high-OOV scenarios but is inferior to vocabulary adaptation, which outperforms CPT-Only in 13 out of 16 high-OOV configurations.
- **RQ3**: Vocabulary adaptation beats BASE in 14 out of 16 settings under high-OOV scenarios, achieving average improvements of 8.74% for Llama-2 and 14.64% for Llama-3.1.
- **RQ4**: Vocabulary adaptation beats BASE in 6 out of 8 settings under high-novelty scenarios, achieving average improvements of 11.92% for Llama-2 and 18.03% for Llama-3.1.
- **Human Evaluation**: Medical experts consensus indicates that vocabulary adaptation produces more relevant, coherent, and faithful summaries.
- **Concept Score** shows that vocabulary adaptation improves factual consistency in 5 out of 6 settings: 18.75% on average for Llama-2 and 14.82% for Llama-3.1.

### Key Findings

- **Even Llama-3.1 with a 128K vocabulary size still requires vocabulary adaptation**—the medical domain fragment score is 13.08% higher than that of the generic domain.
- **ScafFix performs best in high-novelty scenarios** (winning in 5 out of 8 setups), while **MEDVOC-LLM is the best in high-OOV scenarios**.
- **Scaffolding tokens are a real concern**: Approximately 20% redundant intermediate tokens are generated when expanding the vocabulary of Llama-3.1.
- **Two-Stage pre-training is generally more stable than End-to-End**.

## Highlights & Insights

- **Fine-grained evaluation framework** is highly valuable—existing medical evaluations of LLMs mostly report average performance on full datasets only, whereas this paper systematically reveals performance degradation in regions with dense OOV/novel words.
- **The insight of ScafFix is highly practical**: Directly adding complete medical terms to the vocabulary instead of relying on BPE subword decomposition avoids the issue of under-trained scaffolding tokens.
- **Fragment score acts as a diagnostic tool**: It quantifies the severity of vocabulary mismatch and guides whether vocabulary adaptation is necessary.
- **Continual pre-training requires only 20K samples** (6 hours on an A100), yielding comparable performance to 100K samples (40 hours), which represents high cost-effectiveness.

## Limitations & Future Work

- Only evaluates ICL (in-context learning) and does not test the effect of vocabulary adaptation after QLoRA fine-tuning.
- Vocabulary adaptation shows limited improvement on BioASQ-M, indicating that the effect diminishes in multi-document, long-context scenarios.
- The AdaptBPE scheme of ScafFix introduces extra tokenization logic during inference, and its compatibility with standard HuggingFace ecosystem remains undiscussed.
- Only English medical text is considered; vocabulary adaptation for multilingual scenarios may be of greater significance.
- Lacks a comparison with instruction-tuned models (e.g., Med-Gemini, Med-PaLM).

## Related Work & Insights

- **MEDVOC** (Balde et al., 2024b) is the most critical baseline and starting point for this study, and ScafFix is its improved variant for LLMs.
- **ClinSumm** (Van Veen et al., 2024) provides a benchmark framework for LLM-based medical summarization.
- **AdaptBPE** (Balde et al., 2024a) provides the alternative tokenization scheme used by ScafFix.
- **Insight**: LLM tokenizers are a severely underestimated bottleneck, particularly in specialized domains. Improving tokenization might be more effective than scaling model size.

## Rating

- **Novelty**: ⭐⭐⭐ — ScafFix has some novelty, but overall this is a benchmarking study, and the methodological contribution is limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 360 evaluation configurations, 4 models × 3 datasets × 5 scenarios × multiple strategies; extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structured RQ analysis and rich tables.
- **Value**: ⭐⭐⭐⭐ — Reveals the tokenization bottleneck of LLMs in the medical domain, offering guiding significance for practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge](biore_llm_judge_evaluation.md)
- [\[NeurIPS 2025\] Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs](../../NeurIPS2025/medical_nlp/shallow_robustness_deep_vulnerabilities_multi-turn_evaluation_of_medical_llms.md)
- [\[ICML 2026\] MedCase-Structured: A Text-to-FHIR Dataset for Benchmarking Diagnostic Reasoning in Clinically Realistic EHR Settings](../../ICML2026/medical_nlp/medcase-structured_a_text-to-fhir_dataset_for_benchmarking_diagnostic_reasoning_.md)
- [\[NeurIPS 2025\] Faithful Summarization of Consumer Health Queries: A Cross-Lingual Framework with LLMs](../../NeurIPS2025/medical_nlp/faithful_summarization_of_consumer_health_queries_a_cross-lingual_framework_with.md)
- [\[ICLR 2026\] Critic-Adviser-Reviser Cyclic Refinement: Towards High-Quality EMR Corpus Generation with LLMs](../../ICLR2026/medical_nlp/criticadviserreviser_cyclic_refinement_towards_high-quality_emr_corpus_generatio.md)

</div>

<!-- RELATED:END -->
