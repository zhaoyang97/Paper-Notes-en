---
title: >-
  [Paper Note] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification
description: >-
  [ACL 2026][NLP Understanding][Multi-label Classification] This paper proposes MADE—a "living" multi-label text classification benchmark based on FDA medical device adverse event reports, containing 1…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Multi-label Classification"
  - "Uncertainty Quantification"
  - "Medical Devices"
  - "Living Benchmark"
  - "Long-tail Distribution"
date: 2026-05-08
content_hash: fff63e6286986337
---

# MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification

**Conference**: ACL 2026  
**arXiv**: [2604.15203](https://arxiv.org/abs/2604.15203)  
**Code**: [https://hhi.fraunhofer.de/aml-demonstrator/made-benchmark](https://hhi.fraunhofer.de/aml-demonstrator/made-benchmark)  
**Area**: LLM Evaluation  
**Keywords**: Multi-label Classification, Uncertainty Quantification, Medical Devices, Living Benchmark, Long-tail Distribution

## TL;DR

This paper proposes MADE—a "living" multi-label text classification benchmark based on FDA medical device adverse event reports, containing 1,154 hierarchical labels and strict temporal splitting. It systematically evaluates the prediction performance and uncertainty quantification (UQ) capabilities of 20+ encoder/decoder models under discriminative fine-tuning, generative fine-tuning, and few-shot prompting. The study reveals critical trade-offs: small discriminative fine-tuned decoders are optimal for head-to-tail accuracy, generative fine-tuning yields the most reliable UQ, and large reasoning models improve rare labels but exhibit unexpectedly weak UQ.

## Background & Motivation

**Background**: Multi-label text classification (MLTC) is a core task in the healthcare domain (patient triage, clinical coding, event reporting, etc.), requiring the selection of multiple labels from a large set. Existing benchmarks (e.g., MIMIC-III, EUR-LEX) have reached saturation and are likely contaminated by LLM pre-training data.

**Limitations of Prior Work**: (1) Existing MLTC benchmarks are static and prone to inflated zero-/few-shot performance due to data contamination; (2) real-world MLTC data suffers from severe intra-/inter-class imbalance (a few frequent classes dominate while safety-critical classes reside in the long tail); (3) in high-stakes domains (medical), models require not only strong prediction performance but also reliable uncertainty quantification (UQ) to support human oversight, yet UQ research in MLTC is almost non-existent.

**Key Challenge**: Practitioners face unanswered questions—which model architecture should be chosen (encoder vs. decoder)? Which learning paradigm (fine-tuning vs. in-context learning) best balances frequent and rare classes? How reliable are the predictions? There is a lack of a unified, pollution-free benchmark to systematically answer these questions.

**Goal**: (1) Create a continuously updated, pollution-free MLTC benchmark; (2) establish comprehensive baselines covering 20+ models; (3) systematically evaluate the effectiveness of various UQ methods in MLTC.

**Key Insight**: Utilize medical device adverse event reports regularly published by the FDA as a continuously updated data source. Ensure test data does not leak into the pre-training of future models through strict temporal splitting.

**Core Idea**: Construct a "living" benchmark—as the FDA continues to release new reports, future models can always be evaluated on data generated after their training cutoff without risk of contamination.

## Method

### Overall Architecture

The MADE benchmark consists of three main components: (1) Data Pipeline—extracts event descriptions and IMDRF hierarchical labels from FDA adverse event reports, generating train/val/test sets after deduplication, downsampling, and temporal splitting; (2) Model Baselines—covers three paradigms: discriminative fine-tuning (encoder/decoder + classification head), generative fine-tuning (decoders generating label tokens), and few-shot prompting (instruction/thinking models); (3) UQ Evaluation—compares four categories of methods: information-level (entropy, perplexity), consistency-level (graph Laplacian eigenvalues), combined-level, and self-reported uncertainty.

### Key Designs

1.  **Living Benchmark Data Construction**:
    - **Function**: Provide continuous, pollution-free assessment data for MLTC.
    - **Mechanism**: Extracts event descriptions and labels from FDA adverse event reports from 2015-2025. Product problem and patient problem labels for each report are mapped to IMDRF hierarchical codes (3 levels) and propagated to all ancestor codes. The training set spans 2015-2023 (298,825 samples), the validation set is H1 2024 (71,271), and the test set is 2024.7-2025.6 (118,177). The final label set contains 1,154 labels with an average of 8.79 labels per sample, showing a significant long-tail distribution.
    - **Design Motivation**: The FDA releases new reports quarterly, allowing future models to be evaluated on data post-dating their training cutoff, fundamentally avoiding data contamination.

2.  **Multi-Paradigm Model Baselines**:
    - **Function**: Comprehensively compare MLTC performance across different architectures and learning paradigms.
    - **Mechanism**: (a) Discriminative fine-tuning—adding a classification head to Llama 3.2-1B/3B, 3.1-8B, and Ettin 150M/400M/1B using hierarchical BCE loss; (b) Generative fine-tuning—allowing Llama and Ettin decoders to generate label tokens, comparing full-parameter and LoRA fine-tuning; (c) Few-shot prompting—10-shot prompting using kNN retrieval for 10+ models including Llama, DeepSeek-R1, Qwen3, GPT-4.1/5. Labels are categorized by training frequency into head (>1%), medium (0.1-1%), tail (0.01-0.1%), and extreme tail (<0.01%).
    - **Design Motivation**: Practitioners must make decisions regarding model size, training cost, and performance, necessitating fair head-to-head comparisons.

3.  **Systematic UQ Evaluation**:
    - **Function**: Evaluate the quality of uncertainty estimation for different models/paradigms.
    - **Mechanism**: Uses per-label entropy for discriminative models. For generative models, it uses information-level $U_{\text{info}}$ (entropy, improbability, avg-log-prob, perplexity), consistency-level $U_{\text{cons}}$ (sum of graph Laplacian eigenvalues from multiple random samples), combined-level $U_{\text{combined}} = U_{\text{info}} \times U_{\text{cons}}$, and self-reported $U_{\text{self}}$ (prompting the model for confidence). UQ quality is evaluated using PRR (Prediction Rejection Rate), Spearman $\rho$, and positive class ECE$_+$.
    - **Design Motivation**: High-risk medical scenarios require routing uncertain cases to human review; UQ quality directly impacts system safety.

### Loss & Training

Discriminative fine-tuning uses a hierarchical binary cross-entropy loss, calculating BCE separately at each level and summing them. Optimization uses AdamW + cosine learning rate scheduler, batch size 512, for 20 epochs. Classification thresholds for each label are independently selected on the validation set to maximize F1. Generative fine-tuning uses standard autoregressive language modeling loss for 4 epochs, supporting both full-parameter and LoRA fine-tuning.

## Key Experimental Results

### Main Results

**Prediction Performance and UQ Quality across Paradigms (Truncated test set n=10,288)**

| Paradigm/Model | Macro F1 | Head F1 | Tail F1 | ET F1 | PRR↑ | ρ↓ |
|-----------|---------|---------|---------|-------|------|-----|
| Discriminative Llama-3.1-8B | **0.54** | **0.74** | **0.53** | 0.12 | 0.47 | -0.40 |
| Generative Llama-3.1-70B | 0.53 | 0.73 | 0.51 | 0.16 | 0.55 | -0.27 |
| Generative Llama-3.2-3B | 0.48 | 0.67 | 0.46 | 0.12 | **0.60** | **-0.46** |
| Prompting Qwen3-235B-Think | 0.49 | 0.62 | 0.48 | 0.33 | 0.34 | -0.09 |
| Prompting GPT-5 | 0.54 | 0.68 | 0.53 | **0.34** | N/A | N/A |
| Prompting DeepSeek-R1 | 0.48 | 0.62 | 0.47 | 0.30 | 0.24 | -0.09 |

### Ablation Study

**Comparison of UQ Methods (Generative Fine-tuning vs. Prompting)**

| UQ Metric | Gen FT PRR | Instruct PRR | Thinking PRR |
|---------|-------------|-------------|-------------|
| Avg. Log-Prob | 0.54±0.05 | 0.37±0.25 | 0.18±0.12 |
| Entropy | **0.58±0.03** | **0.45±0.15** | 0.19±0.12 |
| Improbability | 0.54±0.05 | 0.43±0.15 | 0.17±0.12 |
| Perplexity | 0.54±0.06 | 0.37±0.25 | 0.18±0.11 |

### Key Findings

- Discriminative fine-tuning consistently outperforms generative fine-tuning of equivalent size in head-tail accuracy (Wilcoxon test $p \leq 0.05$), achieving optimal comprehensive F1 with only 8B parameters.
- Generative fine-tuning performs best in UQ—Generative Llama-3.2-3B achieves the best PRR (0.60) and Spearman $\rho$ (-0.46).
- Reasoning models (GPT-5, Qwen3-235B-Think) excel in extreme tail classes (F1=0.34) but exhibit unexpectedly weak UQ (PRR only 0.21±0.10); their head-class performance is consistently lower than the best fine-tuned models.
- Self-reported confidence is not a reliable proxy for uncertainty—it shows very low correlation with actual error rates.
- Entropy as a $U_{\text{info}}$ metric is the best choice for both generative fine-tuning and instruction models.

## Highlights & Insights

- The "living benchmark" concept accurately addresses the fundamental problem of benchmark contamination in the LLM era by leveraging continuous public data streams from the government for evergreen testing.
- Reveals a counter-intuitive trade-off between prediction performance and UQ quality—reasoning models perform excellently on rare classes but have the worst UQ, implying their "high performance" may be untrustworthy in high-risk scenarios.
- Discovers that discriminative fine-tuning requires only 8B parameters to reach or even exceed the comprehensive performance of GPT-5, providing a highly cost-effective solution for practitioners.

## Limitations & Future Work

- FDA annotation consistency has not been formally verified via inter-annotator agreement studies; labels may contain noise.
- Self-reported UQ only tested simple prompting strategies; more refined calibration prompts might improve results.
- The test set was limited to 10,288 samples to control inference costs, which limits the statistical power of evaluation for extreme tail labels.
- Potential gains from multimodal inputs (e.g., device images) for classification were not evaluated.

## Related Work & Insights

- **vs MIMIC-III ICD Coding**: MIMIC is an ICD coding benchmark for clinical notes but has been widely used for over a decade, posing a serious contamination risk; MADE avoids this through continuous updates and temporal splitting.
- **vs EUR-LEX**: EUR-LEX is an MLTC benchmark for legal documents with a smaller label space and different domain; MADE's 1,154 labels and 3-level hierarchy are more challenging.
- **vs Ettin (Weller et al. 2025)**: Ettin provides matched encoder-decoder model comparisons but was not evaluated on MLTC; this paper fills that gap.

## Rating

- Novelty: ⭐⭐⭐⭐ The "living benchmark" concept is novel and practical; the systematic evaluation of UQ in MLTC fills a gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison of 20+ models, 4 paradigms, and multiple UQ methods with rigorous statistical testing.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and explicit conclusions, though information density is high; some details require referring to the appendix.
- Value: ⭐⭐⭐⭐⭐ Provides a practical guide for model selection and UQ methods in high-stakes MLTC applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)
- [\[ACL 2026\] MTSQL-R1: Towards Long-Horizon Multi-Turn Text-to-SQL via Agentic Training](mtsql-r1_towards_long-horizon_multi-turn_text-to-sql_via_agentic_training.md)
- [\[ACL 2026\] MSMO-ABSA: Multi-Scale and Multi-Objective Optimization for Cross-Lingual Aspect-Based Sentiment Analysis](msmo-absa_multi-scale_and_multi-objective_optimization_for_cross-lingual_aspect-.md)
- [\[ACL 2026\] HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction](hcre_llm-based_hierarchical_classification_for_cross-document_relation_extractio.md)
- [\[ACL 2026\] Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations](agree_disagree_explain_decomposing_human_label_variation_in_nli_through_the_lens.md)

</div>

<!-- RELATED:END -->
