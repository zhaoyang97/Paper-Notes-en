---
title: >-
  [Paper Note] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice
description: >-
  [ACL 2026][Tax Practice] This paper introduces TaxPraBen, the first LLM evaluation benchmark targeting Chinese real-world tax practice. It comprises 14 datasets with 7.3K samples spanning three authentic scenarios—tax risk prevention, tax inspection analysis, and tax planning. The paper proposes a scalable evaluation paradigm based on structured parsing, field-aligned extraction, and hybrid numerical–textual matching. Evaluation of 19 LLMs reveals that closed-source models and Chinese-optimized models outperform others, while YaYi2, a tax-domain fine-tuned model, yields only marginal improvements.
tags:
  - ACL 2026
  - Tax Practice
  - LLM Evaluation Benchmark
  - Structured Evaluation
  - Chinese Taxation
  - Bloom's Taxonomy
date: 2026-05-08
content_hash: c151c2d5db745e9c
---

# TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice

**Conference**: ACL 2026
**arXiv**: [2604.08948](https://arxiv.org/abs/2604.08948)
**Code**: [https://github.com/Yating-Chen/TaxPraBen](https://github.com/Yating-Chen/TaxPraBen)
**Area**: Information Retrieval
**Keywords**: Tax Practice, LLM Evaluation Benchmark, Structured Evaluation, Chinese Taxation, Bloom's Taxonomy

## TL;DR

This paper introduces TaxPraBen, the first LLM evaluation benchmark targeting Chinese real-world tax practice. It comprises 14 datasets with 7.3K samples spanning three authentic scenarios—tax risk prevention, tax inspection analysis, and tax planning. The paper proposes a scalable evaluation paradigm based on structured parsing, field-aligned extraction, and hybrid numerical–textual matching. Evaluation of 19 LLMs reveals that closed-source models and Chinese-optimized models outperform others, while YaYi2, a tax-domain fine-tuned model, yields only marginal improvements.

## Background & Motivation

**State of the Field**: LLMs demonstrate strong performance on general NLP tasks, yet remain notably deficient in highly specialized, knowledge-intensive, and regulation-driven domains such as taxation. Existing domain-specific benchmarks—FinBen (finance), MedBench (medicine), and LAiW (law)—cover several vertical domains, but evaluation benchmarks for the tax domain are extremely scarce.

**Limitations of Prior Work**: (1) Existing tax-related benchmarks (e.g., TaxBen) focus primarily on isolated NLP tasks (text classification, generation, and reasoning), failing to reflect the dual demands of semantic understanding and numerical computation inherent in real-world tax practice. (2) Overseas tax data differs substantially from Chinese tax administration practices and cannot be directly transferred. (3) Some models perform well on isolated tasks but poorly in real-world scenarios requiring integrated semantic reasoning and numerical computation, leading to inflated rankings.

**Root Cause**: Tax practice requires LLMs to simultaneously command policy-level semantic understanding and precise numerical computation. Conventional NLP benchmarks, which evaluate these abilities through separate tasks, fail to capture this compound requirement, causing the practical applicability of models to be overestimated.

**Paper Goals**: To construct the first comprehensive evaluation benchmark for Chinese real-world tax practice, covering the full cognitive spectrum from knowledge memorization to understanding to application, thereby filling the gap in practical tax scenario assessment.

**Starting Point**: Grounded in Bloom's Taxonomy, the paper categorizes tax tasks into three cognitive levels—Knowledge Memorization (KM), Knowledge Understanding (KU), and Knowledge Application (KA)—and introduces three practice-oriented task scenarios closely aligned with real professional workflows: tax risk prevention, tax inspection analysis, and tax planning.

**Core Idea**: To design a structured evaluation paradigm that enables end-to-end assessment of LLMs' tax practice capabilities through a pipeline of structured parsing → field-aligned extraction → hybrid numerical–textual matching, while remaining extensible to other domains such as law, medicine, and finance.

## Method

### Overall Architecture

TaxPraBen contains 14 datasets with 7.3K instances, covering 10 conventional application tasks and 3 novel practical scenario tasks. Data are collected via three pipelines: (A) Book data acquisition—extracting content from tax examination guides and tax planning casebooks via OCR; (B) Official document downloading—obtaining policy regulations from the State Taxation Administration; (C) Web data processing—crawling news articles, risk control reports, and inspection cases from tax websites. All data undergo manual validation and ChatGPT-assisted annotation.

### Key Designs

1. **Bloom's Taxonomy-Based Task Framework**:

    - Function: Systematically organizes tax capability assessment into three cognitive levels.
    - Mechanism: KM (Knowledge Memorization) tests the model's ability to accurately reproduce tax law provisions and policy regulations, exemplified by the TaxRecite dataset; KU (Knowledge Understanding) tests the model's ability to identify key information from tax materials and comprehend policy implications, including TaxSum, TaxTopic, and TaxRead; KA (Knowledge Application) tests the model's ability to integrate regulations and computational methods in real tax scenarios, encompassing 10 datasets including TaxCalc, TaxSCQ, and TaxMCQ.
    - Design Motivation: Compared to benchmarks such as TaxBen that focus solely on isolated NLP tasks, organizing evaluation by cognitive level enables systematic assessment of the complete capability chain from "knowing" to "applying."

2. **Structured Evaluation Paradigm**:

    - Function: Enables end-to-end tax practice evaluation combining semantic accuracy with structural alignment.
    - Mechanism: A unified JSON output protocol is defined, with standardized output schemas designed for three practical scenarios—TaxRisk extracts risk points and remediation strategies; TaxInspect extracts criminal behaviors, charges, and penalty outcomes; TaxPlan generates tax planning strategies and computes tax savings. Given that open-source models produce irregularly formatted outputs, ChatGPT-3.5 is employed as a structured parser to convert free-form text into standard JSON before automated evaluation.
    - Design Motivation: Real-world tax practice requires hybrid outputs integrating semantic reasoning and numerical computation, which conventional single-metric approaches cannot comprehensively assess.

3. **Multi-Source Data Fusion and Quality Control**:

    - Function: Ensures that datasets reflect authentic tax scenarios rather than purely academic constructions.
    - Mechanism: Three data pipelines cover diverse sources including examination questions, policy regulations, and web-based case studies. ChatGPT assists with structured information extraction and data augmentation, while all data must pass manual usability checks. Instruction annotation is conducted by a professional tax team using a 4-point rating scale, with annotation consistency ensured via Fleiss' Kappa and Krippendorff's Alpha.
    - Design Motivation: Existing domain benchmarks largely rely on publicly available datasets and lack coverage of real-world application scenarios.

### Loss & Training

TaxPraBen is an evaluation benchmark and involves no model training. Evaluation metrics are designed according to output type: Accuracy/F1/Macro-F1 for classification tasks, BERTScore/BARTScore for generation tasks, EM Accuracy for structured prediction, and a weighted average of EM Accuracy and BERTScore for hybrid matching tasks.

## Key Experimental Results

### Main Results

**Zero-shot performance of 19 LLMs across tax cognitive levels**

| Model | KM (Memorization) | KU (Understanding) | KA (Application) | Type |
|-------|------------------|-------------------|-----------------|------|
| ERNIE-3.5 | 0.667 | 0.599 | 0.475 | Closed-source Chinese |
| Grok3 | 0.519 | 0.579 | 0.482 | Closed-source Multilingual |
| GPT-4o | 0.478 | 0.637 | 0.472 | Closed-source Multilingual |
| ChatGPT | 0.488 | 0.602 | 0.415 | Closed-source Multilingual |
| Qwen2.5 | 0.499 | 0.538 | 0.375 | Open-source Chinese |
| DeepSeek-R1 | 0.461 | 0.455 | 0.324 | Open-source Chinese |
| YaYi2 (Tax Fine-tuned) | 0.485 | 0.307 | 0.239 | Domain Fine-tuned |
| Mistral-v0.3 | 0.400 | 0.277 | 0.114 | Open-source Multilingual |

### Ablation Study

| Dimension | Finding | Remarks |
|-----------|---------|---------|
| Zero-shot vs. One-shot | 11/19 models improve under one-shot | Some models, e.g., GLM4, degrade |
| NLP-CLS Tasks | Most models perform poorly | Tax classification requires deep domain knowledge |
| NLP-REA Tasks | All models struggle | Multi-step numerical reasoning is the bottleneck |
| NLP-GEN Tasks | Relatively best performance | ERNIE-3.5 achieves 0.670 zero-shot |
| Chinese vs. Multilingual LLMs | Chinese models generally superior | Language optimization is critical for tax terminology |

### Key Findings

- Knowledge Application (KA) is the most challenging level; all models perform worst here, reflecting the reliance of tax practice on contextual reasoning about economic activities.
- ERNIE-3.5 leads on KM tasks owing to its knowledge-enhanced pretraining strategy, demonstrating the value of domain knowledge integration.
- YaYi2, despite fine-tuning on tax data, underperforms general-purpose LLMs, indicating that insufficient fine-tuning data volume and task coverage limit effectiveness.
- Reasoning tasks (REA) challenge all models, exposing systemic weaknesses in LLMs with respect to numerical computation and tax logic comprehension.

## Highlights & Insights

- This is the first work to systematically apply Bloom's Taxonomy to Chinese tax domain evaluation, establishing a complete capability assessment chain from memorization to application.
- The structured evaluation paradigm exhibits strong transferability—the pipeline of structured parsing → field alignment → hybrid matching can be directly applied to domains such as law, finance, and medicine that require mixed outputs of explanatory text and key numerical values.
- Using ChatGPT-3.5 as a lightweight structured parser in place of fragile regex matching is a practical engineering technique that reduces evaluation sensitivity to output formatting.

## Limitations & Future Work

- Data sourced from the internet and publicly available books carry the risk of test data contamination, as LLMs may have encountered portions of the test content during pretraining.
- The evaluated models are predominantly around 7B parameters; comparisons with larger open-source models are absent.
- Automatic semantic similarity metrics may not fully capture human judgments of answer quality.
- Future work could extend to more complex scenarios such as multi-turn conversational tax consultation and cross-year policy change tracking.

## Related Work & Insights

- **vs. TaxBen**: TaxBen focuses on isolated NLP tasks; TaxPraBen adds three real-world practice scenarios and proposes structured evaluation, more closely reflecting actual applications.
- **vs. FinBen/LAiW**: Benchmarks for finance and law are relatively mature; TaxPraBen fills the gap in the tax domain, and its hybrid evaluation methodology can provide feedback to these fields.
- **vs. MMLU/C-Eval**: General-purpose benchmarks lack domain depth and cannot assess specialized numerical reasoning capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ First Chinese tax practice benchmark; the structured evaluation paradigm offers broad applicability.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive cross-evaluation of 19 models, though comparisons with larger models are missing.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed descriptions of task taxonomy and data construction.
- Value: ⭐⭐⭐⭐ Fills the gap in tax evaluation; the structured evaluation paradigm is transferable.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention](../../AAAI2026/information_retrieval/sr-ki_scalable_and_real-time_knowledge_integration_into_llms_via_supervised_atte.md)
- [\[ACL 2026\] Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection](understanding_structured_financial_data_with_llms_a_case_study_on_fraud_detectio.md)
- [\[ACL 2026\] ChunQiuTR: Time-Keyed Temporal Retrieval in Classical Chinese Annals](chunqiutr_time-keyed_temporal_retrieval_in_classical_chinese_annals.md)
- [\[ACL 2026\] CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge](curate_continual_unlearning_in_real_time_with_ensured_preservation_of_llm_knowle.md)
- [\[ACL 2026\] FLARE: Task-Agnostic Embedding Model Evaluation via Normalizing Flows](flare_task-agnostic_embedding_model_evaluation_through_a_normalization_process.md)

<!-- RELATED:END -->
