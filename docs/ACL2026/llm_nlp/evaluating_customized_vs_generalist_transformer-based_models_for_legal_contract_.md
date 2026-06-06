---
title: >-
  [Paper Note] Evaluating Customized vs. Generalist Transformer-based Models for Legal Contract Classification
description: >-
  [ACL2026][LLM/NLP][legal-specific models] This paper systematically compares 13 legal-domain customized Transformer models with 9 general-purpose models across three English contract classification tasks. It finds that s…
tags:
  - "ACL2026"
  - "LLM/NLP"
  - "legal-specific models"
  - "contract classification"
  - "long-tail labels"
  - "macro-F1"
  - "domain pretraining"
date: 2026-05-08
content_hash: 298f4e47e56c792a
---

# Evaluating Customized vs. Generalist Transformer-based Models for Legal Contract Classification

**Conference**: ACL2026  
**arXiv**: [2508.07849](https://arxiv.org/abs/2508.07849)  
**Code**: No public code link found in cache  
**Area**: Legal NLP / Contract Classification / Domain Model Evaluation  
**Keywords**: legal-specific models, contract classification, long-tail labels, macro-F1, domain pretraining

## TL;DR
This paper systematically compares 13 legal-domain customized Transformer models with 9 general-purpose models across three English contract classification tasks. It finds that smaller models pretrained on contract-related corpora, such as Legal-BERT and Contracts-BERT, generally outperform larger generalist models on long-tail legal labels.

## Background & Motivation
**Background**: Various open-source legal models have emerged in Legal NLP, such as Legal-BERT, Contracts-BERT, CaseLaw-BERT, and LexLM. Simultaneously, general encoder/decoder models are widely applied to tasks like contract clause classification, topic identification, and deontic modality recognition (identifying obligations, rights, and prohibitions).

**Limitations of Prior Work**: Although contract tasks inherently depend on legal semantics, many existing studies primarily evaluate generalist models and often exclude legal-domain models entirely. Consequently, the community remains unclear whether legal-specific models are truly more suitable for contract classification, and whether model scale, pretraining corpora, or task distribution is the dominant factor.

**Key Challenge**: General large language models possess more parameters and broader knowledge, but contract classification often faces challenges like long-tail labels, specialized legal terminology, and fine-grained clause semantics. Larger models are not necessarily superior to smaller in-domain encoder models, particularly as macro-F1 scores often expose errors in rare classes.

**Goal**: To answer a direct question: do legal-specific Transformer models outperform generalist models in contract classification tasks? The authors aim to provide benchmark results across tasks, model types, and metrics while identifying strong baselines for future legal contract classification.

**Key Insight**: The paper selects three public English contract classification datasets covering multi-label, multi-class, varying scales, and different levels of legal semantic granularity. It then performs a unified comparison of 13 legal models and 9 general models, focusing on micro-F1, macro-F1, and misclassifications of rare classes.

**Core Idea**: Instead of a generic comparison of "legal models vs. general models," the study decomposes the evaluation of model types, corpus domain alignment, and long-tail label performance specifically on contract tasks.

## Method
As a benchmark/evaluation paper, this work does not propose a new model. The methodological contribution lies in task selection, model coverage, unified fine-tuning, and error analysis.

### Overall Architecture
The input consists of three types of contract classification tasks: UNFAIR-ToS for identifying unfair terms in Terms of Service (9-class multi-label); LEDGAR for SEC Exhibit 10 contract clause topic classification (100-class multi-class); and LEXDEMOD for subject-specific deontic modality detection in lease contracts (7-class multi-label).

Models are divided into two groups. Generalist models include BERT, RoBERTa, DeBERTa, Longformer, BigBird, DistilBERT, RoBERTa-large, and decoder models such as Llama-3.2 and Mistral. Legal-specific models include Legal-BERT, Contracts-BERT, Legal-RoBERTa, CaseLawBERT, PoL-BERT, InLegalBERT, InCaseLawBERT, CustomInLawBERT, LexLM, Legal-XLM-R, LexT5, AdaptLLM, and SaulLM.

The evaluation process involves task-specific fine-tuning for each task, using micro-F1 and macro-F1 to measure performance. Micro-F1 reflects overall classification accuracy, while macro-F1 is more sensitive to rare classes, thus better representing robustness on the long-tail categories of legal contracts.

### Key Designs
1.  **Multi-task Contract Benchmark Coverage**:
    *   Function: Prevents drawing overly narrow conclusions from a single contract task.
    *   Mechanism: UNFAIR-ToS, LEDGAR, and LEXDEMOD cover terms of service, SEC contract clauses, and lease obligation/right detection respectively. Test samples range from 1.6k to 10k, with label counts from 7 to 100.
    *   Design Motivation: The effectiveness of a legal model depends on task semantics and data distribution. A multi-task setup allows observing model stability across rare classes, multi-label settings, and long-text clauses.

2.  **Head-to-Head Comparison of Legal-Specific and Generalist Models**:
    *   Function: Directly addresses whether domain-specific models are worth using.
    *   Mechanism: 13 legal models and 9 general models are compared under identical tasks and metrics, incorporating encoder, decoder, and encoder-decoder architectures.
    *   Design Motivation: Previous work often evaluated only general models or a few legal models, making it impossible to judge the relative contributions of domain pretraining, model size, and architectural category.

3.  **Long-tail Error Analysis**:
    *   Function: Explains the source of legal models' advantages rather than reporting aggregate scores only.
    *   Mechanism: The authors analyze misclassifications of RoBERTa-large and observe whether legal models like Contracts-BERT can correct errors in rare categories of UNFAIR-ToS, such as "Limitation of Liability" and "Unilateral Termination."
    *   Design Motivation: The difficulty in contract tasks often lies in rare but legally significant clause types. Macro-F1 and instance-level error analysis reveal deployment risks more effectively than micro-F1.

### Loss & Training
No specialized loss functions are introduced; the experiments utilize task-specific fine-tuning with standard classification evaluation. For multi-label tasks, micro-F1 and macro-F1 are the primary metrics. For the multi-class LEDGAR task, the same F1 metrics are used for comparison. Regarding methodology, the authors emphasize that contract text may exceed 512 subword tokens, meaning truncation or long-context handling affects results, though this paper focuses on the comparison of model families and domain pretraining.

## Key Experimental Results

### Main Results
| Dataset | Metric | Best/Representative Legal Model (Ours) | Strongest Generalist Model | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| UNFAIR-ToS | micro-F1 / macro-F1 | Contracts-BERT 96.2 / 83.4; Legal-BERT 96.0 / 82.2 | RoBERTa-large 95.8 / 81.6; Mistral 96.0 / 80.7 | Legal models are stronger in macro-F1 |
| LEDGAR | micro-F1 / macro-F1 | Legal-BERT 88.2 / 82.5; Contracts-BERT 87.9 / 82.2 | RoBERTa-large 88.6 / 83.6 | RoBERTa-large remains best on large-scale LEDGAR |
| LEXDEMOD | micro-F1 / macro-F1 | Legal-BERT 81.23 / 78.01; InLegalBERT 80.21 / 77.89 | RoBERTa-large macro-F1 77.88; Llama-3.2 76.2 / 71.4 | Legal encoders significantly outperform decoders |
| Average Performance | Mean micro-F1 | Legal-BERT 88.48±6.03 | Generalist models not summarized as main subjects in Table 2 | Legal-BERT has the strongest composite micro-F1 |
| Average Performance | Mean macro-F1 | Contracts-BERT 81.10±2.45 | Generalist models not summarized as main subjects in Table 2 | Contracts-BERT is more stable for long-tail labels |

### Ablation Study
| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| Legal-BERT | Mean micro-F1 88.48±6.03, macro-F1 80.90±2.05 | Ranked 1st in composite micro-F1 |
| Contracts-BERT | Mean micro-F1 88.09±6.55, macro-F1 81.10±2.45 | Ranked 1st in composite macro-F1 |
| CaseLawBERT | Mean micro-F1 88.01±6.45, macro-F1 80.62±2.23 | Serves as a strong baseline for contract classification |
| LexLM | Mean micro-F1 88.03±6.33, macro-F1 80.15±1.91 | Another stable legal baseline |
| PoL-BERT | LEXDEMOD micro-F1 41.35, macro-F1 15.75 | Larger/newer legal models do not necessarily fit contract distributions |

### Key Findings
*   Legal-specific models established new SOTAs on two tasks. Legal-BERT and Contracts-BERT, despite having only 110M parameters (69% fewer than RoBERTa-large), outperformed the latter on UNFAIR-ToS and LEXDEMOD.
*   RoBERTa-large remains the strongest on LEDGAR, indicating that domain pretraining is not a panacea for all tasks; data scale, label count, and model capacity also influence performance.
*   Decoder-based generalist models exhibit weaker performance on rare classes, particularly in macro-F1. This supports the continued importance of encoder-based discriminative fine-tuning for long-tail legal classification.
*   Older but more contract-focused models like Legal-BERT/Contracts-BERT outperformed some newer, larger models with more heterogeneous legal corpora. This suggests that in-distribution pretraining is more critical than pure corpus scale.

## Highlights & Insights
*   The most practical conclusion is that legal contract classification should not default to larger generalist models. Smaller legal encoders offer realistic advantages in terms of privacy, cost, long-tail class handling, and deployability.
*   The analysis of macro-F1 is highly valuable. In legal scenarios, rare classes often correspond to high-risk clauses; high micro-F1 scores can be misleading if frequent rare class errors make actual deployment unreliable.
*   "More legal corpus" does not equate to "more contract-relevant." If pretraining data is diluted with case law, statutes, or patents, the contract signal may weaken, potentially leading to poorer performance than smaller, more focused models.
*   This benchmark serves as a reminder that model selection should be based on task semantic alignment rather than simply sorting by the "legal" label or parameter count.

## Limitations & Future Work
*   The experiments only cover English contract data; non-English, cross-jurisdictional, and multilingual legal documents remain unverified.
*   The paper focuses specifically on contract language and does not evaluate other legal text types like statutes, judgments, or legal opinions; thus, conclusions cannot be directly extrapolated to all Legal NLP.
*   The study primarily examines domain generalization without systematically investigating alternative routes such as RAG, long-context handling, hierarchical classification, or prompt-based decoder inference.
*   Many passages in LEDGAR exceed 512 subword tokens, and truncation strategies may affect model performance. Future work could include more controlled experiments with long-text models like Longformer/BigBird or combine retrieval and chunk aggregation.

## Related Work & Insights
*   **vs. Original Legal-BERT / Contracts-BERT**: While original models proved the value of legal pretraining, this work further demonstrates that "contract-relevant corpora" are more important than general legal corpora for contract tasks.
*   **vs. General Encoder Baselines**: RoBERTa-large’s advantage in LEDGAR suggests that when data volume is large and label coverage is sufficient, the capacity of large general encoders still provides value.
*   **vs. Decoder-based Legal/Generalist LLMs**: The instability of Llama, Mistral, SaulLM, and AdaptLLM in classification suggests that generative models are not yet ready to replace discriminative encoders, especially in long-tail label classification.
*   **Insight**: Selection of Legal AI systems should prioritize intra-task benchmarks and report macro-F1 and rare class errors rather than relying solely on average accuracy or model scale.

## Rating
*   Novelty: ⭐⭐⭐ (Benchmark papers innovate through systematic coverage and clear problem settings rather than model architecture).
*   Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive task, model, and metric coverage; lacks non-English and wider legal genres).
*   Writing Quality: ⭐⭐⭐⭐ (Information-dense tables with clear conclusions; error analysis complements the raw score comparisons).
*   Value: ⭐⭐⭐⭐ (Highly practical for Legal NLP research and contract classification deployment, providing direct guidance for baseline selection).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PersonaArena: Dynamic Simulation for Evaluating and Enhancing Persona-Level Role-Playing in Large Language Models](personaarena_dynamic_simulation_for_evaluating_and_enhancing_persona-level_role-.md)
- [\[ACL 2026\] Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)
- [\[ACL 2026\] Nürnberg NLP at PsyDefDetect: Multi-Axis Voter Ensembles for Psychological Defence Mechanism Classification](nürnberg_nlp_at_psydefdetect_multi-axis_voter_ensembles_for_psychological_defenc.md)
- [\[ACL 2026\] MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models](muldimif_a_multi-dimensional_constraint_framework_for_evaluating_and_improving_i.md)
- [\[NeurIPS 2025\] Characterizing the Expressivity of Fixed-Precision Transformer Language Models](../../NeurIPS2025/llm_nlp/characterizing_the_expressivity_of_fixed-precision_transformer_language_models.md)

</div>

<!-- RELATED:END -->
