---
title: >-
  [Paper Note] Evaluating Customized vs. Generalist Transformer-based Models for Legal Contract Classification
description: >-
  [ACL2026][LLM (Other)][legal-specific models] This paper systematically compares the performance of 13 customized legal Transformer models and 9 general-purpose models on 3 English contract classification tasks. It finds that smaller models with contract-relevant pretraining, such as Legal-BERT and Contracts-BERT, generally outperform larger general-purpose models on long-tail legal labels.
tags:
  - "ACL2026"
  - "LLM (Other)"
  - "legal-specific models"
  - "contract classification"
  - "long-tail labels"
  - "macro-F1"
  - "domain pretraining"
date: 2026-05-08
content_hash: d0b46ad0a4a91ef5
---

# Evaluating Customized vs. Generalist Transformer-based Models for Legal Contract Classification

**Conference**: ACL2026  
**arXiv**: [2508.07849](https://arxiv.org/abs/2508.07849)  
**Code**: No public code link found in cache  
**Area**: Legal NLP / Contract Classification / Domain Model Evaluation  
**Keywords**: legal-specific models, contract classification, long-tail labels, macro-F1, domain pretraining

## TL;DR
This paper systematically compares the performance of 13 customized legal Transformer models and 9 general-purpose models on 3 English contract classification tasks. It finds that smaller models with contract-relevant pretraining, such as Legal-BERT and Contracts-BERT, generally outperform larger general-purpose models on long-tail legal labels.

## Background & Motivation
**Background**: In legal NLP, several open-source legal models have emerged, such as Legal-BERT, Contracts-BERT, CaseLaw-BERT, and LexLM. Simultaneously, general encoder/decoder models are widely utilized for tasks like contract clause classification, clause topic identification, and deontic modality recognition (e.g., obligations, rights, prohibitions).

**Limitations of Prior Work**: Although contract tasks naturally depend on legal semantics, many existing works predominantly evaluate general models or omit legal domain models entirely. Consequently, the community remains unclear whether legal-specific models are truly superior for contract classification or whether model scale, pretraining corpora, or task distribution is the primary performance driver.

**Key Challenge**: While general large models possess more parameters and broader knowledge, contract classification frequently encounters long-tail labels, specialized legal terminology, and fine-grained clause semantics. Larger models are not necessarily superior to smaller but in-domain encoder models, particularly as macro-F1 metrics tend to expose errors in rare classes.

**Goal**: To answer a straightforward question: do legal-specific Transformer models outperform generalist models in contract classification? The authors aim to provide benchmark results across tasks, model types, and metrics to identify robust baselines for future legal contract classification.

**Key Insight**: The paper selects three public English contract classification datasets covering multi-label, multi-class, varying scales, and different levels of legal semantic granularity. It then evaluates 13 legal models against 9 general models, focusing on micro-F1, macro-F1, and misclassifications of rare classes.

**Core Idea**: Instead of a vague comparison of "legal vs. general models," the study decomposes model types, domain alignment of corpora, and performance on long-tail labels within the context of contract tasks.

## Method

### Overall Architecture
This is a benchmark/evaluation paper that does not propose a new model; its contribution lies in a controlled comparative evaluation protocol. The inputs consist of three public English contract classification tasks: UNFAIR-ToS (identifying unfair terms in terms of service, 9-class multi-label), LEDGAR (SEC Exhibit 10 clause topic classification, 100-class multi-class), and LEXDEMOD (detecting subject-specific obligations/rights/prohibitions in lease contracts, 7-class multi-label). 13 legal Transformers and 9 general models (BERT, RoBERTa, DeBERTa, Longformer, BigBird, and decoders like Llama-3.2 and Mistral) are integrated into a unified pipeline. Task-specific fine-tuning is performed for each task, using micro-F1 and macro-F1 to determine if domain pretraining is truly beneficial for contract classification.

### Key Designs

**1. Multi-task Contract Benchmark Coverage: Preventing Single-Task Overfitting Conclusions**  
Evaluating models on a single contract task can lead to narrow conclusions. Therefore, the authors deliberately selected three tasks with varying semantic granularities and scales: UNFAIR-ToS, LEDGAR, and LEXDEMOD. These cover terms of service, SEC contract topics, and lease contract modalities, with test sets ranging from 1.6k to 10k samples and labels from 7 to 100. This cross-scale, cross-label, and cross-length setup exposes differences in model stability regarding rare classes, multi-label coupling, and long clause texts.

**2. Direct Comparison of Legal-specific and Generalist Models: Decoupling Domain, Scale, and Architecture**  
Prior works often evaluated only general models or a small selection of legal models, making it difficult to distinguish the roles of domain pretraining, model capacity, and architecture. This paper places all 22 models under identical tasks, fine-tuning protocols, and F1 metrics. By including encoders, decoders, and encoder-decoders, the comparison reveals trade-offs (e.g., "110M Contracts-BERT vs. 355M RoBERTa-large") and whether decoder-based generative models are disadvantaged in discriminative classification.

**3. Long-tail Error Analysis: Explaining the Source of Superiority**  
The primary challenge in contract tasks resides in rare but legally significant clause types; high-frequency categories often mask these in micro-F1 scores. The authors analyze misclassified samples from RoBERTa-large to observe whether legal models like Contracts-BERT can correct errors in rare categories such as "Limitation of Liability" or "Unilateral Termination" in UNFAIR-ToS. Combined with macro-F1, this instance-level analysis reveals deployment risks that purely numerical comparisons might miss.

### Loss & Training
The study does not introduce new loss functions, utilizing standard task-specific fine-tuning and classification evaluation. Both multi-label (UNFAIR-ToS, LEXDEMOD) and multi-class (LEDGAR) tasks report micro-F1 (for overall accuracy) and macro-F1 (for rare class robustness). Note that contract clauses often exceed 512 subword tokens; while truncation or long-context handling affects performance, the focus remains on model families and domain pretraining rather than long-text modeling per se.

## Key Experimental Results

### Main Results

| Dataset | Metric | Best/Representative Legal Model (Ours) | Strongest General Model | Conclusion |
|---------|--------|----------------------------------------|-------------------------|------------|
| UNFAIR-ToS | micro-F1 / macro-F1 | Contracts-BERT 96.2 / 83.4; Legal-BERT 96.0 / 82.2 | RoBERTa-large 95.8 / 81.6; Mistral 96.0 / 80.7 | Legal models are stronger on macro-F1 |
| LEDGAR | micro-F1 / macro-F1 | Legal-BERT 88.2 / 82.5; Contracts-BERT 87.9 / 82.2 | RoBERTa-large 88.6 / 83.6 | RoBERTa-large remains best on large-scale LEDGAR |
| LEXDEMOD | micro-F1 / macro-F1 | Legal-BERT 81.23 / 78.01; InLegalBERT 80.21 / 77.89 | RoBERTa-large macro-F1 77.88; Llama-3.2 76.2 / 71.4 | Legal encoders significantly outperform decoders |
| Average Performance | Mean micro-F1 | Legal-BERT 88.48±6.03 | Not summarized as main body of Table 2 | Legal-BERT has the strongest overall micro-F1 |
| Average Performance | Mean macro-F1 | Contracts-BERT 81.10±2.45 | Not summarized as main body of Table 2 | Contracts-BERT is more stable for long-tail labels |

### Ablation Study

| Configuration | Key Metrics | Description |
|---------------|-------------|-------------|
| Legal-BERT | Mean micro-F1 88.48±6.03, macro-F1 80.90±2.05 | Ranked 1st in overall micro-F1 |
| Contracts-BERT | Mean micro-F1 88.09±6.55, macro-F1 81.10±2.45 | Ranked 1st in overall macro-F1 |
| CaseLawBERT | Mean micro-F1 88.01±6.45, macro-F1 80.62±2.23 | Robust baseline for contract classification |
| LexLM | Mean micro-F1 88.03±6.33, macro-F1 80.15±1.91 | Another stable legal baseline |
| PoL-BERT | LEXDEMOD micro-F1 41.35, macro-F1 15.75 | Larger/newer legal models may not fit contract distributions |

### Key Findings
- Legal-specific models established new SOTA results on two tasks. Legal-BERT and Contracts-BERT (110M parameters) outperformed RoBERTa-large on UNFAIR-ToS and LEXDEMOD despite having 69% fewer parameters.
- RoBERTa-large remains strongest on LEDGAR, suggesting domain pretraining is not a universal solution; data scale, label count, and model capacity also influence results.
- Decoder-based generalist models perform poorly on rare classes, especially as measured by macro-F1. This supports the continued importance of encoder-based discriminative fine-tuning for long-tail legal classification.
- Older models with more focused contract-related corpora (e.g., Legal-BERT/Contracts-BERT) outperformed some newer, larger models trained on heterogeneous legal data, indicating that in-distribution pretraining is more critical than raw corpus size.

## Highlights & Insights
- The most practical conclusion: Legal contract classification should not default to larger general models. Small legal encoders offer real-world advantages in privacy, cost, long-tail performance, and deployability.
- Macro-F1 analysis is highly valuable. In legal scenarios, rare classes often correspond to high-risk clauses; while high micro-F1 is visually appealing, frequent errors in rare classes make a model unreliable for deployment.
- "More legal data" does not equal "more contract relevance." Pretraining corpora mixed with excessive case law, statutes, or patents may dilute contract signals, potentially performing worse than models focused on contracts.
- This benchmark serves as a reminder that model selection should align with task semantics rather than being based strictly on "legal" labels or parameter counts.

## Limitations & Future Work
- The experiments only cover English contract data; non-English, cross-jurisdictional, and multilingual legal documents remain unverified.
- The paper focuses on contract language and does not evaluate other legal text types like statutes, judgments, or legal opinions; thus, conclusions cannot be directly extrapolated to all legal NLP.
- The study primarily investigates domain generalization and does not systematically explore RAG, long-context handling, hierarchical classification, or prompt-based decoder inference.
- Many segments in LEDGAR exceed 512 subword tokens. Truncation strategies might affect model performance differently. Future work could include controlled experiments with long-text models like Longformer or BigBird, or incorporate retrieval/chunk aggregation.

## Related Work & Insights
- **vs. Legal-BERT / Contracts-BERT Original Work**: While original models proved the value of legal pretraining, this study further demonstrates that "contract-relevant corpora" are more important than general legal corpora for contract tasks.
- **vs. General Encoder Baselines**: RoBERTa-large's advantage on LEDGAR shows that when data volume and label coverage are sufficient, the capacity of a large general encoder remains influential.
- **vs. Decoder-based Legal/Generalist LLMs**: The instability of Llama, Mistral, SaulLM, and AdaptLLM in classification suggests that generative LLMs do not necessarily replace discriminative encoders, particularly in long-tail classification.
- **Insight**: Selection for legal AI systems should prioritize in-task benchmarks reporting macro-F1 and rare class errors rather than just comparing average accuracy or model scale.

## Rating
- Novelty: ⭐⭐⭐ (Innovation lies in systematic coverage and clear problem setting rather than architecture).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive coverage of tasks, models, and metrics; lacking non-English and diverse legal genres).
- Writing Quality: ⭐⭐⭐⭐ (Dense information in tables with clear conclusions; error analysis complements the scores).
- Value: ⭐⭐⭐⭐ (Highly practical for legal NLP research and contract classification deployment, providing direct guidance on baseline selection).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Nürnberg NLP at PsyDefDetect: Multi-Axis Voter Ensembles for Psychological Defence Mechanism Classification](nürnberg_nlp_at_psydefdetect_multi-axis_voter_ensembles_for_psychological_defenc.md)
- [\[ICLR 2026\] Parameters vs. Context: Fine-Grained Control of Knowledge Reliance in Language Models](../../ICLR2026/llm_nlp/parameters_vs_context_fine-grained_control_of_knowledge_reliance_in_language_mod.md)
- [\[ACL 2026\] Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)
- [\[ACL 2025\] TESS 2: A Large-Scale Generalist Diffusion Language Model](../../ACL2025/llm_nlp/tess_2_a_large-scale_generalist_diffusion_language_model.md)
- [\[ACL 2025\] CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models](../../ACL2025/llm_nlp/cognibench_cognitive_faithfulness.md)

</div>

<!-- RELATED:END -->
