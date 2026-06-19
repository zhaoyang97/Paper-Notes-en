---
title: >-
  [Paper Note] Evaluating Customized vs. Generalist Transformer-based Models for Legal Contract Classification
description: >-
  [ACL 2026][LLM (Other)][legal-specific models] Ours systematically compares 13 legal-domain customized Transformer models with 9 generalist models across 3 English contract classification tasks. It finds that small-scale but contract-relevant pretrained models like Legal-BERT and Contracts-BERT typically outperform larger generalist models on long-tail legal labels
tags:
  - ACL 2026
  - LLM (Other)
  - legal-specific models
  - contract classification
  - long-tail labels
  - macro-F1
  - domain pretraining
date: 2026-05-08
content_hash: 2d11fd35e544e463
---
# Evaluating Customized vs. Generalist Transformer-based Models for Legal Contract Classification

**Conference**: ACL2026  
**arXiv**: [2508.07849](https://arxiv.org/abs/2508.07849)  
**Code**: No public code link found in cache  
**Area**: Legal NLP / Contract Classification / Domain Model Evaluation  
**Keywords**: legal-specific models, contract classification, long-tail labels, macro-F1, domain pretraining

## TL;DR
Ours systematically compares 13 legal-domain customized Transformer models with 9 generalist models across 3 English contract classification tasks. It finds that small-scale but contract-relevant pretrained models like Legal-BERT and Contracts-BERT typically outperform larger generalist models on long-tail legal labels.

## Background & Motivation
**Background**: In legal NLP, numerous open-source legal models exist, such as Legal-BERT, Contracts-BERT, CaseLaw-BERT, and LexLM; simultaneously, general encoder/decoder models are widely used for contract clause classification, topic identification, and deontic modality recognition (e.g., Identifying obligations, rights, and prohibitions).

**Limitations of Prior Work**: Despite the inherent dependence of contract tasks on legal semantics, many existing studies primarily evaluate generalist models, often failing to include legal domain-specific models. Consequently, the community remains unclear whether legal-specific models are truly more suitable for contract classification, or whether model scale, pretraining corpora, or task distribution is the dominant factor.

**Key Challenge**: General large models possess more parameters and broader knowledge, but contract classification often deals with long-tail labels, legal jargon, and fine-grained clause semantics. Larger models are not necessarily superior to smaller but in-domain encoder models, particularly when macro-F1 exposes errors in rare classes.

**Goal**: To answer a direct question: do legal-specific Transformer models outperform generalist models in contract classification? The authors aim to provide benchmark results across tasks, model types, and metrics while identifying strong baselines for future legal contract classification.

**Key Insight**: Three public English contract datasets are selected, covering multi-label, multi-class, varying scales, and different legal semantic granularities. 13 legal models and 9 generalist models are compared uniformly, focusing on micro-F1, macro-F1, and misclassifications of rare classes.

**Core Idea**: Instead of a general "legal model vs. generalist model" comparison, the study decomposes evaluations by model type, corpus domain alignment, and long-tail label performance within contract tasks.

## Method

### Overall Architecture
Ours is a benchmark/evaluation paper that does not propose a new model; its contribution lies in a controlled comparative evaluation protocol. Inputs are three types of public English contract classification tasks: UNFAIR-ToS (identifying unfair terms in terms of service with 9 multi-label categories), LEDGAR (SEC Exhibit 10 clause topic classification with 100 multi-class categories), and LEXDEMOD (detecting party-specific obligations/rights/prohibitions in lease contracts with 7 multi-label categories). 13 legal-specific Transformers (Legal-BERT, Contracts-BERT, InLegalBERT, LexLM, SaulLM, etc.) and 9 generalist models (BERT/RoBERTa/DeBERTa/Longformer/BigBird, plus Llama-3.2, Mistral, etc.) are integrated into the same pipeline. Task-specific fine-tuning is performed for each task, using micro-F1 and macro-F1 to answer whether domain pretraining is truly worthwhile for contract classification.

### Key Designs

**1. Multi-task Contract Benchmark Coverage: Preventing "single-task overfit conclusions" through task diversity**

Comparing models on a single contract task often leads to narrow conclusions. Therefore, the authors deliberately selected three tasks with different semantic granularities and scales: UNFAIR-ToS, LEDGAR, and LEXDEMOD. These cover terms of service, SEC contract topics, and deontic modality in lease contracts, respectively. Test set sizes range from approximately 1.6k to 10k, and label counts from 7 to 100, spanning both multi-label and multi-class formats. The effectiveness of legal models depends on whether task semantics and data distributions align with their pretraining corpora; this cross-scale, cross-label setup exposes stability differences in rare classes, multi-label coupling, and long clause texts.

**2. Side-by-Side Comparison of legal-specific vs. generalist: Decomposing domain, scale, and architecture**

Previous works often evaluated only generalist models or a few selected legal models, failing to distinguish whether performance gains stemmed from domain pretraining, model capacity, or architecture type. Ours places 13 legal models and 9 generalist models under identical tasks, fine-tuning protocols, and F1 metrics for side-by-side comparison, deliberately mixing encoder, decoder, and encoder-decoder architectures. Consequently, the results illustrate trade-offs such as "110M Contracts-BERT vs. 355M RoBERTa-large" while revealing whether decoder-style generative models are disadvantaged in discriminative classification, allowing relative contributions to be attributed individually.

**3. Long-tail Error Analysis: Explaining the origin of advantages beyond average scores**

The true difficulty in contract tasks often lies in rare clause types with high legal stakes, which micro-F1 can mask by emphasizing high-frequency categories. To address this, the authors analyze misclassified samples from RoBERTa-large to observe if legal models like Contracts-BERT can correct errors in rare categories of UNFAIR-ToS, such as Limitation of Liability and Unilateral Termination. Combined with macro-F1, which is more sensitive to rare classes, this case-level analysis reveals deployment risks—the true reliability of models on high-risk, low-frequency clauses—that pure score comparisons fail to show.

### Loss & Training
No new loss functions are introduced, utilizing standard task-specific fine-tuning and classification evaluation. Both multi-label tasks (UNFAIR-ToS, LEXDEMOD) and the multi-class task (LEDGAR) report micro-F1 and macro-F1. The former measures overall accuracy, while the latter reflects long-tail robustness. An engineering detail to note is that contract texts often exceed 512 subword tokens; while truncation or long-context handling affects performance, the evaluation focus remains on comparing model families and domain pretraining rather than long-text modeling itself.

## Key Experimental Results

### Main Results

| Dataset | Metric | Best/Representative Legal Model | Strongest Generalist Model | Conclusion |
|--------|------|------|----------|------|
| UNFAIR-ToS | micro-F1 / macro-F1 | Contracts-BERT 96.2 / 83.4; Legal-BERT 96.0 / 82.2 | RoBERTa-large 95.8 / 81.6; Mistral 96.0 / 80.7 | Legal models are stronger in macro-F1 |
| LEDGAR | micro-F1 / macro-F1 | Legal-BERT 88.2 / 82.5; Contracts-BERT 87.9 / 82.2 | RoBERTa-large 88.6 / 83.6 | RoBERTa-large remains best on large-scale LEDGAR |
| LEXDEMOD | micro-F1 / macro-F1 | Legal-BERT 81.23 / 78.01; InLegalBERT 80.21 / 77.89 | RoBERTa-large macro-F1 77.88; Llama-3.2 76.2 / 71.4 | Legal encoders significantly outperform decoders |
| Average Performance | Mean micro-F1 | Legal-BERT 88.48±6.03 | N/A | Legal-BERT has the strongest overall micro-F1 |
| Average Performance | Mean macro-F1 | Contracts-BERT 81.10±2.45 | N/A | Contracts-BERT is more stable for long-tail labels |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Legal-BERT | Mean micro-F1 88.48±6.03, macro-F1 80.90±2.05 | Ranked 1st in overall micro-F1 |
| Contracts-BERT | Mean micro-F1 88.09±6.55, macro-F1 81.10±2.45 | Ranked 1st in overall macro-F1 |
| CaseLawBERT | Mean micro-F1 88.01±6.45, macro-F1 80.62±2.23 | Strong baseline for contract classification |
| LexLM | Mean micro-F1 88.03±6.33, macro-F1 80.15±1.91 | Another stable legal baseline |
| PoL-BERT | LEXDEMOD micro-F1 41.35, macro-F1 15.75 | Large/new legal models aren't always contract-compatible |

### Key Findings
- Legal-specific models establish new SOTAs in two tasks. Legal-BERT and Contracts-BERT, with only 110M parameters (69% fewer than RoBERTa-large), outperform larger models on UNFAIR-ToS and LEXDEMOD.
- RoBERTa-large remains strongest on LEDGAR, indicating domain pretraining is not a panacea for all tasks; data scale, label count, and model capacity also influence results.
- Decoder-based generalist models perform poorly on rare classes, particularly in macro-F1. This supports the continued importance of encoder-based discriminative fine-tuning for long-tail legal classification.
- Older but more contract-focused models like Legal-BERT/Contracts-BERT outperform newer, larger models with diverse legal corpora, suggesting that in-distribution pretraining is more critical than raw corpus scale.

## Highlights & Insights
- The most practical insight is that legal contract classification should not default to larger generalist models. Small legal encoders offer reality-based advantages in privacy, cost, long-tail category handling, and deployability.
- Macro-F1 analysis is highly valuable. In legal scenarios, rare classes often correspond to high-risk clauses; while micro-F1 may look promising, frequent rare-class errors make models unreliable for deployment.
- "More legal data" does not equal "more contract-relevant." If pretraining corpora are diluted with case law, statutes, or patents, the contract signal may weaken, making the model inferior to smaller, more focused ones.
- This benchmark serves as a reminder that model selection should be guided by task semantic alignment rather than being sorted by the "legal" label or parameter count.

## Limitations & Future Work
- Experiments cover only English contract data; non-English, cross-jurisdictional, or multilingual legal documents remain unverified.
- The paper focuses on contract language and does not evaluate other legal text types like statutes, court judgments, or legal opinions. Therefore, conclusions cannot be directly extrapolated to the entire Legal NLP domain.
- Ours primarily examines domain generalization and does not systematically research alternative routes like RAG, long-context handling, hierarchical classification, or prompt-based decoder inference.
- Many passages in LEDGAR exceed 512 subword tokens; truncation strategies might affect performance across models. Future work could include controlled experiments with long-text models like Longformer/BigBird or chunk aggregation.

## Related Work & Insights
- **vs. Legal-BERT / Contracts-BERT original work**: While original models proved legal pretraining is valuable, Ours further clarifies that "contract-relevant corpora" are more important than general legal corpora for contract tasks.
- **vs. Generalist encoder baselines**: RoBERTa-large's advantage in LEDGAR suggests that when data and label coverage are sufficient, the capacity of large general encoders still plays a role.
- **vs. Decoder-based legal/generalist LLMs**: Llama, Mistral, SaulLM, and AdaptLLM are unstable in classification, indicating that generative large models may not replace discriminative encoders, especially for long-tail label classification.
- **Insight**: Selection for legal AI systems should prioritize intra-task benchmarking and report macro-F1 and rare-class errors rather than just comparing average accuracy or model scale.

## Rating
- Novelty: ⭐⭐⭐ (As a benchmark paper, innovation lies in systematic coverage and clear problem setting).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Task, model, and metric coverage is complete, though lacking non-English documents).
- Writing Quality: ⭐⭐⭐⭐ (Dense information with clear conclusions; error analysis complements pure score comparisons).
- Value: ⭐⭐⭐⭐ (Highly practical for Legal NLP research and contract classification deployment, providing direct guidance for baseline selection).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Nürnberg NLP at PsyDefDetect: Multi-Axis Voter Ensembles for Psychological Defence Mechanism Classification](nürnberg_nlp_at_psydefdetect_multi-axis_voter_ensembles_for_psychological_defenc.md)
- [\[ACL 2025\] TESS 2: A Large-Scale Generalist Diffusion Language Model](../../ACL2025/llm_nlp/tess_2_a_large-scale_generalist_diffusion_language_model.md)
- [\[ACL 2026\] Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)
- [\[ACL 2025\] CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models](../../ACL2025/llm_nlp/cognibench_cognitive_faithfulness.md)
- [\[ACL 2026\] PersonaArena: Dynamic Simulation for Evaluating and Enhancing Persona-Level Role-Playing in Large Language Models](personaarena_dynamic_simulation_for_evaluating_and_enhancing_persona-level_role-.md)

</div>

<!-- RELATED:END -->
