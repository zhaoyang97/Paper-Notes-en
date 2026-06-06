---
title: >-
  [Paper Note] JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew
description: >-
  [ACL 2026][Model Compression][LLM Personalization] The study proposes a synthetic-organic supervision pipeline that transforms raw judicial decisions into reasoning instruction tuning data. Using a Chain-of-LoRA strategy…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "LLM Personalization"
  - "Judicial Reasoning"
  - "Low-resource Languages"
  - "Parameter-Efficient Fine-Tuning"
  - "Synthetic Instruction Data"
date: 2026-05-08
content_hash: c637c0de8aef7c1f
---

# JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew

**Conference**: ACL 2026  
**arXiv**: [2604.18041](https://arxiv.org/abs/2604.18041)  
**Code**: [GitHub](https://github.com/Socially-Embedded-Lab/JudgeMeNot)  
**Area**: Model Compression  
**Keywords**: LLM Personalization, Judicial Reasoning, Low-resource Languages, Parameter-Efficient Fine-Tuning, Synthetic Instruction Data

## TL;DR
The study proposes a synthetic-organic supervision pipeline that transforms raw judicial decisions into reasoning instruction tuning data. Using a Chain-of-LoRA strategy (CLM→Instruction Tuning), the approach achieves high-fidelity emulation of individual judges' reasoning styles, producing content indistinguishable from real judgments in low-resource Hebrew scenarios.

## Background & Motivation

**Background**: Research on LLM personalization has grown rapidly, yet most focus on user preferences (style, recommendations) rather than modeling the reasoning processes of specific decision-makers. In the legal domain, a judge's ruling is not a mechanical application of laws but reflects individual reasoning patterns, argumentative focus, and rhetorical structures.

**Limitations of Prior Work**: (1) Raw judicial decisions are unstructured long-form texts where reasoning is intertwined with procedural templates and factual statements, making them difficult to use directly for training; (2) Judicial reasoning is "unprompted" in the text—there are no explicit trigger questions; (3) Data for individual judges is limited, making it a core challenge to learn strong individual signals while maintaining computational efficiency.

**Key Challenge**: Personalization requires sufficient reasoning supervision signals, but these are diluted by large amounts of non-reasoning text in legal judgments. Direct Causal Language Modeling (CLM) on raw text is inefficient.

**Goal**: Design a non-human-annotated, scalable framework for large numbers of judges that enables LLMs to faithfully emulate specific judicial reasoning styles and content.

**Key Insight**: The legal domain naturally provides decomposable reasoning traces—judges regularly process complex decisions and write detailed arguments. By decomposing judgments into fine-grained reasoning statements (rather than just final rulings), rich reasoning training signals can be obtained.

**Core Idea**: Use an agentic workflow to automatically extract reasoning statements from judgments and generate synthetic questions to construct a reasoning instruction set. This is followed by a two-stage Chain-of-LoRA (CLM→Instruction Tuning) for efficient personalization.

## Method

### Overall Architecture
The framework consists of two phases: The first phase is data generation—using multi-LLM agents to extract reasoning statements and generate synthetic Q&A pairs from raw judgments. The second phase is model training—comparing personalization schemes like CLM, instruction tuning, Chain-of-LoRA (CLM→Instruction Tuning), and RAG.

### Key Designs

1. **Synthetic-Organic Alignment Pipeline**:

    - **Function**: Transforms unstructured judicial documents into high-quality reasoning instruction pairs.
    - **Mechanism**: Employs GPT-4.1-mini for reasoning extraction (temperature=0.3) and GPT-4o-mini for verification (temperature=0.1). Through a multi-round agentic workflow, it extracts reasoning statements → verifies extraction quality → generates synthetic questions → verifies question fidelity. This resulted in 62,051 reasoning sentences and corresponding synthetic questions.
    - **Design Motivation**: Training directly on raw judgments dilutes reasoning signals, while human annotation is not scalable. Synthetic questions compensate for the missing "implicit trigger questions" in judgments, allowing models to learn reasoning in a Q&A format.

2. **Chain-of-LoRA (CoLA) Two-Stage Training**:

    - **Function**: Integrates general writing style adaptation with reasoning specialization.
    - **Mechanism**: The first step uses QLoRA for CLM on all of a judge's raw judgments (learning writing style), merging adapter weights back into the base model. The second step performs another round of QLoRA fine-tuning on the synthetic reasoning instruction set (learning reasoning patterns). This follows the Chain of LoRA concept.
    - **Design Motivation**: The CLM stage familiarizes the model with the judge's vocabulary and stylistic features, while the instruction tuning stage focuses on reasoning logic. This separation allows the model to learn "how to write" and "how to think" independently.

3. **Multi-dimensional Evaluation System**:

    - **Function**: Comprehensively measures personalization quality.
    - **Mechanism**: Includes lexical similarity (BLEU, ROUGE), semantic similarity (BERTScore), stylistic similarity (JSD of POS distributions), and an authorship identification test (training a binary classifier to distinguish real vs. generated text).
    - **Design Motivation**: A single metric cannot capture the multi-faceted nature of personalization—surface style and deep reasoning require different measures.

### Loss & Training
Gemma 3 (4B) is used as the base with a QLoRA configuration (rank=8). A separate LoRA adapter is trained for each judge, with base weights frozen. The CLM stage uses standard causal language modeling loss, while the instruction tuning stage uses standard SFT loss.

## Key Experimental Results

### Main Results (Q&A Task, Delta improvement of CoLA relative to baselines)

| Method | BLEU↑ | BS-F↑ | R-L↑ | POS-JSD↓ |
|------|-------|-------|------|----------|
| Vanilla-Gemma (Baseline) | 0 | 0 | 0 | 0 |
| Gemini-3-Pro RAG | -3.22 | -0.09 | -0.12 | +0.02 |
| Pers-CLM | -0.25 | -0.03 | -0.01 | +0.02 |
| Pers-IT | -7.02 | -0.09 | -0.15 | +0.02 |
| **CoLA (Ours)** | **Best** | **Best** | **Best** | **Best** |

### Authorship Identification Test

| Method | Accuracy | Description |
|------|--------|------|
| Random Guess | 50.0% | Baseline |
| Human vs. Human | 84.3% | Differences exist between judges |
| Vanilla-Gemma | 70.3% | Easily identified |
| CLM-only | 56.2% | Still distinguishable |
| **CoLA** | **49.8%** | Indistinguishable from random |
| **IT-only** | **49.6%** | Indistinguishable from random |

### Key Findings
- **CoLA-generated text is indistinguishable from real judges**: The authorship identification classifier's accuracy dropped to random levels (49.8%), indicating extremely high generation quality.
- **Data quantity is more important than model size**: Ablations show doubling data yields a +2.68 BLEU gain, while doubling LoRA rank only yields +0.77 BLEU.
- **The CLM+IT combination outperforms standalone use**: Cross-judge specificity tests confirmed that personalization effects are judge-specific rather than general improvements.
- **RAG excels at surface style but fails at reasoning**: RAG performed well on POS-JSD but lagged in semantic metrics, suggesting that parameter adaptation is necessary to capture reasoning.

## Highlights & Insights
- The decomposition of **"persona = style layer + reasoning layer"** is insightful: RAG captures surface style but not reasoning, while parameter fine-tuning does the opposite. This suggests personalization may require combining both paths.
- The **synthetic supervision pipeline** is highly practical: The pattern of extracting reasoning and generating questions via multi-agents from unstructured documents can be migrated to any field requiring expert decision-making extraction (e.g., medicine, education).
- Achieving high-fidelity personalization on a **4B parameter model** challenges the notion that "reasoning requires large models and huge data"—the key lies in the structural organization of supervision signals.

## Limitations & Future Work
- Focuses only on fine-grained reasoning statements without modeling the global case-level reasoning chain.
- Does not consider the temporal drift of a judge's reasoning style over time.
- Validated only within the Hebrew legal system; cross-language/cross-jurisdiction generalization remains unknown.
- Model weights are intentionally not released (to prevent misuse), which limits reproducibility.
- Future work could explore explicit modeling of reasoning chain dependencies and reasoning enhancement combined with factual grounding.

## Related Work & Insights
- **vs OnePeFTPerUser**: The latter combines PEFT and retrieval for user personalization but targets labeled tasks (classification/tagging) rather than reasoning.
- **vs DRAFT**: DRAFT improves tool documentation through trial-and-error, similar to this synthetic data pipeline, but with different goals (tool use vs. reasoning emulation).
- **vs General Reasoning Models (e.g., o3)**: Reasoning models typically require verifiable steps (math/code), whereas legal reasoning lacks such objective verification signals. This work substitutes correctness with emulation fidelity as the optimization objective.

## Rating
- Novelty: ⭐⭐⭐⭐ The synthetic-organic pipeline and CoLA strategy are innovative, though individual components are relatively mature.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with three tasks, multiple baselines, ablation studies, cross-judge validation, and robustness tests.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, thorough ethical discussion, and smooth motivation derivation.
- Value: ⭐⭐⭐⭐ Important insights for personalized LLM reasoning, though the application scenario is niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](../../AAAI2026/model_compression/efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](../../ICLR2026/model_compression/landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[ACL 2026\] GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models](grasprune_global_gating_for_budgeted_structured_pruning_of_large_language_models.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](training-free_test-time_contrastive_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
