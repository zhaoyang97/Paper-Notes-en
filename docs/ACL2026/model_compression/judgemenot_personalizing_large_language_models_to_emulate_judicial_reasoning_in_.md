---
title: >-
  [Paper Note] JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew
description: >-
  [ACL 2026][Model Compression][LLM Personalization] This paper proposes a synthetic-organic supervision pipeline that transforms raw judicial opinions into reasoning instruction-tuning data. Through a Chain-of-LoRA strate…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "LLM Personalization"
  - "Judicial Reasoning"
  - "Low-Resource Languages"
  - "Parameter-Efficient Fine-Tuning"
  - "Synthetic Instruction Data"
date: 2026-05-08
content_hash: bd7d780d1e343fb0
---

# JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew

**Conference**: ACL 2026
**arXiv**: [2604.18041](https://arxiv.org/abs/2604.18041)
**Code**: [GitHub](https://github.com/Socially-Embedded-Lab/JudgeMeNot)
**Area**: Model Compression
**Keywords**: LLM Personalization, Judicial Reasoning, Low-Resource Languages, Parameter-Efficient Fine-Tuning, Synthetic Instruction Data

## TL;DR
This paper proposes a synthetic-organic supervision pipeline that transforms raw judicial opinions into reasoning instruction-tuning data. Through a Chain-of-LoRA strategy (CLM → instruction tuning), the framework achieves high-fidelity emulation of individual judges' reasoning styles, producing outputs indistinguishable from authentic judicial writing in the low-resource Hebrew setting.

## Background & Motivation

**Background**: Research on LLM personalization has grown rapidly in recent years, but most efforts focus on user preferences (style, recommendations) rather than modeling the reasoning processes of specific decision-makers. In the legal domain, judicial opinions are not mere mechanical applications of statutes; they reflect each judge's idiosyncratic reasoning patterns, argumentative emphases, and rhetorical structures.

**Limitations of Prior Work**: (1) Raw judicial opinions are unstructured long-form texts in which reasoning content is interleaved with procedural boilerplate and factual narration, making them difficult to use directly for training. (2) Judicial reasoning decisions are "unprompted" in the text—there are no explicit trigger questions. (3) The volume of data available per individual judge is limited, making it a core challenge to extract sufficiently strong individual signals in a computationally efficient manner.

**Key Challenge**: Personalization requires adequate reasoning supervision signals, yet the reasoning signal in legal opinions is diluted by large amounts of non-reasoning text. Applying causal language modeling (CLM) directly on raw text is inefficient.

**Goal**: Design a personalization framework that requires no manual annotation, scales to large numbers of judges, and enables LLMs to faithfully emulate specific judges' reasoning styles and content.

**Key Insight**: The legal domain naturally provides abundant decomposable reasoning traces—judges routinely handle complex decisions and produce detailed written justifications. By decomposing opinions into fine-grained reasoning claims (rather than attending only to final rulings), rich reasoning supervision signals can be obtained.

**Core Idea**: An agentic workflow automatically extracts reasoning claims from opinions and generates synthetic questions to construct reasoning instruction sets; efficient personalization is then achieved via a two-stage Chain-of-LoRA (CLM → instruction tuning).

## Method

### Overall Architecture
The framework comprises two stages. The first is data generation: a multi-LLM agent pipeline extracts reasoning claims from raw opinions and generates synthetic question–answer pairs. The second is model training: the framework compares personalization strategies including CLM, instruction tuning, Chain-of-LoRA (CLM → instruction tuning), and RAG.

### Key Designs

1. **Synthetic-Organic Alignment Pipeline**:

    - **Function**: Transforms unstructured judicial opinions into high-quality reasoning instruction pairs.
    - **Mechanism**: GPT-4.1-mini (temperature=0.3) is used for reasoning extraction and GPT-4o-mini (temperature=0.1) for validation. A multi-round agentic workflow proceeds as follows: extract reasoning claims → validate extraction quality → generate synthetic questions → validate question fidelity. The pipeline ultimately produces 62,051 reasoning sentences and their corresponding synthetic questions.
    - **Design Motivation**: Training directly on raw opinions dilutes the reasoning signal, while manual annotation does not scale. Synthetic questions compensate for the absence of explicit trigger questions in judicial opinions, enabling the model to learn reasoning in a question-answering format.

2. **Chain-of-LoRA (CoLA) Two-Stage Training**:

    - **Function**: Integrates general writing-style adaptation with reasoning specialization.
    - **Mechanism**: In the first step, QLoRA is applied to perform CLM on all of a judge's raw opinions (learning writing style), and the adapter weights are merged back into the base model. In the second step, another round of QLoRA fine-tuning is performed on the synthetic reasoning instruction set (learning reasoning patterns). The approach draws inspiration from Chain of LoRA.
    - **Design Motivation**: The CLM stage familiarizes the model with the judge's vocabulary and stylistic features, while the instruction-tuning stage focuses on reasoning logic. The two-stage separation allows the model to learn "how to write" and "how to reason" independently.

3. **Multi-Dimensional Evaluation Framework**:

    - **Function**: Comprehensively measures personalization quality.
    - **Mechanism**: Evaluation encompasses lexical similarity (BLEU, ROUGE), semantic similarity (BERTScore), stylistic similarity (JSD divergence over POS distributions), and an authorship attribution test (training a binary classifier to distinguish real from generated text).
    - **Design Motivation**: A single metric cannot capture the multiple facets of personalization—surface style and deep reasoning require different measures.

### Loss & Training
Gemma 3 (4B) is used as the base model with QLoRA (rank=8). A separate LoRA adapter is trained for each judge while the base model weights remain frozen. The CLM stage uses the standard causal language modeling loss; the instruction-tuning stage uses the standard SFT loss.

## Key Experimental Results

### Main Results (QA Task, CoLA Gains Relative to Baselines)

| Method | BLEU↑ | BS-F↑ | R-L↑ | POS-JSD↓ |
|--------|-------|-------|------|----------|
| Vanilla-Gemma (baseline) | 0 | 0 | 0 | 0 |
| Gemini-3-Pro RAG | -3.22 | -0.09 | -0.12 | +0.02 |
| Pers-CLM | -0.25 | -0.03 | -0.01 | +0.02 |
| Pers-IT | -7.02 | -0.09 | -0.15 | +0.02 |
| **CoLA (Ours)** | **Best** | **Best** | **Best** | **Best** |

### Authorship Attribution Test

| Method | Accuracy | Notes |
|--------|----------|-------|
| Random guessing | 50.0% | Baseline |
| Human vs. Human | 84.3% | Genuine inter-judge differences exist |
| Vanilla-Gemma | 70.3% | Easily identified |
| CLM-only | 56.2% | Still distinguishable |
| **CoLA** | **49.8%** | Indistinguishable from random |
| **IT-only** | **49.6%** | Indistinguishable from random |

### Key Findings
- **CoLA-generated text is indistinguishable from authentic judicial writing**: The authorship attribution classifier drops to chance level (49.8%), indicating extremely high generation quality.
- **Data quantity matters more than model size**: Ablation shows that doubling the data yields a +2.68 BLEU improvement, whereas doubling the LoRA rank yields only +0.77 BLEU.
- **Combining CLM and IT outperforms either alone**: Cross-judge specificity tests confirm that personalization effects are judge-specific rather than a general improvement.
- **RAG excels at surface style but is weak at reasoning**: RAG performs well on POS-JSD but lags on semantic metrics, indicating that parametric adaptation is necessary to truly capture reasoning.

## Highlights & Insights
- The decomposition of **"persona = style layer + reasoning layer"** is highly insightful: RAG can capture surface style but not reasoning, while parametric fine-tuning exhibits the opposite pattern. This suggests that personalization may require combining both approaches.
- The **synthetic supervision pipeline** design is highly practical: the pattern of using multi-agent systems to extract reasoning and generate questions from unstructured documents is transferable to medicine, education, and any domain requiring the extraction of decision-making reasoning from expert documents.
- **Achieving high-fidelity personalization on a 4B-parameter model** challenges the assumption that "reasoning requires large models and large data"—the key lies in the structured nature of the supervision signal.

## Limitations & Future Work
- The framework focuses exclusively on fine-grained reasoning claims and does not model case-level holistic reasoning chains.
- Temporal drift in a judge's reasoning style over time is not considered.
- Validation is conducted only in the Hebrew legal system; generalization across languages and jurisdictions remains unknown.
- Model weights are intentionally not released (to prevent misuse), which limits reproducibility.
- Future work could explore explicitly modeling reasoning chain dependencies and incorporating fact-grounded reasoning augmentation.

## Related Work & Insights
- **vs. OnePeFTPerUser**: The latter combines PEFT and retrieval for user personalization but targets labeled classification tasks without modeling reasoning.
- **vs. DRAFT**: DRAFT improves tool documentation through trial and error, a process conceptually similar to the synthetic data pipeline proposed here, but with a different objective (tool use vs. reasoning emulation).
- **vs. General Reasoning Models (e.g., o3)**: Reasoning models typically require verifiable steps (mathematics/code), whereas legal reasoning lacks such objective verification signals. This paper substitutes emulation fidelity for correctness as the optimization objective.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The synthetic-organic pipeline and CoLA training strategy are innovative, though individual components are relatively mature.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three tasks, multiple baselines, ablation studies, cross-judge validation, and robustness checks—extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure, thorough ethical discussion, and well-motivated argumentation.
- **Value**: ⭐⭐⭐⭐ Important implications for LLM reasoning personalization, though the application scope is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](selar_selective_latent_reasoning_in_large_language_models.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](../../AAAI2026/model_compression/efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](../../ICLR2026/model_compression/landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](training-free_test-time_contrastive_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
