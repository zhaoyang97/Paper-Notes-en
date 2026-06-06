---
title: >-
  [Paper Note] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text
description: >-
  [LLM Evaluation] This paper proposes the SLGM framework, which reformulates structured prediction tasks for generative language models as classification problems via three components: **structured input format**…
tags:
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 5de455950514ea45
---

# Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text

- **Conference**: AAAI 2026
- **arXiv**: [2402.08971](https://arxiv.org/abs/2402.08971)
- **Code**: Not provided
- **Area**: LLM Evaluation
- **Keywords**: Structured generation, format-aware decoding, loss calibration, named entity recognition, information extraction, low-resource adaptation

## TL;DR

This paper proposes the SLGM framework, which reformulates structured prediction tasks for generative language models as classification problems via three components: **structured input format**, **format loss**, and **format-aware decoding**. Without introducing additional model parameters, SLGM significantly improves structural prediction performance of sub-1B models across 13 datasets spanning 5 task categories, including NER, RE, and SRL.

## Background & Motivation

- **Background**: Generative pre-trained language models (GLMs) excel at open-ended text generation but fall substantially short of encoder models of comparable scale on **structured prediction tasks** (NER, relation extraction, semantic role labeling, etc.).
- **Limitations of Prior Work**: The conventional explanation attributes this gap to a lack of structural knowledge; however, GLMs already generate syntactically correct and semantically coherent text. Methods such as DeepStruct and TANL rely on implicit task/dataset name prompts, requiring the model to retrieve format information from memory—an approach that is unreliable and difficult to generalize. GPT-4, for instance, achieves only 57.7 Entity F1 on CoNLL-04 joint entity-relation extraction, far below smaller specialized models.
- **Key Challenge**: The authors hypothesize that the true bottleneck lies in the missing connection between the model's **internal structural representations** and its **output token space**.
- **Goal**: To bridge this gap through explicit format specifications, format-aware training losses, and constrained decoding, enabling small generative models to perform competitively on structured prediction tasks without increasing model parameters.

## Method

### 1. Task-specific Output Format

SLGM defines an explicit output format string for each task/dataset, comprising:
- **Slot separators** `<;>` and **object separators** `</>`
- Each slot may contain: `<ANY>` (arbitrary token), `<SOURCE>` (must originate from the input text), or a predefined label set.

Example NER format: `<SOURCE> <;> instance of <;> tagset </>`

Prior to inference, SLGM parses the format string into a **format mask tensor**—a boolean matrix marking valid token IDs for each slot—used for loss computation and decoding constraints.

### 2. Format Loss

Standard cross-entropy loss operates over the entire vocabulary, making the target space excessively broad. SLGM introduces two complementary losses:

**Structure Loss** — penalizes incorrect generation of separators:

$$L_{st} = \sum_{t \in S} l_t \cdot missed \cdot w_{miss}$$

where $l_t$ is the log-probability of token $t$, $S$ is the set of separator positions, $w_{miss}$ is the penalty weight, and $missed$ counts how often a separator fails to receive the highest probability. This loss grows **exponentially**: three missing separators yield a total multiplier of $3^2 = 9$.

**Slot Loss** — analogous to cross-entropy but with the denominator covering only the candidate token set for the current slot rather than the full vocabulary. This effectively **reformulates sequence generation as a classification task**, concentrating gradients within the correct label subspace.

The final training loss is a weighted sum of all three: $L = \alpha \cdot L_{CE} + \beta \cdot L_{st} + \gamma \cdot L_{slot}$.

### 3. Formatted Decoding

At inference time, a **finite state machine** (FSM) tracks the current generation stage:
- Each generated slot separator advances the state counter.
- Each generated object separator resets the counter to zero.
- The format mask is retrieved based on the current state, and large negative penalties are applied to logits of invalid tokens.

This ensures the model can only generate valid tokens at each position (correct labels, source tokens, or separators), fundamentally eliminating format errors.

## Experiments

### Experimental Setup

- **Backbone models**: Flan-T5 (small 77M / base / large 0.8B)
- **Training pipeline**: Structured pre-training (TEKGEN + KELM, 400k sentences) → multi-task training (13 datasets, 2 epochs)
- **5 task categories**: NER, RE, SRL, Intent Detection, Dialogue State Tracking
- **Baselines**: CE (no meta-information), CE+task (with task name), CE+data (with dataset name), DeepStruct

### Table 1: Main Results (Multi-task Setting)

| Task | Dataset | SLGM F1 | SLGM FE | CE+data F1 | CE+data FE | DeepStruct F1 |
|------|---------|---------|---------|------------|------------|---------------|
| NER | CoNLL-03 | 80.28 | 43 | 87.11 | 5 | 93.1 |
| NER | OntoNotes | 75.87 | 142 | 81.12 | 348 | 87.6 |
| NER | GENIA | 69.88 | 5 | 66.48 | 30 | 80.2 |
| RE | TACRED | 63.11 | 2 | 59.07 | 13 | 74.9 |
| JER | CoNLL-04 Ent | 71.74 | 0 | 74.88 | 14 | 88.4 |
| JER | CoNLL-04 Rel | 27.87 | 2 | 48.53 | 46 | 72.8 |
| SRL | CoNLL-12 | 83.45 | 0 | 82.35 | 159 | 60.6 |
| ID | ATIS | 93.96 | 0 | 94.21 | 9 | 97.3 |
| DST | MultiWOZ | 38.87 | 0 | 37.18 | 0 | 53.5 |
| **Average** | - | **70.88** | **28** | **73.07** | **51** | **82.9** |

Without dataset name information, SLGM achieves performance comparable to CE+data while producing **half the format errors**.

### Table 2: Ablation on Formatted Decoding

| Configuration | Avg. F1 | Avg. Format Errors |
|--------------|---------|-------------------|
| CE | 63.03 | 625 |
| CE + FD | 69.87 | 45 |
| CE + FL | 63.51 | 582 |
| CE + FL + FD (=SLGM) | **70.86** | **29** |
| CE+task | 52.73 | 1405 |
| CE+task + FD | 59.32 | 569 |

Formatted decoding improves the CE baseline by 6 F1 points and reduces format errors by 94%.

## Key Findings

1. **No dataset name required**: SLGM replaces implicit dataset-name prompts with explicit format information, achieving comparable performance without dataset-specific engineering.
2. **Synergy of format loss and format decoding**: Each component yields individual gains, but their combination produces the best results.
3. **Smaller models benefit more**: On Flan-T5-small (77M), SLGM even surpasses CE+data, suggesting that capacity-constrained models benefit more from the strong supervision provided by format loss.
4. **Zero-parameter adapter**: SLGM can be viewed as a zero-parameter adapter that approximates the effect of dataset-specific fine-tuning in low-resource scenarios.
5. **Gains persist after fine-tuning**: After fine-tuning, SLGM reduces format errors to 1, achieving 94.8 F1 on CoNLL-03 NER and 98.3 F1 on ATIS intent detection.
6. **Format error analysis**: CE+task predominantly produces label-set mismatch errors, while CE+data produces source mismatch errors, indicating that different types of missing information lead to distinct structural error patterns.

## Highlights & Insights

- Reframes structured prediction as an "output space alignment" problem—a novel and principled perspective.
- The framework is **model-agnostic and task-agnostic**, transferable to any generative language model.
- The format mask tensor + FSM decoding design is elegant and incurs zero additional parameter overhead.
- Systematic validation across 5 task categories and 13 datasets ensures comprehensive experimental coverage.
- Low-resource ablations (1%–20% of data) demonstrate the framework's robustness under data-constrained settings.

## Limitations & Future Work

- A gap of approximately 12 F1 points remains relative to SOTA methods such as DeepStruct; SLGM functions more as a general-purpose enhancement framework than a performance ceiling.
- When the true label is absent from the predefined label set (e.g., unseen entity types), formatted decoding may produce worse results than unconstrained decoding.
- Format information only takes effect at the decoding stage; the model's attention layers cannot leverage format information during reasoning.
- The framework has not been validated in practical application scenarios such as RAG systems or regex-guided generation.
- Evaluation is limited to the Flan-T5 family; generalization to larger-scale or architecturally distinct LLMs remains unverified.

## Related Work & Insights

- **Structured pre-training**: DeepStruct, TANL — convey structural information implicitly via task/dataset names.
- **Constrained decoding**: Outlines (Willard et al.) — FSM-based regex-guided generation; SGLang — structured generation language.
- **Information extraction**: UIE — unified IE framework; OIA/OIX — lossless representations of predicate-argument structures.
- **Structure-augmented retrieval**: StructRAG — structuring scattered knowledge prior to retrieval; KELM/TEKGEN — knowledge graph corpora.

## Rating

⭐⭐⭐ — The technical contributions are solid (the combined design of format loss and formatted decoding is thought-provoking), and the experiments are systematic and comprehensive. However, a notable gap relative to SOTA persists, and validation on large-scale models is absent, leaving practical applicability to be further demonstrated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Towards a Rigorous Understanding of the Population Dynamics of the NSGA-III: Tight Runtime Bounds](towards_a_rigorous_understanding_of_the_population_dynamics_of_the_nsga-iii_tigh.md)
- [\[ICLR 2026\] Soft Quality-Diversity Optimization](../../ICLR2026/llm_evaluation/soft_quality-diversity_optimization.md)
- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](../../ICLR2026/llm_evaluation/towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](../../ACL2026/llm_evaluation/minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)
- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](../../ACL2026/llm_evaluation/attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)

</div>

<!-- RELATED:END -->
