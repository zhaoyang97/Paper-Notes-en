---
title: >-
  [Paper Note] The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation
description: >-
  [NeurIPS 2025][LLM Pretraining][In-Context Learning] This paper systematically dissects the internal mechanisms of LLMs in in-context retrieval augmented QA using the AttnLRP attribution method. Three functionally specia…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "In-Context Learning"
  - "Attention Head Analysis"
  - "Retrieval Augmentation"
  - "AttnLRP"
  - "Knowledge Attribution"
date: 2026-05-08
content_hash: 3f9c0c2cd0dabe25
---

# The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation

**Conference**: NeurIPS 2025
**arXiv**: [2505.15807](https://arxiv.org/abs/2505.15807)  
**Code**: [https://github.com/pkhdipraja/in-context-atlas](https://github.com/pkhdipraja/in-context-atlas)  
**Area**: LLM Pretraining
**Keywords**: In-Context Learning, Attention Head Analysis, Retrieval Augmentation, AttnLRP, Knowledge Attribution

## TL;DR
This paper systematically dissects the internal mechanisms of LLMs in in-context retrieval augmented QA using the AttnLRP attribution method. Three functionally specialized attention head types are identified — Task heads (middle layers, parsing instructions/questions), Retrieval heads (later layers, verbatim copying of contextual answers), and Parametric heads (encoding parametric knowledge) — and their functions are validated via Function Vector injection and source-tracking probes, achieving ROC AUC ≥94% on Llama-3.1/Mistral/Gemma.

## Background & Motivation
**Background**: LLMs answer questions by retrieving knowledge directly from in-context prompts via in-context learning. Prior work has identified patterns such as induction heads, but systematic analysis is lacking — particularly regarding how different attention heads divide labor in retrieval augmentation scenarios.

**Limitations of Prior Work**: (a) It remains unclear when LLMs use contextually provided information versus parametric memory; (b) existing analysis methods (e.g., inspecting attention weights alone) fail to capture the causal contribution of attention heads to outputs; (c) a consistent functional taxonomy across multiple models is absent.

**Key Challenge**: LLMs simultaneously rely on two knowledge sources (context vs. parameters) during in-context QA, yet how these compete and cooperate at the attention level remains opaque.

**Goal**: Construct an "attention head atlas" for LLM in-context retrieval — identifying which heads are responsible for which functions, and leveraging these findings to control and diagnose model behavior.

**Key Insight**: AttnLRP (attention-aware Layer-wise Relevance Propagation) is used for causal attribution, providing a more accurate measure of each head's contribution to outputs than inspecting attention weights alone.

**Core Idea**: The three head types serve distinct roles — Task heads parse *what is being asked*, Retrieval heads execute *what to copy and from where*, and Parametric heads supply *what has been memorized* — all verifiable via Function Vector injection and probing.

## Method

### Overall Architecture
(1) Apply AttnLRP to compute the positive contribution of each attention head to the output; (2) design contrastive experiments (open-book vs. closed-book, oracle vs. counterfactual context) to disentangle in-context and parametric heads; (3) further subdivide in-context heads into Task heads and Retrieval heads; (4) validate functions via Function Vector injection, head ablation, and linear probes.

### Key Designs

1. **AttnLRP Attribution Method**:

    - Function: Computes the causal contribution of each attention head to the final output token.
    - Mechanism: $\mathcal{R}^+(x|y) = \max(\mathcal{R}(x|y), 0)$ — positive relevance indicates amplification; negative relevance indicates suppression.
    - Head output: $z_i^h = \sum_{j=1}^{S} A_{i,j}^h(W_V^h x_j)$
    - Novelty: More accurate than raw attention weights — high attention weight does not imply large contribution to output.

2. **Attention Head Identification and Classification**:

    - Function: Partitions heads into in-context heads and parametric heads.
    - Method: Computes the contribution differential for each head under open-book (with context) vs. closed-book (without context) conditions: $\mathcal{D} = \mathbb{E}_{X_{OB}}[\mathcal{R}^h(y_{cf})] - \mathbb{E}_{X_{CB}}[\mathcal{R}^h(y_{gold})]$
    - The top-100 heads by $\mathcal{D}$ are designated in-context heads; the bottom heads are designated parametric heads (comprising 10–15% of all heads).
    - In-context heads are further subdivided: Task heads (high contribution to question tokens) vs. Retrieval heads (high contribution to answer tokens).

3. **Function Vector (FV) Validation**:

    - Function: Extracts output vectors from a specific head type and injects them into scenarios lacking that function, verifying whether the corresponding behavior is induced.
    - Core Finding: Injecting Task head FV → recall increases from 18% to 94.75% (+76.75%); injecting Retrieval head FV → 15.94% to 93.45% (+77.51%).
    - Design Motivation: If injecting a head group's FV induces the corresponding function, those heads are confirmed as core carriers of that function.

4. **Source-Tracking Probes**:

    - Function: Linear probes trained on Retrieval head activations to classify whether an answer originates from context or parametric memory.
    - ROC AUC: Llama 95%, Mistral 98%, Gemma 94%.
    - Localization Accuracy: Using aggregated attention weights + logit lens, Top-1 answer position accuracy is Llama 97%, Mistral 96%, Gemma 84%.

### Layer-wise Distribution
Experiments reveal a consistent layer-wise pattern: Parametric heads are distributed across all layers → Task heads concentrate in middle layers → Retrieval heads concentrate in later layers.

## Key Experimental Results

### Main Results
Effect of head ablation on QA performance (Llama-3.1-8B):

| Configuration | Open-Book Oracle↓ | Counterfactual↓ |
|---|---|---|
| Baseline | ~95% | ~60% |
| Remove 20 in-context heads | −13.86% | slight increase |
| Remove 100 in-context heads | −44.26% | −51% |
| Remove 100 parametric heads | −68.66% | slight increase |

Function Vector zero-shot injection (Biography dataset):

| Injection Type | Llama (Random→+FV) | Mistral | Gemma |
|---|---|---|---|
| Task heads | 18%→94.75% | 9.5%→88.5% | 7.5%→88.0% |
| Retrieval heads | 15.94%→93.45% | 8.56%→97.03% | 3.89%→87.36% |
| Parametric heads | 6.68%→38.84% | 12.95%→44.04% | 6.79%→34.77% |

### Ablation Study

| Validation | Result |
|---|---|
| Cross-dataset transfer (NQ-Swap→TQA) | Head sets transfer with minimal performance drop |
| Removing in-context heads in closed-book setting | Still degrades by 10–25% (heads also process input) |
| Removing parametric heads in open-book setting | Forces the model to rely more heavily on context |

### Key Findings
- **Three head types are functionally distinct and separable**: FV injection for Task/Retrieval heads yields performance jumps of ~78% and ~83%, respectively.
- **Parametric head FV efficacy is weaker** (~30%) — because parametric knowledge is distributed throughout the model rather than concentrated like in-context operations.
- **Source tracking achieves high accuracy**: Linear probes on Retrieval heads reach 94–98% AUC, making them directly applicable to hallucination detection.
- **Consistent layer-wise distribution**: All three architectures (Llama/Mistral/Gemma) exhibit the same layered pattern.
- **Preliminary finding**: 15 heads can induce zero-shot cross-lingual translation, suggesting similar functional specialization may exist for other tasks.

## Highlights & Insights
- **Functional atlas of ICL**: The three-tier taxonomy (Parametric→Task→Retrieval) provides the first clear picture of ICL's internal mechanisms, analogous to an "attention head map."
- **Practical value of source tracking**: The 94%+ AUC source classifier can be directly applied to hallucination detection in RAG systems — determining whether an answer genuinely originates from the context.
- **Controllability via Function Vectors**: Injecting or removing FVs enables precise behavioral control, offering new tools for model editing and safety.
- **AttnLRP vs. Attention Weights**: Relying solely on attention weights leads to misattribution — AttnLRP's causal attribution is strictly more informative.

## Limitations & Future Work
- The computational cost of AttnLRP is high, limiting applicability to large models and large datasets.
- The three-way head taxonomy is derived from QA tasks; its generalizability to generation, reasoning, and other tasks requires further validation.
- The threshold of 100 heads is heuristic and may require tuning across different models.
- The source-tracking probe is a post-hoc analysis tool and has not yet been integrated into online inference-time detection.

## Related Work & Insights
- **vs. Induction Heads**: Induction heads are an early-discovered pattern-matching mechanism; the Retrieval heads identified in this work represent a higher-level functional specialization thereof.
- **vs. Attention Visualization**: Traditional attention visualization examines weight distributions, whereas AttnLRP examines causal contributions — the latter is more revealing of true functionality.
- **Relation to RAG Systems**: The source-tracking and Retrieval head analyses presented here are directly applicable to reliability evaluation in RAG systems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematic discovery and validation of three head types
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 models × multiple validation methods × cross-dataset evaluation
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, elegant illustrations
- Value: ⭐⭐⭐⭐⭐ Provides foundational insights for ICL mechanistic understanding and model controllability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models](ricl_temporal_credit.md)
- [\[ACL 2026\] Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective](../../ACL2026/llm_pretraining/fine-tuning_vs_in-context_learning_in_large_language_models_a_formal_language_le.md)
- [\[NeurIPS 2025\] How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?](how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)
- [\[NeurIPS 2025\] Learning in Compact Spaces with Approximately Normalized Transformer](learning_in_compact_spaces_with_approximately_normalized_transformer.md)
- [\[NeurIPS 2025\] Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding](learning_to_flow_from_generative_pretext_tasks_for_neural_architecture_encoding.md)

</div>

<!-- RELATED:END -->
