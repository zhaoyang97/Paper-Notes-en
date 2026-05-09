---
title: >-
  [Paper Note] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation
description: >-
  [ACL 2026][Factual Hallucination] This paper systematically analyzes factual hallucinations induced by new-knowledge learning during SFT using a controlled synthetic dataset, Biography-Reasoning. It identifies the root mechanism as the attenuation of attention to key entities, and proposes KnownPatch—injecting a small amount of known-knowledge samples at the end of training to restore attention patterns—effectively mitigating hallucinations.
tags:
  - ACL 2026
  - Factual Hallucination
  - New Knowledge Learning
  - Attention Mechanism
  - SFT
  - KnownPatch
date: 2026-05-08
content_hash: 77147dacf688dcc0
---

# Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation

**Conference**: ACL 2026
**arXiv**: [2511.02626](https://arxiv.org/abs/2511.02626)
**Code**: None
**Area**: Interpretability
**Keywords**: Factual Hallucination, New Knowledge Learning, Attention Mechanism, SFT, KnownPatch

## TL;DR

This paper systematically analyzes factual hallucinations induced by new-knowledge learning during SFT using a controlled synthetic dataset, Biography-Reasoning. It identifies the root mechanism as the attenuation of attention to key entities, and proposes KnownPatch—injecting a small amount of known-knowledge samples at the end of training to restore attention patterns—effectively mitigating hallucinations.

## Background & Motivation

**Background**: LLMs acquire rich world knowledge during pre-training and learn to follow instructions during SFT. Prior work has shown that introducing knowledge not covered during pre-training into SFT increases the risk of factual hallucinations—models erroneously generate newly learned information in irrelevant contexts.

**Limitations of Prior Work**: Prior work has focused primarily on closed-form QA settings with mixed knowledge types, leaving the specific manifestations and underlying mechanisms of hallucinations poorly understood. In particular: (1) the propagation patterns of hallucinations across different knowledge and task types are unclear; (2) the attention-level causes of hallucinations have not been identified; (3) lightweight mitigation methods are lacking.

**Key Challenge**: When a particular knowledge type is composed entirely of new knowledge, severe hallucinations arise even when the total volume of new knowledge is small. This diverges from the prior simplistic understanding that hallucination severity scales monotonically with the proportion of new knowledge—the critical factor is the degree of unfamiliarity within a specific knowledge type, not the global ratio of new knowledge.

**Goal**: (1) Construct a controlled dataset for fine-grained analysis of hallucination manifestations; (2) reveal the attention mechanism underlying hallucinations; (3) propose a lightweight mitigation method.

**Key Insight**: A synthetic biographical dataset is constructed to precisely control the proportion and type of known/unknown knowledge, and attention analysis is used to trace the generation and propagation mechanisms of hallucinations.

**Core Idea**: Learning new knowledge weakens the model's attention to key entities in the input, causing over-reliance on other tokens in context and thereby inducing hallucinations. Injecting known knowledge at the end of training restores attention patterns.

## Method

### Overall Architecture

A synthetic dataset, Biography-Reasoning (fictional entities × 4 attribute types × 4 QA tasks + 12 reasoning tasks), is constructed to analyze hallucinations by controlling the proportion of known/unknown knowledge. The analysis proceeds at three levels: (1) fine-grained characterization of hallucination phenomena; (2) interpretability analysis of attention mechanisms; (3) the KnownPatch mitigation method.

### Key Designs

1. **Controlled Synthetic Dataset: Biography-Reasoning**

    - **Function**: Precisely control the type and proportion of known/unknown knowledge to isolate causal factors of hallucination.
    - **Mechanism**: Four attributes are defined for fictional entities (birth year, death year, profession, university), each corresponding to a distinct knowledge type. Four QA tasks and twelve reasoning tasks (single-step, comparative, and novel reasoning) are constructed. Continued pre-training makes a subset of knowledge "known," while the remainder stays "unknown"; SFT then mixes these at varying proportions.
    - **Design Motivation**: Real datasets do not permit precise control over which knowledge is known to the model; the synthetic dataset eliminates this confound.

2. **Attention Analysis and KnownPatch**

    - **Function**: Reveal the hallucination mechanism and provide a lightweight mitigation.
    - **Mechanism**: Attention changes directed at key entities (name tokens) are analyzed in middle-to-late layers (layers 12–24). Key findings: learning new knowledge significantly reduces attention to key entities (the drop in attention values strongly correlates with hallucination severity), while learning known knowledge enhances attention to key entities. Based on this, KnownPatch is proposed: a small proportion of known-knowledge samples (5–20%) is injected at the end of training, leveraging the attention-boosting effect of known knowledge to repair the attention patterns disrupted by new knowledge.
    - **Design Motivation**: If hallucinations stem from disrupted attention patterns, restoring correct attention patterns should alleviate hallucinations without requiring the removal of all new knowledge from training data.

3. **Hallucination Propagation Mechanism Analysis**

    - **Function**: Reveal how hallucinations propagate from training tasks to test tasks.
    - **Mechanism**: Task variants that are lexically similar but semantically different, and semantically similar but lexically different, are constructed. Results show that hallucination propagation is driven primarily by lexical similarity (token overlap) rather than semantic similarity. As attention weights are normalized over all input tokens, a drop in attention to key entities causes excess attention to flow to surrounding context tokens; test samples sharing vocabulary with unknown-knowledge training samples are thus more susceptible.
    - **Design Motivation**: Understanding the propagation mechanism helps predict which tasks are most vulnerable to hallucinations, enabling targeted countermeasures.

### Loss & Training

Standard SFT uses cross-entropy loss. KnownPatch injects known-knowledge samples into the final stage of training (appended at the end rather than shuffled throughout), exploiting the effect of training order to repair attention. Ablations also test adding a KL-divergence constraint ($\alpha=25$) to directly preserve consistency in attention module outputs.

## Key Experimental Results

### Main Results

| Condition | STQA Accuracy Drop | Wiki Accuracy Drop | Notes |
|-----------|-------------------|-------------------|-------|
| All known (baseline) | 0% | 0% | No hallucination |
| One type fully unknown | >50% | Significant drop | Severe hallucination |
| KeepKnown 50% | Moderate drop | Moderate drop | Retaining known data mitigates hallucination |
| RemoveKnown 5% | Severe drop | Severe drop | Fully unknown type is extremely harmful |

### Ablation Study

| Configuration | STQA | Wiki | Notes |
|---------------|------|------|-------|
| KnownPatch 5% | Significant recovery | Significant recovery | Effective with only 5% injection |
| KnownPatch 20% | Near baseline | Slightly above baseline | Approaches upper bound |
| Shuffled 20% | Moderate recovery | Moderate recovery | Shuffling is less effective than end-of-training injection |
| KL constraint | Partial mitigation | Partial mitigation | Direct attention constraint also works but with side effects |

### Key Findings

- **Intra-type unfamiliarity matters more than global proportion**: Even when the total volume of new knowledge is small, if an entire knowledge type consists exclusively of unknown knowledge (RemoveKnown), hallucinations become extremely severe. KeepKnown with 50% replacement substantially outperforms RemoveKnown with only 5% replacement.
- **Cross-type hallucination propagation**: Learning new knowledge of one type not only causes hallucinations in same-type QA (STQA drops >50%), but also propagates to different-type QA (DTQA drops ~5%) and OOD Wiki test sets.
- **Reverse propagation from reasoning to QA**: When reasoning tasks containing unknown knowledge are trained on, hallucinations in QA test sets are more severe than in other reasoning test sets, because QA contexts share greater lexical overlap with reasoning traces.
- **Strong correlation between attention and hallucination**: Higher proportions of unknown knowledge correspond to lower attention to key entities and more severe hallucinations; the two curves track each other almost perfectly.
- **Non-replay nature of KnownPatch**: Even when injected known-knowledge samples do not cover all unknown knowledge types, hallucinations in uncovered types are still mitigated, indicating that KnownPatch operates by restoring attention patterns rather than by knowledge replay.

## Highlights & Insights

- **"Fully unknown within a type" is more dangerous than "high global ratio"**: This finding overturns the prior simplistic understanding that higher proportions of new knowledge are uniformly more hazardous, and has direct practical implications for SFT data construction—each knowledge type should retain some samples that the model already knows.
- **Lexical similarity drives hallucination propagation**: This explains why seemingly unrelated tasks are affected by hallucinations—as long as they share sufficient vocabulary tokens with training samples containing new knowledge.
- **Lightweight nature of KnownPatch**: Injecting only 5% known-knowledge samples at the end of training significantly mitigates hallucinations, without requiring costly known/unknown classification of the entire training corpus.

## Limitations & Future Work

- Experiments are conducted primarily on Qwen2.5-1.5B; consistency on Llama3.2-1B, Qwen3-8B, and Qwen2.5-32B is verified in the appendix.
- The use of a synthetic dataset means that the complexity and distribution of real-world knowledge may differ from the synthetic setting.
- KnownPatch requires access to known-knowledge samples; determining in practice whether a piece of knowledge is known to the model remains an open problem.
- Non-factual hallucinations (e.g., logical errors, format errors) are not examined.

## Related Work & Insights

- **vs. Gekhman et al. (2024)**: They find that hallucination severity increases with the proportion of new knowledge, but use a mixed-knowledge-type setting. This paper controls for knowledge type and reveals a more fine-grained pattern—intra-type unfamiliarity is the decisive factor.
- **vs. Sun et al. (2025)**: They analyze the over-generalization of new knowledge from a token probability perspective; this paper provides a complementary explanation from the attention mechanism perspective.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The controlled experimental design is elegant, and the "intra-type unfamiliarity" finding is genuinely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-dimensional ablations, multi-model validation, attention analysis, and propagation mechanism analysis are all highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The logical chain from phenomena to mechanism to mitigation is very clear.
- **Value**: ⭐⭐⭐⭐⭐ — Provides important practical guidance for understanding and mitigating hallucinations during SFT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](interpretable_traces_unexpected_outcomes_investigating_the_disconnect_in_trace-b.md)
- [\[NeurIPS 2025\] An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations](../../NeurIPS2025/interpretability/an_analysis_of_concept_bottleneck_models_measuring_understanding_and_mitigating_.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)
- [\[ACL 2026\] Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models](revitalizing_black-box_interpretability_actionable_interpretability_for_llms_via.md)

</div>

<!-- RELATED:END -->
