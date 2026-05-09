---
title: >-
  [Paper Note] FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing
description: >-
  [ACL 2026][Knowledge Editing][Model Editing] This paper discovers that existing unstructured model editing methods can holistically recall edited text but cannot perform fine-grained fact access, and proposes FABLE, a framework that uses a two-stage hierarchical strategy to anchor fine-grained facts in shallow layers and integrate holistic narratives in deep layers, along with the UnFine diagnostic benchmark for systematic evaluation.
tags:
  - ACL 2026
  - Knowledge Editing
  - Model Editing
  - Unstructured Knowledge
  - Fine-grained Fact Injection
  - Hierarchical Key-Value Storage
  - UnFine Benchmark
date: 2025-04-17
content_hash: 8b51a5712d400ba8
---

# FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing

**Conference**: ACL 2026  
**arXiv**: [2604.12559](https://arxiv.org/abs/2604.12559)  
**Code**: [https://github.com/caskcsg/FABLE](https://github.com/caskcsg/FABLE)  
**Area**: Knowledge Editing / LLM  
**Keywords**: Model Editing, Unstructured Knowledge, Fine-grained Fact Injection, Hierarchical Key-Value Storage, UnFine Benchmark

## TL;DR
This paper discovers that existing unstructured model editing methods can holistically recall edited text but cannot perform fine-grained fact access, and proposes FABLE, a framework that uses a two-stage hierarchical strategy to anchor fine-grained facts in shallow layers and integrate holistic narratives in deep layers, along with the UnFine diagnostic benchmark for systematic evaluation.

## Background & Motivation

**Background**: Model editing aims to update specific knowledge in LLMs by modifying a small number of parameters. Structured editing (e.g., ROME, MEMIT) has succeeded on <subject, relation, object> triplets. Recent methods like UnKE and AnyEdit have extended editing to unstructured text — enabling models to memorize and holistically recall complete paragraphs.

**Limitations of Prior Work**: Existing unstructured editing methods can holistically recall edited text but cannot support fine-grained fact access. As illustrated, models edited with UnKE can recite the complete text, but when asked about specific details within the text, they fail to provide accurate answers. The model learns a high-level mapping from questions to surface-form representations rather than encoding the underlying atomic facts into knowledge stores.

**Key Challenge**: A mismatch exists between holistic recall and fine-grained fact access. In the Transformer's unidirectional information flow, surface-form generation amplifies rather than corrects underlying fact representations — if shallow layers have not correctly encoded facts, deep-layer narrative generation cannot compensate.

**Goal**: Design a model editing method that simultaneously supports holistic text recall and fine-grained fact access.

**Key Insight**: Leverage the Transformer's "early decoding" phenomenon — shallow layers excel at capturing local fine-grained features while deep layers integrate into global semantic representations. Therefore, fine-grained facts should first be anchored in shallow layers, followed by surface-form integration in deep layers.

**Core Idea**: Decouple the key generator into two levels — a fine-grained fact key generator (shallow layers, injecting discrete facts) and a holistic semantic key generator (deep layers, integrating into coherent narratives) — achieving "facts first, generation second."

## Method

### Overall Architecture
FABLE decomposes the N-layer Transformer's key generator into two hierarchical levels: (1) fine-grained key generator $\mathcal{F}_{\text{fine}}$ (layers 1 to $L_f$) and holistic key generator $\mathcal{F}_{\text{hol}}$ (layers $L_f+1$ to $L_h$), plus value generator $\mathcal{V}$ (layers $L_h+1$ to N). Editing proceeds in two stages: first injecting fine-grained facts into shallow layers, then making minimal adjustments to deep layers to ensure narrative coherence.

### Key Designs

1. **Fine-grained Fact Anchoring (Stage 1)**:

    - Function: Inject discrete facts extracted from unstructured text into the model's shallow layer parameters
    - Mechanism: For each fine-grained QA pair $(q_f, a_f^*)$, first optimize residual vector $\delta_f$ to find a key that triggers the target fact $k_{\text{fine}}^* = k_{\text{fine}} + \delta_f$, then distribute the parameter update across multiple layers (layers 4, 5, 6), with each layer bearing a portion of the offset. The optimization objective simultaneously considers editing efficacy (last token offset), prefix consistency (first n-1 tokens unchanged), and locality preservation (unrelated samples unchanged)
    - Design Motivation: Transformer shallow layers excel at capturing local fine-grained features. Anchoring facts in shallow layers ensures they become the foundation for information flow in all subsequent layers, rather than relying on deep-layer holistic memory. Distributed updates prevent any single layer from bearing excessive offset

2. **Holistic Surface-Form Integration (Stage 2)**:

    - Function: Adjust deep-layer parameters to enable fluent, coherent unstructured narrative generation while protecting already-injected fine-grained facts
    - Mechanism: Similar to Stage 1, but only updates single layer $L_h=7$, using holistic QA pairs $(q_h, a_h^*)$. The key difference is the addition of a "fine-grained preservation constraint" — ensuring that updating $\mathcal{F}_{\text{hol}}$ does not overwrite fine-grained fact signals injected in Stage 1. The optimization objective adds a fine-grained preservation term beyond editing efficacy, prefix consistency, and locality preservation
    - Design Motivation: The first stage ensures facts are correctly encoded; the second stage adds narrative capability on top. The fine-grained preservation constraint resolves signal conflicts between the two stages

3. **UnFine Diagnostic Benchmark**:

    - Function: Systematically evaluate fine-grained fact recall capabilities of model editing
    - Mechanism: Based on three existing unstructured editing datasets (UnKEBench, AKEW-CF, AKEW-MQ), adding fine-grained QA pairs and key knowledge phrase extraction. Two fact-level metrics are designed — Hit Rate (exact phrase matching) and $C_{\text{LCS}}$ (longest common subsequence coverage) — evaluating whether the model truly masters specific facts within edited content
    - Design Motivation: Existing evaluation only checks holistic output (ROUGE-L, BERT-Score), unable to distinguish "truly understanding facts" from "memorizing surface forms." UnFine fills this evaluation gap

### Loss & Training
Two-stage closed-form optimization. Stage 1 updates layers 4, 5, 6, using 5× the number of seed QA pairs as fine-grained QA. Stage 2 updates layer 7, using 1 holistic QA. Each edit sample uses 20 randomly sampled unrelated samples from Alpaca dataset for locality preservation.

## Key Experimental Results

### Main Results

| Method | Holistic (BERT-Score) | Holistic (Rouge-L) | Fine-grained (HR) | Fine-grained ($C_{\text{LCS}}$) |
|------|-------------------|----------------|-----------|------------------------|
| UnKE | High | High | Low | Low |
| AnyEdit | High | High | Low | Low |
| FABLE | **High** | **High** | **Significantly Improved** | **Significantly Improved** |

### Ablation Study

| Config | Holistic | Fine-grained | Note |
|------|--------|--------|------|
| Full FABLE | High | High | Both stages complete |
| Stage 2 only | High | Low | Missing fine-grained anchoring |
| Stage 1 only | Low | High | Missing narrative integration |
| w/o fine-grained preservation constraint | High | Medium | Stage 2 overwrites some fact signals |

### Key Findings
- FABLE substantially improves fine-grained fact access capability while maintaining SOTA holistic editing performance
- Existing methods show high holistic recall scores but low fine-grained scores, validating the hypothesis that "memorizing surface forms ≠ understanding facts"
- Injecting facts into shallow layers (4-6) outperforms deep layers, validating the practical value of the "early decoding" phenomenon
- The fine-grained preservation constraint is critical for two-stage synergy — without it, Stage 2 overwrites Stage 1 signals

## Highlights & Insights
- **Holistic recall vs fine-grained access distinction**: Identifying a neglected fundamental problem in unstructured model editing — being able to recite text ≠ understanding the facts within it. This insight extends to broader areas such as RAG and knowledge augmentation
- **Theoretical basis for hierarchical editing**: Using Transformer information flow direction and the early decoding phenomenon, providing theoretical support for the "shallow-layer facts + deep-layer narratives" design
- **UnFine benchmark contribution**: The proposed HR and $C_{\text{LCS}}$ metrics directly evaluate fact-level editing effectiveness, more precise than ROUGE/BERT-Score

## Limitations & Future Work
- Currently requires manual or LLM-based extraction of fine-grained QA pairs, adding complexity to the editing pipeline
- Layer selection (layers 4-6 for facts, layer 7 for narratives) may vary across model architectures
- Cumulative effects of multiple edits are not sufficiently discussed
- Validated only on a single model architecture; cross-architecture applicability is unknown

## Related Work & Insights
- **vs ROME/MEMIT**: Focus on structured triplet editing; FABLE extends to fine-grained editing of unstructured text
- **vs UnKE**: UnKE achieves holistic unstructured editing but lacks fine-grained fact access. FABLE solves this through hierarchical decoupling
- **vs AnyEdit**: AnyEdit extends editing applicability but similarly suffers from unreliable fine-grained facts

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Precisely identifies the core limitation of unstructured editing; hierarchical decoupling design is elegant
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, multiple baselines, detailed ablation
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is precise, theoretical analysis is thorough, method description is clear
- Value: ⭐⭐⭐⭐ Significant advancement for the model editing field; UnFine benchmark will drive more precise evaluation

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Fine-tuning Done Right in Model Editing](../../ICLR2026/knowledge_editing/fine-tuning_done_right_in_model_editing.md)
- [\[ACL 2026\] CLaRE-ty Amid Chaos: Quantifying Representational Entanglement to Predict Ripple Effects in LLM Editing](clare-ty_amid_chaos_quantifying_representational_entanglement_to_predict_ripple_.md)
- [\[ICLR 2026\] Energy-Regularized Sequential Model Editing on Hyperspheres](../../ICLR2026/knowledge_editing/energy-regularized_sequential_model_editing_on_hyperspheres.md)
- [\[ICLR 2026\] Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing](../../ICLR2026/knowledge_editing/bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed.md)
- [\[AAAI 2026\] Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior](../../AAAI2026/knowledge_editing/model_editing_as_a_double-edged_sword_steering_agent_ethical_behavior_toward_ben.md)

<!-- RELATED:END -->
