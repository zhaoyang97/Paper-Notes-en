---
title: >-
  [Paper Note] FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing
description: >-
  [ACL 2026][Knowledge Editing][Model Editing] This paper finds that existing unstructured model editing methods can recall edited text holistically but fail in fine-grained fact access. It proposes the FABLE framework…
tags:
  - "ACL 2026"
  - "Knowledge Editing"
  - "Model Editing"
  - "Unstructured Knowledge"
  - "Fine-grained Fact Injection"
  - "Hierarchical Key-Value Storage"
  - "UnFine Benchmark"
date: 2026-05-08
content_hash: 5da4bf89b98f9edb
---

# FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing

**Conference**: ACL 2026  
**arXiv**: [2604.12559](https://arxiv.org/abs/2604.12559)  
**Code**: [https://github.com/caskcsg/FABLE](https://github.com/caskcsg/FABLE)  
**Area**: Knowledge Editing / LLM  
**Keywords**: Model Editing, Unstructured Knowledge, Fine-grained Fact Injection, Hierarchical Key-Value Storage, UnFine Benchmark

## TL;DR
This paper finds that existing unstructured model editing methods can recall edited text holistically but fail in fine-grained fact access. It proposes the FABLE framework, which uses a two-stage hierarchical strategy to anchor fine-grained facts to shallow layers and integrate holistic narratives into deep layers, while introducing the UnFine diagnostic benchmark for systematic evaluation.

## Background & Motivation

**Background**: Model editing aims to update specific knowledge in LLMs by modifying a minimal set of parameters. Structured editing (e.g., ROME, MEMIT) has succeeded with <subject, relation, object> triplets. Recent works like UnKE and AnyEdit extend editing to unstructured text, allowing models to memorize and holistically recall complete paragraphs.

**Limitations of Prior Work**: Existing unstructured editing methods, while capable of holistic recall, do not support fine-grained fact access. As shown in Figure 1, an UnKE-edited model can recite the full text but fails to provide accurate details when asked specific questions about the content. The model learns a high-level mapping from questions to surface form representations rather than encoding underlying atomic facts into the knowledge store.

**Key Challenge**: There is a mismatch between holistic recall and fine-grained fact access. In the unidirectional information flow of Transformers, surface form generation amplifies rather than corrects underlying fact representations—if facts are not correctly encoded in shallow layers, narrative generation in deep layers cannot fix the error.

**Goal**: Design a model editing method that supports both holistic text recall and fine-grained fact access.

**Key Insight**: Leveraging the "early decoding" phenomenon in Transformers—shallow layers capture local fine-grained features, while deep layers integrate them into global semantic representations. Therefore, fine-grained facts should be anchored in shallow layers first, followed by surface form integration in deep layers.

**Core Idea**: Decouple the key generator into two levels: a fine-grained fact key generator (shallow layers, injecting discrete facts) and a holistic semantic key generator (deep layers, integrating facts into coherent narratives), achieving a "facts first, generation later" mechanism.

## Method

### Overall Architecture
FABLE decomposes the key generator of an $N$-layer Transformer into a two-level hierarchy: (1) a fine-grained key generator $\mathcal{F}_{\text{fine}}$ (Layers 1 to $L_f$) and a holistic key generator $\mathcal{F}_{\text{hol}}$ (Layers $L_f+1$ to $L_h$), followed by a value generator $\mathcal{V}$ (Layers $L_h+1$ to $N$). Editing is performed in two stages: injecting fine-grained facts into shallow layers first, then applying minimal adjustments to deep layers to ensure narrative coherence.

### Key Designs

1. **Fine-grained Fact Anchoring (Stage 1)**:

    - **Function**: Injects discrete facts extracted from unstructured text into the model's shallow parameters.
    - **Mechanism**: For each fine-grained QA pair $(q_f, a_f^*)$, the method finds a key $k_{\text{fine}}^* = k_{\text{fine}} + \delta_f$ that triggers the target fact by optimizing the residual vector $\delta_f$. The parameter update is then distributed across multiple layers (e.g., Layers 4, 5, 6), with each layer sharing a portion of the shift. The optimization objective considers edit efficacy (last token shift), prefix consistency (first $n-1$ tokens unchanged), and locality preservation (unrelated samples unchanged).
    - **Design Motivation**: Shallow layers are proficient at capturing local fine-grained features. Anchoring facts here ensures they serve as the foundation for the information flow in all subsequent layers, rather than relying on holistic memory in deep layers. Distributed updates prevent excessive shifts in a single layer.

2. **Holistic Surface Form Integration (Stage 2)**:

    - **Function**: Adjusts deep parameters to enable the model to generate fluent, coherent unstructured narratives while protecting the already injected fine-grained facts.
    - **Mechanism**: Similar to Stage 1, but updates only a single deep layer $L_h=7$ using holistic QA pairs $(q_h, a_h^*)$. A critical addition is the "fine-grained preservation constraint," which ensures that updating $\mathcal{F}_{\text{hol}}$ does not override the signals injected during Stage 1. The optimization adds a fine-grained preservation term to the existing efficacy, consistency, and locality objectives.
    - **Design Motivation**: Stage 1 ensures facts are correctly encoded, while Stage 2 builds narrative capability on top. The preservation constraint resolves potential signal conflicts between the two stages.

3. **UnFine Diagnostic Benchmark**:

    - **Function**: Systematically evaluates the model's ability to recall fine-grained facts post-editing.
    - **Mechanism**: Based on three existing unstructured editing datasets (UnKEBench, AKEW-CF, AKEW-MQ), it adds fine-grained QA pairs and key knowledge phrase extractions. It introduces two fact-level metrics—Hit Rate (exact phrase matching) and $C_{\text{LCS}}$ (longest common subsequence coverage)—to evaluate whether the model truly grasps specific facts within the edited content.
    - **Design Motivation**: Existing metrics check only holistic outputs (ROUGE-L, BERT-Score) and cannot distinguish between "truly understanding facts" and "memorizing surface forms." UnFine fills this evaluative gap.

### Loss & Training
The framework utilizes two-stage closed-form optimization. Stage 1 updates Layers 4, 5, and 6 using 5 times the number of fine-grained QA pairs compared to seed QAs. Stage 2 updates Layer 7 using one holistic QA. Each edited sample uses 20 unrelated samples randomly sampled from the Alpaca dataset for locality preservation.

## Key Experimental Results

### Main Results

| Method | Holistic (BERT-Score) | Holistic (Rouge-L) | Fine-grained (HR) | Fine-grained ($C_{\text{LCS}}$) |
|------|-------------------|----------------|-----------|------------------------|
| UnKE | High | High | Low | Low |
| AnyEdit | High | High | Low | Low |
| FABLE | **High** | **High** | **Significant Gain** | **Significant Gain** |

### Ablation Study

| Configuration | Holistic | Fine-grained | Description |
|------|--------|--------|------|
| Full FABLE | High | High | Complete two-stage process |
| Only Stage 2 | High | Low | Lacks fine-grained anchoring |
| Only Stage 1 | Low | High | Lacks narrative integration |
| w/o Fine-grained Preservation | High | Medium | Stage 2 overrides some fact signals |

### Key Findings
- FABLE maintains SOTA holistic editing performance while substantially improving fine-grained fact access.
- Existing methods exhibit high holistic scores but low fine-grained scores, validating the hypothesis that "memorizing surface forms $\neq$ understanding facts."
- Injecting facts into shallow layers (Layers 4-6) is superior to deep layers, confirming the utility of the "early decoding" phenomenon.
- The fine-grained preservation constraint is vital for two-stage synergy—without it, Stage 2 tends to override Stage 1 signals.

## Highlights & Insights
- **Distinction between Holistic Recall vs. Fine-grained Access**: Identifies a fundamental, overlooked issue in unstructured model editing—the ability to recite text does not equate to understanding the facts within it. This insight applies to broader fields like RAG and knowledge enhancement.
- **Theoretical Basis for Hierarchical Editing**: Utilizes Transformer information flow and early decoding to provide theoretical support for the "shallow facts + deep narrative" design.
- **Contribution of the UnFine Benchmark**: The proposed HR and $C_{\text{LCS}}$ metrics directly assess fact-level editing efficacy, offering greater precision than ROUGE or BERT-Score.

## Limitations & Future Work
- Currently requires manual or LLM-based extraction of fine-grained QA pairs, increasing the complexity of the editing pipeline.
- Layer selection (Layers 4-6 for facts, Layer 7 for narrative) may vary across different model architectures.
- The cumulative effects of sequential multiple edits have not been fully explored.
- Validated only on a single model architecture; cross-architecture generalizability remains unknown.

## Related Work & Insights
- **vs. ROME/MEMIT**: While they focus on structured triplet editing, FABLE extends to fine-grained editing of unstructured text.
- **vs. UnKE**: UnKE achieves holistic unstructured editing but lacks fine-grained access. FABLE addresses this through hierarchical decoupling.
- **vs. AnyEdit**: AnyEdit broadens the scope of editing but similarly suffers from unreliable fine-grained fact retrieval.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Precise identification of core limitations in unstructured editing; elegant hierarchical decoupling design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation across three datasets, multiple baselines, and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Precise problem definition, thorough theoretical analysis, and clear methodological description.
- **Value**: ⭐⭐⭐⭐ Significant advancement in the field of model editing; the UnFine benchmark will drive more accurate evaluations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Fine-tuning Done Right in Model Editing](../../ICLR2026/knowledge_editing/fine-tuning_done_right_in_model_editing.md)
- [\[ACL 2026\] HiEdit: Lifelong Model Editing with Hierarchical Reinforcement Learning](hiedit_lifelong_model_editing_with_hierarchical_reinforcement_learning.md)
- [\[ACL 2026\] CLaRE-ty Amid Chaos: Quantifying Representational Entanglement to Predict Ripple Effects in LLM Editing](clare-ty_amid_chaos_quantifying_representational_entanglement_to_predict_ripple_.md)
- [\[ACL 2026\] The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models](the_model_agreed_but_didn39t_learn_diagnosing_surface_compliance_in_large_langua.md)
- [\[ICLR 2026\] Energy-Regularized Sequential Model Editing on Hyperspheres](../../ICLR2026/knowledge_editing/energy-regularized_sequential_model_editing_on_hyperspheres.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] Fine-tuning Done Right in Model Editing](../../ICLR2026/knowledge_editing/fine-tuning_done_right_in_model_editing.md)
- [\[ACL 2026\] HiEdit: Lifelong Model Editing with Hierarchical Reinforcement Learning](hiedit_lifelong_model_editing_with_hierarchical_reinforcement_learning.md)
- [\[ACL 2026\] The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models](the_model_agreed_but_didn39t_learn_diagnosing_surface_compliance_in_large_langua.md)
- [\[ACL 2026\] CLaRE-ty Amid Chaos: Quantifying Representational Entanglement to Predict Ripple Effects in LLM Editing](clare-ty_amid_chaos_quantifying_representational_entanglement_to_predict_ripple_.md)
- [\[AAAI 2026\] Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior](../../AAAI2026/knowledge_editing/model_editing_as_a_double-edged_sword_steering_agent_ethical_behavior_toward_ben.md)

</div>

<!-- RELATED:END -->
