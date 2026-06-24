---
title: >-
  [Paper Note] Mitigating Negative Interference in Multilingual Sequential Knowledge Editing through Null-Space Constraints
description: >-
  [ACL 2025][Knowledge Editing][Multilingual Knowledge Editing] This paper proposes the LangEdit framework, which projects parameter updates of each language onto the null space of previously edited languages to achieve mathematical isolation between language updates in multilingual sequential knowledge editing, effectively mitigating negative interference and preserving multilingual generalization capability.
tags:
  - "ACL 2025"
  - "Knowledge Editing"
  - "Multilingual Knowledge Editing"
  - "Null-Space Projection"
  - "Negative Interference"
  - "Sequential Editing"
  - "LLMs"
date: 2026-05-08
content_hash: 29464eda5db536da
---

# Mitigating Negative Interference in Multilingual Sequential Knowledge Editing through Null-Space Constraints

**Conference**: ACL 2025  
**arXiv**: [2506.10800](https://arxiv.org/abs/2506.10800)  
**Code**: [Yes (GitHub)](https://github.com/VRCMF/LangEdit.git)  
**Area**: NLP / Knowledge Editing / Multilingual LLMs  
**Keywords**: Multilingual Knowledge Editing, Null-Space Projection, Negative Interference, Sequential Editing, LLMs

## TL;DR

This paper proposes the LangEdit framework, which projects parameter updates of each language onto the null space of previously edited languages to achieve mathematical isolation between language updates in multilingual sequential knowledge editing, effectively mitigating negative interference and preserving multilingual generalization capability.

## Background & Motivation

Large Language Models (LLMs) excel at encoding and retrieving factual knowledge. Knowledge editing is an efficient method to update knowledge without full retraining. However, in **multilingual scenarios**, efficient knowledge updates face significant challenges: performing knowledge editing in one language may damage the model's performance in other languages.

Maintaining separate edited models for each language is prohibitively expensive. A more realistic paradigm is to integrate knowledge updates of all languages into a unified model. However, experiments show that cross-lingual sequential editing leads to **destructive parameter interference**, also known as "**negative interference**"—where editing in one language degrades the accuracy of previously edited languages and the model's overall multilingual generalization capability.

Existing monolingual knowledge editing methods (such as ROME, MEMIT, AlphaEdit) underperform in this novel scenario. In particular, although AlphaEdit also utilizes null-space projection, it uses a **static null space** (constructed from original pre-trained knowledge), whereas the dynamic interactions among different languages require distinct projection spaces.

## Method

### Overall Architecture

LangEdit performs knowledge editing on the MLP layers of the LLM (specifically the `W_out` matrix). A multilingual data stream `{L_1, L_2, ..., L_T}` is sequentially injected across time steps, with each step corresponding to a knowledge update in a specific language. The core idea is: when editing knowledge in language $j$, the parameter updates are projected into the null space of all previously edited languages, ensuring that new edits do not destroy existing knowledge.

### Key Designs

1. **MLP as Knowledge Storage**: Factual knowledge (subject, relation, object) is encoded as key-value pairs in MLP layers. The output of the `W_in` layer corresponds to the key (subject + relation), and the output of the `W_out` layer corresponds to the value (object). The goal of knowledge editing is to optimize `W_out`.

2. **Null-Space Projection**:

    - Calculate the uncentered covariance matrix `K̄_{t-1}` for all previously edited key matrices.
    - Incrementally update the covariance matrix using a recurrence formula (avoiding storing the complete history).
    - Perform SVD decomposition on the covariance matrix, preserving the eigenvectors corresponding to zero eigenvalues to construct the projection matrix `P_{t-1}`.
    - Project the new parameter update `ΔW_t` into the null space represented by `P_{t-1}`.

3. **Optimization Objective (Closed-form Solution)**:

    - The objective function contains two terms: a regularization term (limiting the parameter perturbation magnitude) and an editing precision term (ensuring key-value matching after editing).
    - Closed-form solution: `ΔW_t = R_t · K_t^T · P_{t-1} · (K_t · K_t^T · P_{t-1} + I)^{-1}`
    - Where `R_t = V_t - W_{t-1} · K_t` is the residual.

4. **Core Difference with AlphaEdit**:

    - AlphaEdit: Static null space (based only on pre-trained knowledge `K_0`), without language-specific distinction.
    - LangEdit: Dynamic null space (different projection matrix `P_{t-1}` at each step `t`), enabling language-specific isolation.

### Loss & Training

- Edits 100 samples per step.
- `K_0` is computed from 100,000 triplets randomly sampled from Wikipedia.
- Selects key layers to edit for GPT-J-6B, Llama3-8B, and Qwen2.5-7B, respectively.
- Key layers are determined using the causal tracing technique.

## Key Experimental Results

### Main Results: Multilingual Sequential Knowledge Editing (mzsre dataset, 6 languages × 400 samples = 2400 edits)

| Model | Method | Efficacy↑ | Generality↑ | Specificity↑ | XTREME F1↑ |
|------|------|-----------|-------------|-------------|-----------|
| Llama3-8B | Pre-edited | 31.15 | 31.01 | 31.93 | 69.30 |
| Llama3-8B | MEMIT | 1.45 | 1.46 | 0.67 | 4.54 |
| Llama3-8B | AlphaEdit | 80.34 | 75.84 | 30.91 | 60.59 |
| Llama3-8B | **LangEdit** | **82.54** | **77.53** | **31.90** | **66.24** |
| Qwen2.5-7B | AlphaEdit | 93.50 | 87.18 | 42.58 | 73.01 |
| Qwen2.5-7B | **LangEdit** | **93.90** | **87.02** | **42.64** | **74.06** |
| GPT-J-6B | AlphaEdit | 83.59 | 78.34 | 26.55 | 36.74 |
| GPT-J-6B | **LangEdit** | **84.27** | **79.74** | **27.23** | **38.59** |

### Ablation Study: Quantifying Negative Interference (Llama3-8B)

| Method | Efficacy Gap vs. Monolingual | F1 Gap vs. Monolingual |
|------|---------------------|----------------|
| AlphaEdit (multi) | +0.10 ~ +3.27 | +3.20 ~ +15.19 |
| **LangEdit** | Partially outperforms monolingual | **+0.20 ~ +9.29** |

LangEdit outperforms monolingual AlphaEdit in Efficacy on four languages: English, German, Dutch, and French.

### Key Findings

1. **LangEdit Outperforms SOTA Globally**: Across three model architectures and two datasets, the average Efficacy improves by +2.20, and the average XTREME F1 improves by +5.65 (with the largest gains on Llama3-8B).

2. **ROME and FT Suffer Catastrophic Collapse under Sequential Multilingual Editing**: After scaling up edits, their XTREME F1 drops to single digits, demonstrating that existing methods are completely incapable of handling this scenario.

3. **Editing Performance Negatively Correlates with Pre-training Data Volume**: English (which has abundant pre-training data) shows minor improvement, whereas Spanish (with less pre-training data) achieves an improvement of up to +9.00 F1.

4. **Multilingual Editing Enhances Generalization Capability**: For GPT-J-6B and Qwen2.5-7B, the XTREME scores after LangEdit editing even outperform the unedited base models, indicating that injecting multilingual knowledge itself yields positive transfer effects.

5. **Null-Space Projection Effectively Isolates Cross-Lingual Interference**: Without null-space projection, multilingual editing performs significantly worse than monolingual editing; with it, the performance gap is dramatically reduced or even reversed.

## Highlights & Insights

- **New Task Definition**: First to formalize the task of "multilingual sequential knowledge editing," providing a new paradigm for the maintenance of multilingual LLMs.
- **Mathematically Guaranteed Language Isolation**: Null-space projection theoretically guarantees that parameter updates across different languages do not overwrite one another.
- **Incremental Covariance Update**: Avoids storing and recomputing the entire history of key matrices, leading to high computational efficiency.
- **Closed-Form Solution**: Eliminates the need for iterative optimization, achieving one-step computation.
- **Practical Scenarios**: Holds direct value for applications such as multilingual information retrieval and factual updates in multilingual LLMs.

## Limitations & Future Work

- The dimension of the null space shrinks as the number of edits increases, potentially resulting in "null-space exhaustion" during extremely large-scale editing.
- Edits are performed restrictedly on the `W_out` of MLP layers; knowledge residing in attention layers remains unedited.
- The selection of key layers relies on causal tracing, which may require different configurations across various models and languages.
- Experiments only cover 6 languages; the efficacy on more diverse language combinations (e.g., low-resource languages) remains unexplored.
- The edited knowledge type is limited to factual triplets, while more complex knowledge types are not yet addressed.

## Related Work & Insights

- **ROME** (Meng et al., 2022): A pioneering locate-and-edit framework that edits a single MLP layer via rank-one updates.
- **MEMIT** (Meng et al., 2023): Extends ROME to multi-layer parallel editing.
- **RECT** (Gu et al., 2024): Controls update magnitude through relative weight variations.
- **PRUNE** (Ma et al., 2025): Limits weight degradation via condition number constraints.
- **AlphaEdit** (Fang et al., 2025): Monolingual null-space projection editing, serving as a direct baseline for this work.
- **Null-Space Methods in Continual Learning** (Wang et al., 2021): The core technology borrowed and adapted in this work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Multilingual sequential editing is a novel task, and dynamic null-space projection is a meaningful extension of AlphaEdit.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 3 models × 2 datasets × 6 languages × 4 downstream tasks × multiple baselines × language-by-language analysis × negative interference quantification.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem motivation, rigorous mathematical derivations, and highly informative figures and tables.
- **Value**: ⭐⭐⭐⭐ — Provides a practical framework for the knowledge maintenance of multilingual LLMs, offering clear engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing](../../ACL2026/knowledge_editing/evoedit_evolving_null-space_alignment_for_robust_and_efficient_knowledge_editing.md)
- [\[ACL 2025\] Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning](memorizing_is_not_enough_deep_knowledge_injection_through_reasoning.md)
- [\[ACL 2025\] ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains](chainedit_propagating_ripple_effects_in_llm.md)
- [\[ACL 2025\] ToxEdit: Adaptive Detoxification Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing](adaptive_detoxification_safeguarding_general_capabilities_of_llms_through_toxici.md)
- [\[ACL 2025\] Neuron-Level Sequential Editing for Large Language Models](neuron-level_sequential_editing_for_large_language_models.md)

</div>

<!-- RELATED:END -->
