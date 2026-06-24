---
title: >-
  [Paper Note] CompKe: Complex Question Answering under Knowledge Editing
description: >-
  [ACL 2025][Knowledge Editing][Complex Question Answering] Proposes the CompKe benchmark—containing 11,924 complex questions—to evaluate the performance of knowledge editing methods in complex reasoning scenarios involving **one-to-many relations, logical operations (intersection/union), and condition confirmation**, revealing the significant deficiencies of existing methods in complex question answering.
tags:
  - "ACL 2025"
  - "Knowledge Editing"
  - "Complex Question Answering"
  - "Knowledge Graph"
  - "Multi-Hop Reasoning"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: f14bedb441a2e07e
---

# CompKe: Complex Question Answering under Knowledge Editing

**Conference**: ACL 2025  
**arXiv**: [2506.00829](https://arxiv.org/abs/2506.00829)  
**Code**: [GitHub](https://github.com/kzjkzj666/CompKE)  
**Area**: Knowledge Editing  
**Keywords**: Knowledge Editing, Complex Question Answering, Knowledge Graph, Multi-Hop Reasoning, LLM Evaluation

## TL;DR

Proposes the CompKe benchmark—containing 11,924 complex questions—to evaluate the performance of knowledge editing methods in complex reasoning scenarios involving **one-to-many relations, logical operations (intersection/union), and condition confirmation**, revealing the significant deficiencies of existing methods in complex question answering.

## Background & Motivation

**Background**: Knowledge Editing (KE) aims to efficiently update the knowledge of LLMs without full fine-tuning. Existing benchmarks like ZsRE and COUNTERFACT mainly test rote memorization, and while MQuAKE introduces multi-hop Q&A, it remains highly limited.

**Limitations of Prior Work**:
   - **Linear Question Structures**: Questions in benchmarks like MQuAKE only feature sequential step-by-step reasoning chains.
   - **Limited to One-to-One Relations**: Factual triples only involve one-to-one relations, failing to reflect real-world one-to-many relations (e.g., "Who are the major shareholders of the company?").
   - **Limited Edit Operations**: Only substitution operations are supported, overlooking additions and deletions.

**Key Challenge**: Multi-hop Q&A cannot adequately evaluate edited models in real-world scenarios that **require logical composition (intersection, union) and condition filtering**.

**Goal**: To design a more realistic complex Q&A benchmark featuring diverse reasoning structures, one-to-many relations, and addition/deletion/modification edit types.

**Key Insight**: Formalize complex questions as graph-structured reasoning $Q = (\mathbf{S}, \mathbf{L})$, where $\mathbf{S}$ is a sequence of entity sets and $\mathbf{L}$ represents reasoning links (knowledge mapping + condition confirmation + logical operations).

**Core Idea**: Fill the gap in knowledge editing evaluation for complex reasoning from the perspectives of one-to-many relations and logical compositions.

## Method

### Overall Architecture

Extract relation templates and entity triples from Wikidata $\to$ Construct complex question reasoning structures $\to$ Introduce counterfactual edits $\to$ Filter conflicts $\to$ Convert into natural language questions.

### Key Designs

1. **Graph-Structured Definition of Complex Questions**:

    - A complex question $Q = (\mathbf{S}, \mathbf{L})$
    - $\mathbf{S} = \{S_1, S_2, \dots\}$: A sequence of intermediate entity sets, where each $S_i$ is itself an entity set (supporting one-to-many).
    - $\mathbf{L} = \{L_1, L_2, \dots\}$: Reasoning links, categorized into **knowledge-related links** and **logical links**.
    - Design Motivation: Break the limits of linear reasoning chains to support more complex reasoning topologies.

2. **Types of Reasoning Links**:

    - **Knowledge Mapping**: Given $S_i$, find all adjacent entities through relation $r$: $S_j = \bigcup_{s \in S_i} A_r(s)$
    - **Condition Confirmation**: Filter entities in $S_i$ that satisfy specific relation constraints.
    - **Intersection**: $S_j = \bigcap_{k=1}^n S_k$
    - **Union**: $S_j = \bigcup_{k=1}^n S_k$

3. **Formalization of Knowledge Editing**:

    - An edit is defined as $e = (s, r, \mathcal{O} \to \mathcal{O}')$, supporting changes to one-to-many object sets.
    - Decomposed into three basic operations: addition $\mathcal{O}_{\text{add}} = \mathcal{O}' \setminus \mathcal{O}$, deletion $\mathcal{O}_{\text{del}} = \mathcal{O} \setminus \mathcal{O}'$, and retention $\mathcal{O}_{\text{ret}} = \mathcal{O} \cap \mathcal{O}'$.

4. **Dataset Construction Pipeline** (6 steps):

    - Step 1: Select 37 relation templates from Wikidata.
    - Step 2: Sample factual triples based on access frequency, and filter out knowledge that models cannot recall using GPT-3.5.
    - Step 3: Abstract reasoning structure templates from seed questions, and then instantiate them with real-world facts.
    - Step 4: Randomly select knowledge mappings to introduce counterfactual edits.
    - Step 5: Detect and filter conflicting edits.
    - Step 6: Generate 3 natural language variants for each question using GPT-4o-mini.

### Evaluation Metrics

- **Aug (Augment Accuracy)**: The ratio of newly added entities correctly included in the answer.
- **Ret (Retain Accuracy)**: The ratio of original entities correctly retained in the answer.
- **Acc = (Aug + Ret) / 2**

## Key Experimental Results

### Main Results

| Model | Method | 1-edit Acc | 100-edit Acc | 3000-edit Acc |
|------|------|-----------|-------------|--------------|
| Qwen2.5-3B | ROME | 15.26 | 4.60 | 1.21 |
| Qwen2.5-3B | MEMIT | **22.43** | 7.27 | 2.64 |
| Qwen2.5-3B | MeLLo | 3.83 | 3.23 | 1.35 |
| Qwen2.5-7B | MEMIT | **28.56** | **24.46** | 1.97 |
| Qwen2.5-7B | MeLLo | 15.58 | 13.84 | **10.79** |
| LLaMA-3.1-8B | MEMIT | 19.06 | 17.14 | 17.12 |
| LLaMA-3.1-8B | MeLLo | 16.00 | 13.51 | 11.58 |
| GPT-3.5-turbo | MeLLo | **47.05** | **40.60** | **35.60** |
| GPT-4o-mini | PokeMQA | **39.47** | **38.39** | **31.69** |

### Ablation Study

- **Comparison with MQuAKE-T and MQuAKE-CF-3k**: The accuracy of MeLLo and PokeMQA on CompKe is significantly lower, proving that CompKe is more challenging.
- Parametric methods (ROME/MEMIT) collapse sharply when the batch edit size $k \geq 100$.
- Memory-based methods (MeLLo/PokeMQA) experience a gradual decline.

### Key Findings

- **Parametric methods outperform memory-based methods on small models**: MEMIT (22.43) $\gg$ MeLLo (3.83) on Qwen2.5-3B. This is because smaller models exhibit weaker instruction-following capabilities, making the decomposed reasoning required by memory-based methods difficult to execute.
- **Memory-based methods are more robust on large models**: MeLLo (47.05) leads by a wide margin on GPT-3.5-turbo.
- **Overfitting in parametric methods**: MEMIT seems to score high on small models, but in reality, the model indiscriminately outputs newly injected knowledge (overfitting), leading to an artificially inflated Aug.
- **Omission phenomenon in MeLLo**: When decomposing questions, MeLLo often skips logical intersection steps because the original prompt demonstrations do not contain such operations.
- **Catastrophic effects of batch editing**: Parametric methods almost completely collapse when $k \geq 100$, producing incoherent outputs.

## Highlights & Insights

- **Fills an important gap in knowledge editing evaluation**: Methodatically introduces one-to-many relations, logical operations, and addition/deletion edit types for the first time.
- **Reveals the interaction effect between methods and model scales**: Parametric methods are suitable for smaller models, whereas memory-based methods require strong reasoning capabilities.
- Deep analysis of the **overfitting and omission phenomena** provides directions for future method designs.
- **Rigorous dataset design**: The 6-step construction pipeline includes conflict detection and quality filtering.

## Limitations & Future Work

1. **Realism of edits**: Counterfactual edits are introduced randomly, which may not align with the distribution of real-world knowledge changes.
2. **Limited relation types**: Only one-to-one and one-to-many relations are involved, leaving many-to-many and many-to-one relations uncovered.
3. **Limited evaluated methods**: Only four KE methods were tested, leaving newer approaches (such as GRACE or IKE) untested on CompKe.
4. **Scalable directions**: Future versions can introduce more complex scenarios such as temporal edit chains and multi-round continuous editing.

## Related Work & Insights

- **CompKe vs MQuAKE**: MQuAKE only supports linear multi-hop, one-to-one substitutions, whereas CompKe covers graph-structured reasoning, logical operations, and addition/deletion/modification operations.
- **Insights for knowledge editing methods**: (1) Parametric methods need to solve the catastrophic forgetting in batch editing; (2) Memory-based methods require a more robust question decomposition mechanism.
- **Inspiration**: The coverage of prompt examples directly determines decomposition quality—MeLLo misses intersections precisely because of insufficient examples.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of complex reasoning and KE provides a novel and meaningful evaluation perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 5 models $\times$ 4 methods $\times$ 4 batch sizes, paired with detailed case analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous formalization and clear example-driven explanations.
- **Value**: ⭐⭐⭐⭐⭐ — Plays an important role in driving the KE community forward; the benchmark holds long-term utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SAME: Sparse and Anchored Model Editing for Heterogeneous Incremental Learning under Limited Data](../../CVPR2026/knowledge_editing/same_sparse_and_anchored_model_editing_for_heterogeneous_incremental_learning_un.md)
- [\[ACL 2025\] ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains](chainedit_propagating_ripple_effects_in_llm.md)
- [\[NeurIPS 2025\] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](../../NeurIPS2025/knowledge_editing/uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)
- [\[ACL 2025\] SAKE: Steering Activations for Knowledge Editing](sake_steering_activations_for_knowledge_editing.md)
- [\[ACL 2025\] Efficient Knowledge Editing via Minimal Precomputation](efficient_knowledge_editing.md)

</div>

<!-- RELATED:END -->
