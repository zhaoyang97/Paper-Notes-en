---
title: >-
  [Paper Note] Commonsense Abductive Reasoning using Knowledge from Multiple Sources
description: >-
  [ACL 2025][Reasoning][Abductive Reasoning] This paper proposes a commonsense abductive reasoning method that integrates multi-source knowledge (knowledge graphs, pre-trained language models, and rule bases). By jointly utilizing structured and unstructured knowledge to generate more accurate and explainable best explanations, the method achieves significant improvements on abductive reasoning benchmarks.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Abductive Reasoning"
  - "Commonsense Reasoning"
  - "Multi-Source Knowledge"
  - "Knowledge Fusion"
  - "Explainable Reasoning"
date: 2026-05-08
content_hash: 2c391cf01ae123e6
---

# Commonsense Abductive Reasoning using Knowledge from Multiple Sources

**Conference**: ACL 2025  
**Area**: LLM Reasoning  
**Keywords**: Abductive Reasoning, Commonsense Reasoning, Multi-Source Knowledge, Knowledge Fusion, Explainable Reasoning

## TL;DR
This paper proposes a commonsense abductive reasoning method that integrates multi-source knowledge (knowledge graphs, pre-trained language models, and rule bases). By jointly utilizing structured and unstructured knowledge to generate more accurate and explainable best explanations, the method achieves significant improvements on abductive reasoning benchmarks.

## Background & Motivation

**Background**: Abductive reasoning is the process of inferring the most plausible explanation from observed outcomes, which is a core capability of human everyday reasoning (e.g., inferring it might have rained based on seeing that the ground is wet). Commonsense abductive reasoning requires a model to select or generate the most reasonable intermediate hypothesis (explanation) given an initial observation and a final observation. Representative benchmarks include the αNLI (Abductive NLI) and ART tasks.

**Limitations of Prior Work**: Current commonsense abductive reasoning methods mainly rely on a single source of knowledge: (1) PLM-based methods leverage the implicit knowledge in pre-trained language models for reasoning, but implicit knowledge struggles to cover all commonsense scenarios and lacks explainability; (2) Knowledge Graph (KG) based methods (e.g., ConceptNet) utilize explicit commonsense relation triplets, but KGs have limited coverage and are unfriendly to novel concepts; (3) LLM-based methods directly generate explanations using models like GPT, but they are prone to hallucinations and lack factual constraints.

**Key Challenge**: Each knowledge source has unique advantages and limitations—PLMs excel at semantic understanding but lack explicit knowledge, KGs provide accurate factual relations but have incomplete coverage, and LLMs are good at generation but lack reliability. How to integrate the benefits of multiple knowledge sources is the key challenge.

**Goal**: Design a unified framework that effectively fuses multiple knowledge sources (knowledge graphs, pre-trained models, LLM-generated knowledge, and general rule bases) for commonsense abductive reasoning.

**Key Insight**: Different types of abductive problems require different types of knowledge—physical events require physical commonsense (rich in KGs), social interactions require social commonsense (implicit in PLMs), and causal inference requires causal rules (explicit in rule bases). Adaptively selecting the most relevant knowledge source for each problem is the key.

**Core Idea**: Construct a "multi-source knowledge-enhanced" abductive reasoning framework that retrieves relevant knowledge from multiple sources for each reasoning problem, and adaptively fuses them through an attention mechanism to generate knowledge-enhanced hypothetical explanations.

## Method

### Overall Architecture
Given a pair of observations (initial observation O1 and final observation O2), the framework first retrieves knowledge snippets related to the observations from multiple knowledge sources in parallel, integrates these pieces of knowledge through a multi-source knowledge fusion module, and finally conducts hypothesis selection or generation based on the fused knowledge representation.

### Key Designs

1. **Multi-Source Knowledge Retrieval**:

    - **Function**: Efficiently retrieve knowledge snippets related to the current reasoning problem from different knowledge sources.
    - **Mechanism**: Tailor specific retrieval strategies for each knowledge source—for knowledge graphs (ConceptNet), extract key concepts from O1 and O2, and retrieve multi-hop paths between these concepts as structured knowledge; for pre-trained LMs (e.g., RoBERTa), extract relevant implicit knowledge using probing techniques to generate candidate knowledge sentences via cloze tasks; for LLMs (e.g., GPT-3.5), design specific prompts to generate knowledge descriptions regarding potential causal chains O1→O2; for rule bases (e.g., ATOMIC), retrieve if-then commonsense rules matching the observations. Each source returns the top-K most relevant knowledge snippets.
    - **Design Motivation**: Retrieval from a single source may miss critical information, while parallel retrieval from multiple sources ensures comprehensive knowledge coverage.

2. **Knowledge Quality Scoring and Filtering**:

    - **Function**: Evaluate the relevance and reliability of retrieved knowledge snippets and filter out high-quality knowledge.
    - **Mechanism**: For each retrieved knowledge piece $k_i$, a lightweight knowledge scorer is used to calculate its relevance score with the current reasoning problem: $s_i = \sigma(MLP([h_{O1}; h_{O2}; h_{k_i}]))$, where $h$ is the encoded vector representation. Knowledge with relevance scores below a certain threshold is filtered out. Meanwhile, source reliability weights are introduced—knowledge from verified KGs receives a higher prior reliability score, while LLM-generated knowledge receives a lower prior score (requiring higher relevance to pass the filtering).
    - **Design Motivation**: Not all retrieved knowledge is useful, and noisy knowledge can mislead reasoning. In particular, LLM-generated knowledge might contain hallucinations, necessitating stricter filtering.

3. **Attention-Driven Multi-Source Fusion**:

    - **Function**: Adaptively integrate knowledge snippets from different sources into a unified knowledge representation.
    - **Mechanism**: Utilizing a cross-attention mechanism, the representation of the current reasoning problem $Q = [h_{O1}; h_{O2}]$ serves as the query, and the set of filtered knowledge snippets $K = \{k_1, ..., k_n\}$ serves as the key/value. After calculating attention weights, the fused knowledge representation is obtained as $h_{knowledge} = \text{CrossAttn}(Q, K)$. The fused representation is concatenated with the reasoning problem representation and fed into the final classification/generation head for hypothesis selection or generation.
    - **Design Motivation**: Different problems rely on different knowledge sources to varying degrees. The attention mechanism allows the model to automatically learn "when to trust KGs more and when to trust LLMs more."

### Loss & Training
Cross-entropy classification loss is used for the hypothesis selection task, and sequence-to-sequence generation loss is used for the hypothesis generation task. The knowledge scorer is trained via distant supervision (using the final reasoning correctness as a feedback signal for backpropagation).

## Key Experimental Results

### Main Results

| Method | αNLI Acc↑ | ART Acc↑ | δ-CAUSAL F1↑ | Average↑ |
|------|----------|---------|-------------|------|
| RoBERTa-large | 83.5 | 71.2 | 62.8 | 72.5 |
| KG-Augmented RoBERTa | 85.2 | 73.8 | 65.1 | 74.7 |
| GPT-3.5 zero-shot | 80.1 | 68.5 | 59.3 | 69.3 |
| GPT-3.5 + CoT | 84.8 | 74.2 | 66.5 | 75.2 |
| MICO (Best Single Source) | 86.3 | 75.1 | 67.2 | 76.2 |
| **Ours** | **89.1** | **78.6** | **71.8** | **79.8** |

### Ablation Study

| Configuration | αNLI Acc↑ | ART Acc↑ | Description |
|------|----------|---------|------|
| Full Method (4 sources) | 89.1 | 78.6 | All knowledge sources |
| ConceptNet Only | 85.8 | 74.3 | Limited structured knowledge |
| PLM Implicit Knowledge Only | 84.2 | 72.5 | Implicit knowledge is not precise enough |
| LLM-generated Knowledge Only | 86.5 | 75.8 | LLM knowledge is comprehensive but noisy |
| ATOMIC Rules Only | 85.0 | 73.1 | Rule coverage is limited |
| Without Knowledge Filtering | 87.3 | 76.2 | Noisy knowledge causes -1.8 / -2.4 |
| Uniform Fusion (no attention) | 87.8 | 77.0 | Adaptive fusion outperforms uniform by +1.3 |

### Key Findings
- Multi-source fusion achieves significant improvements (+2.8 to +4.9 Acc) compared to any single-source method, demonstrating the complementarity of different knowledge sources.
- LLM-generated knowledge is the most effective single source (αNLI 86.5), but with quality filtering, the multi-source method far exceeds pure reliance on LLMs.
- Knowledge quality filtering has the greatest impact on LLM-sourced knowledge (a 2.3 gap before and after filtering), validating the presence of LLM hallucinations.
- Attention-based fusion outperforms uniform fusion (+1.3 / +1.6); the model learns to automatically select more reliable knowledge sources based on the problem type.
- On samples requiring causal reasoning (δ-CAUSAL), the advantage of multi-source knowledge is even more pronounced.

## Highlights & Insights
- Advancing abductive reasoning from a "single knowledge source" to "multi-source knowledge fusion" is a highly valuable direction, revealing the complementarity of different knowledge sources.
- The design of "source reliability priors" in knowledge quality filtering is highly practical—stricter on LLM-generated knowledge, more lenient on KG knowledge.
- Explainability of attention-driven fusion: Analyzing attention weights reveals which knowledge source the model favors for different types of problems.

## Limitations & Future Work
- Multi-source retrieval increases inference latency and computational overhead; efficiency must be considered for practical deployment.
- Knowledge graphs and rule bases require manual maintenance and updates, which entails high long-term maintenance costs.
- For niche domains not covered by knowledge graphs, the improvement from multi-source fusion may be limited.
- A knowledge conflict detection module could be introduced to decide how to proceed when knowledge from different sources contradicts each other.

## Related Work & Insights
- **vs MICO (Wang et al., 2023)**: MICO uses a single knowledge source to enhance abductive reasoning; this work extends it to multi-source fusion.
- **vs Abductive Commonsense (Du et al., 2023)**: Du et al. utilize mutually exclusive explanations for abductive reasoning, whereas this work focuses on knowledge enhancement.
- **vs RAG (Lewis et al., 2020)**: RAG retrieves from a single document corpus for generation enhancement, while this work retrieves from multiple heterogeneous knowledge sources.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of multi-source knowledge fusion is clear, although the individual component technologies are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The ablation study is highly detailed, with the contribution of each knowledge source quantified.
- Writing Quality: ⭐⭐⭐⭐ The motivation is well-articulated, and the framework diagram is clear.
- Value: ⭐⭐⭐⭐ Highly valuable for both commonsense reasoning and knowledge fusion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Balanced Neuro-Symbolic Approach for Commonsense Abductive Logic](../../ICLR2026/llm_reasoning/a_balanced_neuro-symbolic_approach_for_commonsense_abductive_logic.md)
- [\[ACL 2025\] Complex Reasoning with Natural Language Contexts and Background Knowledge](complex_reasoning_with_natural_language_contexts_and_background_knowledge.md)
- [\[NeurIPS 2025\] Curriculum Abductive Learning](../../NeurIPS2025/llm_reasoning/curriculum_abductive_learning.md)
- [\[AAAI 2026\] NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models](../../AAAI2026/llm_reasoning/nestr_a_neuro-symbolic_abductive_framework_for_temporal_reasoning_in_large_langu.md)
- [\[ACL 2025\] Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](chain_of_reasoning_unified_math.md)

</div>

<!-- RELATED:END -->
