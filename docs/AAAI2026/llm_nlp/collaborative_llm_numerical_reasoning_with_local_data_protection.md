---
title: >-
  [Paper Note] Collaborative LLM Numerical Reasoning with Local Data Protection
description: >-
  [AAAI 2026][LLM/NLP][privacy-preserving reasoning] This paper proposes a large-small model collaboration framework that protects sensitive local data through a two-stage anonymization pipeline — topic shifting followed b…
tags:
  - "AAAI 2026"
  - "LLM/NLP"
  - "privacy-preserving reasoning"
  - "numerical reasoning"
  - "model collaboration"
  - "topic shifting"
  - "code-as-tool"
date: 2026-05-08
content_hash: 2659d97a8c2fbf85
---

# Collaborative LLM Numerical Reasoning with Local Data Protection

**Conference**: AAAI 2026
**arXiv**: [2504.00299](https://arxiv.org/abs/2504.00299)  
**Area**: LLM/NLP
**Keywords**: privacy-preserving reasoning, numerical reasoning, model collaboration, topic shifting, code-as-tool

## TL;DR

This paper proposes a large-small model collaboration framework that protects sensitive local data through a two-stage anonymization pipeline — topic shifting followed by numerical substitution — applied to local queries. The remote GPT-4 returns reasoning solutions as executable Python code (plug-and-play tools), and the local model only needs to perform numerical back-substitution to obtain the final answer. The framework achieves 16–44% accuracy improvements on FinQA and MultiHiertt while reducing data leakage by 2–45%.

---

## Background & Motivation

**Demand for document numerical reasoning**: Numerical reasoning over financial reports, medical documents, and research papers requires deep contextual understanding and logical inference, which exceeds the capabilities of local small models (e.g., 3–4B parameters).

**Privacy risks of directly querying remote models**: Sending local data directly to APIs such as GPT-4 exposes sensitive information including company details, operational figures, and strategic analyses, posing serious data leakage risks.

**Two major challenges with existing privacy-preserving methods**:
- **Difficulty generating logically consistent synthetic queries**: Existing methods (high-level descriptions, similar examples, differential privacy perturbations) struggle to preserve the integrity of reasoning logic in numerical reasoning scenarios.
- **Local answer reconstruction constrained by reasoning capacity**: Even with remote hints, small models have difficulty correctly integrating local data with remote guidance to reconstruct answers.

**Core insight**: Sensitive queries can be transformed into entirely different semantic domains while preserving the underlying mathematical structure. For instance, "deriving a total from counts and percentages of fuel consumption" and "deriving a total from counts and percentages of advertising revenue" are equivalent problems with different topics but identical reasoning patterns.

---

## Method

### Overall Architecture

The framework consists of three main steps: (1) local topic shifting — a small synthesizer is trained via distillation to transform queries into different topics while preserving reasoning patterns; (2) numerical substitution — the numerical values in the synthetic query are replaced via a mapping to achieve complete anonymization; (3) remote reasoning + local reconstruction — the remote model processes the anonymized query and returns Python code, which the local model executes via numerical back-substitution to obtain the answer.

### Key Design 1: Topic Shifter

- **Function**: Transforms the original query $(C, q)$ into a synthetic query $(\tilde{C}, \tilde{q}) = \mathcal{M_S}(C, q)$ that covers a different topic while preserving format, logic, and numerical values.
- **Rationale for preserving numerical values**: Retaining numerical values (a) enables validation of mathematical abstraction consistency — if the synthetic problem yields the same answer as the original, the reasoning pattern is correctly preserved; and (b) establishes clear entity mappings (e.g., "fuel expense ↔ advertising revenue") for subsequent numerical substitution.
- **Distillation training**: GPT-4o (teacher) generates high-quality topic-shifted data to train Llama-3.2-3B (student) as the local synthesizer. Training data undergoes triple quality filtering — leakage assessment, conflicting evidence detection, and answer consistency verification — reducing 6,360 samples to 5,762.
- **Key advantage**: Decouples semantic protection from numerical protection; the synthesizer is responsible only for topic transformation and does not handle numerical values, reducing task complexity.

### Key Design 2: Data Switch

- **Function**: Applies a mapping $h: n_i \mapsto \tilde{n_i}$ to all numerical values $\mathcal{N} = \{n_1, n_2, \dots, n_k\}$ in the synthetic query to achieve complete anonymization.
- **Three quality assurance strategies**:
    - **Special value handling**: Semantically sensitive values such as month-end dates (28–31) are preserved unchanged.
    - **Year offset transformation**: Year-related values are shifted to maintain relative differences between years.
    - **Order-preserving transformation**: Other values are sorted by interval and mapped to a new range, preserving original magnitude relationships.
- **Bridging role**: Numerical substitution serves as the bridge between topic shifting and answer reconstruction — the inverse mapping $h^{-1}$ is used for numerical back-substitution during local reconstruction.

### Key Design 3: Remote Assistance — Code as Tool

- **Function**: Sends the anonymized query $(\tilde{C_h}, \tilde{q_h})$ to the remote model $\mathcal{M_R}$, requesting Python code rather than a natural language answer.
- **Formulation**: $f(m_1, m_2, \dots, m_t) = \mathcal{M_R}(\tilde{C_h}, \tilde{q_h})$, where $m_i \in \{\tilde{n_1}, \tilde{n_2}, \dots, \tilde{n_k}\}$.
- **Key innovation**: The code represents intermediate computation steps as variables, functioning as a reusable plug-and-play tool rather than a one-time answer.
- **Distinction from Program-of-Thought**: PoT uses code to enhance reasoning; this method focuses on **reusing** code under privacy constraints — a success that depends on the logical consistency of the query and the decoupled protection of semantics and numerical values.

### Key Design 4: Local Answer Reconstruction

- **Function**: Directly executes $f(h^{-1}(m_1), h^{-1}(m_2), \dots, h^{-1}(m_t))$ in a Python interpreter, substituting the anonymized values in the remote code with the original numerical values.
- **Why it works**: Because topic shifting preserves the reasoning pattern and numerical substitution preserves magnitude relationships, the same code applies to the original query.
- **No local re-inference required**: This approach completely circumvents the core limitation of small model reasoning capacity, replacing uncertain model inference with deterministic numerical back-substitution.

---

## Key Experimental Results

### Experimental Setup

- **Datasets**: FinQA (financial question answering) and MultiHiertt (multi-level tabular reasoning), both containing explicit and implicit sensitive information
- **Local models**: Phi-3-mini-128k (3.8B), Llama-3.2-3B-Instruct
- **Remote model**: GPT-4o
- **Evaluation**: Normalized accuracy (divided by direct remote inference accuracy) + LLM-as-judge leakage evaluation (GPT-4o-mini, 96% agreement with human annotations)

### Main Results

| Method | MultiHiertt Accuracy | MultiHiertt Leakage | FinQA Accuracy | FinQA Leakage |
|--------|----------------------|---------------------|----------------|---------------|
| Local single inference (Phi-3) | 54.4% | 0% | 71.1% | 0% |
| Hint method | 42.7% | 6.0% | 63.9% | 28.8% |
| Example method | 57.0% | 16.1% | 71.4% | 38.9% |
| **Ours** | **80.1%** | **3.7%** | **87.6%** | **6.4%** |

### Key Findings

1. **Substantial accuracy gains**: 23–44% higher than the Hint method and 16–40% higher than the Example method, approaching the upper bound of direct remote processing.
2. **Significant leakage reduction**: 2–45% less leakage than the Example method, owing to the decoupled design of topic shifting and numerical substitution enabling more thorough anonymization.
3. **Hint/Example methods degrade accuracy**: Logically inconsistent synthetic queries fail to effectively elicit the remote model and further confuse small model inference.
4. **Strong generalization**: The synthesizer is trained only on MultiHiertt yet remains effective on the unseen FinQA dataset.
5. **Ablation Study**: Removing the tool component (reverting to local inference) reduces accuracy from 90.1% to 65.0%; removing distillation reduces accuracy to 42.2%, confirming both core components are indispensable.
6. **Error analysis**: The largest error source is the remote model itself (33.3%); topic shifting errors account for only 16.7%, numerical substitution errors for 6.7%, and answer reconstruction errors for 0%.

---

## Highlights & Insights

### Strengths

- The decoupled design of semantic and numerical protection is the key innovation, substantially reducing the difficulty of the synthesis task.
- Replacing natural language answers with executable code entirely bypasses the fundamental limitation of small model reasoning capacity.
- The framework exhibits model-agnostic and dataset-agnostic generalizability.

### Limitations & Future Work

- The framework relies on a local retriever to shorten context; retrieval errors account for 20% of total errors.
- The topic shifter occasionally makes errors on similar phrases and units of measurement.
- Validation is limited to numerical reasoning tasks; applicability to non-numerical reasoning (causal reasoning, commonsense reasoning) remains unclear.

---

## Related Work & Insights

- **InferDPT / TextObfuscator**: Differential privacy-based text perturbation methods that preserve semantics but do not guarantee logical consistency, yielding poor performance in numerical reasoning scenarios.
- **CogEnesis**: Has the local model generate high-level descriptions to send to the remote model, but descriptions fail to convey complex reasoning logic.
- **Program-of-Thought (PoT)**: A paradigm for code-enhanced reasoning; this work repurposes it as a privacy-preserving tool for cross-domain code reuse.
- **Model Cascading**: Routes queries to remote models based on difficulty; the proposed method can be embedded in a cascading framework to provide privacy protection for hard samples routed to remote models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation](c3tg_conflict-aware_composite_and_collaborative_controlled_text_generation.md)
- [\[AAAI 2026\] Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)
- [\[AAAI 2026\] From Classification to Ranking: Enhancing LLM Reasoning for MBTI Personality Detection](from_classification_to_ranking_enhancing_llm_reasoning_capabilities_for_mbti_per.md)
- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](../../ACL2026/llm_nlp/memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Models](../../ICLR2026/llm_nlp/predicting_llm_reasoning_performance_with_small_proxy_models.md)

</div>

<!-- RELATED:END -->
