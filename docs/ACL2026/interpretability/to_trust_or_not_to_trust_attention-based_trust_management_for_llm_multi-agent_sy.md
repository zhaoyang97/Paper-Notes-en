---
title: >-
  [Paper Note] To Trust or Not to Trust: Attention-Based Trust Management for LLM Multi-Agent Systems
description: >-
  [ACL 2026][LLM multi-agent trust management] This paper proposes the first comprehensive definition of "trustworthiness" for LLM multi-agent systems (LLM-MAS), grounded in six orthogonal dimensions derived from Grice's Cooperative Principle. It demonstrates that LLM attention patterns can distinguish different types of trustworthiness violations, and on this basis introduces A-Trust, a lightweight attention-based evaluation method, and an end-to-end Trust Management System (TMS) that achieves malicious message detection rates of 77–90% across diverse attack scenarios.
tags:
  - ACL 2026
  - LLM multi-agent trust management
  - attention pattern analysis
  - trustworthiness evaluation
  - malicious message detection
  - trusted communication
date: 2026-05-08
content_hash: d0a97915c0735932
---

# To Trust or Not to Trust: Attention-Based Trust Management for LLM Multi-Agent Systems

**Conference**: ACL 2026
**arXiv**: [2506.02546](https://arxiv.org/abs/2506.02546)
**Code**: [GitHub](https://anonymous.4open.science/r/multi-com-1808)
**Area**: Interpretability / Multi-Agent Security
**Keywords**: LLM multi-agent trust management, attention pattern analysis, trustworthiness evaluation, malicious message detection, trusted communication

## TL;DR

This paper proposes the first comprehensive definition of "trustworthiness" for LLM multi-agent systems (LLM-MAS), grounded in six orthogonal dimensions derived from Grice's Cooperative Principle. It demonstrates that LLM attention patterns can distinguish different types of trustworthiness violations, and on this basis introduces A-Trust, a lightweight attention-based evaluation method, and an end-to-end Trust Management System (TMS) that achieves malicious message detection rates of 77–90% across diverse attack scenarios.

## Background & Motivation

**Background**: LLM multi-agent systems have demonstrated strong capabilities in code generation, mathematical reasoning, and scientific simulation, and the Agent2Agent (A2A) protocol is gaining widespread adoption. However, LLM agents accept all incoming messages indiscriminately, lacking the information-vetting ability that humans typically exercise.

**Limitations of Prior Work**: (1) LLM-MAS is highly vulnerable to malicious message attacks, including Adversary-in-the-Middle (AiTM), AutoInject, and NetSafe attacks; (2) existing work addresses only isolated aspects of trustworthiness (e.g., factual correctness or logical consistency), lacking a systematic multi-dimensional evaluation framework; (3) prompt-based trust evaluation is susceptible to hallucination, while external verification tools introduce latency and depend on the quality of external data.

**Key Challenge**: LLM attention patterns already "perceive" untrustworthy information—evidenced by markedly elevated attention weights—yet the model's final output fails to exploit this signal, revealing a mismatch between attention patterns and output behavior, a form of hallucination.

**Goal**: (1) Establish a multi-dimensional trustworthiness definition framework; (2) leverage LLM internal attention signals for efficient trustworthiness evaluation; (3) construct an end-to-end trust management system.

**Key Insight**: Six trust dimensions are defined from Grice's Cooperative Principle. Different types of violations are found to produce distinct patterns in LLM attention, with certain attention heads specializing in sensitivity to particular violation types.

**Core Idea**: Extract cross-layer aggregated feature vectors from LLM multi-head attention weights, train lightweight logistic regression classifiers to score each of the six trust dimensions independently, and build a message-level and agent-level trust management system.

## Method

### Overall Architecture

The system comprises three layers: (1) **Trust dimension definitions + Trust Violation dataset**—a six-dimensional framework and a controlled dataset of >20k samples; (2) **A-Trust evaluation**—a lightweight classifier based on attention vectors; (3) **TMS trust management**—message-level evaluation → trust-aware action strategy → agent-level trust records.

### Key Designs

1. **Six-Dimensional Trust Framework + Trust Violation Dataset**:

    - Function: Provides systematic, interpretable trustworthiness evaluation criteria and training data.
    - Mechanism: Six orthogonal trust dimensions are defined based on Grice's Cooperative Principle—Factual Accuracy, Logical Consistency, Relevance, Bias, Quality, and Clarity. A dataset of >20k samples is constructed such that each untrustworthy sample violates exactly one dimension. The Factual dimension reuses FEVER, Bias reuses StereoSet, and the remaining four dimensions are generated with GPT-4o.
    - Design Motivation: Untrustworthy samples in existing datasets typically conflate multiple violations, precluding fine-grained analysis. Single-dimension violation design enables attribution of specific attention heads to specific dimensions.

2. **A-Trust Attention-Based Trust Evaluation**:

    - Function: Lightweight trustworthiness evaluation of messages using LLM internal signals.
    - Mechanism: Given message $M$ and context $C$, the attention weights from the last token to the message tokens are extracted and aggregated across layers into a scalar per head: $Attn^h(M) = \text{Mean}(\{Attn^{l,h}(M)\}_{l=1}^{L})$, forming an $H$-dimensional feature vector. One-vs-rest logistic regression classifiers $(f_{\text{fact}}, ..., f_{\text{clarity}})$ are trained independently for each of the six dimensions; each classifier's output $\in [0,1]$ serves as the trust score for that dimension.
    - Design Motivation: Analysis reveals that LLMs assign higher attention to untrustworthy messages, and specific attention heads respond selectively to specific dimensions (e.g., head 2 for Relevance, head 21 for Clarity, head 27 for Quality). Logistic regression is lightweight and efficient, suitable for real-time deployment.

3. **Trust Management System (TMS)**:

    - Function: Enables end-to-end trust-aware communication in LLM-MAS.
    - Mechanism: Three components—(a) **Message-level evaluation**: computes six-dimensional A-Trust scores for each inter-agent message; (b) **Trust-aware action strategy**: sets a threshold $\tau_r$ per dimension; messages exceeding the threshold are flagged as untrustworthy and filtered, with a dual-threshold strategy for the Factual dimension (uncertain cases invoke Gemini-2.0-flash + Google Search verification); (c) **Agent-level trust records**: maintains timestamped trust records $R_{A_i}$ and performs periodic agent-level evaluation via the violation rate $VR_r$ within a sliding time window.
    - Design Motivation: Message-level evaluation enables real-time filtering, while agent-level records support long-term behavioral tracking—together they identify and isolate persistently malicious agents.

### Loss & Training

A-Trust employs logistic regression without complex training procedures. For each dimension, one-vs-rest sampling uses 1,500 positive examples plus an equal number of negatives. Attention matrices are extracted using LLaMA3.1-8B-Instruct.

## Key Experimental Results

### Main Results

**Malicious Message Detection Rate MDR (Chain topology, %)**

| Dataset | Method | AiTM ↑ | AutoTrans ↑ | AutoInj ↑ | NetSafe ↑ | Clean ↓ |
|--------|------|--------|------------|----------|----------|---------|
| MMLUPhy | A-Trust | **84.3** | **77.5** | **90.1** | **79.6** | 7.1 |
| | PPL | 43.1 | 51.8 | 57.9 | 50.5 | 4.8 |
| | Prompt | 52.9 | 57.7 | 55.3 | 50.3 | 9.4 |
| MATH | A-Trust | **84.1** | **79.3** | **85.6** | **80.4** | 7.3 |
| | PPL | 52.4 | 54.2 | 53.1 | 57.3 | 4.7 |
| | Prompt | 62.9 | 60.7 | 61.8 | 64.9 | 7.5 |

**Attack Success Rate ASR (Chain topology, %, ↓ is better)**

| Dataset | Configuration | Clean | AiTM | AutoTrans | AutoInj | NetSafe |
|--------|------|-------|------|-----------|---------|---------|
| MMLUPhy | No trust | 41.7 | 92.5 | 69.3 | 62.6 | 67.7 |
| | A-Trust TMS | 43.8 | **14.1** | **51.4** | **47.1** | **46.4** |
| MATH | No trust | 47.6 | 93.6 | 69.8 | 67.2 | 68.2 |
| | A-Trust TMS | 50.6 | **18.4** | **54.2** | **55.2** | **59.2** |

### Ablation Study

- The agent-level trust record strategy achieves 100% malicious agent detection rate (ADR) across all attacks.
- A-Trust demonstrates clear separation between violation and non-violation cases across all six dimensions; prompt-based methods show near-zero discriminability on multiple dimensions due to hallucination.
- A-Trust substantially outperforms PPL and Prompt baselines across Chain, Complete, and Tree topologies.

### Key Findings

- LLMs assign significantly higher attention to untrustworthy messages than to normal ones—the model "perceives" the anomaly but fails to act on it.
- Specific attention heads exhibit specialization for specific trust dimensions (head 2 → Relevance, head 21 → Clarity, head 27 → Quality).
- Prompt-based trust evaluation suffers from severe hallucination—internal attention signals are more reliable than model outputs.
- AiTM attack ASR drops from 92.5% to 14.1%, the largest reduction, indicating that DoS-type attack message patterns are most readily identified via attention.

## Highlights & Insights

- The finding that "LLMs perceive untrustworthy content but fail to utilize the signal" reveals a mismatch between the attention mechanism and model output—a novel form of hallucination.
- The six-dimensional framework grounded in Grice's Cooperative Principle provides a standardized evaluation vocabulary for LLM-MAS security research.
- The minimalist design of logistic regression over attention vectors outperforms GPT-4o prompt-based evaluation, demonstrating that simple methods can surpass complex prompting.

## Limitations & Future Work

- The Trust Violation dataset is partially generated by GPT-4o (four dimensions), which may introduce distributional bias.
- Adaptive adversaries may learn to evade specific attention patterns; robustness under adaptive attacks warrants further investigation.
- The thresholding strategy is relatively simple and does not account for inter-dimensional correlations.
- A-Trust is trained solely on LLaMA3.1-8B; cross-model generalizability has not been thoroughly validated.

## Related Work & Insights

- **vs. Perplexity-based evaluation**: Perplexity is a single scalar and cannot distinguish different types of trustworthiness violations; A-Trust provides fine-grained six-dimensional evaluation.
- **vs. Prompt-based evaluation**: Susceptible to LLM hallucination, yielding near-zero discriminability across multiple dimensions.
- **vs. FEVER/StereoSet**: These datasets cover only a single dimension (factual/bias); the Trust Violation dataset is the first six-dimensional controlled benchmark.
- **vs. NetSafe**: NetSafe focuses on attack methodology, while this paper focuses on defense—the two are complementary.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First attention-pattern-based trust management framework for LLM-MAS; the six-dimensional definition fills a clear gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 attacks × 4 datasets × 3 topologies, comparisons against 3 baselines, agent-level evaluation, and a MetaGPT case study.
- Writing Quality: ⭐⭐⭐⭐ Complete framework with persuasive attention analysis visualizations.
- Value: ⭐⭐⭐⭐⭐ Provides a practical and deployable solution for LLM-MAS security, particularly timely in the era of A2A protocols.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Auditing Cascading Risks in Multi-Agent Systems via Semantic–Geometric Co-evolution](../../ICLR2026/interpretability/auditing_cascading_risks_in_multi-agent_systems_via_semanti-geometric_co-evolut.md)
- [\[AAAI 2026\] iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference](../../AAAI2026/interpretability/imad_intelligent_multi-agent_debate_for_efficient_and_accura.md)
- [\[AAAI 2026\] ToC: Tree-of-Claims Search with Multi-Agent Language Models](../../AAAI2026/interpretability/toc_tree-of-claims_search_with_multi-agent_language_models.md)
- [\[CVPR 2026\] Neurodynamics-Driven Coupled Neural P Systems for Multi-Focus Image Fusion](../../CVPR2026/interpretability/neurodynamics-driven_coupled_neural_p_systems_for_multi-focus_image_fusion.md)
- [\[ICLR 2026\] MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning](../../ICLR2026/interpretability/mata_a_trainable_hierarchical_automaton_system_for_multi-agent_visual_reasoning.md)

<!-- RELATED:END -->
