---
title: >-
  [Paper Note] To Trust or Not to Trust: Attention-Based Trust Management for LLM Multi-Agent Systems
description: >-
  [ACL 2026][Multi-Agent][LLM multi-agent trust management] This paper proposes the first comprehensive "trustworthiness" definition for LLM Multi-Agent Systems (LLM-MAS) based on six orthogonal dimensions of Grice's Coope…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "LLM multi-agent trust management"
  - "attention pattern analysis"
  - "trustworthiness evaluation"
  - "malicious message detection"
  - "trusted communication"
date: 2026-05-08
content_hash: 21e74d8138d719f7
---

# To Trust or Not to Trust: Attention-Based Trust Management for LLM Multi-Agent Systems

**Conference**: ACL 2026  
**arXiv**: [2506.02546](https://arxiv.org/abs/2506.02546)  
**Code**: [GitHub](https://anonymous.4open.science/r/multi-com-1808)  
**Area**: Interpretability / Multi-Agent Safety  
**Keywords**: LLM multi-agent trust management, attention pattern analysis, trustworthiness evaluation, malicious message detection, trusted communication

## TL;DR

This paper proposes the first comprehensive "trustworthiness" definition for LLM Multi-Agent Systems (LLM-MAS) based on six orthogonal dimensions of Grice's Cooperative Principle. It discovers that LLM attention patterns can distinguish different types of trustworthiness violations. Accordingly, it designs A-Trust, a lightweight evaluation method and an end-to-end Trust Management System (TMS), improving malicious message detection rates to 77-90% under various attacks.

## Background & Motivation

**Background**: LLM-MAS has demonstrated powerful capabilities in tasks such as code generation, mathematical reasoning, and scientific simulation, with Agent2Agent (A2A) protocols becoming increasingly popular. However, LLM agents tend to accept all input information indiscriminately, lacking the information screening capability typically possessed by humans.

**Limitations of Prior Work**: (1) LLM-MAS is highly vulnerable to malicious message attacks, including Adversary-in-the-Middle (AiTM), Auto-Injection (AutoInject), and Cyber-Security (NetSafe) attacks; (2) existing work focuses only on a single aspect of trustworthiness (e.g., factual accuracy or logical consistency), lacking a systematic multi-dimensional evaluation framework; (3) prompt-based trust evaluation is prone to hallucinations, while external verification tools introduce latency and depend on internal data quality.

**Key Challenge**: LLM attention patterns actually "perceive" untrustworthy information (indicated by significantly higher attention weights), yet the model's final output fails to utilize this signal—a mismatch between attention patterns and output (a form of hallucination).

**Goal**: (1) Establish a multi-dimensional trustworthiness definition framework; (2) utilize LLM internal attention signals for efficient trustworthiness evaluation; (3) build an end-to-end trust management system.

**Key Insight**: By defining six trust dimensions based on Grice's Cooperative Principle, it was found that violations in different dimensions generate unique patterns in LLM attention, where specific attention heads are specialized for particular violation types.

**Core Idea**: Extract cross-layer aggregated feature vectors from LLM multi-head attention weights to train lightweight logistic regression models for scoring the six trust dimensions, constructing a message-level and agent-level trust management system.

## Method

### Overall Architecture

The system consists of three layers: (1) **Trust dimension definition + Trust Violation dataset**—a six-dimension framework and a controlled dataset with >20k samples; (2) **A-Trust Evaluation**—a lightweight classifier based on attention vectors; (3) **TMS Trust Management**—message-level evaluation → trust-aware action policy → agent-level trust records.

### Key Designs

1.  **Six-dimension Trust Framework + Trust Violation Dataset**:
    - **Function**: Provides systematic, interpretable trustworthiness evaluation standards and training data.
    - **Mechanism**: Defines six orthogonal trust dimensions based on Grice's Cooperative Principle: Factual Accuracy, Logical Consistency, Relevance, Bias, Quality, and Clarity. A dataset of >20k samples was constructed where each untrustworthy sample violates only one dimension. The factual dimension reuses FEVER, bias reuses StereoSet, and the remaining four were generated using GPT-4o.
    - **Design Motivation**: Untrustworthy samples in existing datasets often mix multiple violations, preventing fine-grained analysis. Single-dimension violations allow for the analysis of which attention head corresponds to which dimension.

2.  **A-Trust Attention Trust Evaluation**:
    - **Function**: Uses LLM internal signals for lightweight message trustworthiness evaluation.
    - **Mechanism**: For input message $M$ and context $C$, it extracts attention weights from the last token to the message tokens, aggregated across layers into a scalar for each head $Attn^h(M) = \text{Mean}(\{Attn^{l,h}(M)\}_{l=1}^{L})$, forming an $H$-dimensional feature vector. One-vs-rest logistic regression classifiers $(f_{\text{fact}}, ..., f_{\text{clarity}})$ are trained for each dimension, where the output $\in [0,1]$ serves as the trust score.
    - **Design Motivation**: Analysis reveals that LLMs assign higher attention to untrustworthy messages, and specific heads are sensitive to specific dimensions (e.g., head 2 for relevance, head 21 for clarity, head 27 for quality). Logistic regression is lightweight and efficient for real-time deployment.

3.  **Trust Management System (TMS)**:
    - **Function**: Implements end-to-end trust-aware communication in LLM-MAS.
    - **Mechanism**: Three components—(a) **Message-level evaluation**: Computes six-dimensional A-Trust scores for every inter-agent message; (b) **Trust-aware action policy**: Sets a threshold $\tau_r$ for each dimension; messages exceeding this are flagged and filtered. The factual dimension uses a dual-threshold strategy (uncertain cases trigger Gemini-2.0-flash + Google Search verification); (c) **Agent-level trust records**: Maintains timestamped records $R_{A_i}$, performing periodic agent-level evaluations via the violation rate $VR_r$ within a time window.
    - **Design Motivation**: Message-level evaluation enables real-time filtering, while agent-level records support long-term behavior tracking. Their combination identifies and isolates persistently malicious agents.

### Loss & Training

A-Trust utilizes logistic regression without complex training requirements. For each dimension, 1500 positive and 1500 negative samples were selected via one-vs-rest sampling. LLaMA3.1-8B-Instruct was used to extract attention matrices.

## Key Experimental Results

### Main Results

**Malicious Message Detection Rate (MDR) (Chain Topology, %)**

| Dataset | Method | AiTM ↑ | AutoTrans ↑ | AutoInj ↑ | NetSafe ↑ | Clean ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MMLUPhy | A-Trust | **84.3** | **77.5** | **90.1** | **79.6** | 7.1 |
| | PPL | 43.1 | 51.8 | 57.9 | 50.5 | 4.8 |
| | Prompt | 52.9 | 57.7 | 55.3 | 50.3 | 9.4 |
| MATH | A-Trust | **84.1** | **79.3** | **85.6** | **80.4** | 7.3 |
| | PPL | 52.4 | 54.2 | 53.1 | 57.3 | 4.7 |
| | Prompt | 62.9 | 60.7 | 61.8 | 64.9 | 7.5 |

**Attack Success Rate (ASR) (Chain Topology, %, lower is better)**

| Dataset | Configuration | Clean | AiTM | AutoTrans | AutoInj | NetSafe |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MMLUPhy | No trust | 41.7 | 92.5 | 69.3 | 62.6 | 67.7 |
| | A-Trust TMS | 43.8 | **14.1** | **51.4** | **47.1** | **46.4** |
| MATH | No trust | 47.6 | 93.6 | 69.8 | 67.2 | 68.2 |
| | A-Trust TMS | 50.6 | **18.4** | **54.2** | **55.2** | **59.2** |

### Ablation Study

- The agent-level trust record strategy achieves a 100% Malicious Agent Detection Rate (ADR) across all attacks.
- A-Trust demonstrates clear separation between violation and non-violation across all six dimensions, whereas prompt-based methods show almost no discriminative power in several dimensions due to hallucinations.
- A-Trust significantly outperforms PPL and Prompt baselines across Chain, Complete, and Tree topologies.

### Key Findings

- LLM attention to untrustworthy messages is significantly higher than to normal messages—the model "perceives" the anomaly but fails to utilize this information.
- Specific attention heads exhibit specialized responses to specific trust dimensions (head 2 → relevance, head 21 → clarity, head 27 → quality).
- Prompt-based trust evaluation suffers from severe hallucinations; internal attention signals are more reliable than model outputs.
- The ASR for AiTM attacks dropped from 92.5% to 14.1%, the largest decrease, proving that message patterns of DoS-type attacks are the easiest for the attention mechanism to recognize.

## Highlights & Insights

- The discovery that "LLMs perceive untrustworthiness but fail to utilize it" reveals a mismatch between attention mechanisms and outputs—a new form of hallucination.
- The six-dimension framework based on Grice's Cooperative Principle provides a standardized evaluation language for LLM-MAS safety research.
- The minimalist design of logistic regression combined with attention vectors outperforms GPT-4o prompt evaluation—simple methods prove superior to complex prompting in this context.

## Limitations & Future Work

- The Trust Violation dataset was generated by GPT-4o (for four dimensions), which may contain distributional bias.
- Adaptive attackers might learn to evade specific attention patterns, necessitating further robustness evaluation.
- Threshold strategies are relatively simple and do not account for correlations between different dimensions.
- A-Trust was trained only on LLaMA3.1-8B; cross-model generalization has not been fully verified.

## Related Work & Insights

- **vs Perplexity**: Perplexity is a single scalar and cannot distinguish between different types of trustworthiness violations; A-Trust provides fine-grained, six-dimension evaluation.
- **vs Prompt-based**: Affected by LLM hallucinations and unable to distinguish violations from normal messages in multiple dimensions.
- **vs FEVER/StereoSet**: These datasets cover only factual or bias dimensions; the Trust Violation dataset is the first six-dimension controlled benchmark.
- **vs NetSafe**: NetSafe focuses on attack methods, while this work focuses on defense—they are complementary.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First attention-based trust management framework for LLM-MAS; six-dimension definition fills a research gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 attacks × 4 datasets × 3 topologies + comparison with 3 baselines + agent-level evaluation + MetaGPT case study.
- Writing Quality: ⭐⭐⭐⭐ Complete framework with convincing visualization of attention analysis.
- Value: ⭐⭐⭐⭐⭐ Provides a practical and deployable security solution for LLM-MAS, especially important in the era of A2A protocols.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)
- [\[ACL 2026\] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey](llm-based_human-agent_collaboration_and_interaction_systems_a_survey.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)
- [\[ACL 2026\] MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing](masfactory_a_graph-centric_framework_for_orchestrating_llm-based_multi-agent_sys.md)

</div>

<!-- RELATED:END -->
