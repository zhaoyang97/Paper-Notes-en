---
title: >-
  [Paper Note] Federated In-Context Learning: Iterative Refinement for Improved Answer Quality
description: >-
  [ICML 2025][LLM Safety][Federated Learning] This paper proposes Fed-ICL, a federated In-Context Learning framework. By leveraging multi-round iterative collaboration between clients and the server, it progressively improves answer quality using high-quality examples scattered across clients without transmitting model parameters, while establishing theoretical convergence guarantees.
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "Federated Learning"
  - "In-Context Learning"
  - "Question Answering"
  - "Iterative Optimization"
  - "Communication Efficiency"
date: 2026-05-08
content_hash: 9d693aea58297c66
---

# Federated In-Context Learning: Iterative Refinement for Improved Answer Quality

**Conference**: ICML 2025  
**arXiv**: [2506.07440](https://arxiv.org/abs/2506.07440)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Federated Learning, In-Context Learning, Question Answering, Iterative Optimization, Communication Efficiency

## TL;DR
This paper proposes Fed-ICL, a federated In-Context Learning framework. By leveraging multi-round iterative collaboration between clients and the server, it progressively improves answer quality using high-quality examples scattered across clients without transmitting model parameters, while establishing theoretical convergence guarantees.

## Background & Motivation

**Background**: In-Context Learning (ICL) enables language models to perform tasks using in-context examples without modifying parameters. The effectiveness of ICL heavily depends on the quality and diversity of the examples. In practical scenarios, high-quality examples are often distributed across different client devices.

**Limitations of Prior Work**: Existing methods either require transmitting model parameters (e.g., FedAvg), which incurs massive communication overhead and is unsuitable for ultra-large language models, or only utilize local data for ICL, failing to exploit example diversity across clients. Centralizing scattered examples to the server violates data privacy principles.

**Key Challenge**: How to enable language models to benefit from the performance gains of distributed high-quality examples in ICL without transmitting model parameters or raw data?

**Goal**: Design a communication-efficient federated ICL method that leverages scattered client examples to continuously improve the answer quality of the central server.

**Key Insight**: Instead of transmitting model parameters or raw data, the framework transmits the generated "answers" from ICL. Through multi-round iterations, clients optimize the server's answers using their local examples.

**Core Idea**: The server broadcasts the current best answer to clients, which then use local examples as control contexts to refine the answer. The refined answers are returned to the server for aggregation, executing iteratively.

## Method

### Overall Architecture
Input: Question $q$, $K$ clients each holding a local example set $\{(q_i^k, a_i^k)\}$  
Output: The server's final answer $a^*$  

Each round of iteration:
1. **Server Broadcast**: Send the current answer $a^t$ to all clients
2. **Client Refinement**: Each client $k$ constructs an ICL prompt using local examples + the current answer to generate an improved answer $a_k^{t+1}$
3. **Server Aggregation**: Collect refined answers from all clients and apply an aggregation strategy to obtain $a^{t+1}$

### Key Designs

1. **Iterative Refinement Mechanism**:

    - **Function**: Progressively improve answer quality through multi-round interactions
    - **Mechanism**: The prompt is constructed as: $[\text{Question}: q, \text{Current Answer}: a^t, \text{Examples}: \{(q_i, a_i)\}, \text{Instruction}: \text{Improve the answer}]$
    - **Design Motivation**: Single-round ICL is limited by the quality of examples on a single client, whereas the iterative mechanism allows the knowledge of multiple clients to fuse progressively. Each round of iteration acts as an optimization step within the "answer space".

2. **Answer Aggregation Strategy**:

    - **Function**: Combine multiple refined answers from clients into a better unified answer at the server
    - **Mechanism**: Supports multiple aggregation methods—(i) Voting Aggregation: Majority voting to choose the best answer, (ii) LLM Aggregation: Utilizing an LLM to synthesize multiple answers into a final response, (iii) Scoring Aggregation: Utilizing an LLM to score each answer followed by a weighted average
    - **Design Motivation**: Different aggregation strategies suit different tasks—voting is suitable for classification tasks, while LLM aggregation is ideal for open-ended QA

3. **Convergence Guarantee**:

    - **Function**: Prove that the iterative refinement process of Fed-ICL converges under certain conditions
    - **Mechanism**: Model answer quality as a potential function and prove that the refinement step in each round satisfies a diminishing condition: $V(a^{t+1}) - V(a^t) \geq -c \cdot \|a^t - a^*\|^2$
    - **Design Motivation**: Theoretical guarantees ensure that Fed-ICL is not just an empirically effective heuristic, but an algorithm backed by rigorous theory

### Loss & Training
Model training is not involved. The core metrics are answer quality (e.g., BLEU, ROUGE, EM score). The communicated content consists solely of text answers, leading to extremely low communication overhead.

## Key Experimental Results

### Main Results

| Dataset | Metric | Fed-ICL | Single-Client ICL | FedAvg | Centralized ICL |
|--------|------|---------|-------------|--------|-----------|
| NQ (Natural Questions) | EM | **47.3** | 38.2 | 43.8 | 49.1 |
| TriviaQA | EM | **62.8** | 51.4 | 58.3 | 65.2 |
| SQuAD 2.0 | F1 | **78.5** | 68.9 | 74.2 | 80.1 |
| WebQuestions | EM | **43.7** | 35.1 | 40.6 | 45.9 |

### Ablation Study

| Configuration | NQ EM | Communication Volume (Relative) | Description |
|------|-------|-------------|------|
| Fed-ICL (5 rounds) | **47.3** | 1x | Best performance |
| Fed-ICL (1 round) | 41.5 | 0.2x | Single round is insufficient to fuse multi-client knowledge |
| Fed-ICL (10 rounds) | 47.6 | 2x | Diminishing marginal returns |
| Voting Aggregation | 44.8 | 1x | Simple but loses information |
| LLM Aggregation | **47.3** | 1x | Best aggregation strategy |
| FedAvg (LLaMA-7B) | 43.8 | 1000x+ | Transmits model parameters, communication volume is much larger |

### Key Findings
- Fed-ICL approaches the performance of Centralized ICL on QA tasks (with a gap of 2-4 EM) but reduces communication volume by 3 orders of magnitude.
- Iterative refinement is key—5 rounds of interaction yield an improvement of approximately 6-10 EM.
- LLM Aggregation outperforms simple voting aggregation, as it synthesizes multiple refinement directions better.
- Fed-ICL scales well with the number of clients, showing stable performance across 10-50 clients.

## Highlights & Insights
- **Paradigm Innovation**: The federated ICL idea of "transmitting answers instead of parameters" is highly novel and practical.
- **Extremely Low Communication Overhead**: The communicated content consists solely of brief texts, which is thousands of times more efficient than transmitting model parameters.
- **Privacy-Friendly**: Refraining from transmitting raw data and model parameters naturally preserves privacy.
- **Model-Agnostic**: Compatible with any LLM that supports ICL, without requiring model modifications.

## Limitations & Future Work
- Iterative refinement of answers may leak indirect information (e.g., the distribution of answers may reflect client data characteristics).
- The applicability to non-QA tasks (e.g., generation, translation) has yet to be verified.
- The choice of aggregation strategy requires manual tuning based on the specific task.
- The assumptions supporting the theoretical guarantees (e.g., the probabilistic model of LLM refining answers) may not be fully satisfied in practice.

## Related Work & Insights
- FedAvg (McMahan et al., 2017): Classical federated learning.
- Self-Refine (Madaan et al., 2023): Single-model self-refinement.
- Ours extends the concept of self-refinement to the federated setting, presenting a meaningful cross-disciplinary innovation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Federated ICL" is a brand-new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 QA datasets, with extensive ablation of different aggregation strategies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and solid experimental design.
- Value: ⭐⭐⭐⭐ Holds practical significance for privacy-preserving LLM applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks](iclshield_exploring_and_mitigating_in-context_learning_backdoor_attacks.md)
- [\[ICML 2025\] POPri: Private Federated Learning using Preference-Optimized Synthetic Data](popri_private_federated_learning_using_preference-optimized_synthetic_data.md)
- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](../../NeurIPS2025/llm_safety/fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [\[ICCV 2025\] Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning](../../ICCV2025/llm_safety/geminio_language-guided_gradient_inversion_attacks_in_federated_learning.md)
- [\[ACL 2025\] PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative Prompts](../../ACL2025/llm_safety/pig_privacy_jailbreak.md)

</div>

<!-- RELATED:END -->
