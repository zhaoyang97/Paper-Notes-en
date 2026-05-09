---
title: >-
  [Paper Note] ReliabilityRAG: Effective and Provably Robust Defense for RAG-based Web-Search
description: >-
  [NeurIPS 2025][LLM Safety][RAG security] ReliabilityRAG proposes a RAG framework that leverages document reliability signals (e.g., search ranking) for adversarial defense. It identifies a consistent document subset by finding the Maximum Independent Set (MIS) on a contradiction graph while prioritizing high-reliability documents, providing provable robustness guarantees alongside high accuracy on benign scenarios and long-form generation tasks.
tags:
  - NeurIPS 2025
  - LLM Safety
  - RAG security
  - adversarial robustness
  - maximum independent set
  - provable defense
  - document reliability
date: 2026-05-08
content_hash: 4280ed7add001792
---

# ReliabilityRAG: Effective and Provably Robust Defense for RAG-based Web-Search

**Conference**: NeurIPS 2025
**arXiv**: [2509.23519](https://arxiv.org/abs/2509.23519)
**Code**: None
**Area**: AI Security / RAG Defense
**Keywords**: RAG security, adversarial robustness, maximum independent set, provable defense, document reliability

## TL;DR

ReliabilityRAG proposes a RAG framework that leverages document reliability signals (e.g., search ranking) for adversarial defense. It identifies a consistent document subset by finding the Maximum Independent Set (MIS) on a contradiction graph while prioritizing high-reliability documents, providing provable robustness guarantees alongside high accuracy on benign scenarios and long-form generation tasks.

## Background & Motivation

RAG (Retrieval-Augmented Generation) enhances LLM outputs with timeliness and accuracy by retrieving external documents and has been widely adopted in search products such as Google AI Overview, Bing Chat, and Perplexity. However, the retrieval corpus of RAG systems is vulnerable to adversarial attacks including corpus poisoning and prompt injection.

**Limitations of Prior Work**:

**RobustRAG**: Majority voting based on keywords or next-token probabilities incurs large information loss, yields poor performance on benign scenarios, and degrades severely on long-form generation tasks.

**AstuteRAG / InstructRAG**: Designed for benign performance rather than adversarial robustness, offering insufficient defense under attack.

**Shared Blind Spot**: All existing defenses treat retrieved documents as an unordered set, ignoring the ranking/reliability signals provided by retrieval systems.

**Key Opportunity**: RAG search systems inherently contain document ranking signals refined over decades of anti-SEO optimization. Lower-ranked documents are more likely to be noisy and more susceptible to adversarial manipulation. Exploiting this reliability signal enables a defense-in-depth strategy.

**Core Idea**: From a graph-theoretic perspective, identify the "consistent majority" among retrieved documents — finding the Maximum Independent Set on a contradiction graph, i.e., the largest subset of mutually non-contradictory documents, while prioritizing higher-ranked documents.

## Method

### Overall Architecture

ReliabilityRAG operates in three stages:
1. **Retrieval**: Obtain $k$ documents ranked by reliability.
2. **Ranking-Aware MIS Selection**: Construct a contradiction graph, find the Maximum Independent Set, and prioritize high-ranked documents.
3. **Query**: Use the selected document subset to query the LLM for answer generation.

### Key Designs

1. **Contradiction Graph Construction**:

    - **Isolated Answering**: Query the LLM independently for each document $x_i$ to obtain an isolated answer $y_i$.
    - **Contradiction Detection**: Apply an NLI model (DeBERTa-v3-large) to each answer pair $(y_i, y_j)$ to determine whether they contradict each other (threshold $\beta = 0.5$).
    - **Graph Encoding**: Documents serve as nodes (inheriting retrieval ranks), contradiction relations serve as edges, forming an undirected graph $G = (V, E)$.

2. **Ranking-Aware MIS Search**:

    - Enumerate all $2^{|V|}$ subsets (completable in milliseconds for $k \leq 20$) and filter for independent sets.
    - Select the largest independent set; if multiple MISs of equal size exist, select the lexicographically smallest one (i.e., preferring higher-ranked documents).
    - $S^* = \arg\max_{S \text{ independent}} (|S|, -\text{lex}(S))$
    - Generate the final answer using documents in $S^*$.

3. **Weighted Sampling Aggregation Framework (Sampling + MIS)**:

    - A scalable scheme for large-scale retrieval ($k > 20$).
    - Each round samples $m$ documents according to document weights to form a context and generates an intermediate answer.
    - After $T$ rounds of sampling, MIS is applied to aggregate the intermediate answers.
    - Weights follow exponential decay: $w(x_i) \propto \gamma^{i-1}$ ($\gamma = 0.9$).
    - Reduces MIS computation from $k$ documents to $T$ intermediate answers.

### Provable Robustness Guarantees

**Theorem 1**: Assume the adversary corrupts at most $k' \leq k/5$ documents, the NLI model's error rate among benign documents is $\leq \epsilon_1$, and the error rate between benign and malicious documents is $\leq \epsilon_2$. When $\epsilon_1, \epsilon_2$ satisfy certain conditions, the probability that the MIS contains no malicious documents is at least $1 - e^{-O(k)}$. That is, as the number of retrieved documents $k$ increases, the probability of mistakenly selecting a malicious document decreases exponentially.

Under a perfect NLI model ($\epsilon_1 = \epsilon_2 = 0$), as long as $k' < k/2$, the MIS is exactly the set of all benign documents.

## Key Experimental Results

### Main Results ($k=10$, Prompt Injection Attack)

| Model | Method | RQA (Benign) | RQA (@Pos1) | RQA (@Pos10) | Bio (Benign) | Bio (@Pos1) | Bio (@Pos10) |
|-------|--------|-------------|-------------|-------------|-------------|-------------|-------------|
| Mistral-7B | Vanilla RAG | 64% | 49% | 12% | 72.9 | 65.5 | 11.5 |
| Mistral-7B | RobustRAG | 56% | 53% | 55% | 58.6 | 56.5 | 57.1 |
| Mistral-7B | **MIS** | **70%** | **68%** | **60%** | **73.5** | **69.7** | **71.5** |
| GPT-4o-mini | Vanilla RAG | 77% | 49% | 64% | 81.0 | 65.6 | 9.8 |
| GPT-4o-mini | RobustRAG | 71% | 68% | 70% | 61.2 | 60.4 | 61.4 |
| GPT-4o-mini | **MIS** | **76%** | **70%** | **76%** | **80.1** | **77.9** | **79.0** |

### Long-Form Generation Performance (Bio Dataset)

| Method | Llama3.2-3B Benign | Llama3.2-3B @Pos1 | Llama3.2-3B @Pos10 |
|--------|--------------------|-------------------|-------------------|
| RobustRAG | 56.0 | 53.0 | 51.9 |
| AstuteRAG | 62.7 | 46.7 | 38.6 |
| **MIS** | **73.0** | **71.0** | **72.1** |

### Key Findings
- MIS matches or exceeds Vanilla RAG accuracy on benign scenarios, substantially outperforming RobustRAG.
- Advantages on long-form generation (Bio) are substantial: MIS 73.5 vs. RobustRAG 58.6 (Mistral-7B, benign).
- Ranking-awareness is evident: defense performance is generally stronger when attacks target lower-ranked documents (Pos 10).
- Sampling + MIS scales to $k=50$ documents while maintaining comparable robustness.
- MIS consistently achieves best performance across all three LLMs (Mistral-7B, Llama3.2-3B, GPT-4o-mini).

## Highlights & Insights

- **Graph-theoretic solution to an NLP problem**: Reformulating "selecting a reliable subset from multiple documents" as MIS on a contradiction graph is both elegant and theoretically grounded.
- **Leveraging existing signals rather than constructing new ones**: Search ranking is a readily available reliability indicator, exploited here to build defense-in-depth.
- **Dual benefit of benign performance and robustness**: Unlike RobustRAG, which trades benign performance for robustness, MIS achieves both simultaneously.
- **Isolated answering strategy**: Querying the LLM independently per document prevents cross-document contamination, enabling more precise contradiction detection.
- **Provable robustness**: Under reasonable assumptions, an exponentially decaying risk bound is established — going beyond purely empirical defense.

## Limitations & Future Work

- Isolated answering requires a separate LLM call per document; with $k=10$, this incurs 10 additional LLM calls, resulting in non-trivial latency overhead.
- The NLI model (DeBERTa-v3-large) may itself be bypassed by adversarial attacks.
- MIS computation for $k > 20$ requires approximate methods, partially sacrificing theoretical guarantees.
- The framework assumes adversaries aim to produce outputs contradicting benign documents — defense may be insufficient against stealthier attacks that align with benign content while conveying misleading information.
- Validation is primarily on QA datasets; applicability to multi-step reasoning, code generation, and other scenarios remains to be explored.
- Irrelevant but benign documents require an additional "I don't know" filtering mechanism.

## Related Work & Insights

- **vs. RobustRAG**: RobustRAG relies on keyword/token-level voting, incurring large information loss; MIS detects contradictions at the sentence level, preserving substantially more information.
- **vs. AstuteRAG**: AstuteRAG addresses knowledge conflicts but is not designed for adversarial settings; MIS targets adversarial robustness with theoretical guarantees.
- **vs. InstructRAG**: InstructRAG instructs the LLM to self-denoise, relying on LLM capability; MIS employs an independent NLI model, offering greater reliability.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of graph theory, NLI, and reliability signals is highly original; provable robustness in the RAG domain is pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three LLMs × multiple QA datasets + long-form generation + multiple attack positions.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is rigorous, theoretical proofs are complete, and experimental interpretation is clear.
- Value: ⭐⭐⭐⭐⭐ Significant practical implications for secure RAG deployment, particularly in search engine scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Adversarial Robust Memory-Based Continual Learner](../../ICCV2025/llm_safety/adversarial_robust_memory-based_continual_learner.md)
- [\[ICLR 2026\] SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense](../../ICLR2026/llm_safety/shield_suppressing_hallucinations_in_lvlm_encoders_via_bias_and_vulnerability_de.md)
- [\[CVPR 2026\] Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP](../../CVPR2026/llm_safety/perturb_and_recover_fine-tuning_for_effective_backdoor_removal_from_clip.md)
- [\[AAAI 2026\] Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting](../../AAAI2026/llm_safety/learning_from_the_undesirable_robust_adaptation_of_language_models_without_forge.md)
- [\[ICLR 2026\] Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning](../../ICLR2026/llm_safety/erase_or_hide_suppressing_spurious_unlearning_neurons_for_robust_unlearning.md)

<!-- RELATED:END -->
