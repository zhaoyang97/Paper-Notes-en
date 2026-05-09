---
title: >-
  [Paper Note] Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards
description: >-
  [NeurIPS 2025][RAG consistency] This paper proposes Con-RAG, a framework that trains RAG generators to produce informationally consistent outputs under paraphrased inputs by computing group similarity rewards across multiple generations of semantically equivalent queries via Paraphrased Set GRPO (PS-GRPO), simultaneously improving both consistency and accuracy without requiring explicit ground-truth supervision.
tags:
  - NeurIPS 2025
  - RAG consistency
  - GRPO
  - information consistency
  - semantically equivalent queries
  - reinforcement learning alignment
date: 2026-05-08
content_hash: 9ff86321dc96a391
---

# Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards

**Conference**: NeurIPS 2025
**arXiv**: [2510.04392](https://arxiv.org/abs/2510.04392)
**Code**: To be confirmed
**Area**: Information Retrieval
**Keywords**: RAG consistency, GRPO, information consistency, semantically equivalent queries, reinforcement learning alignment

## TL;DR

This paper proposes Con-RAG, a framework that trains RAG generators to produce informationally consistent outputs under paraphrased inputs by computing group similarity rewards across multiple generations of semantically equivalent queries via Paraphrased Set GRPO (PS-GRPO), simultaneously improving both consistency and accuracy without requiring explicit ground-truth supervision.

## Background & Motivation

RAG systems are widely deployed in high-stakes domains (medical, financial, legal), yet suffer from severe output inconsistency:

**Retrieval-side inconsistency**: Semantically equivalent queries (e.g., "how to close a savings account" vs. "what are the steps to close a savings account") may retrieve different document sets, leading to divergent downstream generations.

**Generation-side inconsistency**: Even given identical retrieved documents, LLMs are sensitive to phrasing variations and may produce different answers.

**Trust crisis**: In domains such as finance, divergent answers to the same question severely undermine user trust and regulatory compliance.

Existing work primarily focuses on RAG accuracy and faithfulness, while **information consistency**—ensuring outputs convey the same core content under semantically equivalent inputs—has been largely overlooked. Unlike lexical consistency, which may penalize legitimate paraphrasing, information consistency focuses on factual-level agreement.

Key observation: In short-form QA, consistency and accuracy are positively correlated; in long-form QA, they are orthogonal—a model may be accurate yet inconsistent.

## Method

### Overall Architecture

Con-RAG comprises two components: (1) a hierarchical consistency evaluation framework for diagnosing sources of inconsistency, and (2) PS-GRPO, a training method that optimizes the generator via group similarity rewards.

### Consistency Measurement Framework

RAG consistency is decomposed into three levels:

1. **Retriever consistency**: Jaccard similarity between document sets retrieved by paraphrased queries:
$$\mathcal{C}_{\text{ret}}(q_0) = \frac{2}{n(n-1)} \sum_{i,j} \frac{|R(p_i) \cap R(p_j)|}{|R(p_i) \cup R(p_j)|}$$

2. **End-to-end consistency**: Similarity between outputs produced by the full RAG pipeline under each paraphrased input:
$$\mathcal{C}_{\text{e2e}}(q_0) = \frac{1}{n(n-1)} \sum_{i \neq j} \text{sim}(y_i, y_j)$$

3. **Generator consistency**: With retrieved documents fixed, measures the LLM's intrinsic stability under input phrasing variation.

### Key Design: Paraphrased Set GRPO (PS-GRPO)

PS-GRPO extends the GRPO (Group Relative Policy Optimization) framework. The core idea is: given $n$ paraphrases $\{p_1,...,p_n\}$ of a canonical query $q_0$, each paraphrase generates $g$ rollouts, forming an $n \times g$ output matrix.

**Group similarity reward**: The reward for each rollout $o_{ij}$ is computed as the average similarity against all rollouts from other paraphrases:

$$r_{ij} = \frac{1}{(n-1)g} \sum_{\substack{u=1 \\ u \neq i}}^{n} \sum_{m=1}^{g} \text{sim}(o_{ij}, o_{um})$$

BLEU is used as the similarity function in practice, as experiments confirm it outperforms ROUGE-L, Exact Match, and other alternatives.

**Joint reward with accuracy** (when ground-truth labels are available):

$$r_{ij}^{\text{final}} = \alpha \cdot r_{ij}^{\text{cons}} + \gamma \cdot \text{Acc}(o_{ij}, y^\star)$$

where Acc is measured by token F1. For open-ended long-form tasks, only the consistency reward is used.

**Advantage normalization**: Normalization is performed within each paraphrase group as $\hat{A}_{ij} = (r_{ij} - \mu_i) / \sigma_i$, followed by the standard GRPO clipped objective for policy updates:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{g} \sum_{i=1}^{g} \sum_{t=1}^{|o_i|} \min(\rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

### Efficient Approximation

Naïve computation requires $n(n-1)g^2$ similarity comparisons (720 per query when $n=5, g=6$). The paper introduces a **relaxed approximation**: for each rollout, only $\kappa$ paraphrases and $s$ rollouts per paraphrase are randomly sampled:

$$\tilde{r}_{ij} = \frac{1}{\kappa s} \sum_{u \in K} \sum_{m \in S_k} \text{sim}(o_{ij}, o_{um})$$

Complexity is reduced from $O(n(n-1)g^2)$ to $O(ng\kappa s)$, where $\kappa \ll n-1$ and $s \ll g$.

### Experimental Setup

- Generators: LLaMA-3.1-8B, Qwen-2.5-3B
- Retriever: e5-base-v2, KILT Wikipedia corpus, top-$k$=5
- Paraphrase generation: LLaMA-3.1-70B generates $n=6$ paraphrases per query
- Training: $g=4$ rollouts per paraphrase, $\kappa=3$, $s=1$, AdamW lr=1e-6

## Key Experimental Results

### Main Results: Short-Form QA Consistency and Accuracy (LLaMA-3.1-8B)

| Dataset | Method | EM↑ | F1↑ | RM↑ | E2E Consistency (Lexical)↑ | E2E Consistency (LLM-Judge)↑ | Generator Consistency (Lexical)↑ |
|--------|------|-----|-----|-----|---------------------|----------------------|---------------------|
| TriviaQA | RAG | 56.0 | 66.1 | 74.0 | 53.0 | 77.8 | 67.3 |
| TriviaQA | DRAG | 54.0 | 63.7 | 72.0 | 56.8 | 78.7 | 68.2 |
| TriviaQA | SFT | 24.0 | 27.5 | 29.0 | 51.3 | 58.2 | 77.8 |
| TriviaQA | **Con-RAG** | **77.0** | **81.0** | **83.0** | **87.3** | **91.3** | **91.2** |
| HotpotQA | RAG | 37.0 | 44.1 | 42.0 | 42.5 | 62.5 | 53.7 |
| HotpotQA | **Con-RAG** | **45.0** | **51.9** | **48.0** | **63.9** | **73.6** | **80.9** |
| MuSiQue | RAG | 8.0 | 15.3 | 12.0 | 27.9 | 48.2 | 44.4 |
| MuSiQue | **Con-RAG** | **23.0** | **30.8** | **25.0** | **72.5** | **72.3** | **91.4** |

On TriviaQA, Con-RAG improves end-to-end consistency from 53.0→87.3 (lexical) and 77.8→91.3 (LLM-Judge), while simultaneously improving accuracy from 56.0→77.0 EM.

### Long-Form Results (ELI5, No Ground-Truth Supervision)

| Method | ROUGE↑ | LLM-Acc↑ | E2E Consistency (Lexical)↑ | E2E Consistency (Info)↑ | Generator Consistency (Lexical)↑ | Generator Consistency (Info)↑ |
|------|--------|----------|---------------------|------------------|---------------------|------------------|
| RAG | 21.9 | 74.0 | 8.6 | 62.8 | 15.1 | 74.2 |
| DRAG | 22.0 | 76.0 | 8.0 | 62.2 | 15.0 | 72.5 |
| SFT | 23.5 | 51.0 | 15.3 | 40.8 | 16.6 | 41.7 |
| **Con-RAG** | **24.2** | **78.0** | **14.6** | **72.7** | **21.7** | **80.8** |

Without ground-truth labels, Con-RAG improves both accuracy (74→78 LLM-Acc) and consistency (62.8→72.7 information consistency) using only the consistency reward.

### Ablation Study: Reward Similarity Metric Comparison (ELI5, Qwen-2.5-3B)

| Metric | LLM-Acc↑ | E2E Consistency (LLM-Judge)↑ | Generator Consistency (LLM-Judge)↑ |
|------|----------|----------------------|----------------------|
| BLEU-1 | 54.0 | 38.2 | 69.8 |
| **BLEU-2** | **58.0** | **42.0** | **67.5** |
| BLEU-3 | 49.0 | 36.3 | 66.0 |
| ROUGE-L | 46.0 | 35.2 | 65.2 |
| Exact Match | 49.0 | 37.7 | 66.2 |

Lower-order BLEU (1–2) performs best, emphasizing lexical choice and local fluency in a manner well-aligned with the information consistency objective.

### Key Findings

1. **The retriever is the primary source of inconsistency**: Jaccard consistency is only 27–52%, indicating that paraphrased queries frequently retrieve non-overlapping documents.
2. **Consistency training implicitly enhances accuracy**: Con-RAG improves EM, F1, and RM simultaneously across all short-form QA datasets, likely benefiting from the data augmentation effect of paraphrase-based training.
3. **SFT performs poorly on open-ended tasks**: On ELI5, SFT achieves only 51.0 LLM-Judge accuracy, far below RAG's 74.0, demonstrating that rigid supervised training is ill-suited for open-ended QA.
4. **Joint consistency + F1 reward is optimal**: Training with either reward alone is inferior to joint training (TriviaQA EM: 51.5 vs. 54.0 vs. 60.0).

## Highlights & Insights

1. **Precise problem definition**: The paper clearly distinguishes information consistency from lexical consistency—the former tolerates legitimate paraphrasing while the latter is overly strict.
2. **Hierarchical diagnostic framework**: Decomposing RAG inconsistency into retriever, generator, and end-to-end layers enables precise bottleneck identification.
3. **Novel application of GRPO**: The method cleverly leverages GRPO's multi-rollout structure, forming larger comparison groups across paraphrases to compute similarity rewards.
4. **Unsupervised accuracy gains**: The ELI5 experiments demonstrate that consistency rewards alone (without ground-truth labels) can improve accuracy, suggesting that consistency serves as a proxy signal for accuracy.
5. **Computational efficiency**: The relaxed approximation reduces reward computation from quadratic to linear complexity, enabling large-scale training.

## Limitations & Future Work

1. **Paraphrase generation depends on a strong LLM**: High-quality paraphrase generation requires LLaMA-70B, which incurs non-trivial cost.
2. **Retriever left unoptimized**: The framework only optimizes the generator; retrieval-side inconsistency remains (Jaccard similarity only 27–52%).
3. **BLEU reward may harm generalization**: Over-optimization of surface-level similarity may sacrifice response diversity.
4. **Evaluation limitations**: The optimal trade-off between consistency and accuracy may vary across tasks, and a unified standard is lacking.
5. **Limited model scale coverage**: Only 3B and 8B models are evaluated; performance on larger models remains unverified.

## Related Work & Insights

- GRPO (Shao et al., 2024) serves as the algorithmic foundation; PS-GRPO is a natural extension along the consistency dimension.
- Unlike Self-Consistency (Wang et al., 2023), Con-RAG improves consistency through training rather than inference-time sampling alone.
- The group reward mechanism suggests a natural extension to multilingual consistency (responses to the same question across languages should be consistent).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Extending GRPO to cross-paraphrase consistency optimization is an elegant design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers short-form, multi-hop, and long-form QA with two model sizes and multiple baselines.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to high-stakes RAG deployments; the relaxed approximation ensures scalability.
- **Writing Quality**: ⭐⭐⭐⭐ — Architecture diagrams are clear and theory–experiment correspondence is well maintained.
- **Overall**: ⭐⭐⭐⭐ — Fills a gap in RAG consistency optimization with solid methodology and experimental validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](../../ACL2026/information_retrieval/end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[NeurIPS 2025\] Chain-of-Retrieval Augmented Generation (CoRAG)](chain-of-retrieval_augmented_generation.md)
- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)
- [\[NeurIPS 2025\] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)
- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](benchmarking_retrievalaugmented_multimodal_generation_for_do.md)

</div>

<!-- RELATED:END -->
