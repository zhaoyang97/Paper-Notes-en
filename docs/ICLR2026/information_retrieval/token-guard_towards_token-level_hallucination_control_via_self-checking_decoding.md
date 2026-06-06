---
title: >-
  [Paper Note] Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding
description: >-
  [ICLR 2026][Information Retrieval & RAG][LLM hallucination control] This paper proposes Token-Guard, a token-level hallucination control method based on self-checking decoding…
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "LLM hallucination control"
  - "token-level decoding"
  - "self-checking"
  - "segment-level scoring"
  - "iterative refinement"
date: 2026-05-08
content_hash: 70ad35aefeabba05
---

# Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding

**Conference**: ICLR 2026
**arXiv**: [2601.21969](https://arxiv.org/abs/2601.21969)  
**Code**: [https://github.com/rhq945/Token-Guard](https://github.com/rhq945/Token-Guard)  
**Area**: Information Retrieval
**Keywords**: LLM hallucination control, token-level decoding, self-checking, segment-level scoring, iterative refinement

## TL;DR

This paper proposes Token-Guard, a token-level hallucination control method based on self-checking decoding, which detects and suppresses hallucinations during decoding via token-level/segment-level scoring in the hidden space and an iterative refinement mechanism, achieving an average F1 improvement of 16.3%.

## Background & Motivation

- **LLM hallucination problem**: Large models frequently generate content inconsistent with the input, which is especially severe in knowledge-intensive scenarios.
- **Limitations of Prior Work**:
    - RAG and RLHF require expensive external retrieval or large-scale fine-tuning.
    - Existing decoding methods (CoT, ToT, etc.) lack explicit token-level hallucination checking mechanisms.
    - Hallucination risk is not explicitly quantified, and token selection lacks directional guidance.
    - Most methods support only single-pass generation and lack dynamic refinement capability.
- **Core Problem**: How to achieve fine-grained hallucination control during decoding with low overhead?

## Method

### Overall Architecture

Token-Guard comprises three levels of hallucination control:
1. Token-level self-checking → 2. Segment-level representation and scoring → 3. Global iterative refinement

### Key Design 1: Token-Level Hallucination Self-Checking

A composite hallucination score is computed for each candidate token:

$$F_{\text{halu}}^{\text{token}}(a_t^{(i)} \mid s_t) = \lambda \cdot \frac{h_t^{(i)} \cdot \bar{h}_{<t}}{|h_t^{(i)}| |\bar{h}_{<t}|} + (1-\lambda) \cdot P(a_t^{(i)} \mid a_{<t}, x)$$

- First term: cosine similarity between the candidate token's hidden state and the mean hidden state of accepted tokens (semantic consistency).
- Second term: model-assigned conditional probability (token probability).
- $\lambda = 0.6$ balances the two terms.
- Threshold $\tau_{\text{token}} = 0.4$; tokens below the threshold are discarded.

Hidden states are taken from the model's second-to-last layer $\text{LLM}_{\text{hidden}}^{(L-1)}$; the mean hidden state of the input context serves as the anchor for the first token.

### Key Design 2: Segment-Level Candidate Representation and Scoring

Candidate segments $C_k$ are formed from consecutive tokens, and segment representations are computed via weighted averaging:

$$H_k = \sum_{i=1}^{n} w_i h_t^{(i)}, \quad w_i = \frac{\exp(F_{\text{halu}}^{\text{token}}(a_{t_i} \mid s_{t_i}))}{\sum_j \exp(F_{\text{halu}}^{\text{token}}(a_{t_j} \mid s_{t_j}))}$$

The segment-level score integrates three dimensions:

$$F_{\text{halu}}^{\text{seg}}(C_k) = \alpha F_{\text{halu}}^{\text{token}}(C_k) + \beta \text{Consistency}(C_k) + \gamma \text{Alignment}(C_k)$$

- **Token aggregation**: weighted token reliability ($\alpha = 0.5$)
- **Local consistency**: smoothness of adjacent token hidden states ($\beta = 0.3$)
- **Global alignment**: semantic alignment with the input context ($\gamma = 0.2$)

Segment-level thresholds: $\tau_{\text{seg}}^{\text{low}} = 0.55$ (discard), $\tau_{\text{seg}}^{\text{high}} = 0.75$ (accept); segments in between undergo local refinement.

### Key Design 3: Local Refinement and Global Iteration

**Local refinement**: The lowest-scoring token and its neighboring window $W_k^{(l)}$ within a segment are identified, and the LLM regenerates replacement tokens conditioned on the surrounding context:

$$W_k^{(l)'} = \text{LLM\_refine}(W_k^{(l)} \mid a_{<i-1}, a_{>i+1}, H_k)$$

**Global iteration**: Reliable segments are assembled into a reasoning chain $R$, and a global score is computed:

$$F_{\text{global}}(R) = \frac{F_{\text{fact}}(R) \cdot F_{\text{logic}}(R)}{F_{\text{fact}}(R) + F_{\text{logic}}(R) - F_{\text{fact}}(R) \cdot F_{\text{logic}}(R)}$$

If $F_{\text{global}} < 0.7$, global regeneration is triggered; if both $F_{\text{fact}}$ and $F_{\text{logic}}$ fall below 0.5, the system outputs "unable to answer."

### Memory Efficiency

- Token level: only the running mean $\bar{h}_{<t}$ is maintained; complexity $\mathcal{O}(L_{\max} \cdot K_{\text{active}} \cdot d)$.
- Segment level: temporary hidden states are released after segment formation; only compact segment vectors are retained.
- Global level: only segment vectors $\{H_k\}$ are operated on; complexity $\mathcal{O}(K \cdot d)$.

## Key Experimental Results

### Main Results (Meta-Llama-3.1-8B-Instruct)

| Method | FinanceBench F1 | DROP_hist F1 | DROP_nfl F1 | HaluEval F1 | Avg F1 |
|--------|----------------|-------------|-------------|-------------|--------|
| BaseModel | 16.00 | 44.21 | 39.10 | 42.16 | 28.29 |
| Guided Decoding | 16.44 | 55.95 | 36.71 | 57.41 | 34.73 |
| Chain-of-Thoughts | 11.01 | 49.26 | 49.21 | 55.32 | 34.63 |
| Tree-of-Thought | 14.44 | 47.73 | 37.69 | 56.02 | 33.33 |
| **Token-Guard** | **30.80** | **68.52** | **58.10** | **78.54** | **51.03** |

### Qwen3-8B Results

| Method | Avg EM | Avg F1 |
|--------|--------|--------|
| BaseModel | 0.22 | 44.25 |
| CoT | 0.23 | 45.10 |
| **Token-Guard** | **0.35** | **53.98** |

### Ablation Study

| Variant | DROP_hist F1 | RAGTruth F1 | Avg BLEU |
|---------|-------------|-------------|----------|
| Full Token-Guard | **68.52** | **43.94** | **51.74** |
| w/o Token-Level | 47.51 | 27.10 | 34.97 |
| w/o Segment-Level | 60.10 | 39.20 | 46.32 |
| w/o Global Iteration | 63.05 | 41.05 | 36.26 |
| w/o Prompt | 55.23 | 32.50 | 39.70 |

### Key Findings

- Token-level scoring contributes most to performance (removing it causes the largest F1 drop).
- Global iteration primarily improves BLEU (linguistic fluency), with additional contributions to EM/F1.
- The advantage is greatest on tasks requiring multi-step reasoning (DROP_nfl).
- Improvements are limited on knowledge-intensive tasks (PubMedQA), as the method cannot compensate for missing domain knowledge.
- The approach is effective across both backbone models (Llama3.1-8B and Qwen3-8B).

## Highlights & Insights

- **Multi-level hallucination control**: A three-tier token→segment→global hierarchy that balances precision and efficiency.
- **No external resources required**: No retrieval system or additional training is needed; the method operates purely at the decoding stage.
- **Modular design**: Can be integrated as a plug-in into any LLM decoding pipeline.
- **Memory-friendly**: Careful state management keeps memory usage independent of generation length.

## Limitations & Future Work

- Multi-level scoring introduces additional computational overhead (each token requires multiple hidden state computations and cosine similarity calculations).
- The method involves numerous hyperparameters ($\lambda$, $\tau_{\text{token}}$, $\alpha/\beta/\gamma$, $\tau_{\text{seg}}$, $\tau_{\text{global}}$, etc.), making tuning complex.
- The hidden-state-similarity-based hallucination detection assumes "consistency with context = factual correctness," which may fail when the model itself contains erroneous knowledge.
- Validation is limited to 8B-scale models; applicability to larger or smaller models remains unknown.
- Global iteration relies on TF-IDF and KMeans clustering, introducing additional dependencies on classical NLP methods.

## Related Work & Insights

- **RAG methods**: External retrieval augmentation; computationally intensive and domain-dependent.
- **RLHF/alignment methods**: Require large-scale fine-tuning with high resource consumption.
- **Decoding methods**: DoLa (inter-layer contrastive decoding), KCTS (knowledge-constrained tree search), Phi-Decoding (look-ahead sampling).
- **Token-Guard**: The first unified hallucination control framework integrating token-level self-checking, segment-level scoring, and global iteration.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ★★★★☆ |
| Theoretical Depth | ★★★☆☆ |
| Experimental Thoroughness | ★★★★☆ |
| Value | ★★★★☆ |
| Writing Quality | ★★★☆☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](tokmem_one-token_procedural_memory_for_large_language_models.md)
- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](query-level_uncertainty_in_large_language_models.md)
- [\[ICLR 2026\] Digging Deeper: Learning Multi-Level Concept Hierarchies](digging_deeper_learning_multi-level_concept_hierarchies.md)
- [\[AAAI 2026\] Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs](../../AAAI2026/information_retrieval/does_less_hallucination_mean_less_creativity_an_empirical_investigation_in_llms.md)
- [\[ACL 2026\] Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search](../../ACL2026/information_retrieval/multi-faceted_self-consistent_preference_alignment_for_query_rewriting_in_conver.md)

</div>

<!-- RELATED:END -->
