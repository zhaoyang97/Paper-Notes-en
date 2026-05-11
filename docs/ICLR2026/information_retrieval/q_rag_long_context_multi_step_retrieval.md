---
title: >-
  [Paper Note] Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training
description: >-
  [ICLR 2026 Oral][Information Retrieval & RAG][multi-step retrieval] Multi-step retrieval is formulated as an MDP and solved via value-based RL (soft Q-learning) to fine-tune the **embedder rather than the LLM**. The Q-fu…
tags:
  - "ICLR 2026 Oral"
  - "Information Retrieval & RAG"
  - "multi-step retrieval"
  - "value-based RL"
  - "embedder training"
  - "long context"
  - "RAG"
date: 2026-05-08
content_hash: 318cebdc59060938
---

# Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training

**Conference**: ICLR 2026 Oral
**arXiv**: [2511.07328](https://arxiv.org/abs/2511.07328)
**Code**: Available
**Area**: LLM / Retrieval-Augmented Generation
**Keywords**: multi-step retrieval, value-based RL, embedder training, long context, RAG

## TL;DR
Multi-step retrieval is formulated as an MDP and solved via value-based RL (soft Q-learning) to fine-tune the **embedder rather than the LLM**. The Q-function is designed as the inner product of state and action embeddings—proven to be a universal approximator—and combined with RoPE relative positional encoding to enable temporal reasoning. Training requires only a single A100 GPU for 12 hours; models trained on 4K-token contexts generalize to 1M+ token contexts, achieving near-perfect NIAH performance on the RULER benchmark.

## Background & Motivation

**Background**: Long-context multi-step retrieval is a central challenge in RAG. Existing approaches fall into two categories: (a) fine-tuning LLMs to generate search queries (Search-R1, R1-Searcher), which requires 8×A100 GPUs and is restricted to open-source LLMs; and (b) fine-tuning retrievers (Beam-Retriever) via supervised learning, which generalizes poorly.

**Limitations of Prior Work**: (a) LLM fine-tuning methods incur prohibitive computational costs and cannot be applied to closed-source LLMs; (b) Beam-Retriever relies on SFT and generalizes poorly to out-of-distribution data and ultra-long contexts; (c) existing retrievers cannot perform temporal reasoning (e.g., "what happened before event X?").

**Key Challenge**: Multi-step retrieval requires dynamically deciding what to retrieve next based on previously retrieved content—a sequential decision-making problem. Yet existing methods either rely on expensive LLMs for decision-making or apply simple SFT with insufficient exploration capacity.

**Goal**: Design a lightweight, general, and generalizable multi-step retrieval agent that (a) modifies only the embedder without touching the LLM; (b) trains via RL rather than SFT; (c) supports temporal reasoning; and (d) generalizes from short training contexts to long inference contexts.

**Key Insight**: The Q-function is designed as an inner product in embedding space—consistent with the similarity search paradigm of retrieval, theoretically proven to be a universal approximator, and enabling efficient inference without per-candidate transformer forward passes.

**Core Idea**: Fine-tune an embedder via RL to learn sequential decision-making in retrieval space, with the inner-product Q-function ensuring both computational efficiency and theoretical soundness.

## Method

### Overall Architecture
The input is a long document (pre-segmented into chunks) along with a query; the output is a set of supporting facts retrieved in multiple steps. MDP formulation: state = ordered list of already-retrieved chunks; action = selecting the next chunk; reward = sparse terminal reward (1 if all supporting facts are found). The embedder is trained via soft Q-learning + PQN.

### Key Designs

1. **Q-Function as Inner Product**

   - **Function**: Parameterizes the Q-function as the inner product of two embedders.
   - **Mechanism**: $Q_\theta(s, a_i) = \langle E_s(s; \theta_1), E_a(a_i, i; \theta_2) \rangle$, where the state embedder encodes previously retrieved content and the action embedder encodes the candidate chunk together with its document position.
   - **Design Motivation**: (a) **Theorem 1** proves this form is a universal approximator via the Stone–Weierstrass theorem; (b) at inference time only a dot product is needed rather than a transformer forward pass, yielding orders-of-magnitude speedup over Beam-Retriever.

2. **RoPE Relative Positional Encoding for Temporal Reasoning**

   - **Function**: Rotary positional encoding is used to express the positional relationship of a candidate chunk relative to already-retrieved facts.
   - **Mechanism**: A relative position mapping $\rho_t(i) = j \cdot \delta + \ell \cdot \frac{i - b_j}{b_{j+1} - b_j}$ is defined, where already-retrieved facts partition the document into intervals and each candidate chunk receives a positional encoding relative to the nearest interval. The action embedder uses $E_a(a_i, \rho_t(i); \theta_2)$.
   - **Design Motivation**: Absolute positional encodings fail under long-context extrapolation. Relative positional encoding directs the model to attend to whether a candidate appears before, after, or between known facts, enabling temporal reasoning that generalizes to arbitrary context lengths.

3. **PQN + Soft Q-Learning**

   - **Function**: Online value-based RL training without a replay buffer.
   - **Mechanism**: PQN (Periodic Q-Network) is adopted to avoid the overhead of re-embedding all chunks at every replay buffer sample. A soft value function $V_{\theta'}(s_t) = \alpha \log \sum_{a} \exp(Q_{\theta'}(s_t, a)/\alpha)$ and a target network are incorporated; $\lambda$-returns replace single-step TD targets to reduce bias.
   - **Design Motivation**: In retrieval settings the number of chunks can reach thousands; a replay buffer would require recomputing Q-values for all chunks at every update. The online nature of PQN eliminates this bottleneck.

### Loss & Training
$\mathcal{L}_Q = \mathbb{E}[(Q_\theta(s_t, a_t) - G_t^\lambda)^2]$, optimized with AdamW (lr = 1.5e-5), temperature $\alpha = 0.05$ annealed to 0, $\lambda = 0.5$; training completes in under 12 hours on a single A100-80GB GPU.

## Key Experimental Results

### Main Results (RULER NIAH)

| Context Length | Q-RAG NIAH Avg | LongRoPE2-8B | Beam-Retriever |
|---|---|---|---|
| 4K | **100** | 99.7 | 98.5 |
| 16K | **100** | 98.8 | 95.3 |
| 32K | **100** | 98.9 | — |
| 128K | **100** | 96.7 | — |
| 1M | **99.7** | — | — |

### Open-Domain QA (HotPotQA → MuSiQue OOD)

| Method | HotPotQA Ans F1 | MuSiQue Ans F1 (OOD) | Avg | Training Resources |
|---|---|---|---|---|
| **Q-RAG** | 0.76 | **0.52** | **0.64** | 1×A100 |
| Beam-Retriever | 0.77 | 0.40 | 0.59 | — |
| Search-R1 | 0.65 | 0.51 | 0.58 | 8×A100 |

### Ablation Study

| Configuration | Key Findings |
|---|---|
| w/o Soft-Q | Performance drops; insufficient exploration |
| w/o Target Network | Training becomes unstable |
| SFT instead of RL | Adequate on short contexts but fails to generalize to long contexts |
| w/o fine-tuning | Significant performance degradation |

### Key Findings
- **4K training → 1M generalization**: NIAH performance remains perfect from 4K up to 1M tokens (2,500× extrapolation), attributed to relative positional encoding.
- **RL > SFT**: Given identical supervision signals, RL training substantially outperforms SFT, especially on OOD and ultra-long contexts.
- **QA3 (hardest subtask)**: Requiring 3+ supporting facts and temporal reasoning, Q-RAG shows almost no performance degradation while Beam-Retriever fails entirely.
- **Efficiency**: At inference time, dot product vs. transformer forward pass gives Q-RAG a large speed advantage under long contexts.

## Highlights & Insights
- **Embedder-only paradigm shift**: Modifying only the embedder while leaving the LLM untouched makes the method compatible with any LLM, including closed-source ones, while reducing training cost by 8×.
- **Q-function as retrieval**: The RL Q-function and retrieval similarity score are unified as an inner product, simultaneously satisfying theoretical guarantees and computational efficiency.
- **Complementarity with LoongRL**: LoongRL teaches the LLM internal reasoning patterns (plan-retrieve-reason), while Q-RAG teaches the embedder external retrieval strategies; the two approaches are naturally complementary and can be combined.

## Limitations & Future Work
- **Supervision limited to supporting facts**: Using LLM answer quality as a reward signal (joint retriever–generator optimization) remains unexplored.
- **Requires pre-segmented chunks**: The method depends on a predefined document chunking strategy.
- **Requires supporting-fact annotations**: Training data must label which chunks constitute supporting facts.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Unifying the RL Q-function with retrieval similarity as an inner product, and applying RoPE relative positional encoding to temporal retrieval, are both novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage across RULER, BabiLong, and Open-Domain QA; 4K→10M generalization is remarkable.
- Writing Quality: ⭐⭐⭐⭐ — Method description is clear, though dense notation requires careful reading.
- Value: ⭐⭐⭐⭐⭐ — Lightweight and deployable, compatible with any LLM; strong potential to become a standard retrieval component in RAG pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)
- [\[ICLR 2026\] Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation](attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)
- [\[ICLR 2026\] Embedding-Based Context-Aware Reranker](embedding-based_context-aware_reranker.md)
- [\[ACL 2026\] CRAFT: Training-Free Cascaded Retrieval for Tabular QA](../../ACL2026/information_retrieval/craft_training-free_cascaded_retrieval_for_tabular_qa.md)
- [\[ACL 2026\] Context Attribution with Multi-Armed Bandit Optimization](../../ACL2026/information_retrieval/context_attribution_with_multi-armed_bandit_optimization.md)

</div>

<!-- RELATED:END -->
