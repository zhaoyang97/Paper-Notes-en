---
title: >-
  [Paper Note] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding
description: >-
  [ICLR 2026][RAG] This paper proposes LDAR (Learning Distraction-Aware Retrieval), a lightweight adaptive retriever that learns to select passages by sampling a continuous quantile band from the query-passage similarity distribution. LDAR surpasses long-context methods while using approximately half the token budget, balancing information coverage against the influence of distracting passages.
tags:
  - ICLR 2026
  - RAG
  - long-context
  - distraction-aware retrieval
  - adaptive passage selection
  - knowledge-intensive QA
date: 2026-05-08
content_hash: 4dcd8aee28192521
---

# Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding

**Conference**: ICLR 2026
**arXiv**: [2509.21865](https://arxiv.org/abs/2509.21865)
**Code**: [https://github.com/ku-dmlab/LDAR](https://github.com/ku-dmlab/LDAR)
**Area**: LLM Reasoning / RAG / Information Retrieval
**Keywords**: RAG, long-context, distraction-aware retrieval, adaptive passage selection, knowledge-intensive QA

## TL;DR
This paper proposes LDAR (Learning Distraction-Aware Retrieval), a lightweight adaptive retriever that learns to select passages by sampling a continuous quantile band from the query-passage similarity distribution. LDAR surpasses long-context methods while using approximately half the token budget, balancing information coverage against the influence of distracting passages.

## Background & Motivation

**Background**: RAG augments LLM generation by retrieving external passages and remains the dominant approach for mitigating factual errors and knowledge staleness in LLMs. The recent expansion of LLM context windows to 128K+ tokens has given rise to "long-context" (LC) alternatives that feed complete documents directly to the model.

**Limitations of Prior Work**: Long-context methods suffer from three issues: (i) poor token efficiency, as processing large amounts of redundant context wastes compute; (ii) the "lost-in-the-middle" phenomenon, where models struggle to recall information at intermediate positions; and (iii) severe distraction for capacity-limited models, ultimately degrading output quality. Although conventional RAG with top-$k$ retrieval is efficient, a fixed $k$ cannot adapt to the varying processing capacities of different LLMs.

**Key Challenge**: Retrieving more passages increases information coverage but simultaneously introduces more distracting passages, yielding an inverted-U performance curve. More critically, the optimal retrieval strategy depends on LLM capacity (stronger models tolerate more distraction) as well as combinatorial interaction effects among passages—passages that each individually support a correct answer may collectively lead to an incorrect one.

**Goal**: How can one adaptively select a passage set that minimizes the influence of distracting passages for a given LLM capacity while ensuring sufficient information coverage?

**Key Insight**: The authors observe that distraction effects depend not only on individual passage relevance but also on the combination of retrieved passages—even when each passage individually yields a correct answer, their joint retrieval may produce an incorrect one. Simple text-based reranking is therefore insufficient; retrieval strategy must be learned from the perspective of the similarity distribution.

**Core Idea**: Train a lightweight neural network that operates solely on the query-passage cosine similarity distribution, selecting passages by sampling a continuous quantile band via Beta distribution, thereby minimizing distraction to the LLM.

## Method

### Overall Architecture
Given a query $q$ and $N$ candidate passages $\{p_i\}$, a pretrained embedding model $f_\phi$ first computes a cosine similarity vector $s \in \mathbb{R}^N$. The adaptive retriever $\pi_\theta$ takes this similarity vector as input and outputs a quantile interval $(q_L, q_U) \subset [0,1]$, from which all passages whose similarity ranks fall within the interval are selected to form the retrieval set $\mathcal{R}$, which is then passed to the LLM for answer generation. The LLM and embedding model parameters are frozen throughout; only the lightweight retriever is trained.

### Key Designs

1. **Band-Based Retrieval Strategy**

   - **Function**: Reduces the passage selection space from an exponentially large subset search to a two-dimensional continuous control problem.
   - **Mechanism**: Rather than applying independent Bernoulli sampling per passage, the retriever predicts a continuous similarity interval $[q_L, q_U]$ and retrieves all passages ranked within it. Concretely: $\ell = \max(1, \lfloor N \cdot q_L \rceil)$, $u = \max(\ell, \lfloor N \cdot q_U \rceil)$, and passages ranked $\ell$ through $u$ (by similarity) are selected.
   - **Design Motivation**: Bernoulli sampling requires exploring a combinatorial space of size $2^N$, making generalization difficult and convergence prone to suboptimal solutions (empirically degenerating to the LC baseline). The band-based strategy reduces the search space to a low-dimensional continuous space, analogous to temporal abstraction in reinforcement learning, enabling more efficient credit assignment and exploration.

2. **Transformer Encoder with Attention Pooling**

   - **Function**: Encodes a variable-length similarity vector into a fixed-dimensional global representation.
   - **Mechanism**: Each similarity score $s_i$ is first embedded via periodic embedding, then processed by a bidirectional self-attention Transformer, and finally aggregated into a global vector through attention pooling. Two output heads respectively predict Beta distribution parameters $(\alpha_L, \beta_L)$ and $(\alpha_U, \beta_U)$, from which $q_L$ and $q_U$ are sampled.
   - **Design Motivation**: Periodic embeddings handle continuous-valued similarities; the Transformer captures relative ordering relationships among passages; and the Beta distribution naturally constrains outputs to $[0,1]$.

3. **Policy Gradient Optimization**

   - **Function**: Uses LLM prediction correctness as a reward signal to optimize the retriever via the log-derivative trick.
   - **Mechanism**: The objective is $\max_\theta J(\theta) = \mathbb{E}[r_\psi(q, \mathcal{R}, y)]$, where $r_\psi = \mathbb{1}_{\text{corr}}(F_\psi(q, \mathcal{R}), y)$ is a binary indicator of whether the LLM prediction is correct. The gradient update is: $\theta_{k+1} = \theta_k + \gamma \cdot r_\psi \cdot \nabla_{\theta_k} \log \pi_{\theta_k}(\cdot|s)$.
   - **Design Motivation**: The LLM is frozen, so no gradient backpropagation through it is required. Since the retriever relies only on the similarity distribution rather than passage text, it can be trained efficiently with lightweight policy gradient methods. Reward signals from different LLMs naturally guide the retriever to learn strategies suited to each model's capacity.

### Additional Implementation Details
- The retriever $\pi_\theta$ deliberately has no access to passage text, relying solely on the similarity distribution, ensuring scalability to large-scale retrieval settings.
- Special handling is applied for the Hallucination task: since the reward for this task is granted for refusing to answer, including it in training would cause the retriever to learn a degenerate strategy of intentionally not retrieving. It is therefore used only as an evaluation benchmark.
- Under the 128K long-context setting, LDAR adaptively retrieves fewer passages (lower token usage ratio), indicating that distraction risk is higher with longer inputs.

## Key Experimental Results

### Main Results
Evaluation is conducted on 6 knowledge-intensive benchmarks: the LaRA benchmark (Location, Reasoning, Comparison, Hallucination), HotpotQA, and NQ, covering 5 open-source LLMs (Qwen-2.5-7B, Llama-3.1-8B, etc.) and 4 closed-source LLMs (GPT-4o, Gemini-2.5-pro, etc.).

| Setting | Method | Avg. Score (Open) | Avg. Score (Closed) | Token Ratio |
|---------|--------|-------------------|---------------------|-------------|
| 32K | LC | 58.62 | 74.00 | 1.000 |
| 32K | RAG (Top-5+Reranker) | 62.12 | 70.62 | 0.094 |
| 32K | BGM | 62.70 | 69.63 | 0.057 |
| 32K | Self-Route | 59.45 | 73.97 | 0.426 |
| 32K | **LDAR** | **70.00** | **79.42** | 0.467/0.629 |
| 128K | LC | 43.65 | 69.97 | 1.000 |
| 128K | RAG | 54.67 | 64.42 | 0.025 |
| 128K | **LDAR** | **61.55** | **76.22** | 0.250/0.517 |

### Ablation Study

| Configuration | Result | Remarks |
|---------------|--------|---------|
| Band-based LDAR | High score + low token usage | Effectively balances coverage and distraction |
| Bernoulli-based LDAR | Degenerates to LC | Combinatorial space too large for effective exploration |
| Open-source LLMs | Token ratio ~0.25–0.47 | Weaker models retrieve fewer passages |
| Closed-source LLMs | Token ratio ~0.52–0.63 | Stronger models tolerate more distraction |
| Location task | Usage ratio 0.45 | Focuses on high-similarity region for localization |
| Comparison task | Usage ratio 0.50 | Cross-region integration requires more passages |

### Zero-Shot Generalization
After training on LaRA, LDAR transfers zero-shot to the HELMET benchmark (HotpotQA, NQ). Average scores for open-source models improve from 44.2 to 60.0 (HotpotQA) and from 37.4 to 49.2 (NQ) compared to LC; closed-source models show consistent gains as well.

### Key Findings
- The band-based strategy is the critical factor for success—the Bernoulli variant completely fails and degenerates to the long-context baseline.
- LDAR adapts its retrieval dynamically: it retrieves fewer passages for weaker models (to avoid distraction) and more for stronger models (to exploit their long-context capability).
- Under the 128K context setting, LDAR intentionally retrieves fewer passages than under 32K, confirming that distraction risk increases with longer context.
- LDAR offers superior inference efficiency compared to LC (3.9s vs. 15.4s for open-source models) and most reranker-based methods.

## Highlights & Insights
- **The temporal abstraction intuition behind band-based retrieval is elegant**: compressing an exponentially large subset selection problem into a two-dimensional continuous control problem, analogous to the options framework in reinforcement learning, simultaneously reduces exploration difficulty and retains sufficient expressive power.
- **The minimalist design of deciding solely from the similarity distribution is striking**: without reading passage text, fine-tuning the LLM, or fine-tuning the embedding model, training only a lightweight network yields substantial improvements over text-reading methods such as BGM and rerankers—demonstrating that the similarity distribution alone contains sufficient information for retrieval decisions.
- **Emergent adaptation to LLM capacity**: the same framework, when trained with different LLMs, naturally learns distinct retrieval strategies (fewer passages for open-source models, more for closed-source models) without any manual tuning.

## Limitations & Future Work
- The Hallucination task requires special treatment (excluded from training), indicating that the framework does not adapt well to scenarios where the correct strategy is to refrain from retrieval.
- The band-based strategy assumes that the optimal passage set forms a contiguous interval in similarity rank order, whereas in practice the optimal subset may be non-contiguous.
- A separate retriever must be trained for each LLM; cross-model transferability remains to be validated.
- Fixing the embedding model may impose an upper bound—if embedding quality is poor, the similarity distribution itself carries insufficient information.

## Related Work & Insights
- **vs. RAG**: Conventional RAG uses a fixed top-$k$; LDAR uses a learned dynamic interval. RAG ignores LLM capacity, whereas LDAR naturally adapts to it.
- **vs. Long-Context**: LC provides complete information but introduces extensive distraction; LDAR finds an optimal trade-off between coverage and distraction.
- **vs. BGM**: BGM performs subset selection among top-5 text candidates (limited combinatorial space but requires reading text); LDAR performs interval selection over all passages (larger search space but more efficient).
- **vs. Self-Route**: Self-Route uses LLM self-evaluation to choose between RAG and LC; LDAR directly learns a continuous retrieval policy.

## Rating
- Novelty: ⭐⭐⭐⭐ — The band-based retrieval perspective is novel, reformulating the retrieval problem as a continuous control task.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 6 benchmarks, 9 LLMs, 8+ baselines, with separate evaluation of open-source and closed-source models; highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly developed, figures are intuitive, and the analysis in the Motivation section is persuasive.
- Value: ⭐⭐⭐⭐ — Provides a practical middle ground in the RAG vs. LC debate with a highly favorable cost-effectiveness ratio.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Embedding-Based Context-Aware Reranker](embedding-based_context-aware_reranker.md)
- [\[ICLR 2026\] Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training](q_rag_long_context_multi_step_retrieval.md)
- [\[ICLR 2026\] Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation](attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)
- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](../../ACL2026/information_retrieval/videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)
- [\[ICLR 2026\] Fine-tuning with RAG for Improving LLM Learning of New Skills](fine-tuning_with_rag_for_improving_llm_learning_of_new_skills.md)

<!-- RELATED:END -->
