---
title: >-
  [Paper Note] A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses
description: >-
  [AAAI 2026][LLM Agent][Multi-Armed Bandit] This paper proposes MACO (Multi-Agent Conversational Online Learning), which formulates LLM response selection as a multi-agent conversational bandit problem. It employs local agents to eliminate low-quality responses and a cloud-side adaptive keyword-based dialogue to collect user preferences, achieving near-optimal online response evaluation and user preference alignment.
tags:
  - AAAI 2026
  - LLM Agent
  - Multi-Armed Bandit
  - Online Learning
  - Preference Alignment
  - Multi-Agent
  - Conversational Selection
date: 2026-05-08
content_hash: 2e4b3099e2775aaa
---

# A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses

**Conference**: AAAI 2026
**arXiv**: [2501.01849](https://arxiv.org/abs/2501.01849)
**Code**: [https://github.com/TarferSoul/MACO](https://github.com/TarferSoul/MACO)
**Area**: LLM Agent
**Keywords**: Multi-Armed Bandit, Online Learning, Preference Alignment, Multi-Agent, Conversational Selection

## TL;DR

This paper proposes MACO (Multi-Agent Conversational Online Learning), which formulates LLM response selection as a multi-agent conversational bandit problem. It employs local agents to eliminate low-quality responses and a cloud-side adaptive keyword-based dialogue to collect user preferences, achieving near-optimal online response evaluation and user preference alignment.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: LLMs can generate stylistically diverse candidate responses via different prompts (e.g., humorous, formal, or code-oriented styles). Selecting the response that best matches user preferences in an online setting is a critical challenge. Offline scoring is computationally expensive (e.g., 78 GPU hours to evaluate 205 prompts), whereas online evaluation can dynamically adjust based on user feedback.

**Four limitations of existing conversational bandit methods**:
1. **High-dimensional feature space**: Semantic embeddings of LLM responses are high-dimensional; traditional SVD-based dimensionality reduction incurs high computational complexity.
2. **Finite arm sets**: Most conversational bandit methods assume infinite arm sets, whereas LLM candidate responses form a finite yet large set.
3. **Fixed conversation frequency**: Existing methods control conversation frequency via predefined functions (linear/logarithmic), which cannot adapt to dynamic needs.
4. **Single-agent setting**: Users access LLMs across multiple devices (phone/tablet/desktop), producing fragmented preference data that existing methods cannot handle in a multi-agent cooperative manner.

## Method

### Overall Architecture

MACO consists of two components: (1) **MACO-A** (local agent): runs on each device and filters low-quality responses via an online elimination mechanism; (2) **MACO-S** (cloud server): aggregates data from all agents and adaptively selects keywords to query users, enabling efficient preference learning.

### Key Designs

1. **Local Agent Elimination Mechanism (MACO-A)**
   - Each agent $m$ maintains an active arm set $\mathcal{A}_m^p$, computes the information matrix $M_m^p$, and performs eigendecomposition.
   - For directions whose eigenvalues fall below threshold $h_p$ (i.e., insufficiently explored directions in the feature space), the corresponding eigenvectors are uploaded to the cloud server.
   - All active arms are pulled uniformly to collect reward feedback.
   - The updated preference estimate $\hat{\theta}_p$ is downloaded from the server, and candidate responses whose expected rewards are significantly lower than the best arm are eliminated.

2. **Cloud-Side Adaptive Dialogue Mechanism (MACO-S)**
   - Upon receiving under-explored directions uploaded by local agents, the server selects the keyword $k$ with the largest inner product with those directions and instructs the corresponding agent to query the user.
   - Keywords represent core stylistic concepts of responses (e.g., "C/C++", "humorous tone"); user feedback on keywords generalizes to related responses.
   - **Adaptive triggering**: Dialogue is initiated only when preference estimates are uncertain, rather than at fixed intervals, reducing unnecessary user interruptions.
   - The server aggregates information matrices $G$ and reward vectors $W$ from all agents, estimating the global preference via linear regression: $\hat{\theta}_p = G^{-1}W$.

3. **Avoiding the Computational Cost of G-Optimal Design**
   - Traditional elimination-based bandits require computing G-optimal designs (determining arm-pulling probability distributions), which is computationally intensive.
   - MACO leverages multi-agent heterogeneity and keyword-based dialogue to compensate for insufficient feature space coverage, eliminating the need for G-optimal design.
   - Communication cost is only $O(d^2 M \log T)$, independent of the arm set size $A$.

### Loss & Training

No training process is involved. The objective is to minimize cumulative regret: $R_M(T) = \sum_{m,t}(x_{a_m^*}^\top \theta^* - x_{a_{m,t}}^\top \theta^*)$.

## Key Experimental Results

### Main Results

| Method | Regret (vs. MACO) | Finite Arm Adaptation | Multi-Agent |
|---|---|---|---|
| LinUCB | Baseline | ✓ | ✗ |
| Conversational Bandit | Better | ✗ (infinite arm assumption) | ✗ |
| PE-Lin (independent) | Worse | ✓ | Nominal |
| **MACO** | **Optimal, outperforms baseline by 8.29%+** | **✓** | **✓** |

### Theoretical Results

| Metric | Bound |
|---|---|
| Regret upper bound | $O(\sqrt{dMT \log(AM\log T / \delta)})$ |
| Regret lower bound | $\Omega(\sqrt{dMT})$ |
| Communication cost | $O(d^2 M \log T)$ |

### Key Findings

- **Near-optimality**: The upper and lower bounds differ only by logarithmic factors, demonstrating that MACO is minimax optimal.
- **Multi-agent cooperation significantly reduces regret**: With $M$ agents sharing information, regret scales as $\sqrt{dMT}$ rather than $M\sqrt{dT}$, yielding a $\sqrt{M}$-fold improvement.
- **Adaptive dialogue outperforms fixed-frequency dialogue**: User interactions are not wasted when preferences are already known; queries are posed precisely when preferences are uncertain.
- Consistent effectiveness is demonstrated across two embedding models (Google and OpenAI) and two LLMs (Llama and GPT-4o).

## Highlights & Insights

- **Formalizing LLM response selection as a conversational bandit problem**: The practical problem of selecting the best LLM response is elegantly mapped to an online learning framework with theoretical guarantees.
- **Keyword-based dialogue as a substitute for G-optimal design**: By asking users questions such as "Do you prefer a humorous or formal tone?", the method efficiently completes preference information, avoiding the computationally intensive G-optimal probability design.
- **Practical consideration for multi-device scenarios**: When the same user accesses an LLM across different devices, preference data becomes fragmented; MACO's multi-agent architecture naturally addresses this issue.

## Limitations & Future Work

- The linear reward assumption is overly strong—user satisfaction with LLM responses may not be a linear function of feature vectors.
- The keyword set must be predefined; automatic discovery of effective keywords is not discussed.
- Non-stationary preferences (i.e., preferences that evolve over time) are not considered.
- Experiments involve hundreds of candidate responses, whereas the actual response space generated by LLMs is far larger.

## Related Work & Insights

- **vs. LinUCB (Abbasi-Yadkori 2011)**: A classic linear bandit that does not support conversational preference collection or multi-agent cooperation.
- **vs. ConUCB (Zhang et al. 2020)**: A single-agent conversational bandit with fixed conversation frequency and an infinite arm set assumption; MACO supports multiple agents, finite arm sets, and adaptive dialogue.
- **vs. RLHF**: RLHF trains a reward model from offline preference data for global alignment; MACO performs online personalized selection. The two approaches are complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ Modeling LLM response selection as a multi-agent conversational bandit is a novel problem formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple embedding models, datasets, theoretical analysis, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Theoretically rigorous, though the heavy notation somewhat limits readability.
- Value: ⭐⭐⭐⭐ Provides a theoretically grounded solution for online personalized LLM response selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation](finrpt_dataset_evaluation_system_and_llm-based_multi-agent_framework_for_equity_.md)
- [\[NeurIPS 2025\] Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](../../NeurIPS2025/llm_agent/automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)
- [\[AAAI 2026\] AutoTool: Efficient Tool Selection for Large Language Model Agents](autotool_efficient_tool_selection_for_large_language_model_agents.md)
- [\[AAAI 2026\] KDR-Agent: A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval](a_multi-agent_llm_framework_for_multi-domain_low-resource_in-context_ner_via_kno.md)
- [\[AAAI 2026\] Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems](parallelism_meets_adaptiveness_scalable_documents_understanding_in_multi-agent_l.md)

</div>

<!-- RELATED:END -->
