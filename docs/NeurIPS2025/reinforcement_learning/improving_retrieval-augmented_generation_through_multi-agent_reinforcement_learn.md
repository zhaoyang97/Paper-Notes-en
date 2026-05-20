---
title: >-
  [Paper Note] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][RAG] This work models multiple components of a complex RAG pipeline (Query Rewriter, Selector, Generator) as a cooperative multi-agent system and jointly optimizes them via MAPPO…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "RAG"
  - "multi-agent reinforcement learning"
  - "MAPPO"
  - "joint optimization"
  - "question answering"
date: 2026-05-08
content_hash: fc6cd8d227cdd60f
---

# Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2501.15228](https://arxiv.org/abs/2501.15228)  
**Code**: [GitHub](https://github.com/chenyiqun/MMOA-RAG)  
**Area**: Reinforcement Learning / NLP
**Keywords**: RAG, multi-agent reinforcement learning, MAPPO, joint optimization, question answering

## TL;DR

This work models multiple components of a complex RAG pipeline (Query Rewriter, Selector, Generator) as a cooperative multi-agent system and jointly optimizes them via MAPPO, using the F1 score of the final answer as a shared reward. The proposed method outperforms existing single-module optimization approaches on multiple QA benchmarks.

## Background & Motivation

**Background**: RAG systems augment LLMs with retrieved external knowledge. Modern RAG pipelines consist of multiple components—query rewriting, document retrieval, document selection, and answer generation.

**Limitations of Prior Work**: Individual components are typically optimized independently via SFT, leading to a misalignment between module-level objectives and the global objective of generating accurate answers. For instance, documents deemed highly relevant under nDCG-optimized retrieval may not contribute to correct answer generation.

**Key Challenge**: Existing end-to-end optimization methods (e.g., applying PPO or DPO to individual RAG components) either address only simple two-component pipelines or optimize modules in isolation, failing to adequately model the cooperative relationships among multiple components.

**Goal**: To jointly optimize the parameters of multiple components in a RAG system so that each module's optimization objective is aligned with final answer quality.

**Key Insight**: Model RAG as a cooperative multi-agent reinforcement learning (Co-MARL) problem and leverage MAPPO for joint multi-agent optimization.

**Core Idea**: Treat RAG as a cooperative game in which each component serves as an agent, all sharing a global reward based on final answer F1, and optimize all agents synchronously via MAPPO.

## Method

### Overall Architecture

MMOA-RAG models RAG as a multi-agent system $\langle \mathcal{G}, \mathcal{O}, \mathcal{A}, \mathcal{R} \rangle$. The four-module pipeline proceeds as: Query Rewriter → Retriever (fixed, not trained) → Selector → Generator. The three trainable agents share a single LLM through parameter sharing, reducing training overhead.

### Key Designs

1. **Multi-Agent Modeling (Co-MARL)**:

    - **Function**: Treats each RAG component as an RL agent sharing a global reward.
    - **Design Motivation**: Independent module optimization leads to objective misalignment; the multi-agent framework naturally captures inter-component cooperation.
    - **Mechanism**: Three agents are defined: the Query Rewriter (QR) receives question $q$ and outputs sub-questions $subq$; the Selector (S) receives $q$ and candidate documents $D$, outputting a selected subset of document IDs $D_{\text{selected}}$; the Generator (G) receives $q$ and $D_{\text{selected}}$ and produces the final answer.
    - **Novelty**: Unlike Rewrite-Retrieve-Read (which optimizes only the rewriter) or BGM (which optimizes only the bridge module), MMOA-RAG jointly optimizes all three components.

2. **Observation / Action / Reward Design per Agent**:

    - **Function**: Defines precise MDP elements for each agent.
    - **Design Motivation**: The distinct roles of different agents necessitate differentiated action spaces and penalty terms.
    - **Mechanism**:
        - QR's action space is the full vocabulary $\mathcal{V}$; its reward is $R_{QR} = R_{\text{shared}} + P_{QR}$, with a penalty of $-0.5$ when the number of sub-questions exceeds 4.
        - Selector's action space is restricted to $\{$"0", "1", ..., "K-1", "Document", ","$\}$, substantially reducing the exploration space; format errors or duplicate IDs incur a penalty of $-1$.
        - Generator's action space is $\mathcal{V}$, with a penalty of $-0.5$ for excessively long outputs.
        - The shared reward $R_{\text{shared}}$ is the F1 score of the predicted answer.
    - **Novelty**: The constrained action space for the Selector is an elegant design choice—converting free-text generation into structured ID selection, which significantly improves training stability.

3. **MAPPO Joint Optimization**:

    - **Function**: Jointly updates all agents using the Multi-Agent PPO algorithm.
    - **Design Motivation**: MAPPO employs a shared global reward in fully cooperative settings to foster inter-agent collaboration, making it more suitable than independent PPO.
    - **Mechanism**: The actor loss adopts the standard PPO clipping objective, extended to multiple agents:
    $$\mathcal{L}_{\text{Actor}}(\theta) = \sum_i \sum_t \min(r_t^i \hat{A}_{\pi_\theta}^{i,t},\ \text{clip}(r_t^i, 1-\epsilon, 1+\epsilon) \hat{A}_{\pi_\theta}^{i,t})$$
    The final reward includes a KL penalty to prevent deviation from the SFT baseline:
    $$R(s_t^i, a_t^i) = R_i - \beta \log \frac{\pi_\theta(Answer_i | O_i)}{\pi_{\theta_{\text{SFT}}}(Answer_i | O_i)}$$
    - **Novelty**: All three agents share a single set of LLM parameters (via parameter sharing), resulting in training efficiency comparable to single-agent PPO.

### Loss & Training

- **Warm Start (SFT)**: Each agent is first fine-tuned via SFT to acquire basic instruction-following capability.
- **MAPPO Joint Training**: Starting from the SFT checkpoint, rollouts are collected by sequentially passing through QR → Retriever → S → G. The shared reward and per-agent penalties are computed, advantage estimates are obtained via GAE, and both the actor and critic are updated accordingly.
- Mini-batch parallelism is employed to accelerate training.

## Key Experimental Results

### Main Results (Contriever Retriever + Llama-3-8B-Instruct)

| Method | HotpotQA F1 | 2Wiki F1 | AmbigQA F1 |
|------|:---:|:---:|:---:|
| LLM w/o RAG | 31.18 | 29.47 | 33.42 |
| Vanilla RAG w/o train | 30.67 | 22.84 | 33.56 |
| Vanilla RAG w SFT | 44.49 | 43.36 | 44.36 |
| SELF-RAG | 38.93 | 38.86 | 39.04 |
| RetRobust | 46.49 | 44.51 | 44.78 |
| Rewrite-Retrieve-Read | 46.32 | 44.17 | 45.92 |
| BGM | 44.54 | 43.29 | 45.76 |
| RAG-DDR | 44.26 | 44.18 | 45.83 |
| **MMOA-RAG** | **48.29** | **46.40** | **48.59** |
| Δ vs. best baseline | +1.80 | +1.89 | +2.67 |

### Ablation Study (Removing Individual Agents from Joint Optimization)

| Configuration | HotpotQA F1 | 2Wiki F1 | AmbigQA F1 |
|------|:---:|:---:|:---:|
| MMOA-RAG (QR+S+G) | **48.29** | **46.40** | **48.59** |
| MMOA-RAG w/o QR | 47.07 | 45.25 | 47.19 |
| MMOA-RAG w/o S | 47.94 | 46.19 | 47.53 |
| MMOA-RAG w/o G | worst (per figure) | worst | worst |

### Generalization (SFT → MAPPO Gains across RAG Configurations)

| Configuration | HotpotQA F1 (SFT→MAPPO) | 2Wiki F1 (SFT→MAPPO) | AmbigQA F1 (SFT→MAPPO) |
|------|:---:|:---:|:---:|
| QR+S+G | 44.69→48.29 (+3.60) | 42.97→46.40 (+3.43) | 46.71→48.59 (+1.88) |
| S+G | 43.14→47.07 (+3.93) | 42.40→45.25 (+2.85) | 45.82→47.19 (+1.37) |
| QR+G | 45.00→47.94 (+2.94) | 42.91→46.19 (+3.28) | 45.31→47.53 (+2.22) |

### Key Findings

- **Joint optimization outperforms isolated optimization**: The full three-agent configuration (QR+S+G) achieves the best performance across all datasets.
- **Generator is the most critical agent**: Removing G from joint optimization causes the largest performance drop, particularly on the single-hop AmbigQA benchmark.
- **Selector can be partially substituted by the Generator**: The performance drop when removing S is smallest, as the jointly trained Generator develops a degree of noise-filtering capability.
- **Multi-hop datasets benefit more**: MAPPO yields larger gains on HotpotQA/2Wiki (multi-hop, ~3.5 F1) than on AmbigQA (single-hop, ~1.9 F1), indicating that multi-module cooperation is more critical for complex reasoning.
- **MAPPO consistently effective**: Across all three configurations (QR+S+G, S+G, QR+G), MAPPO delivers consistent and significant improvements over SFT, demonstrating the generality of the framework.

## Highlights & Insights

- **Novel modeling perspective**: This work is the first to model a RAG system as a cooperative multi-agent task, offering a new paradigm for end-to-end optimization of complex AI pipelines.
- **Selector action space design**: Constraining document selection from free-text generation to structured ID output is an engineering insight that substantially reduces the exploration space and training instability.
- **Parameter sharing**: All three agents share a single LLM (distinguished only by different prompts), bringing training compute overhead close to that of single-agent PPO.
- **Penalty term design**: Lightweight per-agent penalties (number of sub-questions, format compliance, answer length) constrain output quality without interfering with the primary optimization objective.

## Limitations & Future Work

- The Retriever is frozen and excluded from optimization; joint optimization of the retrieval module could potentially yield further gains.
- Experiments are conducted solely on Llama-3-8B-Instruct; larger-scale models and closed-source LLMs remain untested.
- The shared reward is based exclusively on F1 score, without considering multi-objective optimization targets such as latency or cost.
- The training overhead of MAPPO is not quantitatively compared against other methods.
- More complex RAG workflows in DAG form or with iterative retrieval calls are not explored.

## Related Work & Insights

- **MAPPO (Yu et al., 2022)**: A multi-agent PPO algorithm validated on StarCraft II; this paper transfers it to NLP pipelines.
- **Rewrite-Retrieve-Read / BGM**: Pioneering works that apply PPO to optimize individual RAG modules; this paper extends the paradigm to joint multi-module optimization.
- **Search-R1 / R1-Searcher**: Concurrent works applying RL to RAG reasoning, but focused on single-agent settings.
- **InstructGPT**: The source of inspiration for the KL penalty term.
- **Broader Implications**: Co-MARL modeling offers a general approach to optimizing any multi-module AI system, with potential applicability to multi-agent coding, tool-use pipelines, and beyond.

## Rating

- Novelty: ⭐⭐⭐⭐ Modeling RAG as Co-MARL is a fresh perspective; the Selector action space design is particularly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, multiple baselines, agent ablations, configuration generalization experiments, and validation across different retrievers.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical formulations, clear architecture diagrams, and precise definitions of agent MDP elements.
- Value: ⭐⭐⭐⭐ Provides a reproducible general framework for optimizing complex RAG systems, with open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](../../ACL2026/reinforcement_learning/language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[NeurIPS 2025\] Robust and Diverse Multi-Agent Learning via Rational Policy Gradient](robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)
- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[NeurIPS 2025\] Extending NGU to Multi-Agent RL: A Preliminary Study](extending_ngu_to_multi-agent_rl_a_preliminary_study.md)
- [\[NeurIPS 2025\] Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering](knowledge-based_visual_question_answer_with_multimodal_processing_retrieval_and_.md)

</div>

<!-- RELATED:END -->
