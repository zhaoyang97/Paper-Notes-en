---
title: >-
  [Paper Note] Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering
description: >-
  [ACL2025][NLP Understanding][Multi-hop Question Answering] The SiGIR framework is proposed to enable models with iterative question decomposition, retrieval, reasoning, and self-evaluation capabilities through end-to-end training. During inference, it utilizes self-critique feedback to guide iteration-level beam search for selecting optimal reasoning paths, outperforming the state-of-the-art (SOTA) by an average of 8.6% across three multi-hop QA datasets.
tags:
  - "ACL2025"
  - "NLP Understanding"
  - "Multi-hop Question Answering"
  - "Retrieval-Augmented Generation"
  - "Self-Critique"
  - "Iterative Reasoning"
  - "Beam Search"
date: 2026-05-08
content_hash: 781e45fee6d81463
---

# Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering

**Conference**: ACL2025  
**arXiv**: [2505.19112](https://arxiv.org/abs/2505.19112)  
**Code**: [zchuz/SiGIR-MHQA](https://github.com/zchuz/SiGIR-MHQA)  
**Area**: NLP Understanding  
**Keywords**: Multi-hop Question Answering, Retrieval-Augmented Generation, Self-Critique, Iterative Reasoning, Beam Search

## TL;DR

The SiGIR framework is proposed to enable models with iterative question decomposition, retrieval, reasoning, and self-evaluation capabilities through end-to-end training. During inference, it utilizes self-critique feedback to guide iteration-level beam search for selecting optimal reasoning paths, outperforming the state-of-the-art (SOTA) by an average of 8.6% across three multi-hop QA datasets.

## Background & Motivation

Large Language Models (LLMs) face three major challenges in knowledge-intensive multi-hop reasoning tasks: (1) decomposing complex questions into sub-questions in a single pass is highly difficult, where initial errors cause subsequent reasoning to deviate from the correct path; (2) iterative retrieval methods struggle with complex question planning and cannot express retrieval intent clearly, leading to inaccurate retrieval; (3) the lack of guided feedback for intermediate steps easily leads to cascading errors. Although existing methods (such as IRCoT, Self-RAG, Auto-RAG) partially alleviate these issues, they still fail to effectively combine intermediate step quality evaluation to guide the reasoning process. This paper focuses on two key aspects: iterative question decomposition to facilitate precise retrieval, and intermediate step feedback to guide reasoning, expanding the reasoning space and reducing error propagation.

## Method

### Overall Architecture

SiGIR (Self-Critique Guided Iterative Reasoning) consists of two phases: training and inference.

The **training phase** constructs the SC-Reasoner (Self-Critique Reasoner) in three steps:

1. **Iterative Reasoner (R)**: Synthesizes iterative reasoning trajectories (including sub-question decomposition, retrieval triggering, and knowledge reasoning) using DeepSeek-V2.5. The data is organized in an interleaved format to train a small model, yielding the base reasoner.
2. **Critic Model (C)**: Generates reasoning trajectories from non-overlapping corpora using R (instead of the LLM) to ensure a balance of positive and negative samples, followed by LLM evaluation of retrieval quality, reasoning quality, and overall quality, to train an independent critic model.
3. **SC-Reasoner (R_sc)**: Annotates reward signals on sub-processes of reasoning trajectories using C, appending self-scores after special tokens. Finally, end-to-end training is conducted to obtain a model with both reasoning and self-evaluation capabilities.

### SC-Reasoner Inference Mode

The model performs five operations in each step: (1) determining whether the question needs decomposition; (2) decomposing one atomic sub-question at a time; (3) triggering external retrieval and conducting knowledge reasoning; (4) self-evaluating retrieval and reasoning quality (generating $r_{\text{retr}}$ and $r_{\text{reas}}$); (5) reducing the original question to identify unresolved parts. This process is repeated until the final answer is reached.

### Self-Critique Guided Iterative Reasoning (Iteration-level Beam Search)

The inference phase employs iteration-level beam search:

- **Branch Exploration**: At each reasoning step, branch expansion is performed on sub-question decomposition, retrieval, and reasoning via temperature sampling to generate multiple candidate reasoning paths.
- **Candidate Selection**: Candidate paths are scored using cumulative process reward ($r_c = r_{\text{retr}} + r_{\text{reas}}$ incrementally accumulated), keeping the top-$k$ optimal candidates.
- **Final Selection**: After reasoning concludes, the final answer trajectory is selected using cumulative process rewards or outcome rewards.

### Self-Improvement Mechanism

SC-Reasoner can perform reasoning and self-evaluate quality on unlabeled data, retaining high-quality trajectories for subsequent training, thereby achieving self-improvement in data-sparse scenarios.

## Key Experimental Results

### Main Results (Table 1)

F1 results on three multi-hop QA datasets (with Mistral-7B as the default backbone):

| Method | 2WikiMQA | HotpotQA | MuSiQue |
|------|----------|----------|---------|
| IRCoT | 46.80 | 49.03 | 21.43 |
| Self-RAG | 31.60 | 54.60 | 22.00 |
| Auto-RAG | 56.06 | 50.03 | 22.08 |
| DR-Distillation | 70.51 | 58.06 | 22.74 |
| **SiGIR (Ours)** | **74.47** | **63.09** | **37.15** |

SiGIR achieves the most significant improvement on MuSiQue (3-4 hop difficult questions): 3-hop $+47.1\%$, 4-hop $+24.37\%$. Competitive results are also achieved on Qwen2.5 and LLaMA2.

### Ablation Study (Table 2)

| Setting | 2WikiMQA | HotpotQA | MuSiQue |
|------|----------|----------|---------|
| Full SiGIR | 74.47 | 63.09 | 37.15 |
| w/o Guided Search | 72.37 | 60.86 | 35.47 |
| w/o Reward Signals | 67.23 | 54.94 | 29.45 |
| w/o Self-Critique | 66.96 | 56.86 | 28.62 |
| Coarse-grained Reward | 73.40 | 59.85 | 34.32 |

Removing reward signals leads to an approximate 10% performance drop; fine-grained rewards are more effective than coarse-grained rewards in search-based reasoning.

### Search and Selection Strategy Analysis (Table 4)

| Strategy | Average F1 |
|------|---------|
| Guided Search (Full) | 58.23 |
| Keep Only 1 Candidate | 56.81 |
| No Sub-question Branching | 50.03 |
| Cumulative Process Reward Selection | 58.23 |
| Outcome Reward Selection | 55.99 |
| Random Selection | 55.81 |

Performance drops by 13.57% when the sub-question exploration width is set to 1, demonstrating the importance of diverse sub-question exploration.

## Highlights & Insights

- **Unified Framework**: Unifies reasoning and self-critique into a single model, avoiding the inference overhead of decoupled generator-critics (yielding a 4.46x throughput improvement).
- **Iterative Decomposition + Beam Search**: Avoids the high risk of one-pass decomposition, mitigating error propagation through step-by-step decomposition and searching.
- **Self-Improvement Capability**: With only 40% labeled data + two rounds of self-improvement, the model's performance approaches that of training with 100% data (with only a 2.2% gap).
- **Retrieval System Flexibility**: The method naturally supports hybrid retrieval (sparse + dense), where hybrid retrieval yields an average improvement of 5.7%.

## Limitations & Future Work

- Sub-question exploration adopts a sampling approach rather than a dedicated decomposition module, which may limit the breadth of exploration.
- It only employs a Supervised Fine-Tuning (SFT) paradigm, without introducing reinforcement learning to optimize reasoning trajectories.
- Beam search increases computational overhead during inference (although partially mitigated by GenCritic).
- Experiments are only validated on English multi-hop QA datasets; cross-lingual and cross-domain generalizability remains unknown.

## Related Work & Insights

- **Retrieval-Augmented Generation (RAG)**: Evolves from single-pass retrieval to multi-round iterative retrieval (e.g., IRCoT, Self-RAG). SiGIR incorporates self-critique signals on top of this to guide the retrieval direction.
- **Multi-hop Question Answering**: Compares question decomposition methods (e.g., ProbTree, BeamAggR) vs. iterative reasoning methods (e.g., Auto-RAG, DR-Distillation). SiGIR combines iterative decomposition with reward-guided search.
- **Inference-Time Scaling**: Includes sampling ensembles (e.g., Self-Consistency) and MCTS search. SiGIR's iteration-level beam search balances performance and efficiency.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of unifying self-critique and iterative reasoning into beam search is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets + four backbone models + rich ablation analysis + retrieval system experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive diagrams.
- Value: ⭐⭐⭐⭐ Provides practical reference value for knowledge-intensive multi-hop reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [\[ACL 2025\] Multi-Hop Reasoning for Question Answering with Hyperbolic Representations](multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)
- [\[ACL 2025\] BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering](belle_a_bi-level_multi-agent_reasoning_framework_for_multi-hop_question_answerin.md)
- [\[ACL 2025\] Active LLMs for Multi-hop Question Answering](active_llms_for_multi-hop_question_answering.md)
- [\[ACL 2025\] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)

</div>

<!-- RELATED:END -->
