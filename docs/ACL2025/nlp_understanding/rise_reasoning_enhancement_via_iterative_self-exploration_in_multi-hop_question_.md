---
title: >-
  [Paper Note] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering
description: >-
  [ACL 2025][NLP Understanding] Proposes RISE—a multi-hop QA framework combining RAG with self-iterative training. Through a self-exploration loop consisting of three actions—question decomposition, retrieve-and-read, and self-critique—it iteratively generates training data and multi-objectively optimizes the model, outperforming GPT-3.5 and all 8B-tier baselines on 2Wiki, HotpotQA, and MuSiQue.
tags:
  - "ACL 2025"
  - "NLP Understanding"
date: 2026-05-08
content_hash: 6b1a147045395de9
---

# RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering

**Conference**: ACL 2025  
**arXiv**: [2505.21940](https://arxiv.org/abs/2505.21940)  
**Code**: None  
**Area**: NLP Understanding  

## TL;DR

Proposes RISE—a multi-hop QA framework combining RAG with self-iterative training. Through a self-exploration loop consisting of three actions—question decomposition, retrieve-and-read, and self-critique—it iteratively generates training data and multi-objectively optimizes the model, outperforming GPT-3.5 and all 8B-tier baselines on 2Wiki, HotpotQA, and MuSiQue.

## Background & Motivation

1. **Multi-Hop Question Answering (MHQA) remains a challenge for LLMs**: It requires integrating multiple sources of evidence and managing complex logical dependencies, which is particularly error-prone for small models.
2. **RAG suffers from two core types of errors**: (a) Evidence aggregation errors—the model fails to accurately integrate multiple retrieved snippets, leading to hallucinations; (b) Reasoning decomposition errors—sub-questions are inconsistent with the original question's intent, leading to deviation in the reasoning chain.
3. **Full-model gradient methods are too costly**: Distillation and human-annotated fine-tuning are effective but expensive, and human bias may harm performance.
4. **Gap in combining self-iteration and RAG**: Self-iterative methods have succeeded in code generation and agents, but remain unexplored in RAG multi-hop QA.

## Method

### Overall Architecture

RISE is a self-iterative closed-loop framework, where each round consists of two phases: **Self-Exploration** (generating training data) $\rightarrow$ **Iterative Optimization** (multi-objective model fine-tuning).

### 1. Self-Exploration Mechanism

For each question $q_0$, the model executes at most 20 rounds of exploration nodes:

**Question Decomposition**: Based on the existing history $\mathcal{H} = \{(subq_1, suba_1), \ldots\}$ and the original question $q_0$, the model generates the next sub-question $subq_t$; if the historical information is sufficient, it directly outputs the final answer.

**Retrieve-then-Read**: A retriever is used on the sub-question to obtain relevant snippets $r_t$, and the model generates a sub-answer $suba_t$ based on the retrieval results.

**Self-Critique**: The model evaluates the relevance of $(subq_t, suba_t)$ to solving the original question and outputs a binary judgment $\sigma_t \in \{0, 1\}$. If evaluated as False, the model backtracks to the previous valid node and regenerates.

The three actions collect datasets $\mathcal{D}_d$ (decomposition), $\mathcal{D}_r$ (reading), and $\mathcal{D}_c$ (critique), respectively, with 2K to 8K samples per category.

### 2. Multi-Objective Joint Optimization

The three datasets are trained jointly, with the total loss as:
$$\mathcal{L} = \alpha \mathcal{L}_d + \beta \mathcal{L}_r + \gamma \mathcal{L}_c$$

- $\mathcal{L}_d$: Autoregressive loss for sub-question generation
- $\mathcal{L}_r$: Sub-answer generation loss based on retrieval context
- $\mathcal{L}_c$: Cross-entropy loss for True/False binary classification
- In experiments, equal weights $\alpha = \beta = \gamma = 1$ are adopted to avoid overfitting

### 3. Question Expansion

After each round of optimization, the updated model is used to expand seed questions via in-context learning, generating more diverse training questions for the next round of self-exploration.

## Key Experimental Results

### Table 2: Main Results (Accuracy %)

| Method | Model | 2Wiki | HotpotQA | MuSiQue | NQ | WebQ | TriviaQA |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Naive LLM | LLaMA-3.1-8B | 35.90 | 27.30 | 11.30 | 57.50 | 61.25 | 71.50 |
| GPT-3.5-turbo | GPT-3.5 | 47.10 | 41.50 | 19.10 | 57.25 | 58.30 | 80.25 |
| CoT | LLaMA-3.1-8B | 43.00 | 34.60 | 16.20 | 56.75 | 62.00 | 71.75 |
| GenGround | LLaMA-3.1-8B | 37.90 | 36.10 | 17.80 | 48.50 | 44.50 | 75.25 |
| **RISE** | **LLaMA-3.1-8B**| **49.40** | **40.50** | **21.70** | **59.50** | **62.50** | **80.25** |

RISE outperforms GPT-3.5 on all MHQA datasets, achieving a 6-14 percentage point improvement compared to Naive RAG with the same model.

### Ablation Study (Round 1 Data, Accuracy %)

| Configuration | 2Wiki | HotpotQA | MuSiQue |
|:---|:---:|:---:|:---:|
| w/o Decomposition | 37.63 | 33.89 | 11.08 |
| w/o Retrieve-then-Read | 40.59 | 33.06 | 9.46 |
| w/o Self-Critique | 38.98 | 33.89 | 10.27 |
| Separate Training | 40.86 | 34.72 | 10.54 |
| **RISE (Joint Training)** | **41.13** | **35.83** | **11.89** |

All three sub-tasks are indispensable, and joint training outperforms separate training.

### Iterative Improvement

- Accuracy continuously increases with iterative rounds (4 rounds), whereas the reasoning chain length first increases and then decreases, indicating gradual optimization of decomposition capability.
- Critique consistency with GPT-4o improves from 60-74% in Round 1 to 78-81% in Round 4.

## Highlights & Insights

- **Innovative combination of self-iteration and RAG**: Introduces the self-iterative training paradigm into RAG multi-hop QA for the first time, without relying on LLM distillation or human annotation.
- **Three-task collaborative self-exploration**: Question decomposition, retrieve-and-read, and self-critique form a closed loop, automatically generating high-quality training data.
- **Multi-objective joint optimization**: The three types of data enable complementary learning, where joint training performance is superior to separate training.
- **8B model outperforms GPT-3.5**: LLaMA-3.1-8B trained with RISE comprehensively outperforms GPT-3.5 on MHQA tasks.

## Limitations & Future Work

- **Unoptimized retriever**: The framework relies on an external retriever but does not self-improve it; retrieval quality remains a bottleneck.
- **Validation limited to LLaMA-3.1-8B**: The performance on larger or smaller models has not been tested.
- **Self-exploration efficiency**: With at most 20 exploration nodes per question, the training data collection cost is relatively high for large-scale applications.
- **Equal weight strategy is sub-optimal**: The authors chose equal weights to avoid overfitting, but Table 1 shows that $(\alpha=2, \beta=2, \gamma=2)$ reaches 44.27% vs 41.13% with equal weights.

## Related Work & Insights

| Dimension | RISE | Self-RAG | GenGround | CoT |
|:---|:---|:---|:---|:---|
| Retrieval | Multi-turn RAG | Adaptive Retrieval | Alternate Generation & Retrieval | None |
| Self-Improvement | Self-Iterative Fine-Tuning | Reflection Token Training | None | None |
| Question Decomposition | Explicit Decomposition + Critique | None | Sub-question Guidance | Implicit Chain |
| Training Data | Self-Explored Generation | Human Annotation + GPT-4 | None | None |
| Multi-hop Capability | Strong (Iteratively Enhanced) | Weak | Medium | Medium |

## Rating

- ⭐⭐⭐⭐ Novelty: The combination of RAG and self-iteration represents a fresh exploration direction, with a comprehensive design of three-task closed-loop self-exploration.
- ⭐⭐⭐ Utility: Requiring 4 rounds of iterative training, the cost is not yet optimal compared to standard fine-tuning, but it does not depend on LLM annotations.
- ⭐⭐⭐⭐ Experimental Thoroughness: Evaluation across 3 MHQA + 3 SHQA datasets; comprehensive coverage which includes ablation, iterative analysis, and separate evaluation of the three capabilities.
- ⭐⭐⭐ Writing Quality: Clearly structured, but some formula symbols are inconsistent; the distinction between related work and the proposed method could be stronger.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multi-Hop Reasoning for Question Answering with Hyperbolic Representations](multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)
- [\[ACL 2025\] Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering](self-critique_guided_iterative_reasoning_for_multi-hop_question_answering.md)
- [\[ACL 2025\] Active LLMs for Multi-hop Question Answering](active_llms_for_multi-hop_question_answering.md)
- [\[ACL 2025\] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)
- [\[ACL 2025\] BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering](belle_a_bi-level_multi-agent_reasoning_framework_for_multi-hop_question_answerin.md)

</div>

<!-- RELATED:END -->
