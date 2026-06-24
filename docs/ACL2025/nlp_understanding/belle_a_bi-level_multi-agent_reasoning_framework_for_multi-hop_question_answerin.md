---
title: >-
  [Paper Note] BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering
description: >-
  [ACL2025][NLP Understanding][multi-hop QA] This paper proposes BELLE, a bi-level multi-agent debate framework. It first classifies multi-hop questions into four types, and then dynamically plans the optimal combination scheme of operators (such as CoT, single-step retrieval, and iterative retrieval) through a bi-level debate mechanism (a first-level affirmative-negative debate + a second-level fast/slow debater supervision), realizing adaptive multi-hop reasoning tailored to…
tags:
  - "ACL2025"
  - "NLP Understanding"
  - "multi-hop QA"
  - "multi-agent debate"
  - "operator combination"
  - "question type classification"
  - "retrieval-augmented reasoning"
date: 2026-05-08
content_hash: bbb716fa987361f2
---

# BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering

**Conference**: ACL2025  
**arXiv**: [2505.11811](https://arxiv.org/abs/2505.11811)  
**Code**: Unreleased  
**Area**: NLP Understanding  
**Keywords**: multi-hop QA, multi-agent debate, operator combination, question type classification, retrieval-augmented reasoning

## TL;DR

This paper proposes BELLE, a bi-level multi-agent debate framework. It first classifies multi-hop questions into four types, and then dynamically plans the optimal combination scheme of operators (such as CoT, single-step retrieval, and iterative retrieval) through a bi-level debate mechanism (a first-level affirmative-negative debate + a second-level fast/slow debater supervision), realizing adaptive multi-hop reasoning tailored to different question types.

## Background & Motivation

Multi-hop QA requires step-by-step reasoning across multiple documents and remains one of the core challenges for LLMs. Existing methods generally fall into two categories:

- **Closed-book reasoning** (CoT, sub-question decomposition, probabilistic reasoning trees): Completely rely on the internal knowledge of LLMs, which is prone to hallucination when facing knowledge-intensive questions.
- **Retrieval-augmented reasoning** (single-step retrieval, iterative retrieval, adaptive retrieval): Introduce external knowledge but typically employ static workflows, disregarding differences in question types. This leads to excessive computational overhead on simple questions and insufficient retrieval for complex ones.

The key insight of the authors is that **different types of multi-hop questions exhibit significantly varied sensitivities to different reasoning methods**. Specifically:
- **Inference type** (reasoning across documents): Requires a combination of sub-question decomposition and iterative retrieval.
- **Comparison/Temporal type** (comparison or chronological): Only needs sub-question decomposition + single-step retrieval.
- **Null type** (no fixed pattern): Requires an adaptive combination.

This observation drives the authors to design a framework that dynamically selects the operator combination based on the question type, rather than applying a single rigid workflow to all problems.

## Method

### Overall Architecture

BELLE consists of three modules: Question Type Classifier $\rightarrow$ Bi-Level Multi-Agent Debate $\rightarrow$ Multi-Hop QA Executor. This framework treats five methods—CoT, single-step retrieval, iterative retrieval, sub-question decomposition, and adaptive retrieval—as "operators." The objective of the debate system is to plan the optimal operator combination for any given question.

### Module 1: Question Type Classifier

An in-context learning (ICL) method is utilized to classify multi-hop questions into four categories:
- **Inference**: Requires reasoning across multiple documents and involves multiple logical steps.
- **Comparison**: Compares attributes of two entities, and answers are usually concise (e.g., Yes/No).
- **Temporal**: Involves identifying and comparing timestamps.
- **Null**: Does not belong to the above categories and has no fixed pattern.

The classification result is injected as prior knowledge into the meta prompt of the subsequent debate system to guide the operator selection process. For datasets lacking type annotations (such as HotpotQA, 2WikiQA, and MuSiQue), GPT-4 was employed for annotation, which was further verified by humans (achieving 95% accuracy).

### Module 2: Bi-Level Multi-Agent Debate System

This is the core innovation of BELLE. On top of the traditional multi-agent debate (MAD) setup, a second-level memory agent is introduced to monitor the quality of the debate.

**First Level — Affirmative/Negative Debate**:
- The affirmative and negative debaters conduct adversarial debates regarding the operator selection scheme in a turn-by-turn manner.
- Each round of debate is based on their respective historical records $H_{ad}^{t-1}$ along with the feedback from the fast/slow debaters of the previous round.
- The round-$t$ output of the affirmative debater is: $f_{ad}^t = \mathcal{M}(H_{ad}^{t-1}, f_{fast}^{t-1}, f_{slow}^{t-1})$

**Second Level — Fast/Slow Debater Supervision**:
- **Fast Debater**: Focuses on whether the perspectives of both sides in the **current round** are sensible and evaluates the current operator selection scheme. However, it can be easily influenced by the arguments of both sides.  
  $f_{fast}^t = \mathcal{M}(f_{ad}^t, f_{nd}^t, H_{fast}^{t-1})$
- **Slow Debater**: Integrates **all historical information** for a comprehensive judgment, preventing the debaters from losing the correct stance due to mutual influence and causing perspective oscillations.  
  $f_{slow}^t = \mathcal{M}(f_{ad}^t, f_{nd}^t, f_{fast}^t, H_{slow}^{t-1})$

The naming of the fast/slow debater is inspired by the Talker-Reasoner architecture (System 1 vs. System 2 thinking), where the fast debater concentrates on the immediate state and the slow debater is responsible for deep synthesis.

**Judge**: Monitors the entire debate process and has two termination modes:
- **Hard Mode**: A consensus is reached during the debate, and the operator combination scheme is output directly.
- **Soft Mode**: If no consensus is reached after running out of rounds, the judge extracts the most reasonable operator proposal from the history of the slow debater.

### Module 3: Multi-Hop QA Executor

According to the execution plan provided by the debate, the corresponding operators are called in sequence. For example, it first splits the original question using sub-question decomposition, then executes iterative retrieval on each sub-question to obtain external knowledge, and finally merges the sub-answers into the final output.

## Key Experimental Results

### Datasets and Baselines
- **Datasets**: MultiHop-RAG, HotpotQA, 2WikiMultiHopQA, MuSiQue (2/3/4 hops).
- **Baselines**: 13 methods, including closed-book reasoning (SP, CoT), retrieval-augmented reasoning (Single-step, IRCoT, FLARE, ProbTree, BeamAggR), and agent-based methods (LONGAGENT, GEAR, RopMura).
- **LLMs**: GPT-3.5-turbo, Qwen2.5-7B.

### Table 1: Main Results (F1 %)

| Method | MultiHop-RAG | HotpotQA | 2WikiQA | MuSiQue-2hop | MuSiQue-3hop | MuSiQue-4hop |
|------|------|------|------|------|------|------|
| CoT | 50.5 | 46.5 | 42.3 | 30.2 | 22.5 | 13.2 |
| IRCoT | 59.2 | 56.2 | 56.8 | 31.4 | 19.2 | 16.4 |
| ProbTree | 62.5 | 60.4 | 67.9 | 41.2 | 30.9 | 14.4 |
| BeamAggR | 67.2 | 62.9 | 71.6 | 45.9 | 36.8 | 21.6 |
| LONGAGENT | 56.8 | 59.3 | 65.6 | 40.5 | 25.8 | 16.4 |
| **BELLE** | **70.4** (+3.2) | **66.5** (+3.6) | **75.7** (+4.1) | **50.5** (+4.6) | **42.1** (+5.3) | **29.2** (+7.6) |

BELLE achieves the best F1 across all datasets, and **the performance gain increases with the number of hops** (+7.6% on 4-hop), indicating that dynamic operator combination is particularly effective for complex reasoning.

### Table 2: Ablation Study (F1 %, GPT-3.5-turbo)

| Configuration | MultiHop-RAG | HotpotQA | 2WikiQA | MuSiQue | Avg. |
|------|------|------|------|------|------|
| BELLE (Full) | 70.4 | 66.5 | 75.7 | 40.6 | 63.3 |
| w/o Type Classifier | 67.9 | 63.4 | 73.2 | 37.6 | 60.5 |
| w/o First-Level Debate | 68.2 | 63.7 | 73.5 | 38.1 | 60.9 |
| w/o Second-Level Debate | 66.8 | 62.8 | 72.3 | 36.5 | 59.6 |
| w/o Fast Debater | 67.3 | 63.2 | 72.9 | 37.4 | 60.2 |
| w/o Slow Debater | 67.0 | 63.1 | 72.7 | 36.9 | 59.9 |

**Key Findings**: Removing the second-level debate has the most significant impact (-3.7% avg.), showing that the history integration mechanism of the fast/slow debaters is core to the performance. The slow debater is more critical than the fast debater.

### Table 3: Computational Overheads Comparison (Average prompt tokens)

| Method | Avg. Token |
|------|------|
| BELLE | 20,742 |
| LONGAGENT | 48,493 |
| GEAR | 41,931 |
| RopMura | 56,859 |

BELLE's token consumption is only **36% to 57%** of other agent-based methods, as the debate typically determines the operator plan within 2 rounds.

## Key Findings

1. **Operator combinations outperform single operators**: Across all question types, combining operators yields a 3% average improvement compared to using a single operator.
2. **Question types dictate the optimal combination**: Inference is suited for sub-questions + iterative retrieval, while Comparison/Temporal is suited for sub-questions + single-step retrieval.
3. **More hops yield more pronounced advantages**: The improvement of 7.6% on MuSiQue 4-hop shows that dynamic planning delivers greater value at higher complexity levels.
4. **Fast debate convergence**: The operator combination is typically resolved within 2 debate rounds, yielding a token cost far lower than other agent-based methods.
5. **The slow debater is more critical than the fast debater**: The slow debater prevents perspective oscillation by integrating global history, contributing more significantly to performance.

## Highlights & Insights

- **Perspective of operator combinations**: Abstracting existing multi-hop QA methods as "operators" and transforming the method-selection problem into a planning problem offers a clear and inspiring outlook.
- **System 1/2 analogy in bi-level debate**: The fast debater behaves like an intuitive reaction (focusing on current states), while the slow debater represents deliberate thinking (synthesizing history). This designs multi-agent collaboration by borrowing from the dual-system theory in cognitive science.
- **Win-win in cost and effectiveness**: It not only secures the best performance but also consumes the fewest tokens. The primary reason is that precise planning based on question types avoids redundant retrieval rounds.
- **Effectiveness on Qwen2.5-7B**: Achieving substantial improvements on a 7B open-source model (avg. 56.2 vs. 48.3 for BeamAggR) indicates that the framework is not dependent on ultra-large models.

## Limitations & Future Work

- **Optimization space for debate overhead**: Although its token consumption is lower than other agent-based methods, the iterative multi-agent debate itself still incurs overhead. There is room to optimize the debate strategies and rules.
- **Generalizability of question types**: Only 4 types are currently defined. Performance may drop when encountering brand new/unseen question formats, and the scalability of the taxonomy requires verification.
- **Reliance on classifier accuracy**: The pipeline is built on the premise that question types are classified correctly; classification errors will lead to flawed operator planning. However, the paper does not analyze the cascading effects of classification errors in detail.
- **Fixed operator pool**: The current operator pool contains 5 methods. The study does not explore how to dynamically expand to new operators or adapt to the retrieval requirements of different domains.
- **Evaluation limited to English datasets**: The four datasets are all in English, and its applicability in multilingual scenarios remains unknown.

## Related Work & Insights

- **BeamAggR** (ACL 2024): Beam-search reasoning with multi-source knowledge aggregation, which is the strongest baseline (avg. F1 59.1). BELLE further improves upon it by leveraging prior knowledge of question types.
- **Talker-Reasoner Architecture** (Christakopoulou et al., 2024): The inspiration for the fast/slow debaters, bringing the cognitive science concepts of System 1 (fast response) and System 2 (deep reasoning) into multi-agent systems.
- **Evolution of MAD Frameworks**: Moving from the basic three-role setup (affirmative, negative, and judge) to the bi-level five-role setup in this paper, representing an evolutionary direction in the structural design of multi-agent debate systems.
- **Insights**: Modeling the method selection itself as a debatable decision-making problem represents a meta-reasoning approach that can be extended to other NLP tasks requiring dynamic strategy selection (such as summarization and translation).

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective of dynamically combining operators based on question types is novel, and the fast/slow design of the bi-level debate has clear cognitive science motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage with 4 datasets, 13 baselines, detailed ablation, computational overhead analysis, and visualization of operator selection dynamics.
- Writing Quality: ⭐⭐⭐⭐ — Plain and logical motivation analysis, rigorous method description, although some formulas could be simplified due to heavy notation.
- Value: ⭐⭐⭐⭐ — Consistent SOTA performance on multi-hop QA tasks with lower token overhead, providing practical guidance for real-world applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multi-Hop Reasoning for Question Answering with Hyperbolic Representations](multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)
- [\[ACL 2025\] Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering](self-critique_guided_iterative_reasoning_for_multi-hop_question_answering.md)
- [\[ACL 2025\] Active LLMs for Multi-hop Question Answering](active_llms_for_multi-hop_question_answering.md)
- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [\[ACL 2025\] ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision](rescore_multihop_qa.md)

</div>

<!-- RELATED:END -->
