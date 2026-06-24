---
title: >-
  [Paper Note] Self-Taught Agentic Long-Context Understanding
description: >-
  [ACL 2025][LLM Agent][long-context understanding] The AgenticLU framework is proposed, which enables LLMs to autonomously generate clarification questions and retrieve relevant context via a Chain-of-Clarifications (CoC) workflow. By distilling search paths from tree search into the model using a two-stage SFT+DPO fine-tuning, an 8B model significantly outperforms baselines on 128K long-context QA tasks.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "long-context understanding"
  - "agentic workflow"
  - "chain-of-clarifications"
  - "inference-time scaling"
  - "self-taught reasoning"
date: 2026-05-08
content_hash: a4e7df9e429edf0b
---

# Self-Taught Agentic Long-Context Understanding

**Conference**: ACL 2025  
**arXiv**: [2502.15920](https://arxiv.org/abs/2502.15920)  
**Code**: [https://github.com/EvanZhuang/AgenticLU](https://github.com/EvanZhuang/AgenticLU)  
**Area**: LLM Agent  
**Keywords**: long-context understanding, agentic workflow, chain-of-clarifications, inference-time scaling, self-taught reasoning

## TL;DR
The AgenticLU framework is proposed, which enables LLMs to autonomously generate clarification questions and retrieve relevant context via a Chain-of-Clarifications (CoC) workflow. By distilling search paths from tree search into the model using a two-stage SFT+DPO fine-tuning, an 8B model significantly outperforms baselines on 128K long-context QA tasks.

## Background & Motivation

**Background**:
   - While current LLMs support context windows of 128K or even 2M tokens, their performance in actual long-document understanding tasks falls far short of what their nominal capacity suggests.
   - A severe gap exists between the "nominal context size" and the "effective context size."
   - On HotpotQA, the accuracy of Llama3.1-8B-Instruct drops sharply as the context scales from 8K to 128K.

**Limitations of Prior Work**:
   - When directly processing ultra-long texts, models tend to lose key information located in the middle segments (the "lost-in-the-middle" effect).
   - Existing methods like ProLong require fine-tuning on an additional 40B tokens of long-context corpora, incurring extremely high training costs.
   - Prompting-based methods (such as Chain-of-Thought, Plan-and-Solve) suffer from severe performance degradation at extreme lengths (128K).

**Key Challenge**:
   - A huge chasm exists between the model's nominal context capacity (how long an input it can accept) and its effective context capability (how much of the input it can actually utilize).
   - Analogous to computer memory: having a larger capacity alone does not equate to efficient computation; an intelligent "information loading" mechanism is also required.

**Goal**:
   - How to enhance the LLM's capability to understand and utilize long contexts without relying on human annotations or stronger teacher models.
   - How to amortize the high computational overhead during inference into the training phase.

**Key Insight**:
   - Reformulate long-context understanding as an agentic workflow of iterative self-clarification and context localization.
   - Leverage tree search during inference to collect high-quality reasoning paths, which are then distilled back into the model.

**Core Idea**:
   - Allow the model to generate its own clarification questions, retrieve evidence itself, and answer them. This capability is then internalized through SFT+DPO to achieve "self-taught" long-context understanding.

## Method

### Overall Architecture
AgenticLU consists of two core phases:
1. **CoC Path Construction** (inference-time tree search): Generates diverse Chain-of-Clarifications paths using tree search.
2. **CoC Path Distillation** (training-time knowledge transfer): Distills the searched paths into the model via a two-stage fine-tuning of SFT + DPO.

### Key Designs

1. **Chain-of-Clarifications (CoC)**:

    - **Function**: In each CoC step, the model autonomously executes three actions: (1) generating clarification questions to identify potentially misunderstood areas; (2) locating relevant paragraphs using the pointback mechanism; (3) answering the clarification questions and the original question based on the collected evidence.
    - **Mechanism**: Instead of processing the entire long context all at once, the task is decomposed into a series of targeted sub-tasks to progressively refine the understanding.
    - **Design Motivation**: Simulates the natural process of humans reading long documents—referring back to verify whenever encountering uncertainty.

2. **Pointback Mechanism**:

    - **Function**: Highlights key context segments by labeling the index numbers of the relevant paragraphs.
    - **Mechanism**: During the data collection phase, the context is chunked into 512-token segments, and the LLM is queried one by one to determine relevance; after training, the model directly generates the paragraph numbers.
    - **Design Motivation**: Internalizes the computationally intensive block-by-block retrieval process into the model's intuitive capability.

3. **Tree Search Data Construction**:

    - **Function**: Constructs search trees with a branching factor of 8 and a maximum depth of 3, where each node represents a CoC step.
    - **Mechanism**: Selects the optimal paths using a combined scoring of RougeL semantic similarity and GPT-4o-mini binary validation.
    - **Design Motivation**: 92% of the questions can be resolved with just one round of clarification; two rounds resolve an additional 53% of the remainder; and three rounds resolve another 35%, ultimately covering 97.8% of the correct answers.

### Loss & Training

- **First Stage (SFT)**: Trains the model using standard cross-entropy loss to learn CoC reasoning paths, containing the full context + question + step-by-step reasoning chain.
- **Second Stage (DPO)**: Uses incorrect reasoning paths as negative samples (where correctness is determined by GPT-4o-mini) to create preference pairs for Direct Preference Optimization.
- The base model is Llama3.1-8B-Instruct. The training data comes from NarrativeQA (14.7K QA pairs), generating 107,550 traces, with an average context length of 67K, and total generated tokens of 17M.

## Key Experimental Results

### Main Results
- **Long-Context Tasks (128K) Average**: AgenticLU-8B improves over Llama3.1-8B by **+14.7 points** ($53.4 \rightarrow 68.1$).
- **HotpotQA (128K)**: $+31.1$ ($40.0 \rightarrow 71.1$) — the multi-hop reasoning task with the most significant improvement.
- **NaturalQ (128K)**: $+21.7$ ($56.1 \rightarrow 77.8$)
- **TriviaQA (128K)**: $+7.7$ ($80.6 \rightarrow 88.3$)
- **NarrativeQA (128K)**: $+18.0$ ($38.0 \rightarrow 56.0$)
- **Short-Context Tasks Average**: Drops by only $-0.6$ points ($62.3 \rightarrow 61.7$), with almost no impact on general capabilities.
- Consistently outperforms prompting methods and ProLong-8B across all 7 long-context tasks and all context lengths (8K to 128K).

### Ablation Study / Key Findings
- **Effect of Multi-Turn CoC**: $1\text{ turn} \rightarrow 75.7\%$; $2\text{ turns} \rightarrow 76.7\%$; $3\text{ turns} \rightarrow 78.4\%$ (average of 4 RAG tasks on 128K), with the first turn obtaining most of the gains.
- **Removing Self-Clarification**: Average accuracy drops from $75.7\%$ to $62.1\%$ ($-13.6$), and from $71.1\%$ to $57.8\%$ on HotpotQA.
- **Removing Pointback**: Average accuracy drops from $75.7\%$ to $62.2\%$ ($-13.5$), demonstrating that context localization is equally critical.
- **Tree Search Coverage**: Reaches $97.8\%$ answer recall on NarrativeQA with a depth of 3 and a branching factor of 8.
- With prefix caching, the additional inference overhead scales linearly only with the number of newly generated tokens.

## Highlights & Insights

- **Self-taught Paradigm**: It does not rely on human annotations or stronger teacher models; the base model itself generates training data to teach itself. The "self-taught" concept is highly elegant.
- **Amortization of Inference Time $\rightarrow$ Training Time**: Transfers the expensive tree search cost into one-time training, requiring only a single forward pass during inference.
- **Pointback Mechanism**: Ingeniously internalizes RAG-style retrieval capabilities into the generative model, avoiding the introduction of external retrievers.
- **Generalization Preservation**: Performance on short-context tasks remains almost intact after fine-tuning ($-0.6$ points), indicating well-constructed data.

## Limitations & Future Work

- Training data comes solely from a single dataset, NarrativeQA; generalizability depends on data diversity.
- Search depth is restricted to 3 (limited by exponential computational costs); deeper reasoning may require alternative strategies.
- The base model uses 8B parameters; whether larger models can benefit further remains unexplored.
- The quality of CoC paths highly depends on the long-context understanding capability of the initial model; models with weak capabilities may fail to generate effective clarification questions.
- Lacks an in-depth comparison against RAG + external retriever approaches.

## Related Work & Insights

- It shares the same philosophy as the STaR (Self-Taught Reasoner) line of work, but extends from mathematical reasoning to long-context understanding.
- While LongRAG and Chain-of-Agents require multi-component/multi-agent collaboration, AgenticLU uses only a single LLM to orchestrate both reasoning and retrieval.
- ProLong-8B requires 40B tokens of additional training data, whereas AgenticLU is much more data-efficient (17M generation tokens).
- The utilization of DPO continues the successful paradigm of the RLHF family in LLM alignment, presenting a novel application scenario for long-context understanding.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Resolving Evidence Sparsity: Agentic Context Engineering for Long-Document Understanding](../../CVPR2026/llm_agent/resolving_evidence_sparsity_agentic_context_engineering_for_long-document_unders.md)
- [\[ACL 2025\] Agentic Knowledgeable Self-Awareness](agentic_knowledgeable_self-awareness.md)
- [\[NeurIPS 2025\] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](../../NeurIPS2025/llm_agent/deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)
- [\[ACL 2025\] Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement](gödel_agent_a_self-referential_agent_framework_for_recursive_self-improvement.md)

</div>

<!-- RELATED:END -->
