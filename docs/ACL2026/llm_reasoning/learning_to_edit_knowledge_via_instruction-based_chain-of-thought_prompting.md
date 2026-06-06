---
title: >-
  [Paper Note] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting
description: >-
  [ACL 2026][LLM Reasoning][Knowledge Editing] CoT2Edit proposes a new paradigm for teaching LLMs to perform knowledge editing via CoT reasoning. By constructing CoT instruction data for both structured and unstructured ed…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Knowledge Editing"
  - "Chain-of-Thought"
  - "GRPO"
  - "RAG"
  - "Multi-hop Reasoning"
date: 2026-05-08
content_hash: 431ef5b10274a9dd
---

# Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting

**Conference**: ACL 2026  
**arXiv**: [2604.05540](https://arxiv.org/abs/2604.05540)  
**Code**: [https://github.com/FredJDean/CoT2Edit](https://github.com/FredJDean/CoT2Edit)  
**Area**: LLM Reasoning / Knowledge Editing  
**Keywords**: Knowledge Editing, Chain-of-Thought, GRPO, RAG, Multi-hop Reasoning

## TL;DR

CoT2Edit proposes a new paradigm for teaching LLMs to perform knowledge editing via CoT reasoning. By constructing CoT instruction data for both structured and unstructured editing, the model undergoes SFT cold-start followed by GRPO optimization. During inference, RAG is integrated to retrieve edited facts. A single training session achieves SOTA performance across six editing benchmarks and demonstrates strong generalization.

## Background & Motivation

**Background**: Knowledge editing aims to update outdated or incorrect knowledge within LLMs. Mainstream approaches include In-Context Editing (ICE/IKE), parameter modification (ROME/MEMIT/AlphaEdit), and train-retrieve paradigms (LTE/EditCoT).

**Limitations of Prior Work**: (1) Locate-and-edit methods (ROME/MEMIT) modify model parameters directly, which is incompatible with frozen production LLMs and suffers from "rote memorization" issues—where precise queries succeed but semantically equivalent queries fail; (2) LTE does not explicitly model reasoning paths, and requiring single-step generation of correct answers often leads to hallucinations; (3) EditCoT requires a multi-model pipeline (one for CoT generation, one for execution), which is complex and non-scalable; (4) Existing methods primarily handle structured fact triplets, ignoring unstructured knowledge such as news and articles.

**Key Challenge**: Current methods treat knowledge editing as a memory problem ("remembering new facts") rather than a reasoning problem ("understanding and reasoning with new facts"). SFT is prone to overfitting the training distribution, leading to poor generalization on OOD editing data.

**Goal**: To build a knowledge editing method that can generalize to various editing scenarios (structured/unstructured, single-hop/multi-hop) with a single training phase.

**Key Insight**: Knowledge editing is redefined as a two-stage function $f_{\theta'}(e,q) = g_{\theta'}(h_{\theta'}(e,q))$—first generating an interpretable reasoning chain $h$, then producing the answer $g$ based on that reasoning. SFT provides the cold start, while GRPO provides the generalization capability.

**Core Idea**: LLM agents generate CoT instructions for structured and unstructured editing data. SFT is used to learn the editing reasoning paradigm, and GRPO enhances generalization to unseen editing scenarios. During inference, RAG retrieves relevant edited facts.

## Method

### Overall Architecture

The framework consists of three stages: (1) Data construction—generating CoT instruction data from MQuAKE (structured) and MQuAKE-uns (unstructured), with data augmentation via HotpotQA entity relations; (2) Training—Phase 1 SFT cold-start to learn editing reasoning patterns, Phase 2 GRPO optimization on merged data to enhance generalization; (3) Inference—RAG retrieves relevant edited facts, and the model generates answers via CoT reasoning.

### Key Designs

1.  **CoT Instruction Data Construction**:
    - **Function**: Teaches the model to perform step-by-step reasoning starting from edited facts.
    - **Mechanism**: For structured data, an LLM agent generates reasoning chains based on edited facts $\mathcal{E}$ and multi-hop questions $\mathcal{Q}$: $\text{Agent}(\mathcal{Q}, \mathcal{E}, \mathcal{T}) \to \text{CoT}, \mathcal{A}$. For unstructured data, relevant facts are extracted from the context $\mathcal{C}$ before reasoning: $\text{Agent}(\mathcal{Q}, \mathcal{C}, \mathcal{T}) \to \mathcal{E}, \text{CoT}, \mathcal{A}$. Data augmentation synthesizes additional instructions (~10K) through HotpotQA entity relations.
    - **Design Motivation**: To cover both structured and unstructured editing scenarios; CoT provides explicit reasoning paths to reduce hallucinations.

2.  **Two-stage Training (SFT + GRPO)**:
    - **Function**: SFT provides a cold start for editing reasoning, while GRPO enhances OOD generalization.
    - **Mechanism**: Phase 1 SFT involves auto-regressive training on CoT instruction data. Phase 2 GRPO optimizes the model on merged data using a reward function $\mathcal{R} = \mathcal{R}_{acc} + \mathcal{R}_{format}$ (accuracy + format), employing a self-evolution strategy where high-reward samples are added to subsequent training rounds: $\mathcal{D}_{t+1} = \mathcal{D}_t \cup \{s | \mathcal{R}(s) > \theta\}$.
    - **Design Motivation**: Pure SFT easily overfits training editing patterns; GRPO improves generalization by exploring diverse reasoning paths. Self-evolution accelerates convergence.

3.  **RAG Knowledge Injection at Inference**:
    - **Function**: Dynamically retrieves relevant edited facts during inference without retraining.
    - **Mechanism**: The most relevant edited facts are retrieved as context for a user query. The model then uses its learned CoT reasoning ability to answer based on the retrieved facts.
    - **Design Motivation**: Decouples knowledge storage from reasoning capacity—the knowledge base can be updated at any time, while the model only needs to learn "how to reason based on given facts" once.

### Loss & Training

SFT: Standard auto-regressive cross-entropy. GRPO: Accuracy rewards + format rewards (including think/answer tags and keywords). Validated on Llama-3.1-8B, Qwen-2.5-7B, and DeepSeek-R1-Distill-Qwen-7B.

## Key Experimental Results

### Main Results (Comprehensive Performance on 6 Editing Benchmarks)

| Method | Edit Succ | Paraphrase | Neighborhood | Applicability |
| :--- | :--- | :--- | :--- | :--- |
| AlphaEdit | 88.78 | ~81 | ~70 | Structured Only |
| EditCoT | 86.13 | 83.55 | ~70 | Structured Only |
| **CoT2Edit** | **93.17** | **89** | **93** | Structured + Unstructured |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| SFT Only | Overfitting, poor OOD | Cold start but insufficient generalization |
| SFT + GRPO | Overall improvement | GRPO is the core contribution |
| No Data Augmentation | Insufficient GRPO training | 10K augmented data is crucial |
| No RAG | Performance drop | Retrieval provides critical edited facts |

### Key Findings

- A single training session generalizes to six unseen editing benchmarks, proving the model has learned a general "reasoning based on facts" capability.
- Unstructured knowledge editing reaches 92% accuracy (approx. 20% higher than IKE).
- Performance remains stable (89% paraphrase, 93% neighborhood success) even under large-scale editing (20K-30K facts vs. traditional 2K-3K).
- GRPO is applied to the knowledge editing field for the first time; the self-evolution strategy accelerates convergence.
- Demonstrates that RL is superior to pure SFT for achieving generalization in knowledge editing.

## Highlights & Insights

- Knowledge editing is redefined from a "memory problem" to a "reasoning problem"—the model is not required to memorize all edited facts but rather to learn how to reason given those facts. This paradigm shift is fundamental.
- The two-stage training strategy (SFT cold-start + GRPO generalization) is transferable to other tasks requiring OOD generalization.
- The self-evolution strategy (collecting high-reward samples for training) is a simple but effective data augmentation approach.

## Limitations & Future Work

- RAG retrieval quality directly impacts editing efficacy; failure may occur if relevant facts are not retrieved.
- The training data scale is approximately 13K; scaling behavior at much larger magnitudes has not been verified.
- Validated only on 7-8B models; larger models might exhibit different behaviors.
- Conflict resolution between conflicting edited facts is not explicitly addressed.

## Related Work & Insights

- **vs. AlphaEdit/MEMIT**: Parameter modification methods are incompatible with frozen models and suffer from rote memorization. CoT2Edit generalizes to semantic variants via reasoning.
- **vs. EditCoT**: Requires two independent LLMs (CoT generation + edit execution), whereas CoT2Edit works with a single model and supports unstructured editing.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of GRPO to knowledge editing; reasoning paradigm replaces memory paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 benchmarks, 3 models, multiple editing scenarios; comprehensive analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagrams, complete method description.
- Value: ⭐⭐⭐⭐ High practical value due to single-training generalization across multiple scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Does Self-Consistency Improve the Recall of Encyclopedic Knowledge?](does_self-consistency_improve_the_recall_of_encyclopedic_knowledge.md)
- [\[AAAI 2026\] Intention Chain-of-Thought Prompting with Dynamic Routing for Code Generation](../../AAAI2026/llm_reasoning/intention_chain-of-thought_prompting_with_dynamic_routing_for_code_generation.md)
- [\[ACL 2026\] Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization](is_chain-of-thought_really_not_explainability_chain-of-thought_can_be_faithful_w.md)
- [\[ACL 2026\] ETR: Entropy Trend Reward for Efficient Chain-of-Thought Reasoning](etr_entropy_trend_reward_for_efficient_chain-of-thought_reasoning.md)
- [\[ICML 2026\] Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning](../../ICML2026/llm_reasoning/clustering_as_reasoning_a_k-means_interpretation_of_chain-of-thought_graph_learn.md)

</div>

<!-- RELATED:END -->
