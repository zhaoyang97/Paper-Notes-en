---
title: >-
  [Paper Note] Contextual Experience Replay for Self-Improvement of Language Agents
description: >-
  [ACL 2025][Experience Replay] CER (Contextual Experience Replay) proposes a training-free self-improvement framework for language agents. By accumulating and legacy-synthesizing past interaction experiences into a dynamic memory buffer, it allows the agent to retrieve relevant knowledge during inference to enhance decision-making on new tasks, achieving a 51.0% relative success rate improvement over the GPT-4o baseline on WebArena.
tags:
  - "ACL 2025"
  - "Experience Replay"
  - "Self-Improvement"
  - "Language Agents"
  - "In-Context Learning"
  - "Web Navigation"
date: 2026-05-08
content_hash: d233bdce5442bcdc
---

# Contextual Experience Replay for Self-Improvement of Language Agents

**Conference**: ACL 2025  
**arXiv**: [2506.06698](https://arxiv.org/abs/2506.06698)  
**Code**: None  
**Area**: Others  
**Keywords**: Experience Replay, Self-Improvement, Language Agents, In-Context Learning, Web Navigation

## TL;DR

CER (Contextual Experience Replay) proposes a training-free self-improvement framework for language agents. By accumulating and legacy-synthesizing past interaction experiences into a dynamic memory buffer, it allows the agent to retrieve relevant knowledge during inference to enhance decision-making on new tasks, achieving a 51.0% relative success rate improvement over the GPT-4o baseline on WebArena.

## Background & Motivation

**Background**: LLM agents have demonstrated potential in sequential decision-making tasks such as web navigation, with representative methods including ReAct, SteP, and AgentQ. These methods typically place the LLM directly into the environment, defining the task and available operations through prompts to guide step-by-step decision-making. However, most methods treat each task as independent—the agent does not retain any experience after completing a task, leading to repeating the same mistakes when encountering similar situations in the future.

**Limitations of Prior Work**: (1) **Lack of environment-specific knowledge**—The pre-trained knowledge of LLMs is general and lacks understanding of specific website layouts, operational logic, and common pitfalls, leading to frequent failures in complex environments; (2) **Inability to learn from experience**—Current agents lack a mechanism to continuously accumulate and exploit past experiences during the inference stage, starting "from scratch" each time; (3) **High training cost**—Injecting experiential knowledge through fine-tuning requires substantial annotated data and computational resources.

**Key Challenge**: Humans naturally accumulate experience when operating websites (e.g., "the search function of this website is in the upper left corner", "logging in is required before submitting an order"), but the context window of LLM agents is blank at the start of each task. How to enable agents to dynamically accumulate and exploit experience at inference time within the context window, without training, is critical to improving agent practicality.

**Goal**: Design a training-free framework that enables LLM agents to accumulate, synthesize, and retrieve experience within the context window, allowing continuous learning from past task executions to improve performance on subsequent tasks.

**Key Insight**: Starting from the classic concept of "experience replay" in reinforcement learning, the authors adapt it to the in-context learning paradigm of LLM agents. Instead of storing experience through gradient updates, experiences are stored as text in a dynamic memory buffer. When facing a new task, relevant experiences are retrieved to enhance the agent's decision-making.

**Core Idea**: Transfer the concept of reinforcement learning experience replay to the LLM context window. By accumulating, synthesizing, and retrieving past task experiences (environmental dynamics, decision-making patterns), the agent achieves continuous self-improvement without any training.

## Method

### Overall Architecture

The workflow of CER is as follows: as the agent executes a task and interacts with the environment, it processes and stores the interaction trajectory into the memory buffer upon completion (regardless of success or failure). When executing a new task, the most relevant experiences are retrieved from the buffer based on the current task description and added to the prompt as in-context examples to help the agent make better decisions. As more tasks are executed, the buffer continuously enriches, and the agent's performance steadily improves.

### Key Designs

1. **Dynamic Memory Buffer**:

    - **Function**: Stores and manages accumulated interaction experiences of the agent, supporting efficient retrieval.
    - **Mechanism**: Each experience in the buffer is not a simple recording of interaction trajectories, but synthesized and structured knowledge. It contains two types of information: (1) environmental dynamics knowledge—operational logic of specific websites/pages, element locations, common issues, etc.; (2) decision-making patterns—which sequences of actions are effective in specific scenarios. The buffer employs a dynamic update strategy where new experiences can supplement or replace old ones to avoid outdated information.
    - **Design Motivation**: Storing raw trajectories directly would consume too much of the context window and introduce substantial redundancy. Through synthesis and structuring, more useful experiences can be preserved within a limited context space.

2. **Experience Synthesis**:

    - **Function**: Extracts reusable knowledge from raw interaction trajectories.
    - **Mechanism**: After a task is completed, an LLM is utilized to reflect on and summarize the interaction trajectory, extracting key environmental knowledge and decision-making patterns. For instance, from the trajectory of "searching for a post on Reddit", it extracts "the search bar of Reddit is in the top navigation panel, needing a click before entering keywords". Successful trajectories extract positive experiences, while failed trajectories extract lessons and mistake-avoidance guides. Synthesized experiences are more compact and generalizable than raw trajectories.
    - **Design Motivation**: Raw trajectories contain numerous step-level details, making them highly inefficient as few-shot exemplars. Experience synthesis transforms "What happened" into "What to know", rendering the knowledge more compact and generalized.

3. **Contextual Retrieval Augmentation**:

    - **Function**: Quickly retrieves the most relevant historical experiences when executing a new task.
    - **Mechanism**: Given a new task description and the current environment state, similarity matching (e.g., based on semantic similarity of task descriptions, overlap of websites/functions involved) is performed to retrieve the top-k most relevant experiences from the buffer. The retrieved experiences are formatted and inserted into the agent's prompt as a reference for decision-making. As the buffer grows, the quality of retrieval continuously improves.
    - **Design Motivation**: Different tasks require support from different experiences. Blindly stuffing all experiences into the context wastes tokens and introduces noise. Retrieving only relevant experiences strikes a balance between efficiency and effectiveness.

### Loss & Training

CER is a completely training-free framework that does not involve gradient updates or loss functions. All "learning" occurs within the context window through experience accumulation and prompt adjustment, allowing CER to be directly applied to any off-the-shelf LLM without extra training.

## Key Experimental Results

### Main Results

Success rate comparison on WebArena and VisualWebArena benchmarks:

| Method | WebArena Success Rate (%) | VisualWebArena Success Rate (%) | Requires Training |
|------|-------------------|------------------------|------------|
| GPT-4o (baseline) | ~24.3 | ~26-28 | No |
| ReAct | ~15-20 | - | No |
| SteP | ~28-30 | - | Fine-tuning |
| AgentQ | ~35+ | - | RL Training |
| **CER (Ours)** | **36.7** | **31.9** | **No** |

CER reaches 36.7% on WebArena, representing a **51.0%** relative improvement over the GPT-4o baseline.

### Ablation Study

| Configuration | WebArena Success Rate | Note |
|------|----------------|------|
| Full CER | 36.7% | Complete framework |
| w/o Experience Synthesis | Significant decline | Directly uses raw trajectories, inefficient knowledge utilization |
| w/o Retrieval (Stuffing all) | Decline | Irrelevant experiences introduce noise |
| Using successful trajectories only | Slight decline | Failed experiences are also valuable (mistake-avoidance guides) |
| Using failed trajectories only | Moderate decline | Successful experiences are the primary source of gain |
| Scaling with task count | Continuous improvement | More experiences lead to better results, validating continuous learning |

### Key Findings

- **Experience synthesis is critical**: Removing the experience synthesis module significantly degrades performance, demonstrating that translating raw trajectories into structured knowledge is crucial, as using raw trajectories for few-shot learning is highly inefficient.
- **Failed experiences are also highly valuable**: Although successful experiences contribute more, incorporating "lesson summaries" from failed experiences brings extra gains. This aligns with human learning intuition—we also learn from mistakes.
- **More experiences are better, but with diminishing returns**: As the number of experiences in the buffer increases, agent performance continuously improves, but the growth rate gradually slows. This suggests the potential value of smarter experience management strategies (e.g., deduplication, compression).
- **Training-free methods can approach the performance of trained methods**: CER requires no training but achieves performance on par with or even exceeding AgentQ (which requires RL training) on WebArena, demonstrating the potential of the in-context learning paradigm.

## Highlights & Insights

- **Elegant transfer of RL concepts to LLM contexts**: Moving experience replay from the parameter space to the context space is an ingenious analogy. Learning is achieved purely by managing experiences within prompts without updating model parameters, significantly lowering deployment costs.
- **51% relative improvement indicates a severe lack of environmental knowledge**: As a powerful general model, GPT-4o performs poorly primarily due to the lack of environment-specific knowledge. CER achieves a massive boost simply by filling this "knowledge gap", indicating that the core challenge of web navigation lies in environmental knowledge rather than reasoning ability.
- **Experience synthesis is more important than experience storage**: This finding suggests that the key to an agent's memory system is not "how much it remembers" but "what it extracts". This aligns with the concept of "schema theory" in cognitive science.

## Limitations & Future Work

- **Context window constraints**: Storing all experiences in the context is limited by the context length of the LLM. Once the volume of experience exceeds the window capacity, more sophisticated management strategies are needed.
- **Dependency on LLM synthesis capability**: The quality of experience synthesis depends on the LLM's capability to correctly extract key knowledge from trajectories. If the LLM itself misinterprets certain actions, the synthesized knowledge will also be flawed.
- **Environment specificity**: The experiences accumulated by CER are environment-specific; cross-website/platform transferability has not been validated.
- **API Cost**: While training-free, each task execution requires additional LLM calls for experience synthesis and retrieval, increasing inference cost.
- **Future directions**: Exploring cross-environment experience transfer, more efficient experience compression algorithms, and combining CER with training-based methods for further performance gains.

## Related Work & Insights

- **vs ReAct**: ReAct uses a thought-action-observation loop to improve decisions within a single task but does not accumulate experience across tasks. CER adds a cross-task experience reuse dimension on top of ReAct.
- **vs SteP**: SteP injects experience via fine-tuning, requiring training data and compute resources. CER achieves comparable effects while being completely training-free, lowering deployment barrier.
- **vs AgentQ**: AgentQ utilizes RL training to learn decision policies, representing "parameter-space experience utilization." CER stands for "context-space experience utilization." The two are complementary and could theoretically be combined.
- **vs Reflexion**: Reflexion also leverages LLM reflection for improvement, but focuses on trial-and-error loops within a single task. CER's experience buffer supports cross-task knowledge transfer, aligning better with the philosophy of "continuous learning."

## Rating

- Novelty: ⭐⭐⭐⭐ The transfer of experience replay to in-context learning is novel and elegant, and the training-free design lowers the adoption barrier.
- Experimental Thoroughness: ⭐⭐⭐⭐ Verified on both WebArena and VisualWebArena benchmarks with comprehensive ablation studies; the 51% improvement is convincing.
- Writing Quality: ⭐⭐⭐⭐ A content-rich 20-page paper; methodology descriptions are clear and analysis is thoroughly deep.
- Value: ⭐⭐⭐⭐⭐ The practical value of training-free self-improvement is exceptionally high, providing significant progress for continuous learning research in agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Truly Self-Improving Agents Require Intrinsic Metacognitive Learning](../../ICML2025/others/truly_self-improving_agents_require_intrinsic_metacognitive_learning.md)
- [\[ACL 2025\] Inducing Lexicons of In-Group Language with Socio-Temporal Context](inducing_lexicons_of_in-group_language_with_socio-temporal_context.md)
- [\[ACL 2025\] Autalic: A Dataset for Anti-Autistic Ableist Language In Context](autalic_a_dataset_for_anti-autistic_ableist_language_in_context.md)
- [\[ACL 2025\] Learning to Reason Over Time: Timeline Self-Reflection for Temporal Reasoning](tiser_timeline_self_reflection_temporal.md)
- [\[ACL 2025\] Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate](improve_rule_retrieval_and_reasoning_with_self-induction_and_relevance_reestimat.md)

</div>

<!-- RELATED:END -->
