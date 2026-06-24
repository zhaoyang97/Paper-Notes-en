---
title: >-
  [Paper Note] Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks
description: >-
  [ACL 2025][LLM (Other)][Critic-Guided Planning] This paper proposes the CR-Planner framework, which guides the planning of reasoning and retrieval processes via a fine-tuned critic model. By utilizing Monte Carlo Tree Search (MCTS) to train the critic, the framework significantly outperforms baseline methods on competitive programming, theorem-proving mathematical reasoning, and complex domain retrieval tasks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Critic-Guided Planning"
  - "Retrieval-Augmentation"
  - "Monte Carlo Tree Search"
  - "Sub-goal Decomposition"
  - "Competitive Programming"
date: 2026-05-08
content_hash: 9d4bd12eef00a57a
---

# Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks

**Conference**: ACL 2025  
**Link**: [ACL Anthology](https://aclanthology.org/2025.acl-long.1244/)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Critic-Guided Planning, Retrieval-Augmentation, Monte Carlo Tree Search, Sub-goal Decomposition, Competitive Programming

## TL;DR

This paper proposes the CR-Planner framework, which guides the planning of reasoning and retrieval processes via a fine-tuned critic model. By utilizing Monte Carlo Tree Search (MCTS) to train the critic, the framework significantly outperforms baseline methods on competitive programming, theorem-proving mathematical reasoning, and complex domain retrieval tasks.

## Background & Motivation

**Background**: Large Language Models (LLMs) perform exceptionally well on general reasoning tasks, but still struggle with challenging tasks that require complex reasoning and precise factual knowledge (e.g., competitive programming, theorem proving). Chain-of-Thought (CoT) and Retrieval-Augmented Generation (RAG) enhance LLM capabilities from the perspectives of reasoning depth and knowledge breadth respectively, but their combination remains unsatisfying.

**Limitations of Prior Work**: (1) CoT is prone to the accumulation of reasoning errors in multi-step reasoning—an early erroneous reasoning step can deviate the entire chain from the correct path; (2) retrieval results of RAG often contain irrelevant information, which interferes with the reasoning process instead; (3) existing approaches lack an effective "feedback mechanism" to timely detect and correct errors, and filter retrieval results during the reasoning process.

**Key Challenge**: Complex tasks require deep collaboration between "reasoning + retrieval", but LLMs can easily divert to incorrect reasoning paths and be misled by noisy retrieved information during the process. There is a lack of a "judge" to evaluate reasoning quality and retrieval relevance at each step.

**Goal**: Design a unified framework that utilizes a specially trained critic model to simultaneously guide both the reasoning and retrieval processes, improving LLM performance on challenging tasks.

**Key Insight**: Model the reasoning process as a planning problem, use a critic model to evaluate the quality of each reasoning/retrieval action, and train the critic through process supervision data collected via MCTS.

**Core Idea**: Use two specialized critics (sub-goal critic and execution critic) to guide reasoning and retrieval decisions during the planning process, with the training data of critics automatically collected via MCTS.

## Method

### Overall Architecture

The input of CR-Planner is a complex problem requiring reasoning and knowledge (e.g., programming problems, mathematical theorem-proving problems), and the output is the final answer. The framework consists of three components: (1) Planner—decomposes the task into a sequence of sub-goals; (2) sub-goal critic—evaluates the quality of candidate sub-goals; (3) execution critic—evaluates the quality of sub-goal execution results (either reasoning steps or retrieval results). The entire process iterates until the final answer is reached.

### Key Designs

1. **Sub-goal Critic**:

    - **Function**: Evaluate the quality and prospects of candidate sub-goals in each planning step.
    - **Mechanism**: Given the current problem state and completed sub-goals, the Planner generates multiple candidate next-step sub-goals (e.g., "first understand input format", "find relevant algorithms", "design data structures"). The sub-goal critic scores each candidate, evaluating its "prospect of reaching the final answer". The critic is a fine-tuned language model that takes (problem + completed steps + candidate sub-goal) as input and outputs a quality score. The sub-goal with the highest score is selected for execution.
    - **Design Motivation**: In complex tasks, a single wrong step can lead the entire subsequent process astray. The sub-goal critic provides a "forward-looking assessment" to avoid entering inefficient or incorrect reasoning paths.

2. **Execution Critic**:

    - **Function**: Evaluate the execution quality of the selected sub-goals.
    - **Mechanism**: Once a sub-goal is selected, the Planner executes it through three possible execution actions: (a) reasoning (generating reasoning steps directly with the LLM); (b) query generation (generating queries for the retrieval system); (c) retrieval (executing retrieval to acquire relevant knowledge). The execution critic evaluates the quality of each execution result—is the reasoning step logically correct? Are the retrieval results relevant and useful? If the execution quality is below a threshold, the action is regenerated or an alternative execution path is selected.
    - **Design Motivation**: Guidance solely at the sub-goal level is insufficient—a correct goal with faulty execution will still lead to failure. The execution critic provides in-process quality control.

3. **Based on MCTS Critic Training Data Collection**:

    - **Function**: Automatically collect high-quality (state, action, reward) data to train the two critics.
    - **Mechanism**: Utilizing Monte Carlo Tree Search (MCTS) as the core tool—during the training phase, for each training problem, a search tree is expanded starting from the root node (initial problem) via MCTS. Each node in the tree represents an intermediate state (a list of completed sub-goals), and each edge represents a sub-goal or an execution action. MCTS estimates the value of each node/edge through repeated simulations (rollouts). After the search is complete, each node and edge receives a quality estimate validated by extensive simulations. These triplets of (state, sub-goal/execution, quality score) serve as the training data for the critics.
    - **Design Motivation**: Process supervision signals are extremely scarce, as not every reasoning step has human-annotated "correct/incorrect" labels. MCTS automatically generates these signals through search and backtracking, drastically reducing data annotation costs.

### Loss & Training

The two critics are fine-tuned separately using the data collected by MCTS. During inference, a beam-search-style exploration is adopted—keeping the top-$k$ candidate paths at each step and guided by the critics.

## Key Experimental Results

### Main Results

| Task | Metric | CR-Planner | CoT | RAG | Tree-of-Thought | Gain |
|------|------|-----------|-----|-----|-----------------|------|
| Competitive Programming (APPS) | Pass@1 | Best | Baseline | Ineffective | Suboptimal | ~10-15% |
| Theorem-based Math Reasoning | Accuracy | Best | Baseline | Helpful | Suboptimal | ~8-12% |
| Complex Domain Retrieval (Bamboogle) | F1 | Best | Weak | Baseline | Suboptimal | ~5-10% |

### Ablation Study

| Configuration | Competitive Programming Pass@1 | Math Reasoning Acc | Description |
|------|------|-----------|-----|
| CR-Planner (Full) | Best | Best | Dual critics + MCTS training |
| w/o Sub-goal Critic | Significant drop | Significant drop | Unable to select high-quality reasoning directions |
| w/o Execution Critic | Moderate drop | Moderate drop | Unable to filter low-quality reasoning/retrieval |
| w/o Retrieval | Significant drop on knowledge-intensive tasks | Slight drop | Programming tasks have low dependency on retrieval |
| Random Critic (Untrained) | Close to without Critic | Close to without Critic | Proves that a trained Critic is crucial |

### Key Findings
- The Sub-goal Critic contributes more than the Execution Critic, indicating that "correct direction" is more crucial than "correct execution" (if the direction is wrong, even perfect execution is futile).
- CR-Planner achieves the most substantial improvement on competitive programming because competitive programming requires precise algorithm selection and implementation, where the critic’s guidance prevents early algorithm selection errors.
- The value of retrieval augmentation varies by task—mathematical reasoning relies primarily on reasoning capabilities (where retrieval helps less), whereas domain knowledge queries are highly dependent on retrieval.
- The quality of training data collected by MCTS is far superior to simple positive/negative sample annotations, leading to high learning efficiency for the critics.

## Highlights & Insights
- **Automatically collecting process supervision signals using MCTS** is the most significant technical highlight of this paper. It addresses the core challenge of "how to inform the model whether each step is correct or not." MCTS is inherently suitable for evaluating the value of intermediate nodes in tree search, perfectly matching reasoning planning problems.
- The **dual-critic design (sub-goal + execution)** provides hierarchical quality control. The high-level critic is responsible for strategic direction, while the low-level critic manages execution quality, rendering a clear-cut and complementary division of labor.
- The framework demonstrates great generalizability, as it can theoretically be applied to any complex reasoning task that requires "planning + search + critic."

## Limitations & Future Work
- MCTS incurs high computational costs during the training phase, as each training sample requires a large number of search rollouts.
- The generalization capability of the Critic models is limited, potentially requiring retraining on new tasks outside the training domains.
- Search during inference (beam search) increases latency, making it less suitable for real-time applications.
- Future directions: (1) Exploring more efficient Critic training methods; (2) investigating the feasibility of a general Critic (cross-task generalization); (3) exploring relations with self-play and Process Reward Models (PRMs).

## Related Work & Insights
- **vs Tree-of-Thought**: ToT also utilizes search trees for reasoning but lacks a trained Critic to guide the search, relying solely on the self-evaluation capabilities of the LLM itself.
- **vs RAG**: Traditional RAG performs retrieval only once during the input stage, whereas CR-Planner dynamically decides when and what to retrieve during the reasoning process.
- **vs Process Reward Model (PRM)**: PRMs provide reward signals for each reasoning step, sharing a similar concept with the execution Critic of CR-Planner, though CR-Planner additionally provides sub-goal-level guidance.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The framework design featuring dual critics + MCTS training is ingenious, seamlessly integrating mechanisms of planning, search, and critique.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three different types of challenging tasks plus detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, and the experimental analysis is profound.
- Value: ⭐⭐⭐⭐⭐ Provides a systematic methodology for eliciting and boosting LLM reasoning capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [\[ACL 2025\] Learning from Litigation: Graphs and LLMs for Retrieval and Reasoning in eDiscovery](learning_from_litigation_graphs_and_llms_for_retrieval_and_reasoning_in_ediscove.md)
- [\[ACL 2025\] Perspective Transition of Large Language Models for Solving Subjective Tasks](perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)
- [\[ACL 2025\] Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?](graph_descriptive_order_llm.md)
- [\[ACL 2025\] Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](veracity_bias_llm_hidden_beliefs.md)

</div>

<!-- RELATED:END -->
