---
title: >-
  [Paper Note] CodeTool: Enhancing Programmatic Tool Invocation of LLMs via Process Supervision
description: >-
  [ACL 2025][LLM (Other)][Tool Invocation] This paper proposes CodeTool, a step-by-step code generation framework that guides LLMs to select the optimal tool invocation path through two process reward mechanisms: On-the-spot Reward and Latent Reward. It significantly outperforms existing methods on StableToolBench and RestBench-TMDB.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Tool Invocation"
  - "Code Generation"
  - "Process Supervision"
  - "Process Reward Model"
  - "Step-by-step Reasoning"
date: 2026-05-08
content_hash: 48e096808ea2ce58
---

# CodeTool: Enhancing Programmatic Tool Invocation of LLMs via Process Supervision

**Conference**: ACL 2025  
**arXiv**: [2503.20840](https://arxiv.org/abs/2503.20840)  
**Code**: [LimOkii/CodeTool](https://github.com/LimOkii/CodeTool)  
**Area**: LLM/NLP  
**Keywords**: Tool Invocation, Code Generation, Process Supervision, Process Reward Model, Step-by-step Reasoning

## TL;DR

This paper proposes CodeTool, a step-by-step code generation framework that guides LLMs to select the optimal tool invocation path through two process reward mechanisms: On-the-spot Reward and Latent Reward. It significantly outperforms existing methods on StableToolBench and RestBench-TMDB.

## Background & Motivation

Tool invocation enables LLMs to access external APIs and services, greatly expanding their capabilities. However, prior work suffers from three core limitations:

**Limitations of JSON/Text Formats**: Current tool invocation primarily utilizes JSON or text formats. When handling request-intensive tasks, this leads to high token consumption, long reasoning paths, and vulnerability to truncation, causing critical information loss.

**Lack of Step-Level Supervision**: Existing code-based approaches (such as generating the complete code at once) fail to detect and correct errors in intermediate steps within complex scenarios.

**Poor Reliability of Process Rewards**: Existing process reward methods (such as StepTool) rely on powerful LLMs to generate reward signals, making their objectivity and reliability questionable.

**Advantages of Code-Based Approaches**: Code can efficiently handle batch requests using programming structures such as loops (for/while) and arrays, thereby reducing interaction steps. Furthermore, code is inherently executable and verifiable, providing an objective and reliable signal source for process supervision.

## Method

### Overall Architecture

CodeTool is a step-by-step code generation framework with the following core workflow:
1. Given a user query $q$, a tool set $\mathcal{T}$, and their document protocols $\mathcal{D}$.
2. At each step, the LLM generates a block of Python code $\mathcal{C}_t$ to invoke the appropriate tools.
3. The intermediate result $r_t$ is obtained by executing $\mathcal{C}_t$ via a Python interpreter.
4. Multiple candidate actions are sampled at each step, and the optimal action is selected based on process rewards to proceed to the next step.
5. The process iterates until the final answer is obtained.

Formal representation: $\mathcal{C}_t = \mathcal{M}(q, \mathcal{T}, \mathcal{D}, I_c, r_{t-1})$, where $r_0 = \emptyset$.

### Key Designs

**Dual Process Reward Mechanism**:

**1. On-the-spot Reward**:
- Evaluates whether the code generated at the current step can run correctly.
- Automatically verified via a Python interpreter, requiring no external supervision.

$$R_{spot,t} = \begin{cases} 1, & \text{if Execute}(\mathcal{C}_t) \text{ is successful} \\ 0, & \text{otherwise} \end{cases}$$

- Advantage: Grounded in code executability, this provides an objective and reliable signal that does not rely on LLM annotation.

**2. Latent Reward**:
- Evaluates the potential contribution of the current step toward final task completion.
- Estimated using Monte Carlo Tree Search (MCTS): multiple paths are rolled out from the current step to calculate the success rate.
- Introduces a penalty mechanism to suppress redundant invocations and excessively long paths:

$$R_{latent,t}(q, s_{1:t}) = \alpha^{1-LR(q, s_{1:t})} \cdot \beta^{\frac{\tau}{L}}$$

where $\alpha, \beta \in (0,1]$, $\tau$ is the average number of steps, and $L$ is a constant hyperparameter.

**Cumulative Reward**: $R_{total,t} = R_{spot,t} + R_{latent,t}$

At each step, the candidate action with the highest cumulative reward is selected.

**Process Latent Reward Model (PRM) Training**:
- Estimating Latent Reward via MCTS during inference is computationally expensive.
- A PRM (based on Qwen2.5-7B-Instruct) is trained to directly predict the Latent Reward.
- Construction of training data: For queries in the ToolBench training set that are still callable, a binary action tree is constructed using Depth-First Search (DFS).
- A generative PRM training method is adopted, using two special tokens to represent "more promising" and "less promising".
- Fully automated data construction process, with no human annotation required.

### Loss & Training

- The PRM is trained using the standard SFT loss with a learning rate of 1e-6 for 2 epochs.
- The code generation model (e.g., Qwen2.5-Coder-7B-Instruct) does not require extra fine-tuning.
- Separation of training and inference: Only the PRM needs to be trained, while the code generation capability is directly sourced from pre-trained code models.

## Key Experimental Results

### Main Results

**StableToolBench Results (SoPR%)**:

| Model | Strategy | Format | Average SoPR |
|------|------|------|-----------|
| ToolLLaMA-v2 | CoT | JSON | 33.39 |
| ToolLLaMA-v2 | DFSDT | JSON | 53.24 |
| StepTool | DFSDT | JSON | 44.02 |
| Qwen2.5-7B-Instruct | DFSDT | JSON | 60.52 |
| **Qwen2.5-7B + CodeTool** | CodeTool | Code | **64.19** |
| **Qwen2.5-Coder-7B + CodeTool** | CodeTool | Code | **69.75** |
| GPT-4-Turbo | DFSDT | JSON | 62.03 |
| **GPT-4-Turbo + CodeTool** | CodeTool | Code | **71.05** |

CodeTool significantly outperforms existing methods across both open-source and closed-source LLMs.

**RestBench-TMDB Generalization** (no PRM retrained):

| Method | Success Rate | Path Rate |
|------|-------------|-----------|
| ATC | 89% | 84.71% |
| **CodeTool** | **92%** | **91.15%** |

### Ablation Study

**Necessity of Dual Rewards (Qwen2.5-Coder-7B)**:

| Configuration | Average SoPR | Average SCEP |
|------|-----------|-----------|
| Full CodeTool | **69.75** | **86.86%** |
| - W/o On-the-spot Reward | 65.99 (-3.76) | 69.46% (-17.4%) |
| - W/o Latent Reward | 65.41 (-4.34) | 85.34% (-1.52%) |

- Removing On-the-spot Reward has the greatest impact on the code execution success rate (SCEP) (-17.4%), indicating that instant feedback is critical for code correctness.
- Removing Latent Reward has a more significant impact on SoPR, indicating that long-term direction guidance is crucial for task completion.

### Key Findings

1. The stronger the model's coding capability, the better the performance of CodeTool (Qwen2.5-Coder > Qwen2.5-Instruct > CodeLlama).
2. The PRM remains effective on the unseen RestBench-TMDB dataset, demonstrating strong generalization.
3. The advantage of code format over JSON format is more pronounced in request-intensive scenarios (reaching 79.41% with Qwen2.5-Coder+CodeTool in the I3 subset).
4. The objectivity of the On-the-spot Reward is a core advantage of CodeTool, being based on code execution rather than LLM annotation.

## Highlights & Insights

1. **Code as the "Language" of Tool Invocation**: Replacing JSON with code not only reduces token consumption but also inherently provides executability verification, serving as the cornerstone of the entire framework.
2. **Complementary Design of Dual Rewards**: On-the-spot reward ensures the correctness of each individual step, while Latent reward ensures correct global direction; both are indispensable.
3. **Fully Automated Process Data Construction**: Without relying on GPT annotations, training data is automatically constructed using depth-first search and code execution outcomes.
4. **Training-Free Code Generation Model**: Only a lightweight PRM needs to be trained, whereas the code generation model directly leverages the pre-trained model.

## Limitations & Future Work

- The code generation performance heavily depends on the capability of the underlying code model (as evidenced by CodeLlama-7B's poorer performance).
- StableToolBench suffers from missing API cache issues, necessitating manual filtering of the test set, which may affect evaluation fairness.
- The training data of the PRM is solely derived from ToolBench; its effectiveness in completely different tool environments remains to be validated.
- Step-by-step code generation combined with multi-candidate sampling increases inference latency and computational overhead.
- Evaluation relies heavily on GPT-4 as a judge, which introduces evaluation biases.

## Related Work & Insights

- **ToolLLaMA** (Qin et al., 2023) and **StepTool** (Yu et al., 2024) are the primary baselines for tool invocation.
- **ATC** (Shi et al., 2024) adopts a one-pass full-code generation approach.
- The success of **Process Reward Models (PRMs)** (Lightman et al., 2023) in mathematical reasoning inspired this work.
- Reflection and Inspiration: The On-the-spot Reward provided by code execution is essentially a form of "grounded verification." Similar methods can be extended to other verifiable domains (such as mathematical proofs and unit testing).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual process reward design is ingenious, especially exploiting code execution for the On-the-spot Reward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive baseline comparisons and clear ablation studies are conducted, though testing datasets remain somewhat limited.
- **Value**: ⭐⭐⭐⭐ — The training-free nature of the code generation model lowers the barrier to deployment.
- **Writing Quality**: ⭐⭐⭐⭐ — The architectural diagrams are clear, although mathematical notations are slightly dense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation](dta_llama_parallel_tool_invocation.md)
- [\[ACL 2025\] Enhancing Open-Domain Task-Solving Capability of LLMs via Autonomous Tool Integration from GitHub](paper_2312_17294.md)
- [\[ACL 2025\] ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](toolcoder_code_empowered_tool_learning.md)
- [\[ACL 2025\] SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation](skillverse_tree_eval.md)
- [\[ACL 2025\] Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More](lm_graph_search_supervision.md)

</div>

<!-- RELATED:END -->
