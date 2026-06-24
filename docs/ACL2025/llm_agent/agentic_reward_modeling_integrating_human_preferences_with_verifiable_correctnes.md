---
title: >-
  [Paper Note] Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems
description: >-
  [ACL 2025][LLM Agent][Reward Model] This paper proposes the Agentic Reward Modeling paradigm and its implementation, RewardAgent, which integrates traditional human preference-based reward models with verifiable correctness signals from factuality and instruction-following verification. It significantly enhances the reliability of reward models through a three-module architecture consisting of a Router, Verification Agents, and a Judger.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Reward Model"
  - "Verifiable Signal"
  - "Factuality Verification"
  - "Instruction Following"
  - "Agentic Workflow"
date: 2026-05-08
content_hash: c71709a6a4fdf84a
---

# Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems

**Conference**: ACL 2025  
**arXiv**: [2502.19328](https://arxiv.org/abs/2502.19328)  
**Code**: [https://github.com/THU-KEG/Agentic-Reward-Modeling](https://github.com/THU-KEG/Agentic-Reward-Modeling)  
**Area**: LLM Agent / RLHF Alignment  
**Keywords**: Reward Model, Verifiable Signal, Factuality Verification, Instruction Following, Agentic Workflow

## TL;DR
This paper proposes the Agentic Reward Modeling paradigm and its implementation, RewardAgent, which integrates traditional human preference-based reward models with verifiable correctness signals from factuality and instruction-following verification. It significantly enhances the reliability of reward models through a three-module architecture consisting of a Router, Verification Agents, and a Judger.

## Background & Motivation

**Background**: Reward models (RMs) play a central role in post-training (RLHF/DPO) and inference-time scaling (best-of-n sampling) of LLMs. Current mainstream RMs are primarily trained on human preferences, learning preference rankings via the Bradley-Terry model.

**Limitations of Prior Work**: Human preference-based RMs exhibit subjective biases, such as tending to favor longer and more detailed responses (length bias) while overlooking factual errors and instruction violations. For instance, a response containing factual errors but written in a fluent style might receive a higher reward score than a factually correct but concise response.

**Key Challenge**: Human preference signals and verifiable correctness signals are complementary yet distinct dimensions of quality. Human preferences capture subjective elements like linguistic style, whereas verifiable signals provide objective checks on facts and constraints. Existing RMs rely solely on the former, leading to unreliable reward signals.

**Goal**: To design a novel reward system capable of combining human preferences with multidimensional verifiable signals to provide highly reliable reward scoring.

**Key Insight**: Inspired by the potential of verifiable rewards demonstrated in works such as DeepSeek-R1 for LLM training, this work introduces verifiable signals into reward models and flexibly integrates multiple verification dimensions through an agentic workflow.

**Core Idea**: The "Agentic Reward Modeling" paradigm is proposed, where the reward score is defined as the weighted sum of a baseline human preference score and correctness signals from various verification agents, with a Router dynamically selecting the required verification agents.

## Method

### Overall Architecture
The reward formulation of RewardAgent is defined as $r(x,y) = \lambda \cdot r_{RM}(x,y) + \sum_{i \in A_x} w_i \cdot a_i(x,y)$, where $r_{RM}$ represents the baseline preference reward, $a_i$ denotes the correctness signal from the $i$-th verification agent, and $A_x$ is the subset of verification agents selected based on the instruction $x$. The system comprises three modules: Router, Verification Agents, and Judger.

### Key Designs

1. **Router**:

    - **Function**: To analyze the instruction content and dynamically select the verification agents that need to be invoked.
    - **Mechanism**: The functional descriptions and trigger conditions of all verification agents are fed into the LLM backbone, enabling it to determine which dimensions of verification are required for the current instruction. For instance, factual QA tasks trigger the factuality verification agent, while formatting constraint tasks trigger the instruction-following verification agent.
    - **Design Motivation**: Since not all instructions require validation across all dimensions, dynamic selection reduces inference overhead and avoids error accumulation caused by irrelevant verification steps.

2. **Factuality Verification Agent**:

    - **Function**: To evaluate the factual discrepancies between two responses.
    - **Mechanism**: Instead of independent evaluation, a pairwise comparison approach is adopted, consisting of four steps: (1) Difference Proposal: identifying discrepancies in key facts between the two responses; (2) Query Generation: constructing search queries based on these discrepancies to retrieve supporting evidence; (3) Evidence Generation: obtaining supporting evidence using search engines or the LLM's parametric knowledge; (4) Verification: assigning a binary factual score (0 or 1) to each response based on the evidence and the original answers. All steps are executed by the LLM backbone.
    - **Design Motivation**: Compared to atomic-level fact-checking methods like FactScore (which require extensive search engine calls), pairwise comparison only verifies the discrepant parts between the responses, substantially lowering inference costs while successfully capturing nuanced factual differences.

3. **Instruction-Following Verification Agent**:

    - **Function**: To verify whether responses satisfy the hard constraints specified in the instructions.
    - **Mechanism**: A three-step workflow is employed: (1) Constraint Parsing: extracting hard constraints (e.g., length limits, formatting requirements, keyword inclusions) from the instruction; (2) Code Generation & Refinement: generating a Python validation script for each constraint that takes responses as input and outputs a binary score (0 or 1). A self-correction loop is incorporated to allow the LLM to rewrite the script if code execution errors occur; (3) Verification: executing the Python scripts to obtain binary scores for each constraint, with the final score being the mean of all constraint scores.
    - **Design Motivation**: Hard constraints (such as "length within 100 words") can be precisely validated using code, which existing RMs struggle to evaluate accurately. Code-based verification is far more reliable than direct LLM judgment and requires no additional training.

### Loss & Training
The RewardAgent framework itself does not require training as it aggregates modules on-the-fly during inference via an agentic workflow. The baseline RM utilizes a pre-trained ArmoRM. In DPO training experiments, preference pairs constructed by RewardAgent are utilized to conduct DPO on Zephyr-7B.

## Key Experimental Results

### Main Results

| Model | RM-Bench Normal | RM-Bench Hard | JudgeBench | IFBench | Overall |
|------|----------------|---------------|------------|---------|---------|
| ArmoRM-8B | 76.7 | 34.6 | 66.2 | 59.5 | 56.5 |
| GPT-4o | 71.4 | 27.9 | 66.2 | 54.4 | 56.3 |
| Skywork-Gemma-27B | 82.7 | 35.1 | 68.4 | 56.1 | 59.2 |
| DeepSeek-R1 | 83.7 | 50.1 | 74.4 | 64.0 | 69.1 |
| **RewardAgent-mini** | **86.0** | **60.2** | 69.2 | **78.0** | **72.5** |
| **RewardAgent-Llama** | 79.3 | 53.5 | 63.9 | 67.8 | 63.2 |

### Ablation Study

| Configuration | RM-Bench | JudgeBench | IFBench | Description |
|------|---------|------------|---------|------|
| RewardAgent-mini | 73.1 | 68.2 | 75.5 | Complete model |
| – factuality verifier | 54.0 | 52.9 | 73.6 | Factuality verification contributes the most |
| – if verifier | 74.7 | 66.2 | 60.4 | Instruction-following verification contributes significantly to IFBench |
| – both | 55.4 | 58.8 | 58.8 | Degenerates to the baseline RM |
| Oracle setting | 76.7 | 70.1 | 77.5 | Performance upper bound with perfect Router prediction |

### Key Findings
- The factuality verification agent contributes the most: removing it causes RM-Bench performance to drop from 73.1 to 54.0, confirming that existing RMs heavily overlook factuality.
- Powered by an open-source Llama-8B backbone, RewardAgent-Llama outperforms both domain-specific RMs with significantly larger parameters and GPT-4o.
- Search-engine-assisted evidence generation occasionally reduces performance due to noise interference, indicating that internal knowledge retrieval from the LLM might be more stable.
- Models trained via DPO on preference data constructed by RewardAgent consistently outperform those trained on data labeled by the baseline RM across multiple NLP benchmarks.

## Highlights & Insights
- The Agentic Reward Modeling paradigm extends reward modeling from a single-score regression task to an agentic multi-collaboration system. It offers an extensible framework where new verification dimensions (e.g., code execution, mathematical reasoning) can be integrated effortlessly.
- Validating hard constraints via Python code is a clever design that translates non-differentiable constraint checking into executable program verification, offering higher precision and interpretability compared to fuzzy LLM judgments.
- Pairwise factuality verification focuses exclusively on discrepant statements, which is substantially more execution-efficient than FactScore and represents a paradigm that is readily transferable to other pairwise evaluation tasks.

## Limitations & Future Work
- Currently, only factuality and instruction-following verification agents have been implemented; tasks like code generation and mathematical reasoning require different specialized verification agents.
- The overall performance is bounded by the accuracy of the Router, and Oracle experiments indicate that there is still room for improvement in routing decisions.
- The inference cost is relatively high, as each reward evaluation demands multiple LLM calls (Router + Verification Agents + Judger).
- The Judger currently relies on a simple weighted sum ($\lambda = w_i = 1.0$); adaptive weight adjustment is left for future work.

## Related Work & Insights
- **vs ArmoRM**: ArmoRM is a pure preference-based RM. RewardAgent overlays verification signals on top of it, achieving substantial performance gains.
- **vs FactScore**: FactScore conducts fact-checking on an atomic, individual statement level, whereas RewardAgent's pairwise comparison is significantly more efficient.
- **vs Verifiable rewards in DeepSeek-R1**: While DeepSeek-R1 applies verifiable rewards primarily within RL training environments, this work generalizes the concept into a comprehensive, general-purpose RM framework.

## Rating
- Novelty: ⭐⭐⭐⭐ The Agentic Reward Modeling paradigm is pioneering, though individual component technologies are relatively mature.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensively validated across three application scenarios: benchmark evaluation, Best-of-N search, and DPO training.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is well-structured, methods are thoroughly detailed, and diagrams are well-designed.
- Value: ⭐⭐⭐⭐⭐ It opens up a new research direction for reward modeling, with open-source code facilitating subsequent studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)
- [\[ACL 2026\] PRInTS: Process Reward Modeling for Long-range Information Retrieval](../../ACL2026/llm_agent/prints_reward_modeling_for_long-horizon_information_seeking.md)
- [\[ACL 2026\] AdaRubric: Task-Adaptive Rubrics for Reliable LLM Agent Evaluation and Reward Learning](../../ACL2026/llm_agent/adarubric_task-adaptive_rubrics_for_reliable_llm_agent_evaluation_and_reward_lea.md)
- [\[ACL 2026\] Exploring Reasoning Reward Model for Agents](../../ACL2026/llm_agent/exploring_reasoning_reward_model_for_agents.md)
- [\[ACL 2025\] Self-Taught Agentic Long-Context Understanding](self_taught_agentic_long_ctx.md)

</div>

<!-- RELATED:END -->
