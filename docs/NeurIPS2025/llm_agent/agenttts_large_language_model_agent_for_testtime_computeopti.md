---
title: >-
  [Paper Note] AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks
description: >-
  [NeurIPS 2025][LLM Agent][Test-time Scaling] This paper investigates the problem of compute-optimal test-time scaling in multi-stage complex tasks. Through large-scale pilot experiments, three generalizable scaling insights for LLMs on multi-stage tasks are identified. The authors propose AgentTTS—an LLM agent-based framework that autonomously searches for compute-optimal model selection and budget allocation strategies via iterative feedback-driven search.
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "Test-time Scaling"
  - "Multi-stage Tasks"
  - "Compute Budget Allocation"
  - "Hyperparameter Optimization"
date: 2026-05-08
content_hash: 82a1bc5ddf56efbb
---

# AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks

**Conference**: NeurIPS 2025
**arXiv**: [2508.00890](https://arxiv.org/abs/2508.00890)  
**Code**: [https://github.com/FairyFali/AgentTTS/](https://github.com/FairyFali/AgentTTS/)  
**Area**: LLM Agent / Test-time Compute
**Keywords**: Test-time Scaling, Multi-stage Tasks, Compute Budget Allocation, LLM Agent, Hyperparameter Optimization

## TL;DR

This paper investigates the problem of compute-optimal test-time scaling in multi-stage complex tasks. Through large-scale pilot experiments, three generalizable scaling insights for LLMs on multi-stage tasks are identified. The authors propose AgentTTS—an LLM agent-based framework that autonomously searches for compute-optimal model selection and budget allocation strategies via iterative feedback-driven search.

## Background & Motivation

**Background**: Test-time Scaling (TTS) improves LLM performance by allocating additional compute at inference time and has proven effective on single-stage tasks such as mathematical reasoning and code generation. Existing approaches include sequential scaling (iterative refinement) and parallel scaling (repeated sampling with selection/aggregation), with the latter being more favored due to its independence from high-quality initial responses and broader coverage.

**Limitations of Prior Work**: (1) Existing TTS research primarily targets single-stage tasks (e.g., standalone math solving or code generation), whereas many real-world tasks are multi-stage—such as retrieval-augmented QA systems or software development pipelines (requirements → design → coding → testing), where different subtasks require models with different capabilities. (2) The search space in multi-stage tasks grows exponentially (e.g., 3 subtasks × 2 model choices yields up to $10^6$ configurations), and each configuration evaluation may require hours of inference, rendering exhaustive search infeasible. (3) Compute allocation across subtasks is not independent—the quality of earlier subtasks affects the optimal configuration for later ones.

**Key Challenge**: The search space for compute allocation in multi-stage tasks is both large and interdependent, making standard optimization methods (Bayesian optimization, random search) ineffective in non-smooth search landscapes.

**Goal**: Given a fixed total compute budget $B$ and a multi-stage task $\mathcal{T} = [T_1, T_2, ..., T_n]$, the goal is to select an appropriate model $M_i$ and allocate budget $B_i$ (with $\sum B_i = B$) for each subtask to maximize overall performance.

**Key Insight**: The paper first conducts large-scale pilot experiments to derive three generalizable insights, then encodes these insights as search strategy priors for an LLM agent, leveraging the model's reasoning and planning capabilities to efficiently navigate the search space.

**Core Idea**: Based on three empirical insights—subtask-specific model preferences, the existence of an optimal scaling budget, and inter-subtask budget interdependencies—an LLM agent framework is designed to autonomously search for compute-optimal configurations in multi-stage tasks.

## Method

### Overall Architecture

AgentTTS consists of three core components: **Agent** (an LLM-based searcher responsible for generating candidate trials and search guidelines), **Archive** (stores historical trials, guidelines, and feedback), and **Environment** (executes trials on the actual task platform and returns performance feedback). The workflow is iterative: the Agent generates candidate configurations → the Environment evaluates them → feedback is returned to the Agent → the Agent updates guidelines and generates new configurations → this loop repeats until a stopping criterion is met.

### Key Designs

1. **Three Empirical Insights (Pilot Experiment Findings)**:

    - Function: Provide prior knowledge to guide the search strategy and substantially narrow the search space.
    - Mechanism:
        - **Insight 1 (Subtask Model Preference Divergence)**: Different subtasks exhibit different preferences for large vs. small models. For instance, retrieval subtasks require strong long-context understanding, favoring large models; whereas QA subtasks primarily extract information from already-retrieved content, where small models with repeated sampling can be competitive with large models.
        - **Insight 2 (Optimal Scaling Budget Exists)**: Increasing test-time compute yields initial gains, but performance may degrade beyond the optimal point, as aggregating too many candidates becomes more complex—particularly for smaller models.
        - **Insight 3 (Inter-subtask Budget Interdependency)**: Budget allocation in earlier subtasks affects scaling dynamics in later ones. For example, high-quality retrieval (large model) reduces the optimal sampling count for the QA subtask; low-quality retrieval requires more sampling or larger models downstream to compensate.
    - Design Motivation: All three insights are consistently observed across four task types (retrieval QA, knowledge graph QA, task automation, software development) and six datasets, confirming their generalizability.

2. **Initialization Search Strategy (Based on Insight 1)**:

    - Function: Rapidly determine each subtask's model preference to avoid wasting search effort on inferior models.
    - Mechanism: For each subtask $T_i$, all other subtasks are fixed to use the largest model with a single inference pass, and the performance of all candidate models on $T_i$ is compared within budget constraint $B_i^{max}$. Based on the initial feedback, model preference guidelines are summarized: if the large model substantially outperforms the small model, subsequent search prioritizes large models; otherwise, small models are preferred due to their larger sampling budget flexibility.
    - Design Motivation: Establishing the correct model direction early substantially reduces wasted exploration in subsequent search.

3. **Iterative Guideline Generation and Trial Search (Based on Insights 2 & 3)**:

    - Function: Guide the LLM agent to efficiently explore the budget allocation space.
    - Mechanism: At each iteration, the Agent generates search guidelines based on historical trials and feedback. Insight 2 is encoded as a prompt instruction requiring the Agent to "identify the search direction for the optimal sampling count for each subtask," ensuring focus on the appropriate budget range. Insight 3 is encoded as an instruction to "leverage the LLM's planning capabilities to explore budget trade-offs across subtasks," enabling the Agent to identify critical subtasks and adaptively adjust configurations. All three insight instructions are applied in parallel throughout the search process.
    - Design Motivation: The LLM's in-context reasoning capabilities are leveraged to recognize patterns in non-smooth search landscapes, outperforming traditional Bayesian optimization in multi-modal and discontinuous search spaces.

### Budget Normalization Framework

To enable fair comparison across models and subtasks, a unified budget unit is defined: the FLOPs of a single inference pass using the smallest model (LLaMA 3B) on the lowest-compute task serves as the baseline unit. Given model $M_\ell$, sampling count $S_\ell$, and task $T_\ell$, the normalized equivalent budget is:

$$B = \frac{2\alpha\beta_2 S_\ell}{\beta_1} + 2(\alpha\beta_2 - 1)$$

where $\alpha = M_\ell / M_{smallest}$, $\beta_1 = N_{p,\ell} / N_{d,\ell}$ (prompt-to-generation length ratio), and $\beta_2 = N_{p,\ell} / N_{p,lowest}$ (prompt length ratio).

### Loss & Training

Search is conducted using GPT-o3-mini as the LLM agent, executing 50 search iterations on a training set of 50 samples, with final evaluation on a test set of 500 samples. The scaling paradigm is repeated sampling with aggregation (temperature = 0.9). The default total budget is set to the sum of budgets required for each subtask using the largest model with a single inference pass.

## Key Experimental Results

### Main Results

| Method | 2Wiki EM | Hotpot EM | CWQ EM | WebQSP EM | TaskBench p-F1 | ChatDev Cons. | Search Time (h) |
|--------|---------|----------|--------|----------|--------------|-------------|----------------|
| AgentTTS | **0.72** | **0.74** | **0.78** | **0.89** | **0.53** | **0.75** | 2.5–64.3 |
| AgentHPO | 0.70 | 0.74 | 0.78 | 0.89 | 0.49 | 0.74 | 8.3–48.1 |
| MLCopilot | 0.70 | 0.72 | 0.78 | 0.88 | 0.53 | 0.75 | 12.5–48.4 |
| BO | 0.60 | 0.71 | 0.76 | 0.85 | 0.52 | 0.75 | — |
| Random | 0.66 | 0.71 | 0.76 | 0.86 | 0.40 | 0.74 | — |

### Ablation Study

| Ablation Variant | Steps to Optimal Trial | Impact |
|-----------------|----------------------|--------|
| Full AgentTTS | ~10 steps | — |
| w/o Insight 1 (random initialization) | Never converges | Initial model selection is critical |
| w/o Insight 2 (no optimal budget guidance) | Step 29 | Reduced search efficiency |
| w/o Insight 3 (no inter-subtask dependency) | Step 38 | Delayed convergence |

### Key Findings

- **Search Efficiency**: AgentTTS reaches optimal or near-optimal configurations with fewer trials and less time on most tasks (more than 3× faster than AgentHPO on 2Wiki).
- **Generalization**: AgentTTS outperforms the second-best method by 2% on the 2Wiki test set (0.72 vs. 0.70), suggesting that Insight 2 helps avoid redundant sampling and improves generalization.
- **Failure of Traditional Methods**: Bayesian optimization tends to become trapped in local optima in non-smooth landscapes; random search is robust to noise but lacks efficiency.
- **Robustness**: As the training set size decreases from 100 to 75 to 50 samples, AgentTTS maintains search efficiency, while other LLM-based methods and BO exhibit noticeable performance degradation.
- **Interpretability**: Agent-generated guidelines clearly reflect the application of the three insights—e.g., "prefer large models for retrieval, prefer small models for QA" and "search sampling counts for QA in the range of 5–50."

## Highlights & Insights

- **Novel Problem Formulation**: This work is the first to formally define the compute-optimal test-time allocation problem for multi-stage tasks, extending TTS from single-task settings to more realistic composite task scenarios.
- **Insight-Driven Design**: The three empirical insights not only guide the design of AgentTTS but are themselves valuable research contributions, revealing fundamental scaling laws for multi-stage TTS.
- **LLM Advantage in Non-Smooth Landscapes**: Leveraging LLM in-context reasoning to bypass local optima in discontinuous, multi-modal search spaces addresses a known weakness of traditional optimization methods.
- **Budget Normalization Framework**: A unified compute budget definition across models and tasks is provided, which can serve as a standard tool for future TTS research.
- **Supplementary Finding on Temperature**: High temperature (0.9) is superior in multi-sample settings due to increased output diversity, while low temperature (0.1) is preferable for single-sample inference due to greater stability.

## Limitations & Future Work

- The search process itself incurs substantial compute overhead (50 iterations on a training set), requiring a small-scale training set as a prerequisite.
- The search quality of the LLM agent (o3-mini) may be constrained by its understanding of test-time scaling concepts.
- Only four task types are evaluated; more complex multi-stage tasks (e.g., multi-turn dialogue agents) are not explored.
- The framework assumes a linear pipeline dependency structure among subtasks and does not handle branching or parallel subtask architectures.
- The aggregation strategy is fixed to self-aggregation within the same model; cross-model aggregation is not explored.
- Budget normalization is based on FLOPs, which may not fully correspond to actual latency and memory constraints.

## Related Work & Insights

- Brown et al. (2024) demonstrate that small models with repeated sampling can surpass large models on single predictions, forming a foundational basis for TTS.
- Snell et al. (2025) study compute-optimal scaling strategies at test time but focus on single-stage reasoning tasks.
- AgentHPO and MLCopilot apply LLMs to hyperparameter optimization; AgentTTS extends this paradigm to test-time budget allocation.
- ChatDev and TaskBench provide practical platforms for multi-stage tasks, on which the generality of the proposed framework is validated.
- This work is complementary to the concurrent work Multi2, which addresses test-time scaling for multi-agent multi-document processing.

## Rating

⭐⭐⭐⭐ (4/5)

The problem formulation is novel and practically motivated (multi-stage TTS addresses a real-world need), the three insights offer generalizable value, and the agent framework is well-designed. The experiments cover four task types across six datasets, with thorough ablation and robustness analyses; interpretability is a notable strength. Limitations include the substantial compute overhead of the search process itself and the restriction to linearly structured multi-stage tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GTA1: GUI Test-time Scaling Agent](../../ICLR2026/llm_agent/gta1_gui_test-time_scaling_agent.md)
- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](../../ACL2025/llm_agent/metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)
- [\[NeurIPS 2025\] TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration](trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)
- [\[ICLR 2026\] Pushing Test-Time Scaling Limits of Deep Search with Asymmetric Verification](../../ICLR2026/llm_agent/pushing_test-time_scaling_limits_of_deep_search_with_asymmetric_verification.md)

</div>

<!-- RELATED:END -->
