---
title: >-
  [Paper Note] Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling
description: >-
  [ACL 2026][LLM Alignment][Reward Models] Plan-RewardBench is proposed as a trajectory-level preference benchmark for complex tool-augmented scenarios…
tags:
  - "ACL 2026"
  - "LLM Alignment"
  - "Reward Models"
  - "Agent Evaluation"
  - "Trajectory-Level Preference"
  - "Tool Use"
  - "Planning Benchmark"
date: 2026-05-08
content_hash: dff3529c19948f9a
---

# Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling

**Conference**: ACL 2026  
**arXiv**: [2604.08178](https://arxiv.org/abs/2604.08178)  
**Code**: None (To be released after corporate approval)  
**Area**: LLM Alignment  
**Keywords**: Reward Models, Agent Evaluation, Trajectory-Level Preference, Tool Use, Planning Benchmark

## TL;DR

Plan-RewardBench is proposed as a trajectory-level preference benchmark for complex tool-augmented scenarios, designed to evaluate the capability of reward models to distinguish between superior and inferior agent trajectories in contexts such as multi-step planning, tool use, and error recovery.

## Background & Motivation

**Background**: Large Language Models (LLMs) have evolved from passive dialogue systems into agentic systems capable of autonomous tool calling and complex reasoning. Their behavior has expanded from single responses to complete trajectories involving user inputs, reasoning, tool execution, and environmental feedback.

**Limitations of Prior Work**: Existing RM benchmarks (e.g., RewardBench, RM-Bench) primarily focus on response-level preference evaluation, assessing limited dimensions like helpfulness and safety within short-context scenarios. Tool-calling benchmarks (e.g., FC-RewardBench) only verify the correctness of atomic actions, neglecting the evaluation of long-horizon planning behaviors.

**Key Challenge**: While agent systems inherently require multi-turn interactions, current benchmarks fail to assess the judgment capabilities of reward models over long-range, multi-step trajectories, particularly regarding planning consistency, error recovery, and refusal quality.

**Goal**: Construct a trajectory-level preference benchmark to systematically evaluate reward models' capacity to judge planning logic and tool-use faithfulness in complex tool-integration scenarios.

**Key Insight**: Leveraging the MCP tool registry and real execution environments, "indistinguishable" negative pairs are constructed via multi-model natural sampling, rule-based perturbations, and minimal editing.

**Core Idea**: Elevate RM evaluation from the response level to the trajectory level, covering four major scenario families: Safety Refusal, Tool-Irrelevance, Complex Planning, and Robust Recovery.

## Method

### Overall Architecture

Plan-RewardBench models the task as pairwise trajectory preference judgment: given a tool environment $\mathcal{T}$, multi-turn user interactions, and two candidate trajectories $(\tau_A, \tau_B)$, the RM must determine which trajectory is superior. It supports three evaluation protocols: DRM/GRM training preference, inference-time best-of-N re-ranking, and D-PO-style optimization.

### Key Designs

1.  **Four Scenario Families**:
    - **Function**: Covers core challenge dimensions of agent systems.
    - **Mechanism**: Safety Refusal, Tool-Irrelevance, Complex Planning, and Robust Recovery.
    - **Design Motivation**: Existing benchmarks cover only single dimensions, whereas real-world agents must perform correctly across diverse scenarios.

2.  **Multi-source Hard Negative Construction**:
    - **Function**: Generates deceptive negative trajectories.
    - **Mechanism**: 70% natural sampling + 22% minimal edit perturbations + 8% rule injection, controlling for length and format bias to isolate semantic failures.
    - **Design Motivation**: Simple negative samples can be distinguished by surface cues (length/format); hard negatives that are "nearly correct but semantically flawed" must be constructed.

3.  **Multi-judge Annotation and Human Audit**:
    - **Function**: Ensures the reliability of preference labels.
    - **Mechanism**: Median score from $K=3$ LLM judges + meta-review + human auditing (Cohen's $\kappa \in [0.71, 0.86]$).
    - **Design Motivation**: A single judge is prone to bias; multi-judge frameworks combined with human verification guarantee annotation quality.

### Data Construction Workflow

Tasks and tool environments are retrieved from Toucan/MCP. Natural trajectories are obtained through multi-model, multi-parameter sampling using Qwen-Agent and OpenAI-Agent. Negative samples are then constructed via rules and minimal edits. Finally, preference pairs are assembled following multi-judge scoring and human auditing.

## Key Experimental Results

### Main Results

| Model Type | Representative Model | Evaluation Method | Characteristics |
| :--- | :--- | :--- | :--- |
| Discriminative RM | Inf-ORM-Llama3.1-70B | Pointwise scoring → select highest | Evaluates each trajectory independently |
| Generative RM | Skywork-o1, etc. | Generative scoring | Evaluation via the generation process |
| LLM-as-Judge | GPT-o3, Claude, etc. | Pairwise comparison | Direct comparison of two trajectories |

### Dataset Statistics

| Scenario | Pairs | Avg Tokens (Chosen/Rejected) | Max Tokens |
| :--- | :--- | :--- | :--- |
| Tool-Irrelevance | 275 | 1,363 / 1,358 | ~5K |
| Planning-Multi (Hard) | 73 | 6,523 / 6,554 | ~17K |
| Robust Recovery | 361 | 4,545 / 4,462 | ~29K |
| Safety Refusal | 51 | 1,219 / 2,233 | ~11K |

### Key Findings

- Performance across all three types of evaluators (discriminative, generative, LLM-as-judge) drops significantly on long-range trajectories.
- Tool grounding hallucinations (claiming tool use without actual invocation) is the most common failure mode in complex planning.
- In safety refusal scenarios, delayed refusal (partial execution followed by refusal) is the primary source of confusion.
- Blind retrying is the most frequent error pattern in robust recovery.

## Highlights & Insights

- Systematically elevates RM evaluation from response-level to agent-trajectory-level for the first time, filling a critical gap in agent alignment evaluation.
- The hard negative construction methodology serves as a general blueprint for building agent planning preference training data.
- Human audit results (Cohen's $\kappa > 0.7$) validate the reliability of the annotation pipeline.
- Identifies that all mainstream RMs face significant challenges with long-range trajectories, pointing to the necessity of specialized training.

## Limitations & Future Work

- Currently covers only text modality, excluding multi-modal agent scenarios.
- Data scale is constrained by high-quality annotation costs.
- Limited sample size in safety refusal scenarios (51 pairs), restricting statistical significance.
- Future work may extend to multi-modal, longer-horizon, and more complex tool-chain scenarios.

## Related Work & Insights

- RewardBench Series (Lambert et al., 2025): Foundations for response-level RM evaluation.
- AgentRewardBench (Lù et al., 2025): Web agent trajectory evaluation, but lacks tool-augmented scenarios.
- FC-RewardBench (Agarwal et al., 2025): Tool-calling correctness evaluation, limited to single turns.
- This work inspires future research in the direction of RL-from-agent-feedback.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First trajectory-level preference benchmark for tool-augmented agents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple model types, validated by human audit.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with systematic scenario classification.
- **Value**: ⭐⭐⭐⭐⭐ Fills a critical gap in agent RM evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AgentV-RL: Scaling Reward Modeling with Agentic Verifier](agentv-rl_scaling_reward_modeling_with_agentic_verifier.md)
- [\[ICML 2026\] Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling](../../ICML2026/llm_alignment/mitigating_reward_hacking_in_rlhf_via_bayesian_non-negative_reward_modeling.md)
- [\[ACL 2026\] WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback](wildfeedback_aligning_llms_with_in-situ_user_interactions_and_feedback.md)
- [\[ICLR 2026\] Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](../../ICLR2026/llm_alignment/chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)
- [\[NeurIPS 2025\] Provably Efficient Online RLHF with One-Pass Reward Modeling](../../NeurIPS2025/llm_alignment/provably_efficient_online_rlhf_with_one-pass_reward_modeling.md)

</div>

<!-- RELATED:END -->
