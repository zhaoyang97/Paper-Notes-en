---
title: >-
  [Paper Note] MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use
description: >-
  [ICML 2026][LLM Reasoning][Agent safety] MOSAIC transforms "safety decision-making" from an implicit byproduct of reasoning into an explicit first-class action within a plan → check → act/refuse loop (incorporating `<saf…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Agent safety"
  - "Tool use"
  - "Explicit safety checks"
  - "Pairwise preference reinforcement learning"
  - "GRPO"
date: 2026-05-08
content_hash: b2f166ce346cd5c1
---

# MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use

**Conference**: ICML 2026  
**arXiv**: [2603.03205](https://arxiv.org/abs/2603.03205)  
**Code**: To be confirmed  
**Area**: LLM Safety / Agent / Tool Use  
**Keywords**: Agent safety, Tool use, Explicit safety checks, Pairwise preference reinforcement learning, GRPO

## TL;DR
MOSAIC transforms "safety decision-making" from an implicit byproduct of reasoning into an explicit first-class action within a plan → check → act/refuse loop (incorporating `<safety_thoughts>` and `refusal_tool`), trained using pairwise trajectory preferences from an LLM judge and GRPO. Across Qwen2.5-7B / Qwen3-4B-Thinking / Phi-4, it reduces harmful behaviors by 50% in zero-shot OOD scenarios, increases prompt injection refusal rates by 20%, and decreases privacy leaks while maintaining benign task utility.

## Background & Motivation

**Background**: LLMs have evolved from chat assistants to agents—capable of planning, calling tools, and executing multi-step interactions with external systems. Benchmarks such as AgentHarm, Agent Security Bench, and PrivacyLens have demonstrated that a single error (e.g., writing a file, initiating a payment, or leaking credentials) can cause irreversible harm. Small Language Models (SLMs) like Phi-4, Qwen2.5-7B, and Qwen3-4B are often preferred for deployment in agent scenarios due to cost, latency, and privacy considerations.

**Limitations of Prior Work**: (1) Chat safety (RLHF/Constitutional AI) does not reliably transfer to agents; models may refuse harmful chat requests but comply when the same request is wrapped as a tool-based task. (2) Existing agent RL (typically focused on math or coding) rarely performs explicit safety or irreversibility checks within long reasoning traces. (3) Outcome-only scalar rewards fail to distinguish between "early refusal" and "late termination," though the two differ critically in safety. (4) SLMs have tighter context and world models, making them particularly vulnerable to prompt injections, anomalous tool feedback, and cascading failures.

**Key Challenge**: Current agent training objectives focus solely on "task completion." Safety decisions are buried in implicit reasoning, making them neither controllable nor supervisable. The trajectory-level safety distribution is sequence-sensitive relative to outcome-level rewards (e.g., two failures are not equal if one occurs after an unsafe action), which scalar rewards cannot adequately express.

**Goal**: (1) Refactor safety checks and refusals as explicit first-class actions to make them learnable, controllable, and auditable. (2) Replace outcome-level scalar rewards with trajectory-level preferences to capture the temporal differences of "when to refuse." (3) Validate generalization across multiple model families and OOD benchmarks.

**Key Insight**: It is observed that agent insecurity often stems not from "malicious intent" but from a "failure to realize when to stop"—specifically, the lack of an explicit safety check step within long reasoning traces. This implies that training the model on "when to insert a safety check" and "when to refuse" can significantly enhance safety using the same model capacity.

**Core Idea**: A plan → check → act/refuse loop combined with preference RL. Safety checks are triggered via a `<safety_thoughts>` block (dynamically toggled by the model), and refusal is implemented as a termination action via a `refusal_tool`. An LLM judge performs pairwise comparisons of trajectories for the same task to optimize the policy using GRPO.

## Method

### Overall Architecture

At each step $t$:
1. **plan**: Generate a plan and candidate tool calls within a `<think>` block.
2. **gate $g_t \in \{0,1\}$**: The model decides whether to initiate `<safety_thoughts>` (by outputting the opening tag); this is learned end-to-end via RL without external triggers.
3. **safety check** (if $g_t=1$): Perform structured reasoning within `<safety_thoughts>` regarding potential harm, irreversibility, permission changes, and tool feedback risks.
4. **act / refuse**: Select an action from $\{\text{tool\_call}, \text{refusal\_tool}, \text{answer}\}$. The `refusal_tool` serves as a termination action with an accompanying explanation.
5. Trajectory $\tau = \{(o_t, \text{plan}_t, g_t, \text{safety}_t, a_t)\}_{t=1}^{T_{\text{term}}}$.

Training utilizes GRPO (Group Relative Policy Optimization), which requires no critic and employs pairwise rollouts. Tool output tokens are masked, with backpropagation occurring only on model-generated text.

### Key Designs

1. **Explicit safety check + refusal as first-class actions**:
    - **Function**: Transitions safety decision-making from an implicit reasoning byproduct to a learnable, controllable, and auditable discrete action.
    - **Mechanism**: Defines a `<safety_thoughts>` block and a `refusal_tool` termination action. The gate for the former is a learned self-decision (learned gate), allowing the model to skip checks to avoid constant overhead. The latter enters the action space like a standard tool and can be directly rewarded by RL.
    - **Design Motivation**: "Forgetting to check" is a primary vulnerability in long agent reasoning traces. Making checks and refusals explicit ensures they are subject to end-to-end learning, moving safety decisions from hidden logit distributions to explicit token-level actions where RL signals can be precisely applied.

2. **Pairwise trajectory preference RL (replacing scalar reward)**:
    - **Function**: Uses an LLM judge to perform relative preference comparisons between two rollouts of the same task to capture temporal safety differences.
    - **Mechanism**: Several rollouts are sampled for each prompt. The LLM judge performs a pairwise comparison to determine which trajectory is safer and more appropriate (rather than assigning an absolute score). These preference pairs supervise the group advantage in GRPO. Evaluation dimensions include early refusal vs. late termination, adherence to injected instructions, and privacy preservation.
    - **Design Motivation**: Outcome-level scalar rewards might assign nearly identical scores to "refusing before touching a dangerous tool" and "terminating only after executing an unsafe operation." Pairwise comparisons preserve this temporal sensitivity, which is crucial for agent safety RL.

3. **Composite reward + length-aware training (GRPO join optimization)**:
    - **Function**: Balances safety alignment, task utility, structured output formatting, and token efficiency.
    - **Mechanism**: The composite reward includes safety preferences, task success, formatting compliance (correct use of `<think>` and `<safety_thoughts>` tags), and a token length penalty. The group relative advantage in GRPO naturally normalizes reward differences without a critic.
    - **Design Motivation**: Optimizing solely for safety leads to over-refusal, while optimizing solely for task completion ignores safety. Length penalties prevent the chain-of-thought from expanding indefinitely. GRPO proves more stable than PPO for long agent trajectories.

## Key Experimental Results

### Main Results (OOD Benchmarks)

| Model | AgentHarm Harmful Task Reduction | AgentHarm Refusal Rate | PrivacyLens Privacy Leak Reduction | BFCL Benign Completion Rate |
| :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-7B base | – | 35% | – | 78% |
| Qwen2.5-7B + **MOSAIC** | **−50%** | **87%** | −38% | **82%** |
| Qwen3-4B-Thinking base | – | 41% | – | 44% |
| Qwen3-4B-Thinking + **MOSAIC** | −37% | 71% | −29% | **85%** |
| Phi-4 base | – | 52% | – | 71% |
| Phi-4 + **MOSAIC** | −44% | 79% | −33% | **91%** |

The benign completion rate for Qwen3-4B-Thinking improved from 44% to 85% (nearly double) because the base model often fell into "infinite reasoning loops," whereas MOSAIC learned to terminate appropriately.

### Ours vs. Prev. SOTA (Closed-source Models)

| Model | Agent Safety Score ↑ |
| :--- | :--- |
| GPT-4o (No scaffold) | 71 |
| GPT-5 (No scaffold) | 76 |
| Qwen2.5-7B + MOSAIC | **78** |
| Phi-4 + MOSAIC | **74** |

MOSAIC elevates SLMs to agent safety levels comparable to frontier models. When frontier models are equipped with explicit scaffolding, the gap narrows further.

### Prompt Injection Refusal Rate (Agent Security Bench)

| Injection Type | Base | + MOSAIC |
| :--- | :--- | :--- |
| Tool Call Hijacking | 31% | **62%** |
| System Prompt Override | 38% | **68%** |
| Implicit Harmful Subtasks | 44% | **65%** |

MOSAIC achieves an average increase of over 20% in refusal rates, proving particularly effective against prompt injection.

### Key Findings
- **Selective safety invocation works**: Safety tokens account for <20% of total tokens on average, as the model learns to insert safety checks only at critical steps.
- **Simultaneous reduction in under- and over-refusal**: Phi-4 saw a 56% reduction in over-refusal (wrongly refusing benign tasks) while increasing refusal rates for harmful tasks—proving MOSAIC is not simply "more conservative."
- **Ablation of pairwise vs. scalar rewards**: Replacing pairwise signals with scalar rewards dropped the harmful task reduction from 50% to 28%, validating the necessity of pairwise feedback for "when to refuse."
- **Generalization across model families**: Benefits were observed across both Qwen and Phi families and various scales, indicating MOSAIC is a robust paradigm rather than a specific trick.

## Highlights & Insights
- **Paradigm shift to "Safety as a First-Class Action"**: Historically, safety was treated as implicit alignment during RLHF or as an inference-time filter. MOSAIC elevates it to an action category equivalent to tool calls, enabling better supervision and auditing.
- **Insight into Temporal Safety via Pairwise Preferences**: Comparing two rollouts for the same prompt automatically amplifies differences in "timing." This is an undervalued design choice in agent scenarios that can be generalized to any task where trajectory quality is time-dependent.
- **SLM-Friendly**: MOSAIC primarily benefits models in the 4–7B range. This suggests that agent safety can be resolved without relying on massive model capacity, which is crucial for real-world deployment (cost, latency, privacy).
- **Natural Learning of the Selective Gate**: The model autonomously learns to "check during safety-sensitive steps and skip during routine steps" without manual heuristics.

## Limitations & Future Work
- The LLM judge may harbor biases (using GPT-4o/-5 as judges may favor certain styles); future work could explore ensemble judges or self-play critics.
- Validation was limited to three tool/action types; the performance of the selective gate in vast real-world tool spaces remains unknown.
- `<safety_thoughts>` uses a single-segment structured reasoning format without separate scoring for different dimensions (harm, privacy, irreversibility); it could be decomposed into multiple heads.
- Sample efficiency of preference RL in long-horizon tasks (>20 steps) has not been fully verified.

## Related Work & Insights
- **vs. Chat RLHF (e.g., Constitutional AI)**: Traditional methods work for single-turn text but do not transfer to multi-step agents; MOSAIC reforms alignment specifically for agentic contexts.
- **vs. Inference-time Safety Filters**: Filters are post-hoc and cannot prevent unsafe behaviors that have already been initiated; MOSAIC moves the decision point ahead of each action.
- **vs. Scalar Reward Agent RL (e.g., RLVR)**: While scalar rewards suffice for math or code, they cannot express the temporal nuances of agent safety; pairwise signals are a necessary upgrade.
- **Insight**: Treating "when to do something" as a first-class learning objective can be extended to all multi-step decision-making processes (e.g., "when to stop loss" in trading or "when to call a human" for medical agents).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Safety as a first-class action + pairwise temporal preference" is a genuine new paradigm for agent safety.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Coverage across three model families, four OOD benchmarks, and harmful/injection/privacy/benign categories; complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The MOSAIC framework diagram is intuitive, though the composite reward section is slightly brief.
- Value: ⭐⭐⭐⭐⭐ As SLM agents are the mainstream for deployment, this paper provides an industrially viable safety post-training solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents](diversity_over_frequency_rethinking_tool_use_in_visual_chain-of-thought_agents.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[ICML 2026\] The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary](the_deterministic_horizon_when_extended_reasoning_fails_and_tool_delegation_beco.md)
- [\[ICLR 2026\] Generalizable End-to-End Tool-Use RL with Synthetic CodeGym](../../ICLR2026/llm_reasoning/generalizable_end-to-end_tool-use_rl_with_synthetic_codegym.md)
- [\[ACL 2026\] HISR: Hindsight Information Modulated Segmental Process Rewards for Multi-turn Agentic Reinforcement Learning](../../ACL2026/llm_reasoning/hisr_hindsight_information_modulated_segmental_process_rewards_for_multi-turn_ag.md)

</div>

<!-- RELATED:END -->
