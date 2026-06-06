---
title: >-
  [Paper Note] Agent-Omit: Adaptive Context Omission for Efficient LLM Agents
description: >-
  [ICML 2026][LLM Agent][Context Management] Quantify "which turns of thought/observation can be omitted" via Monte-Carlo rollouts, then train an 8B agent using cold-start SFT and dual-sampling omit-aware GRPO to adaptivel…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Context Management"
  - "Thought Omission"
  - "Observation Omission"
  - "GRPO"
  - "Dual Sampling"
date: 2026-05-08
content_hash: 505ecd1e88061445
---

# Agent-Omit: Adaptive Context Omission for Efficient LLM Agents

**Conference**: ICML 2026  
**arXiv**: [2602.04284](https://arxiv.org/abs/2602.04284)  
**Code**: https://github.com/usail-hkust/Agent-Omit (Available)  
**Area**: LLM Agent / Efficient Inference / Agentic Reinforcement Learning  
**Keywords**: Context Management, Thought Omission, Observation Omission, GRPO, Dual Sampling  

## TL;DR
Quantify "which turns of thought/observation can be omitted" via Monte-Carlo rollouts, then train an 8B agent using cold-start SFT and dual-sampling omit-aware GRPO to adaptively skip redundant thoughts and observations. Across five benchmarks, token usage is significantly reduced while maintaining accuracy parity with seven frontier models.

## Background & Motivation

**Background**: LLM agents solve tasks through cycles of thought→action→observation (ReAct / agentic RL). Agents like Kimi-K2 and DeepSeek-V3.2 have demonstrated strong capabilities in deep search, web shopping, embodied decision-making, and scientific discovery. However, multi-turn interactions lead to ever-expanding context, causing token costs to skyrocket.

**Limitations of Prior Work**: Existing efficiency methods fall into three categories: compressing only thoughts (ToolLight, DEPO), pruning only observations (Observation-Mask, DeepMiner), or summarizing both (MEM-Agent, ReSum). These methods treat the entire trajectory "indiscriminately," ignoring the vast differences in contribution across different turns.

**Key Challenge**: The "necessity" of thought and observation is turn-dependent. Early high-level planning often determines subsequent rounds of thinking; by the final round of summarization, most early observations are outdated. One-size-fits-all compression either deletes essential information (impacting accuracy) or retains useless tokens (impacting efficiency).

**Goal**: Two steps: (1) Quantitatively prove the feasibility of "selective turn-based omission" via controlled interventions; (2) Train a policy that adaptively decides "whether to write this round's thought and which previous observations to discard" during interaction.

**Key Insight**: Model the omission behavior itself as part of the action space—outputting an empty string for thoughts and explicitly deleting observations via special tokens like `<omit_tool_response_N>`. This allows omission to be learned naturally within SFT and RL frameworks.

**Core Idea**: Enable agents to actively output "thought omission" and "observation omission" actions. Train them using an omit-aware GRPO that couples "task reward" with "token savings" (with omission rewards reset to zero upon task failure), supported by dual sampling to solve the attribution challenge where "omitted information is no longer visible."

## Method

### Overall Architecture
Two-stage optimization: (a) Agent Omission Behavior Synthesis (Cold-start SFT) — Identify "omittable" thought/observation turns in trajectories via Monte-Carlo rollouts. Construct single-turn and multi-turn synthetic data to teach the base model both "omission formatting" and "continued reasoning under omitted context." (b) Omit-Aware Agentic RL — Introduce dual sampling (sampling full trajectories alongside partial trajectories for each omission point) and omit-aware rewards (task reward + omit reward), optimized via GRPO. Theoretically, the deviation of the learned omission strategy from the optimal strategy is bounded by the KL divergence.

### Key Designs

1.  **Quantitative Analysis + Explicit Omission Actions**:
    - **Function**: Provide quantitative proof that "selective omission significantly reduces tokens without performance loss," then design omission as a token pattern for SFT/RL learning.
    - **Mechanism**: On WebShop + Qwen3-8B, "excavate" thought $\tau_t$ or observation $o_t$ at turn $t$ and let the agent complete the task, tracking tokens and Pass@1. Results: Thought accounts for 45.1%, observation 52.2%, and action only 2.7%. Middle-turn thoughts are omittable, final-round observations are not, and first-round thoughts are essential. Large "gray zones" exist where accuracy remains stable while tokens decrease. Action-wise, thought omission uses empty `<think> </think>`; observation omission uses `<omit_tool_response_N_...>` to explicitly mask historical observation sets $\Gamma \subseteq \{1,\dots,t-1\}$.
    - **Design Motivation**: Upgrading heuristics (time-window based deletion) to a learned strategy requires verifying that the "omission space is non-empty" and providing a clear linguistic interface.

2.  **Omission Behavior Synthesis (Cold-start SFT)**:
    - **Function**: Transform a general LLM into an omission-aware agent, providing an initial policy to prevent RL exploration catastrophe.
    - **Mechanism**: Perform forward rollouts on training trajectories to identify "omittable turns"—marked if omission reduces tokens without dropping accuracy. Construct layered data: (i) Single-turn omission, teaching the agent to output empty thoughts or omit commands via system prompts; (ii) Multi-turn omission, replacing omittable turns with omission symbols to force the agent to maintain reasoning continuity despite historical gaps. Full-parameter SFT is performed with loss: $\mathcal{L} = -\mathbb{E}_{(x,y)\sim \mathcal{D}_{single}\cup\mathcal{D}_{multi}}[\log \mathcal{P}_{\pi_\theta}(y\mid x)]$, applying a loss mask to environmental observations.
    - **Design Motivation**: Direct RL fails if the agent cannot output omission symbols. SFT is the most cost-effective path to instilling the "omission is a valid action" concept.

3.  **Omit-aware Agentic RL: Dual Sampling + Dual Rewards + GRPO**:
    - **Function**: Learn the "omission strategy" as a first-order decision goal while ensuring task accuracy is not sacrificed to reward hacking.
    - **Mechanism**: Dual sampling—For each input, sample a full trajectory $y$ (complete episode with omission actions) and derive partial trajectories $y'$ for each omission turn (context before omission + the thought/action of that turn). Each $y$ spawns $p(y)$ partial trajectories. This allows the agent to "see context before omission" in $y'$ to learn attribution for omission decisions. For rewards: task reward $R_{task}$ is given to both full and partial trajectories; omit reward $R_{omit}=\mathrm{Tok}(\tau_{omitted})/\mathrm{Tok}(y) + \mathrm{Tok}(o_{omitted})/\mathrm{Tok}(y)$ is given only to full trajectories and is forced to zero if $R_{task}=0$. Combined reward: $r(\cdot)=(1-\mu)R_{task}+\mu R_{omit}$ ($\mu=0.2$), $r'(\cdot)=R_{task}$. Optimization via GRPO with KL constraint $-\beta \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]$.
    - **Design Motivation**: Credit assignment for omission requires "counterfactual non-omitted context," which standard agentic RL lacks. Dual sampling fills this gap. Task-conditioned omit rewards are more robust than simple weighted sums.

### Loss & Training
SFT stage: Standard LM loss + environmental observation loss mask. 
RL stage objective:
$$\max_{\pi_\theta} \mathbb{E}_{x,\{y_i,\{y'_{i,j}\}\}}\big[\tfrac{1}{n}\sum_i \big(r(x,y_i) + \tfrac{1}{p(y_i)}\sum_j r'(x,y'_{i,j})\big)\big] - \beta \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]$$
Base model: Qwen3-8B. Theoretically, under the semantic Lipschitz assumption, the performance/efficiency deviation is bounded by $\delta + K' \cdot \mathrm{KL}(\pi^\ast,\pi_\theta)$, showing monotonic approximation to the optimal omission strategy as KL decreases.

## Key Experimental Results

### Main Results
Compared against five agent environments (DeepSearch, WebShop, TextCraft, BabyAI, SciWorld), seven frontier LLMs (DeepSeek-R1-0528, DeepSeek-V3.2, o3/o4-mini, Qwen3-235B-A22B, Qwen3-Next-80B-A3B, Qwen3-32B), and seven efficient agent methods.

| Comparison | Pass@1 Accuracy | Token Cost | Notes |
| :--- | :--- | :--- | :--- |
| Agent-Omit-8B (v.s. Qwen3-8B) | Comparable to 7 frontier LLMs | Significantly lower | 8B reaches parity using half the tokens of large models |
| 7 Efficient Agent Methods | Varies | Varies | Agent-Omit achieves best accuracy-efficiency trade-off |
| Qwen3-8B Native | Baseline | Baseline | Without omission: thought 45.1% + observation 52.2% |

### Ablation Study

| Configuration | Key Observation | Interpretation |
| :--- | :--- | :--- |
| SFT Only (No RL) | Learned format but limited gains | RL is necessary to learn adaptive "when to omit" |
| No dual sampling | Omission strategy fails to converge | Partial trajectories are essential bridges for credit assignment |
| No $R_{omit}$ | Nearly identical to original agent | Lack of explicit efficiency incentive |
| $R_{omit}$ uncoupled from $R_{task}$ | Reward hacking occurs; accuracy drops | Mandatory "zero omit reward on task failure" is necessary |
| Single-turn omission only | Poor generalization in multi-turn | Multi-turn data forces learning "continuation without original info" |
| Post-training behavior | Adaptive omission of 3–4 turns | Highly consistent with the "omittable gray zones" in Section 3 |

### Key Findings
- Fixed TM/OM/TOM methods sacrifice either accuracy or tokens by ignoring turn differences; Agent-Omit captures the optimal frontier for both.
- The pattern "cannot omit start/end, can omit middle" is consistent across five environments, suggesting cross-domain transferability of omission strategies.
- Theoretical KL bounds align with training curves: as GRPO progresses, the agent approaches the optimal omission frontier labeled by Monte-Carlo.
- Learning omission as a first-order action is more effective than post-hoc processing (like summarization), as it utilizes task-aware RL feedback.

## Highlights & Insights
- Shift in paradigm: Context compression moves from static post-processing to a "first-order decision of the agent itself"—the model decides what to omit.
- Dual sampling solves the "omission deadlock" for credit assignment, a reusable trick for any agentic RL task involving "deletion/merging" actions.
- Explicit token interfaces (`<omit_tool_response_N>`) maintain compatibility with existing LLM tokenizers and APIs, offering a low-cost "soft retrofit" for production systems.
- Task-conditioned omit rewards are simple yet critical reward-shaping designs to avoid the collapse patterns common in efficiency-only rewards.

## Limitations & Future Work
- Experiments focused on Qwen3-8B and text-based environments; validation on larger scales, multi-modality, and long-horizon (>20 turns) tasks is needed.
- Omission currently covers full thought removal and historical observation deletion; "fine-grained omission" (partial thoughts or summarized observations) remains unexplored.
- Dual sampling doubles RL sampling costs, presenting a potential bottleneck for training 100B+ models.
- Theoretical analysis relies on semantic Lipschitz assumptions; non-continuity in rewards under minor prompt changes might loosen the bounds.

## Related Work & Insights
- **vs ToolLight / DEPO (Thought Compression)**: They focus on token-level compression; Ours performs turn-level decision-making, which is more precise and RL-learnable.
- **vs Observation-Mask / DeepMiner (Heuristic Pruning)**: They use fixed rules; Ours uses a learned strategy consistent across environments.
- **vs MEM-Agent / ReSum (LLM Summarization)**: Summarization adds LLM invocation costs and potential info distortion; omission uses direct masking with no distortion and greater token savings.
- **vs Agentic RL Mainstream (e.g., GRPO/Verl)**: This work extends agentic RL by introducing dual sampling and omit-aware rewards, orthogonally combinable with ReAct or search agent frameworks.

## Rating
- Novelty: ⭐⭐⭐⭐ Redefining context compression as a first-order action with dual sampling for credit assignment is clearly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison across five heterogeneous environments and multiple baselines is comprehensive, though missing scaling curves for model size.
- Writing Quality: ⭐⭐⭐⭐ Flow from quantitative analysis to framework and theory is smooth; Figure 3 visualization is compelling.
- Value: ⭐⭐⭐⭐⭐ Highly practical for real-world agent deployment; context cost is a major barrier, and this method is plug-and-play with existing RL pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ACON: Optimizing Context Compression for Long-horizon LLM Agents](acon_optimizing_context_compression_for_long-horizon_llm_agents.md)
- [\[ICML 2026\] Adaptive Querying with AI Persona Priors](adaptive_querying_with_ai_persona_priors.md)
- [\[ICML 2026\] Learning Efficient Guardrails for Compliance](learning_efficient_guardrails_for_compliance.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/llm_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[AAAI 2026\] AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search](../../AAAI2026/llm_agent/agentswift_efficient_llm_agent_design_via_value-guided_hierarchical_search.md)

</div>

<!-- RELATED:END -->
