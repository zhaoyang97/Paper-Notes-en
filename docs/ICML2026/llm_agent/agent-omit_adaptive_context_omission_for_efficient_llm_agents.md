---
title: >-
  [Paper Note] Agent-Omit: Adaptive Context Omission for Efficient LLM Agents
description: >-
  [ICML 2026][LLM Agent][Context Management] By using Monte-Carlo rollout to quantify "which rounds of thought/observation can be omitted…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Context Management"
  - "Thought Omission"
  - "Observation Omission"
  - "GRPO"
  - "Dual Sampling"
date: 2026-05-08
content_hash: f4c836dbc7629207
---

# Agent-Omit: Adaptive Context Omission for Efficient LLM Agents

**Conference**: ICML 2026  
**arXiv**: [2602.04284](https://arxiv.org/abs/2602.04284)  
**Code**: https://github.com/usail-hkust/Agent-Omit (available)  
**Area**: LLM Agent / Efficient Inference / Agentic Reinforcement Learning  
**Keywords**: Context Management, Thought Omission, Observation Omission, GRPO, Dual Sampling

## TL;DR
By using Monte-Carlo rollout to quantify "which rounds of thought/observation can be omitted," and then training an 8B agent with cold-start SFT and dual-sampling omit-aware GRPO, the model adaptively skips redundant reasoning and observations. On five benchmarks, token usage drops significantly while accuracy matches seven leading models.

## Background & Motivation

**Background**: LLM agents solve tasks through multi-turn thought→action→observation cycles (ReAct / agentic RL). Agents like Kimi-K2 and DeepSeek-V3.2 have demonstrated strong capabilities in deep search, web shopping, embodied decision-making, and scientific discovery. However, multi-turn interactions lead to ever-lengthening contexts and soaring token costs.

**Limitations of Prior Work**: Existing efficiency methods fall into three categories—compressing only thoughts (ToolLight, DEPO), pruning only observations (Observation-Mask, DeepMiner), or summarizing both (MEM-Agent, ReSum). All treat the entire trajectory uniformly, ignoring the vast differences in contribution across turns.

**Key Challenge**: The "necessity" of thoughts and observations is turn-dependent—early high-level planning often determines subsequent reasoning, while most early observations become obsolete by the final summary turn. Uniform compression risks deleting essential information (hurting accuracy) or retaining useless tokens (hurting efficiency).

**Goal**: Twofold: (1) Use controlled interventions to quantitatively demonstrate the feasibility of selective omission by turn; (2) Train a policy that adaptively decides, during interaction, "whether to write the current thought and which previous observations to discard."

**Key Insight**: Model the agent's omission behavior as part of the action space—outputting empty strings for thoughts and using special tokens `<omit_tool_response_N>` to explicitly delete observations—so omission can be naturally learned within SFT and RL frameworks.

**Core Idea**: The agent actively outputs "thought omission" and "observation omission" actions, and is trained with an omit-aware GRPO that couples "task reward" and "token saving" (with omission reward zeroed if the task fails), using dual sampling to address the attribution challenge of "not seeing omitted information."

## Method

### Overall Architecture
Two-stage optimization: (a) Agent Omission Behavior Synthesis (cold-start SFT)—identify omittable thought/observation rounds in each trajectory via Monte-Carlo rollout, construct both single-turn and multi-turn synthetic data, and jointly teach the base model both the omission format and reasoning continuation under omitted context. (b) Omit-Aware Agentic RL—introduce dual sampling (sampling both full trajectories and partial trajectories at each omission point) and omit-aware reward (task reward + omit reward), optimized with GRPO. Theoretically, the deviation of the learned omission policy from the optimal policy is upper-bounded by the KL divergence.

### Key Designs

1. **Quantitative Analysis + Explicit Omission Actions**:

    - **Function**: First, quantitatively demonstrate that "selective omission can indeed reduce tokens without accuracy loss," then design omission as an explicit token pattern that the agent can output, enabling direct learning in subsequent SFT/RL.
    - **Mechanism**: On WebShop + Qwen3-8B, "remove the $t$-th $\tau_t$ or $o_t$" and let the agent continue, recording tokens and Pass@1. Results: thoughts account for 45.1%, observations 52.2%, actions only 2.7%; middle-turn thoughts are omittable, final-turn observations are not, and first-turn thoughts are not—there is a large "gray area" where accuracy is maintained but tokens are significantly reduced. For actions, thought omission uses empty `<think> </think>`; observation omission uses `<omit_tool_response_N_...>`, explicitly masking historical observation set $\Gamma \subseteq \{1,\dots,t-1\}$.
    - **Design Motivation**: To upgrade from heuristic (time-window-based) deletion to a learnable policy, it is necessary to first verify that "the omission space is indeed non-empty," then provide a clear linguistic interface for omission actions; these elements enable supervision in both SFT and RL.

2. **Cold-Start Data Synthesis (Omission Behavior Synthesis)**:

    - **Function**: Teach a general LLM to become an omission-aware agent, providing an initial policy to prevent catastrophic exploration in RL.
    - **Mechanism**: For training trajectories, perform forward rollout to identify "omittable rounds"—if omitting a round reduces tokens without lowering accuracy, it is marked as omittable (see Figure 4 for examples like $\tau_2,\tau_3,o_3$). Then, construct: (i) Single-Turn omission, using a dedicated system prompt to teach the agent to output empty thoughts or omit_tool_response commands; (ii) Multi-Turn omission, replacing all omittable thoughts/observations in the trajectory with corresponding omission symbols, forcing the agent to maintain reasoning continuity even after context is omitted, avoiding context loss. Finally, perform full-parameter SFT with loss $\mathcal{L} = -\mathbb{E}_{(x,y)\sim \mathcal{D}_{single}\cup\mathcal{D}_{multi}}[\log \mathcal{P}_{\pi_\theta}(y\mid x)]$, applying a loss mask to environment observations.
    - **Design Motivation**: Directly applying RL would fail to sample positive examples since the agent cannot output omission symbols; SFT first opens up the format, embedding the notion that "omission is a legitimate action" into the model at minimal cost.

3. **Omit-aware Agentic RL: Dual Sampling + Dual Reward + GRPO**:

    - **Function**: Learn the "omission policy" as a first-order decision objective, while ensuring task accuracy is not sacrificed due to reward hacking.
    - **Mechanism**: Dual sampling—sample a full trajectory $y$ (the complete episode with omission actions) for each input, and for each omission round, extract "pre-omission context + that round's thought/action" as partial trajectory $y'$, with each $y$ deriving $p(y)$ $y'$s. This allows the agent to "see the context before omission" for credit assignment, avoiding the deadlock where "omitted information is never seen again." For rewards: task reward $R_{task}$ is given to both full and partial trajectories; omit reward $R_{omit}=\mathrm{Tok}(\tau_{omitted})/\mathrm{Tok}(y) + \mathrm{Tok}(o_{omitted})/\mathrm{Tok}(y)$ is only given to full trajectories and is zeroed if $R_{task}=0$, preventing the agent from omitting for omission's sake. The combined reward is $r(\cdot)=(1-\mu)R_{task}+\mu R_{omit}$ ($\mu=0.2$), $r'(\cdot)=R_{task}$. GRPO is used for optimization, with KL constraint $-\beta \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]$.
    - **Design Motivation**: Direct credit assignment for omission decisions requires "counterfactual non-omitted context"—which standard agentic RL cannot access; dual sampling fills this gap, making omission policy learnable. Task-conditioned omit reward explicitly encodes "speedup without accuracy loss," more robust than a simple weighted sum.

### Loss & Training

SFT stage uses standard LM loss with environment observation loss mask; RL stage objective is
$$
\max_{\pi_\theta} \mathbb{E}_{x,\{y_i,\{y'_{i,j}\}\}}\left[\frac{1}{n}\sum_i \left(r(x,y_i) + \frac{1}{p(y_i)}\sum_j r'(x,y'_{i,j})\right)\right] - \beta \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]
$$
Base model is Qwen3-8B. Theoretically, under the semantic Lipschitz assumption, the authors prove that the effect/efficiency deviation is upper-bounded by $\delta + K' \cdot \mathrm{KL}(\pi^\ast,\pi_\theta)$, indicating that as KL decreases, the learned omission policy monotonically approaches the optimal.

## Key Experimental Results

### Main Results

Five agent environments (DeepSearch, WebShop, TextCraft, BabyAI, SciWorld) are compared with seven leading LLMs (DeepSeek-R1-0528, DeepSeek-V3.2, o3/o4-mini, Qwen3-235B-A22B, Qwen3-Next-80B-A3B, Qwen3-32B) and seven efficient agent construction methods.

| Comparison | Pass@1 Accuracy | Token Cost | Notes |
|------------|-----------------|------------|-------|
| Agent-Omit-8B (based on Qwen3-8B) | Comparable to seven leading LLMs | Significantly lower | 8B achieves peer-level accuracy with half or less the tokens of large models |
| Seven efficient agent methods (TM / OM / TOM) | Each has strengths | Each has strengths | Agent-Omit achieves the best effect-efficiency trade-off |
| Qwen3-8B native | Baseline | Baseline | Without omission, thoughts 45.1% + observations 52.2% of tokens |

### Ablation Study

| Configuration | Key Phenomenon | Interpretation |
|---------------|----------------|---------------|
| SFT only (no RL) | Learns omission format but limited gains | RL is needed to learn when to omit adaptively |
| No dual sampling | Omission policy fails to converge | Partial trajectories are necessary for omission credit assignment |
| No $R_{omit}$ | Nearly identical to original agent | Lacks explicit efficiency incentive |
| $R_{omit}$ not coupled with $R_{task}$ | Reward hacking occurs, accuracy drops | Strong constraint "omit reward zero if task fails" is necessary |
| Single-turn omission only | Poor generalization to multi-turn scenarios | Multi-turn synthetic data forces the model to learn to continue reasoning without original information |
| Post-training agent behavior analysis | Adapts to omit 3–4 rounds of thought/observation, concentrated in middle turns | Highly consistent with the "omittable gray area" in Section 3's quantitative analysis |

### Key Findings
- Uniform TM/OM/TOM methods sacrifice either accuracy or tokens due to ignoring turn differences; Agent-Omit achieves the best frontier in both accuracy and token usage.
- The pattern "cannot omit first/last, can omit middle" is consistent across five environments, indicating cross-domain transferability of the omission policy.
- The theoretical KL upper bound matches the actual training curve: as GRPO training proceeds, the agent approaches the Monte-Carlo annotated optimal omission frontier.
- Learning omission as a first-order action is more effective than post-hoc processing (e.g., summarization), as the former leverages RL's task-aware feedback.

## Highlights & Insights
- Reframes "context compression" from a static post-processing problem to a first-order agent decision—a paradigm shift: previous work compressed the model externally, this work lets the model decide what to omit.
- Dual sampling resolves the deadlock of "no attribution after omission," providing a reusable trick for agentic RL with "delete/merge" actions; this approach can transfer to any policy learning scenario involving deletion/merging.
- Explicit token interface (`<omit_tool_response_N>`) ensures full compatibility with existing LLM tokenizers/APIs, making deployment low-cost—a practical "soft retrofit" for production systems.
- The simple but critical reward shaping—"omit reward zero if task fails"—prevents the collapse mode common to efficiency-only rewards.

## Limitations & Future Work
- Experiments are limited to Qwen3-8B and five text-based agent environments; effectiveness on larger models, multimodal, or long-horizon (>20 turns) tasks remains to be validated.
- Omission actions currently cover only full thought omission and historical observation deletion; the space of fine-grained omission (partial thought omission, observation summarization) is unexplored.
- Dual sampling more than doubles RL sampling cost, increasing computational overhead; scaling to 100B+ training is a potential bottleneck.
- Theoretical analysis relies on the semantic Lipschitz assumption; in practice, LLM reward discontinuity under minor prompt changes may loosen the upper bound.

## Related Work & Insights
- **vs ToolLight / DEPO (thought compression)**: They perform token-level compression; this work does turn-level decision, which is more fine-grained and RL-learnable.
- **vs Observation-Mask / DeepMiner (heuristic observation deletion)**: They use fixed rules; this work uses learned policies, consistent across environments.
- **vs MEM-Agent / ReSum (LLM summarization)**: Summarization incurs LLM invocation cost and information distortion; omission directly masks, with no distortion and more thorough token savings.
- **vs Mainstream Agentic RL (e.g., GRPO/Verl for search agents)**: This work extends GRPO with dual sampling and omit-aware rewards, orthogonally stackable with ReAct/search agent frameworks.

## Rating
- Novelty: ⭐⭐⭐⭐ Redefines context compression as a first-order agent action, with clear innovation in dual-sampling credit assignment.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five heterogeneous environments + seven LLMs + seven efficiency methods provide comprehensive comparison, though scaling curves for larger models are missing.
- Writing Quality: ⭐⭐⭐⭐ Smooth logic from quantitative analysis → framework → theory → experiments; Figure 3 visualization is highly convincing.
- Value: ⭐⭐⭐⭐⭐ Directly useful for real-world agent deployment—context cost is one of the most expensive parts of agent landing, and this method is plug-and-play with existing RL pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search](../../AAAI2026/llm_agent/agentswift_efficient_llm_agent_design_via_value-guided_hierarchical_search.md)
- [\[ICML 2026\] Adaptive Querying with AI Persona Priors](adaptive_querying_with_ai_persona_priors.md)
- [\[ICLR 2026\] Efficient Agent Training for Computer Use](../../ICLR2026/llm_agent/efficient_agent_training_for_computer_use.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/llm_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[AAAI 2026\] When Refusals Fail: Unstable Safety Mechanisms in Long-Context LLM Agents](../../AAAI2026/llm_agent/when_refusals_fail_unstable_safety_mechanisms_in_long-context_llm_agents.md)

</div>

<!-- RELATED:END -->
