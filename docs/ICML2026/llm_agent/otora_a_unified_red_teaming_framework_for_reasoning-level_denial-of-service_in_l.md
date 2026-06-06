---
title: >-
  [Paper Note] OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents
description: >-
  [ICML 2026][LLM Agent][Reasoning-Level DoS] OTora proposes a novel attack paradigm called Reasoning-Level Denial-of-Service (R-DoS): without compromising task correctness…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Reasoning-Level DoS"
  - "Red Teaming"
  - "Tool-Use Hijacking"
  - "Reasoning Payload Optimization"
  - "ICL Genetic Search"
date: 2026-05-08
content_hash: 026bba61fdbc9d01
---

# OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents

**Conference**: ICML 2026  
**arXiv**: [2605.08876](https://arxiv.org/abs/2605.08876)  
**Code**: https://github.com/llm2409/OTora  
**Area**: LLM Agent Security / Red Teaming  
**Keywords**: Reasoning-Level DoS, Red Teaming, Tool-Use Hijacking, Reasoning Payload Optimization, ICL Genetic Search

## TL;DR
OTora proposes a novel attack paradigm called Reasoning-Level Denial-of-Service (R-DoS): without compromising task correctness, it employs a two-stage red teaming pipeline (first using insertion-aware optimization to induce the agent to fetch attacker-controlled external resources, then delivering a "reasoning payload" optimized via ICL genetic search) to force the LLM agent into a sustained multi-round over-reasoning state. It achieves 10× reasoning token inflation and order-of-magnitude latency increases across WebShop, Email, and OS agents, while maintaining nearly identical terminal task accuracy.

## Background & Motivation

**Background**: Existing LLM security research generally falls into three categories: (i) jailbreak attacks for illicit content generation; (ii) agent hijacking for incorrect tool invocation or data exfiltration; and (iii) overthinking research observing how misleading inputs cause reasoning models to consume more tokens. Traditional and application-layer DoS exist on the systems side.

**Limitations of Prior Work**: Previous works focus on "correctness" or "behavioral shift," **ignoring a fundamental failure mode: the agent output remains correct, but it is induced into excessive meaningless reasoning, violating latency/SLA/cost budgets**, constituting a Denial-of-Service from an availability perspective. This is critical for real-world LLM agent deployments with strict timeouts and cost budgets.

**Key Challenge**: Traditional DoS features (traffic flooding, erroneous outputs) are easily detected. Reasoning-level DoS, characterized by "correct output + massive latency," bypasses detection systems based on output correctness or safety policies—the system cannot block it because the agent technically does nothing "wrong."

**Goal**: (a) Formally define the R-DoS threat model; (b) construct an **automated**, **unified**, **white/black-box compatible** red teaming framework for stable R-DoS instantiation; (c) quantify its impact on real agent systems and discuss defenses.

**Key Insight**: Agents generally treat "tool returns" and "environment observations" as trusted inputs. Thus, an attack can be decoupled into two stages: first, inducing the agent to fetch an attacker-controlled URL (external access trigger), and then placing a payload at that URL to force the agent into compute-intensive reasoning. This separation is necessary because injection channels in instructions or third-party environments are often too narrow/noisy for long payloads, whereas once a fetch is established, the attacker gains full control over the content delivery.

**Core Idea**: A staged "narrow-channel trigger + wide-channel delivery" red teaming approach combined with a "persistent policy segment" to transform a single hijack into sustained multi-round over-reasoning.

## Method

### Overall Architecture
OTora is a two-stage end-to-end pipeline (Algorithm 1): **Stage I** injects an adversarial suffix $s$ into user instructions or observations to make the victim agent $\mathcal{M}$ naturally include "access attacker.test" in its ReAct plan. **Stage II** hosts a reasoning payload $r$ on attacker.test, optimized via multi-objective genetic search, making the agent persistently over-reason across multiple rounds while preserving final task correctness. The pipeline works for both white-box and black-box agents, using top-$k$ logprobs or surrogate models in black-box settings.

### Key Designs

1.  **Stage I: Attention-aware Insertion Point Scoring + Dynamic Target Co-evolution**:
    -   **Function**: Identifies optimal positions in the agent's response sequence to insert adversarial suffixes while allowing "target token strings" to evolve along the response distribution for easier triggering.
    -   **Mechanism**: A position scoring function is defined as $r_j(t) = \tfrac{1}{|t|+1}(\alpha M_j(t) + \beta P_j(t) + \lambda A_j(t,s))$, representing prefix matching, average probability under distribution $\mathcal{P}$, and average attention allocated to suffix $s$. The attention term filters pseudo-signals stemming from context priors. **Dynamic target co-evolution** allows the target phrase to shift toward high-probability tokens or semantically equivalent candidates $\mathcal{T}$ generated by an auxiliary LLM. The optimal suffix $s$ is optimized via gradient descent (white-box) or log-prob search (black-box) to maximize $\sum_{j\in\mathcal{J}}\log p(t^\star\mid x,o,s,z_{[:j]})$.
    -   **Design Motivation**: Fixed target phrases limit the search space and may not match the agent's style. Co-evolution significantly improves trigger rates. The attention term ensures optimization convergence by focusing on the actual influence of $s$.

2.  **Stage II: Agent-aware Persistent Payload + ICL-guided Genetic Search**:
    -   **Function**: Generates reasoning expansion that persists **across multiple rounds** rather than a single step.
    -   **Mechanism**: The payload is split into a **local sink segment** (an immediate complex problem) and a **persistent policy segment** (meta-instructions injected into history to guide subsequent Thoughts toward verbose reasoning). This exploits the ReAct property where each new Thought is conditioned on the entire interaction history. The optimization objective is $\mathrm{Score}(r) = w_1 S_{\text{RTI}} + w_2 S_{\text{FID}} + w_3 S_{\text{STAB}}$, measuring token inflation, task fidelity, and stability (negative variance). A black-box genetic search uses an ICL-capable model $\mathcal{M}_{\text{ICL}}$ to mutate top-performing payloads conditioned on the agent's context.
    -   **Design Motivation**: Standard overthinking attacks only increase single-step tokens. The "local + persistent" structure and ICL mutations create a sustained policy impact. Multi-objective scoring prevents low-quality payloads that break task logic.

3.  **Unified White/Black-box Interface and Fidelity Evaluation**:
    -   **Function**: Enables gradient optimization for white-box models and probability-based search for black-box APIs (e.g., GPT-3.5/Gemini).
    -   **Mechanism**: In black-box settings, attention $A_j$ is set to zero or approximated via surrogate models, and gradient-based optimization is replaced by discrete search with log-prob feedback. Evaluation metrics include $\mathrm{ASR}_S$ (target sequence rate), $\mathrm{ASR}_H$ (valid tool call rate), Hit (payload execution rate), and Accuracy.
    -   **Design Motivation**: Real-world victims are often black-box; the framework must degrade gracefully. Separating trigger reliability from expansion magnitude avoids overestimating end-to-end effectiveness.

### Loss & Training
Stage I targets $\max_s \sum_{j\in\mathcal{J}}\log p(t^\star\mid x,o,s,z_{[:j]})$, using GCG-like discrete gradient search (white-box) or log-prob search (black-box). Stage II employs black-box multi-objective genetic search without gradients, using iterative evaluation and ICL mutation with $w_1=w_2=w_3=1$.

## Key Experimental Results

### Main Results
Victim agents: WebShop (shopping) + InjecAgent Email/OS (system agents). Backbones: LLaMA-70B, GPT-OSS-120B, Gemini-1.5-Flash, GPT-3.5-Turbo.

| Metric | OTora Results |
|---|---|
| Reasoning Token Inflation | Up to 10× |
| End-to-end latency amplification | Order-of-magnitude slowdown |
| Task accuracy change | Minimal (Preservation of correctness) |
| Stage I trigger $\mathrm{ASR}_H$ | Significantly outperforms SNES baselines on Gemini-1.5-Flash |

Stage I black-box experiments show OTora's insertion-aware scoring + co-evolution achieves the best $\mathrm{ASR}_S/\mathrm{ASR}_H$ across various model-agent combinations with fewer iterations.

### Ablation Study

| Configuration | Impact |
|---|---|
| Remove attention term ($\lambda=0$) | Lower optimization stability; degrades to pure likelihood search. |
| Fixed target phrase (No co-evolution) | Significant drop in trigger success rate. |
| Local sink only (No persistent policy) | RTI increases per step, but multi-round expansion and slowdown vanish. |
| Score optimized for RTI only | FID drops sharply; payloads frequently break task logic. |
| ICL mutation $\rightarrow$ Random mutation | Slower convergence and poorer payload quality. |

### Key Findings
- The "persistent policy segment" is the critical design for multi-round amplification; without it, the end-to-end slowdown magnitude is significantly reduced.
- Attention-aware scoring contributes significantly to optimization stability, especially when white-box attention is available.
- Defenses like budgeted reasoning or relevance filtering can mitigate but not eliminate R-DoS, particularly against "low-and-slow" stealth modes.

## Highlights & Insights
- Formally defines the "Reasoning-level DoS" threat model, shifting the traditional DoS paradigm to the "reasoning budget" dimension of LLM agents.
- The "narrow-channel trigger + wide-channel delivery" paradigm is a universal pattern for bypassing prompt injection size limits.
- The vulnerability in ReAct's history-concatenation architecture naturally amplifies any attack that successfully injects into the interaction history.

## Limitations & Future Work
- Evaluation focused on WebShop/Email/OS; generalizability to complex multi-agent systems (e.g., AutoGen) or advanced toolchains (MCP) needs verification.
- Persistent policies rely on history concatenation; efficacy might decrease for agents using truncated history or pure state-based memory.
- Black-box attacks assume the availability of top-$k$ logprobs, which are increasingly restricted by closed-source providers.
- Defense discussions are preliminary; systematic R-DoS detectors (e.g., reasoning anomaly detection) have yet to be designed.

## Related Work & Insights
- **vs. Jailbreak (GCG / Pliny)**: Jailbreak aims for "illicit content," manipulating input-output layers. OTora maintains correctness but attacks the runtime budget, bypassing common detection surfaces.
- **vs. Agent Hijacking (InjecAgent)**: Hijacking changes actions (wrong tool, data leak), which safety filters can catch. OTora changes only the reasoning path length, bypassing filters.
- **vs. Overthinking Attacks**: Traditional overthinking targets single-step QA models; OTora scales this to multi-turn agents and introduces persistence for cumulative amplification.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to formalize and systematically solve the R-DoS threat model.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage of agents/backbones, though defense experiments are limited.
- Writing Quality: ⭐⭐⭐⭐ Clear definitions and algorithmic structure, though symbol-dense.
- Value: ⭐⭐⭐⭐⭐ Highlights a critical availability blind spot in LLM agent deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](../../AAAI2026/llm_agent/moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)
- [\[ICML 2026\] On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents](on_information_self-locking_in_reinforcement_learning_for_active_reasoning_of_ll.md)
- [\[ACL 2026\] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](../../ACL2026/llm_agent/hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)
- [\[ICML 2026\] Answer Only as Precisely as Justified: Calibrated Claim-Level Specificity Control for Agentic Systems](answer_only_as_precisely_as_justified_calibrated_claim-level_specificity_control.md)
- [\[ICML 2026\] Process Reward Agents for Steering Knowledge-Intensive Reasoning](process_reward_agents_for_steering_knowledge-intensive_reasoning.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2025\] AdvAgent: Controllable Blackbox Red-teaming on Web Agents](../../ICML2025/llm_agent/advagent_controllable_blackbox_red-teaming_on_web_agents.md)
- [\[ICML 2026\] On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents](on_information_self-locking_in_reinforcement_learning_for_active_reasoning_of_ll.md)
- [\[AAAI 2026\] MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](../../AAAI2026/llm_agent/moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)
- [\[ICML 2026\] Answer Only as Precisely as Justified: Calibrated Claim-Level Specificity Control for Agentic Systems](answer_only_as_precisely_as_justified_calibrated_claim-level_specificity_control.md)
- [\[ICML 2026\] Process Reward Agents for Steering Knowledge-Intensive Reasoning](process_reward_agents_for_steering_knowledge-intensive_reasoning.md)

</div>

<!-- RELATED:END -->
