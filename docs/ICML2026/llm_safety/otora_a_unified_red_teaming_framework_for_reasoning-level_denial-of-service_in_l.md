---
title: >-
  [Paper Note] OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents
description: >-
  [ICML 2026][LLM Safety][Reasoning-Level DoS] OTora introduces a novel attack paradigm, Reasoning-Level Denial-of-Service (R-DoS): without compromising task correctness…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Reasoning-Level DoS"
  - "Red Team Attack"
  - "Tool Invocation Hijacking"
  - "Reasoning Payload Optimization"
  - "ICL Genetic Search"
date: 2026-05-08
content_hash: e5b365759278c2b4
---

# OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents

**Conference**: ICML 2026  
**arXiv**: [2605.08876](https://arxiv.org/abs/2605.08876)  
**Code**: https://github.com/llm2409/OTora  
**Area**: LLM Agent Security / Red Team Attacks  
**Keywords**: Reasoning-Level DoS, Red Team Attack, Tool Invocation Hijacking, Reasoning Payload Optimization, ICL Genetic Search  

## TL;DR
OTora introduces a novel attack paradigm, Reasoning-Level Denial-of-Service (R-DoS): without compromising task correctness, it employs a two-stage red teaming pipeline (first using insertion-aware optimization to induce the agent to proactively access attacker-controlled external resources, then deploying "reasoning-type payloads" optimized via ICL genetic search at those resources) to trap LLM agents in prolonged multi-turn overthinking states. On WebShop, Email, and OS agents, this achieves up to 10× reasoning token inflation and orders-of-magnitude latency attacks, with final task accuracy nearly unchanged.

## Background & Motivation

**Background**: Existing LLM security research falls into three main categories—(i) jailbreak attacks that induce models to output prohibited content; (ii) agent behavior hijacking that causes agents to misuse tools or leak data; (iii) overthinking studies observing that misleading inputs can cause reasoning models to consume more tokens. On the system side, there are also traditional DoS and application-layer DoS attacks.

**Limitations of Prior Work**: The above works focus on "correctness" or "behavioral deviation," **overlooking a fundamental failure mode—where the agent's output remains correct, but it is induced to perform excessive, meaningless reasoning, thereby violating latency/SLA/cost budgets and constituting a denial of service from an availability perspective**. This is especially critical in real-world LLM agent deployments, as industrial systems typically enforce strict timeouts and cost budgets.

**Key Challenge**: Traditional DoS attacks (traffic flooding, erroneous outputs) are easily detected; in contrast, reasoning-level DoS, characterized by "correct output + drastic latency increase," renders all detection systems based on output correctness or security policies ineffective—there is nothing to block, as nothing is technically wrong.

**Goal**: (a) Formally define the R-DoS threat model; (b) construct an **automated**, **unified**, **white/black-box compatible** red teaming framework to reliably instantiate R-DoS; (c) quantify its impact on real agent systems and discuss defenses.

**Key Insight**: Agents generally treat "tool returns" and "environment observations" as trusted inputs. Thus, the attack can be divided into two stages: first, induce the agent to proactively fetch an attacker-controlled URL (triggering external access), then have the attacker pre-place a payload at that URL to automatically trap the agent in compute-intensive reasoning. The reason for this two-stage separation is that injection channels into instructions or third-party environments are too narrow and noisy to reliably deliver long payloads; but once fetching succeeds, the attacker fully controls the content and can reliably deliver arbitrarily complex payloads.

**Core Idea**: "Narrow-channel trigger + wide-channel delivery" staged red teaming + "persistent strategy segment" to convert a single hijack into multi-turn sustained overthinking.

## Method

### Overall Architecture
OTora is a two-stage end-to-end pipeline (Algorithm 1): **Stage I** injects an adversarial suffix $s$ into user instructions or environment observations, causing the victim agent $\mathcal{M}$ to naturally include "visit attacker.test" in its ReAct action plan; **Stage II** pre-places a reasoning payload $r$, optimized via multi-objective genetic search, on the already-accessed attacker.test, causing the agent to fall into multi-turn, sustained overthinking upon receiving the fetched content, while maintaining correct final task output. The entire pipeline works for both white-box and black-box agents, with API top-$k$ logprobs or surrogate models replacing gradients in black-box settings.

### Key Designs

1. **Stage I: Attention-Aware Insertion Point Scoring + Dynamic Target Co-evolution**:

    - **Function**: Identify the optimal position in the agent's response sequence to insert the adversarial suffix, while allowing the "target token sequence" to evolve along the response distribution for easier triggering.
    - **Mechanism**: Define a position scoring function $r_j(t) = \tfrac{1}{|t|+1}(\alpha M_j(t) + \beta P_j(t) + \lambda A_j(t,s))$, where the three terms represent "number of matches between the prefix at that position and the target token sequence," "average probability of matched tokens under distribution $\mathcal{P}$," and "average attention assigned to suffix $s$ during generation." The third term avoids false signals from context priors. Then, perform **dynamic target co-evolution**: instead of fixing the target phrase, use high-probability tokens from $\mathcal{P}$ plus an auxiliary LLM to generate a set of semantically equivalent candidates $\mathcal{T}$, select $t^\star = \arg\max_{t^{(k)}}\max_j r_j(t^{(k)})$; then use weighted interval scheduling to select the top-$\ell$ non-overlapping insertion points. Finally, optimize $s$ via gradient descent (white-box) or log-prob search (black-box) to maximize $\sum_{j\in\mathcal{J}}\log p(t^\star\mid x,o,s,z_{[:j]})$.
    - **Design Motivation**: Fixing the target phrase limits the search space, and the agent's own language style may not match manually set phrases; co-evolution allows the target to drift with the agent's response distribution, significantly improving trigger rates. The attention term filters out false matches (apparent target tokens unrelated to $s$), stabilizing optimization convergence.

2. **Stage II: Agent-Aware Persistent Payload + ICL-Guided Genetic Search**:

    - **Function**: Ensure a single hijack produces **multi-turn** reasoning inflation, not just longer single-step responses.
    - **Mechanism**: Decompose the payload into two segments—**local sink segment** (poses a complex math/logic problem at the hijacked step for immediate computation), **persistent strategy segment** (injected as a meta-instruction into the agent's history, guiding subsequent Thoughts to adopt a more elaborate reasoning style). This leverages a key property of ReAct agents: each new Thought-Action is conditioned on the entire interaction history, so once the persistent segment is written into history, future rounds naturally repeat it → single hijack becomes multi-turn amplification. The optimization objective is a multi-objective score $\mathrm{Score}(r) = w_1 S_{\text{RTI}} + w_2 S_{\text{FID}} + w_3 S_{\text{STAB}}$, measuring reasoning token inflation, final task fidelity, and cross-seed stability (expressed as -Var), with equal weights $w_i=1.0$. The backend is black-box genetic search, with each generation using an ICL-capable model $\mathcal{M}_{\text{ICL}}$ to mutate top payloads under agent context.
    - **Design Motivation**: Traditional overthinking attacks only increase single-step tokens, with limited impact on multi-turn agents; the "local + persistent" two-segment structure and "context-aware ICL mutation" enable sustained strategic attacks rather than one-off costs. Multi-objective scoring ensures that "obviously garbage payloads" cannot pass—task fidelity and cross-seed stability must also be maintained.

3. **Unified White/Black-Box Interface and Fidelity Evaluation**:

    - **Function**: Enable the same framework to use gradient optimization in white-box (with logits and attention) and degrade to probability search in black-box APIs like GPT-3.5/Gemini.
    - **Mechanism**: In black-box settings, set the attention $A_j$ in Algorithm 1 to zero ($\lambda=0$) or approximate with surrogate model attribution; replace gradient-based optimization with discrete search using log-prob feedback. For fidelity evaluation, introduce $\mathrm{ASR}_S$ (target token sequence occurrence rate), $\mathrm{ASR}_H$ (actual effective tool invocation rate by the agent), Hit (whether Stage II content is executed), and Accuracy (task correctness), decoupling trigger reliability and latency amplification via multiple metrics.
    - **Design Motivation**: Many real-world victims are black-box APIs, so the attacker must degrade gracefully; to avoid "inflated attack success statistics," trigger reliability and sustained amplification are evaluated separately, with the product $\mathrm{ASR}_H \times \mathrm{Hit}$ estimating end-to-end effectiveness.

### Loss & Training
Stage I objective is $\max_s \sum_{j\in\mathcal{J}}\log p(t^\star\mid x,o,s,z_{[:j]})$, using GCG-like discrete gradient search in white-box and log-prob search in black-box. Stage II uses black-box multi-objective genetic search, no gradients, with evaluation and ICL mutation per generation; weights $w_1=w_2=w_3=1$.

## Key Experimental Results

### Main Results
Benchmark agents: WebShop (shopping agent) + InjecAgent's Email/OS (system agents); backbones include LLaMA-70B, GPT-OSS-120B, Gemini-1.5-Flash, GPT-3.5-Turbo. Core conclusions from the summary:

| Metric | OTora Result |
|---|---|
| Reasoning Token Inflation | Up to 10× |
| End-to-end latency amplification | Orders-of-magnitude slowdown |
| Task accuracy change | Close to baseline (preservation of correctness) |
| Stage I trigger ASR_H | Significantly outperforms baselines like SNES on Gemini-1.5-Flash WebShop black-box |

Stage I black-box experiments (Table 1 excerpt) use ASR_S/ASR_H and Iters as metrics; OTora's insertion-aware scoring + co-evolution achieves best results across multiple (model, agent) combinations, with reduced average iterations.

### Ablation Study

| Configuration | Impact |
|---|---|
| Remove attention term ($\lambda=0$) | Optimization stability drops, black-box degenerates to pure likelihood search |
| Fixed target phrase (no co-evolution) | Trigger success rate drops significantly |
| Sink segment only (no persistent strategy) | RTI increases per step but multi-turn amplification disappears, end-to-end slowdown much weaker |
| Score switched to single-objective RTI | FID drops sharply—often breaks task correctness |
| ICL-guided mutation → random mutation | Genetic search converges slowly, payload quality poor |

### Key Findings
- The "persistent strategy segment" is key to converting a single hijack into multi-turn amplification; removing it reduces overall slowdown by an order of magnitude.
- Attention-aware scoring significantly contributes to optimization stability, especially when attention is available in white-box settings.
- Defenses (budgeted reasoning, relevance filtering, runtime monitoring) can partially mitigate but cannot fundamentally solve R-DoS, especially ineffective against low-and-slow infiltration.

## Highlights & Insights
- First formalization of the "reasoning-level DoS" threat model, transferring traditional DoS thinking to the "reasoning budget" dimension of LLM agents, opening new directions for security research.
- The "narrow-channel trigger + wide-channel delivery" two-stage decomposition is a general pattern for handling all prompt injection channel limitations, directly transferable to other long-payload attack or defense research.
- "Persistent strategy segment + ReAct history reuse" reveals an essential vulnerability of ReAct agents: the history concatenation architecture inherently amplifies any attack that can inject into history.

## Limitations & Future Work
- Evaluation focuses on WebShop / Email / OS agents; transferability to more complex multi-agent systems (e.g., AutoGen), or strong toolchains (MCP), needs validation.
- The persistent strategy segment relies on ReAct history concatenation; effectiveness may diminish for agents with truncated histories or pure state-based memory frameworks.
- Black-box attacks assume access to top-$k$ logprobs, but closed-source APIs are increasingly restricting this, limiting practical feasibility.
- Defense discussion in the paper is preliminary; no systematic design of detectors specifically for R-DoS (e.g., reasoning anomaly detection).

## Related Work & Insights
- **vs Jailbreak Attacks (GCG / Pliny)**: Jailbreaks aim to induce models to output prohibited content, operating at the input-output layer; OTora maintains output correctness, attacks runtime budget, and targets a completely different detection surface.
- **vs Agent Behavior Hijacking (e.g., InjecAgent)**: Behavior hijacking changes actions (misusing tools, leaking data), which can be blocked by security filters; OTora does not alter actions, only the reasoning path length, bypassing all security filters.
- **vs Overthinking Attacks**: Traditional overthinking inflates tokens in single-step QA models; OTora upgrades this to multi-turn agents and introduces the "persistent strategy segment" for cross-turn amplification.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to define the R-DoS threat model and systematically solve it, opening a new direction
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage of agents/backbones/white-black box, but limited depth in defense experiments
- Writing Quality: ⭐⭐⭐⭐ Clear algorithm boxes and definitions, but dense notation requires repeated reading
- Value: ⭐⭐⭐⭐⭐ Reveals a real-world LLM agent availability security blind spot, with important implications for system design

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance](stable-gflownet_toward_diverse_and_robust_llm_red-teaming_via_contrastive_trajec.md)
- [\[ACL 2026\] STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming](../../ACL2026/llm_safety/star-teaming_a_strategy-response_multiplex_network_approach_to_automated_llm_red.md)
- [\[ICML 2026\] STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack](stare_step-wise_temporal_alignment_and_red-teaming_engine_for_multi-modal_toxici.md)
- [\[AAAI 2026\] An LLM-Based Simulation Framework for Embodied Conversational Agents in Psychological Counseling](../../AAAI2026/llm_safety/an_llm-based_simulation_framework_for_embodied_conversationa.md)
- [\[ICLR 2026\] Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks (DialTree)](../../ICLR2026/llm_safety/tree-based_dialogue_reinforced_policy_optimization_for_red-teaming_attacks.md)

</div>

<!-- RELATED:END -->
