---
title: >-
  [Paper Note] Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents
description: >-
  [ICLR2026][LLM Safety][agent security] The authors propose the **threat snapshots** framework—isolating the exact moment an LLM vulnerability manifests within an AI agent's execution flow. Using 190,000 crowdsourced adversarial attacks to build the **b3 benchmark**, they rank the backbone security of 34 mainstream LLMs. They unearth counter-intuitive findings, such as "reasoning capabilities enhance security, while model size is unrelated to security."
tags:
  - "ICLR2026"
  - "LLM Safety"
  - "agent security"
  - "backbone LLM"
  - "threat snapshot"
  - "adversarial attack"
  - "security benchmark"
date: 2026-05-08
content_hash: 761a4345047fc0b6
---

# Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=kga18ld70t](https://openreview.net/forum?id=kga18ld70t)  
**Code**: Open-source benchmark + dataset + evaluation code (The paper promises release, b3 benchmark)  
**Area**: Agent Security / LLM Security / Security Evaluation  
**Keywords**: agent security, backbone LLM, threat snapshot, adversarial attack, security benchmark

## TL;DR
The authors propose the **threat snapshots** framework—isolating the exact moment an LLM vulnerability manifests within an AI agent's execution flow. Using 190,000 crowdsourced adversarial attacks to build the **b3 benchmark**, they rank the backbone security of 34 mainstream LLMs. They unearth counter-intuitive findings, such as "reasoning capabilities enhance security, while model size is unrelated to security."

## Background & Motivation
**Background**: LLM-driven AI agents are being deployed at scale, yet the industry lacks a systematic understanding of how changing a backbone LLM affects agent safety and risk. Existing works (AgentDojo, InjecAgent, etc.) mostly evaluate specific agent security via benchmarks, competitions, or frameworks.

**Limitations of Prior Work**: The authors identify two critical flaws in existing frameworks. First, their threat coverage is incomplete—often focusing only on restricted attack vectors like indirect injection or remote code execution, failing to capture the full spectrum of LLM vulnerabilities. Second, they require **mocking the entire agent along with its complete execution flow**, which is cumbersome and conflates "novel LLM-specific vulnerabilities" with "traditional software issues (permission mismatches, XSS, etc.)," thereby obscuring the true source of risk. Furthermore, many studies confuse **safety** (broad safety like toxicity/reliability) with **security** (exploivability by adversaries in deployment), whereas this paper focuses strictly on the latter.

**Key Challenge**: Systematically modeling AI agent security is difficult due to two factors: ① Agents make decisions based on the **non-deterministic black-box output** of backbone LLMs, making it impossible to map a fixed execution flow like standard programs. ② LLMs are mechanically **unable to programmatically distinguish "data" from "instructions"**; when coupled with traditional software via tools, this new "natural language instruction injection" vulnerability becomes entangled with traditional security flaws.

**Goal**: To systematically answer "how the choice of backbone LLM impacts agent security" by breaking it into two sub-problems: how to **cleanly model only the LLM-layer vulnerabilities** (unencumbered by the full execution flow) and how to **fairly compare the adversarial resistance of different LLMs** when serving as backbones.

**Key Insight**: The authors observe a crucial fact—under a formal definition, **each LLM call is stateless**: it receives all information needed for the next step of reasoning in the current context, independent of hidden internal states. Thus, one need not model the entire execution flow; it suffices to capture the "state slice where the vulnerability manifests."

**Core Idea**: Use **threat snapshots** to freeze the "specific LLM call + the attacker's objective and delivery method at that moment" into an atomic snapshot. This isolates LLM vulnerabilities from traditional ones and ensures comparability across different backbone LLMs under the same snapshot.

## Method

### Overall Architecture
The methodology consists of an "abstract framework + an attack taxonomy + a real-world benchmark." First, a formal definition decomposes the AI agent into "backbone LLM $m$ + four processing components," proving each LLM call is stateless. Based on this, the threat snapshot is proposed as an atomic abstraction. A comprehensive attack taxonomy is then designed to ensure the threat snapshot set is exhaustive. Finally, strong attacks are collected via crowdsourcing to assemble the b3 benchmark, producing vulnerability rankings for 34 LLMs.

Formally, an agent is defined as $A_{m,f}:\mathcal{I}\to\mathcal{R}$, where input request $I$ leads to response $R$ after multiple iterations. The backbone LLM is $m:\mathcal{C}\to\mathcal{O}$ (mapping context to output), with four processing components $f=(f_{\text{proc}}, f_{\text{stop}}, f_{\text{in}}, f_{\text{out}})$: an input processor $f_{\text{in}}$ transforms the request into the initial context; a processing function $f_{\text{proc}}$ parses LLM output, calls tools, and constructs the next context; a stop condition $f_{\text{stop}}$ determines termination; and a response processor $f_{\text{out}}$ generates the final reply. The agent alternates between "LLM steps (calling $m$)" and "process steps (calling $f_{\text{proc}}$)." The paper focuses strictly on backbone $m$; multi-agent systems are integrated by designating one LLM as the evaluation target and treating others' outputs as part of $f_{\text{proc}}$.

### Key Designs

**1. Threat snapshot: Isolating the moment of vulnerability manifestation**

To address the issue that existing frameworks require mocking the full execution flow and conflate LLM and traditional vulnerabilities, the authors define an **LLM vulnerability**: in agent $A_{m,f}$, if an attacker has **partial control** over the context fed to the LLM at time $t$ and can insert attack $a$ into $C_t$ to get poisoned context $C_t^p(a)$, such that output $O_t^p(a):=m(C_t^p(a))$ deviates from the normal output ($O_t^p(a)\neq O_t$), an LLM vulnerability has been exploited. These are "insecure features" resulting from the useful property of following natural language instructions, rather than patchable bugs.

A threat snapshot freezes this event, containing two parts: **Agent state** (description of agent function, current state, and the unpoisoned context $C_t$) and **Threat description** (attack taxonomy label, attack insertion function $C_t\mapsto C_t^p(a)$, and a scoring function for $O_t^p(a)$ in $[0,1]$). Since LLMs are stateless, multi-turn and multi-agent attacks can be **decomposed into a chain of threat snapshots**. This makes threat snapshots a complete atomic abstraction regardless of complexity. This paper uses **single-step threat snapshots**, arguing that models vulnerable in a single step are inherently more vulnerable in multi-step scenarios, providing a lower bound for system security while ensuring broader coverage.

**2. Dual-perspective taxonomy: Ensuring exhaustive threat coverage**

To ensure the benchmark's credibility, the authors construct **two complementary taxonomies**. The **Vector-objective taxonomy** categorizes by "delivery method × attack objective," facilitating threat modeling for specific agents: vectors are **direct** (attacker as "user") or **indirect** (attack hidden in web pages, files, tools); objectives include data exfiltration, content injection, decision manipulation, DoS, system intrusion, and policy bypass. The **Task-type taxonomy** categorizes by "which LLM function is affected": Direct Instruction Overriding (**DIO**), Indirect Instruction Overriding (**IIO**), Direct Tool Injection (**DTI**), Indirect Tool Injection (**ITI**), Direct Context Extraction (**DCE**), and AI Denial of Service (**DAIS**). These taxonomies support the 10 threat snapshots that cover **all vectors, high-level objectives, and task types**.

**3. Crowdsourced strong attacks + b3 benchmark: Powering the framework with human Red Teaming**

The authors emphasize that **strong attacks are the most critical part of security evaluation**. Current automated methods struggle to generate strong, context-aware attacks. Thus, they used **gamified crowdsourcing**: a Red Teaming game, *Gandalf Agent Breaker*, was built where users were randomly assigned one of 7 backbone LLMs. Each threat snapshot had 4 difficulty levels; a submission scoring >75/100 allowed progression. From 947 users and 13,920 sessions, **194,331 unique attacks** were collected, with 10,935 successful ones.

The benchmark was refined by **re-submitting** successful attacks to all 7 backbones and averaging scores. For each "level × threat snapshot," the top 7 scoring attacks were selected, totaling $7\times10\times3=210$ attacks—only 0.1% of the data, highlighting the scarcity of high-quality attacks. The strongest attacks were **withheld from the public set** to prevent overfitting by model providers. The public attacks have a cross-model mean of 0.18, while the withheld set reaches 0.56, demonstrating the necessity of strong attacks for evaluation. The final b3 benchmark consists of 10 agents × 3 defense levels ($L_1$ minimal prompt, $L_2$ stronger prompt + benign history, $L_3$ $L_1$ plus an LLM-as-judge defense).

**4. Vulnerability score: Compressing "vulnerability" into a rankable scalar**

To facilitate horizontal ranking, a vulnerability score is defined. For a backbone $m$, for each $(i,\ell)$ in subset $T$ and each attack $a\in A_i$, $a$ is injected and $m$ is run $N$ times to get outputs $s_k(a, TS_i^\ell)$. The score is a triple average:

$$V(m,T) := \frac{1}{|T|}\sum_{(i,\ell)\in T}\frac{1}{|A_i|}\sum_{a\in A_i}\frac{1}{N}\sum_{k=1}^{N} s_k(a, TS_i^\ell).$$

Lower scores indicate higher security. Using $N=5$ offsets non-determinism, and a **non-parametric bootstrap** provides 95% confidence intervals $[V^{\text{lower}}, V^{\text{upper}}]$.

## Key Experimental Results

### Main Results
Evaluation of **34 mainstream LLMs** on the b3 benchmark using 210 selected attacks and $N=5$. Partial overall vulnerability rankings (lower is safer):

| Model | Vulnerability Score (Lower is safer) | Notes |
|------|------|------|
| claude-haiku-4-5 (R) | 0.13 | Safest on the leaderboard |
| claude-sonnet-4-5 (R) | 0.15 | Second safest |
| claude-haiku-4-5 | 0.17 | Very safe even without reasoning |
| grok-4 (R) | 0.20 | |
| kimi-k2-thinking (R) | 0.34 | Strongest open-weight model |
| gpt-5.1 (No reasoning) | 0.36 | High capability but ranked 8th in safety |
| gpt-4.1 | 0.63 | Most vulnerable on the board |

(Note: Representative values based on Figure 2; ⚠️ refer to the original paper for exact figures.)

### Ablation Study
The paper validates the robustness of **rankings** rather than traditional modules:

| Design Choice | Impact on Final Ranking | Description |
|------|---------|------|
| Attack selection | Highest impact but ranking stays robust | Attack **quality** is the most critical factor |
| Threat snapshot aggregation | No impact | Rankings remain consistent across methods |
| Threat snapshot selection | Low impact | Existing 10 snapshots are sufficiently representative |
| Defense levels $L_1/L_2/L_3$ | Consistent for safest models | Defense type should not dictate backbone choice |

### Key Findings
- **Reasoning enhances security**: Enabling reasoning significantly reduced vulnerability scores for most models—contrary to Zou et al. (2025). However, for very small models (gemini-2.5-flash-lite), reasoning sometimes decreased security, suggesting a size threshold for effectiveness.
- **Size is unrelated to security**: Among comparable series (gpt-oss, llama4, gpt-5, claude-4, gemini-2.5), larger models showed **no significant safety advantage over smaller models** without reasoning; reasoning provided modest gains primarily in the "ultra-small to small" jump.
- **Closed-weight models are generally safer, with a caveat**: Top ranks are closed-weight, but these systems often include integrated guardrails, making this an "asymmetric comparison" of system-level vs. model-level safety. Notably, the open-weight kimi-k2-thinking (0.34) outperformed the flagship gpt-5.1 (no reasoning).
- **Capability correlates with safety, but with exceptions**: Stronger models are generally safer (better instruction following reduces confusion), but the most capable models (gpt-5.1, kimi-k2-thinking) only ranked 8th and 14th—capability $\neq$ security.
- **Safety profiles vary by task type**: Rankings are consistent across defense levels but vary significantly by task type, meaning backbones should be selected **based on specific use cases**.

## Highlights & Insights
- **"Statelessness $\Rightarrow$ Slicing" is the pivot**: Proving that LLM calls are stateless and contexts are self-contained allows for modeling only the "moment of vulnerability" rather than the whole execution flow. This observation decouples agent security from heavy overhead.
- **Distinguishing "insecure features" from "bugs"**: Recognizing that LLM instruction injection stems from the useful feature of "obeying natural language" defines the problem boundary as context-dependent and unpatchable.
- **Withholding strong attacks to prevent overfitting**: The gap between public (0.18) and withheld (0.56) scores quantifies the necessity of strong attacks for realistic evaluation.
- **Dual-perspective taxonomy**: This provides a ready-made threat modeling checklist for developers: list all vector × objective pairs, then build threat snapshots for compatible ones.

## Limitations & Future Work
- **Blind spots**: Any benchmark might miss novel exploits or new attack surfaces as agent architectures evolve; the modular threat snapshot framework allows for incremental expansion.
- **Closed vs. Open-Weight comparison**: Closed-weight models represent "system-level security," while open-weight models represent "raw model security." This asymmetry suggests the findings should be interpreted carefully.
- **Single-step focus**: While single-step acts as a lower bound, the combined harm of multi-turn/multi-agent attacks might be underestimated. Scaling evaluation to "threat snapshot chains" is left for future work.
- **Dependence on LLM-as-judge**: Biases in the scoring models can propagate into the vulnerability scores.

## Related Work & Insights
- **vs. AgentDojo / InjecAgent**: These require full execution flow mocks and focus on limited vectors. Ours uses threat snapshots for single calls with a dual-taxonomy covering all vectors, making it lighter and more comprehensive.
- **vs. Safety works like HarmBench**: Those focus on general safety (toxicity); this paper strictly differentiates and targets **security** (exploitability).
- **vs. Zou et al. (2025) on reasoning**: This work empirically finds that reasoning generally improves security, providing large-scale evidence for this debate.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The decoupling of agent security via threat snapshots is a fresh and effective perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 190k attacks, 34 models, triple defense levels, and robust bootstrap analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear formal definitions and actionable conclusions; dense but well-structured.
- Value: ⭐⭐⭐⭐⭐ Provides an open-source benchmark for selecting secure backbones and encourages treating security as a primary evaluation metric.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Optimizing Agent Planning for Security and Autonomy](optimizing_agent_planning_for_security_and_autonomy.md)
- [\[ICLR 2026\] A2ASecBench: A Protocol-Aware Security Benchmark for Agent-to-Agent Multi-Agent Systems](a2asecbench_a_protocol-aware_security_benchmark_for_agent-to-agent_multi-agent_s.md)
- [\[ICLR 2026\] RedCodeAgent: Automatic Red-teaming Agent against Diverse Code Agents](redcodeagent_automatic_red-teaming_agent_against_diverse_code_agents.md)
- [\[ACL 2026\] A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?](../../ACL2026/llm_safety/a_survey_on_the_safety_and_security_threats_of_computer-using_agents_jarvis_or_u.md)
- [\[ICLR 2026\] From Static Benchmarks to Dynamic Protocol: Agent-Centric Text Anomaly Detection for Evaluating LLM Reasoning](from_static_benchmarks_to_dynamic_protocol_agent-centric_text_anomaly_detection_.md)

</div>

<!-- RELATED:END -->
