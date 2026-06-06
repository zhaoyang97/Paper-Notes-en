---
title: >-
  [Paper Note] Multi-Agent Teams Hold Experts Back: Why Self-Organizing LLM Teams Cannot Retain "Experts"
description: >-
  [ICML 2026][LLM/NLP][Multi-agent] This paper systematically evaluates self-organizing heterogeneous LLM teams using the organizational psychology "strong synergy" standard (team $\ge$ strongest individual). It finds that…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Multi-agent"
  - "self-organizing teams"
  - "expert utilization"
  - "consensus bias"
  - "alignment side effects"
date: 2026-05-08
content_hash: 0086fd43aa975083
---

# Multi-Agent Teams Hold Experts Back: Why Self-Organizing LLM Teams Cannot Retain "Experts"

**Conference**: ICML 2026  
**arXiv**: [2602.01011](https://arxiv.org/abs/2602.01011)  
**Code**: https://github.com/apappu97/multi-agent-teams-hold-experts-back  
**Area**: Multi-Agent / LLM Collaboration / Organizational Psychology  
**Keywords**: Multi-agent, self-organizing teams, expert utilization, consensus bias, alignment side effects

## TL;DR
This paper systematically evaluates self-organizing heterogeneous LLM teams using the organizational psychology "strong synergy" standard (team $\ge$ strongest individual). It finds that even when experts are explicitly identified, teams underperform experts by 6.3%–41.1% on frontier ML benchmarks. The root cause is not an inability to recognize experts, but a refusal to let them lead—LLMs tend toward "middle-ground integration" rather than "epistemic deference." This consensus mechanism becomes more dilutive as team size increases, though it conversely makes teams exceptionally robust against adversarial members.

## Background & Motivation

**Background**: Multi-agent LLM systems are being deployed for open-ended tasks such as coding, research, and decision support. Existing work primarily relies on external coordination mechanisms like pre-designed roles (MetaGPT), fixed communication graphs (GPTSwarm/AFlow), or learned routing (AgentNet) to ensure team performance, effectively hardcoding solutions into the protocol.

**Limitations of Prior Work**: In real-world scenarios, coordination often **cannot be pre-specified** and must emerge through interaction. When different frontier models possess true "differentiated advantages"—e.g., GPT-5 excelling at MMLU and Claude at GPQA—can a role-less, protocol-free LLM team identify and utilize these differences like human teams? Prior work has either used identical model copies (pseudo-heterogeneity) or pre-set roles (externally imposed), never simultaneously addressing "self-organization + true heterogeneity + strong synergy."

**Key Challenge**: Organizational psychology suggests that human teams achieve strong synergy when expert identities are revealed; however, LLMs are trained via RLHF to be "agreeable and consensus-seeking." This creates a natural conflict with "deferring judgment to experts." There is a fundamental tension between needing teams to be robust against noise and needing them to let experts make the final call—these two traits may stem from the same mechanism.

**Goal**: To answer three questions in a role-less, self-organizing setting—RQ1: Can teams achieve strong synergy (match the strongest individual)? RQ2: If not, is the bottleneck "identifying the expert" or "adopting the expert's view"? RQ3: Which team dynamic factors correlate with failure?

**Key Insight**: Pairing classic organizational psychology experiments (NASA Moon Survival, Lost at Sea, Student Body President) with modern ML benchmarks (MMLU Pro/GPQA/SimpleQA/HLE/MATH-500) provides a controllable yet realistic evaluation. By using GEPA to optimize "reveal expert" prompts, the "prompt engineering" confounder is removed. Team failure is then decomposed into an "identification gap" and a "leveraging gap."

**Core Idea**: Utilize "Reveal Expert" ablations to maximize identification capabilities and measure the remaining performance gap. Perform conversation-level coding (Epistemic Deference ED / Integrative Compromise IC / Strategic Persistence SP / Epistemic Flexibility EF) to correlate specific mechanisms with the strong synergy gap. Finally, use adversarial experiments to test whether consensus tendencies produce robustness.

## Method

### Overall Architecture

The experimental framework spans two axes: "expert distribution" and "information conditions." Expert distribution includes "Centralized" (one individual holds ground-truth info) and "Distributed" (information is split among members). Information conditions include No Information (control), Expert Not Mentioned (expert exists but identity is hidden), Reveal Expert (explicit identification), and Best Individual (expert working alone). All experiments default to 4 agents $\times$ 4 discussion rounds, with the final answer derived via majority vote after discussion. Results are standardized across tasks using a "relative synergy gap" formula: $(\max_t f(\{a_t\}) - f(\{a_1,\dots,a_T\})) / \max_t f(\{a_t\})$.

### Key Designs

1.  **Decomposition of Synergy Gap into "Identification / Leveraging"**:
    - **Function**: Decomposes why teams underperform the strongest member into two sub-problems—failing to recognize the expert vs. failing to utilize them once recognized.
    - **Mechanism**: Defines Identification Gap = $f(\text{Expert Not Mentioned}) - f(\text{Reveal Expert})$; Leveraging Gap = $f(\text{Reveal Expert}) - f(\text{Best Individual})$. If the team fails to improve significantly after the expert is revealed, the bottleneck lies in leveraging. GEPA is used to optimize Reveal Expert prompts to isolate prompt engineering issues.
    - **Design Motivation**: Previous multi-agent work only compared "team vs. average" (weak synergy), failing to capture the difference in expert utilization between LLM and human teams. 

2.  **Coding of Four Conversation Dynamics + Correlation Analysis**:
    - **Function**: Explains "why" teams fail at a behavioral rather than just a result level.
    - **Mechanism**: Following philosophical preemption thesis and negotiation theory, turns are coded with four labels: non-experts' "Epistemic Deference ED" and "Integrative Compromise IC"; experts' "Strategic Persistence SP" and "Epistemic Flexibility EF." Automation via Gemini 3.0 Pro achieved 94% agreement with human verification. Pearson correlations were calculated between these behaviors and the synergy gap.
    - **Design Motivation**: This provides the first quantitative evidence that "over-compromise" is an endogenous mechanism of failure in LLM teams.

3.  **Dual Ablation of Team Size $\times$ Adversarial Members**:
    - **Function**: Tests whether "consensus tendency" simultaneously dilutes experts and filters out adversaries.
    - **Mechanism**: With 4 discussion rounds, team sizes were set to 2/4/8 to observe if the synergy gap increases with scale ($p < 0.05$). An "adversarial" member (instructed to sabotage with worst-case rankings) was also introduced to test team robustness.
    - **Design Motivation**: Placing "failure" and "robustness" in the same experimental set demonstrates that robustness is a byproduct of the same mechanism causing expert dilution.

### Evaluation Strategy

For human psychology tasks, L1 ranking error was used. For ML benchmarks, the "At Least One Correct" (ALOC) upper bound was employed: the accuracy achievable if the team deferred to the correct member on every question. The gap relative to ALOC serves as a hard metric for potential improvement through perfect expert utilization.

## Key Experimental Results

### Main Results: Relative Synergy Gap on Frontier ML Benchmarks

| Benchmark | CoT+MV | Debate (Reveal) | Opt-Out (Reveal) | Team (No Mention) | **Team (Reveal)** | Best Ind. | ALOC | Relative Synergy Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MMLU Pro | 83.0% | 86.0% | 88.0% | 86.0% | **86.0%** | 86.5% | 91.8% | 6.3% |
| GPQA Diamond | 73.0% | 83.0% | 81.0% | 76.0% | **83.0%** | 78.0% | 88.8% | 14.4% |
| SimpleQA | 44.0% | 53.0% | 56.0% | 51.0% | **60.0%** | 52.0% | 62.3% | 18.1% |
| HLE Text-Only | 14.0% | 23.0% | 31.0% | 28.0% | **36.0%** | 29.0% | 47.5% | 41.1% |
| MATH-500 | 61.0% | 75.0% | 73.0% | 63.0% | **75.0%** | 73.5% | 79.0% | 20.3% |

Note: All Reveal Expert prompts were optimized via GEPA. None of the coordination protocols reached the ALOC upper bound.

### Ablation Study: Relative Synergy Gap on Human Psychology Tasks

| Task | Centralized-Not Mentioned | Centralized-Reveal | Distributed-Not Mentioned | Distributed-Reveal |
| :--- | :---: | :---: | :---: | :---: |
| NASA Moon Survival | 78.7% | 81.8% | 113.4% | 110.1% |
| Lost at Sea | 55.6% | 58.6% | 50.1% | 42.1% |
| Student Body President | 98.7% | 73.5% | 66.0% | 17.3% |

The gap remains largely unchanged after Reveal for most tasks, indicating the bottleneck is leveraging, not identification.

### Key Findings

- **41.1% Synergy Gap on HLE Text-Only**: Even with the strongest prompt identifying the expert, teams fall short of the 47.5% ALOC potential, reaching only 36%. Expert utilization failure can eliminate 1/4 of a model's effective capability.
- **Conversation-Level Causality**: Integrative Compromise (IC) is strongly positively correlated with the synergy gap (NASA $r=0.55$; SBP $r=0.69$, $p<0.001$), while Epistemic Deference (ED) is negatively correlated (NASA $r=-0.44$; SBP $r=-0.68$). More "compromise" leads to worse performance compared to the expert.
- **Scale Increases Dilution**: For all tasks, the synergy gap increased significantly as team size grew from 2 to 8 ($p<0.05$), even under Reveal Expert conditions.
- **Robustness as a Byproduct**: Inserting a saboteur providing the worst answers barely affected team performance—the same mechanism that dilutes experts also filters adversarial signals.

## Highlights & Insights

- **Clear Distinction Between Strong and Weak Synergy**: Adopting the "team $\ge$ best member" standard from organizational psychology exposes real gaps that "team > single agent" (weak synergy) obscures.
- **"Identification $\neq$ Utilization"**: This is a critical insight. While prior work assumed prompt engineering was the issue, this paper proves the gap remains after optimization, attributing the failure to "alignment-induced refusal to defer."
- **Common Roots of Robustness and Failure**: Explaining "over-consensus" and "adversarial resistance" through the same mechanism presents a genuine dilemma for future alignment paradigms: can we maintain robustness while learning context-dependent deference?
- **Transferable Evaluation Framework**: The open-sourced teamwork harness provides a standardized "thermometer" for alignment failure across different team compositions.

## Limitations & Future Work

- **Correlational Attribution to Alignment**: The assumption that consensus tendency stems from RLHF is currently a hypothesis; comparisons between base and aligned models are needed for causal proof.
- **Narrow Task Set**: 5 ML benchmarks and 3 psychology tasks do not yet cover long-horizon planning, tool usage, or coding collaboration.
- **Lack of Structured Protocol Design**: This study focused on the extreme of self-organization; the "sweet spot" between self-organization and pre-set roles remains unexplored.
- **Authenticity of Expert Identity**: In ML benchmarks, "experts" are defined post-hoc by the strongest model per question. Identifying such conditional experts in real-time is a challenge itself, requiring cautious interpretation.

## Related Work & Insights

- **vs. MetaGPT / Virtual Lab / SiriuS**: These rely on pre-set roles. Ours shows that without roles, LLM performance drops significantly, suggesting that currently successful SOTA multi-agent systems owe more to their frameworks than to the models' innate "collaborative ability."
- **vs. Mixture-of-Agents / GPTSwarm / AgentNet**: These are non-negotiatory aggregations. Ours focuses on "negotiatory" collaboration, linking conversation dynamics to the causal chain.
- **vs. Davidson et al. 2025 "Collaboration Gap"**: Independent concurrent work reaching similar conclusions; Ours further identifies the failure mechanism via "integration vs. deference" behaviors.
- **Insight**: A structural conflict exists between alignment goals (helpful, agreeable) and decision effectiveness (defer to expert). Future training must introduce "contextual epistemic authority," teaching models when to step back.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixing of Experts](../../AAAI2026/llm_nlp/promptmoe_generalizable_zero-shot_anomaly_detection_via_visually-guided_prompt_m.md)
- [\[ICML 2026\] Why Are Linear RNNs More Parallelizable?](why_are_linear_rnns_more_parallelizable.md)
- [\[AAAI 2026\] Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](../../AAAI2026/llm_nlp/scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)
- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](../../NeurIPS2025/llm_nlp/large_language_models_miss_the_multi-agent_mark.md)
- [\[NeurIPS 2025\] SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assemblies](../../NeurIPS2025/llm_nlp/symphony_synergistic_multi-agent_planning_with_heterogeneous_language_model_asse.md)

</div>

<!-- RELATED:END -->
