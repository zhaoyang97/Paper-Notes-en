---
title: >-
  [Paper Note] Multi-Agent Teams Hold Experts Back: Why Self-Organized LLM Teams Fail to Retain "Experts"
description: >-
  [ICML 2026][LLM (Other)][Multi-agent] This paper systematically evaluates self-organized heterogeneous LLM teams using the organizational psychology standard of "strong synergy" (team $\ge$ strongest individual). It finds that even when explicitly informed of expert identities, teams underperform experts by 6.3%–41.1% on frontier ML benchmarks. The root cause is not the inability to recognize experts, but a reluctance to let them lead—LLMs favor "middle-ground integration" ov…
tags:
  - "ICML 2026"
  - "LLM (Other)"
  - "Multi-agent"
  - "self-organized teams"
  - "expert utilization"
  - "consensus bias"
  - "alignment side-effects"
date: 2026-05-08
content_hash: 8f4865c67ce82ec0
---

# Multi-Agent Teams Hold Experts Back: Why Self-Organized LLM Teams Fail to Retain "Experts"

**Conference**: ICML 2026  
**arXiv**: [2602.01011](https://arxiv.org/abs/2602.01011)  
**Code**: https://github.com/apappu97/multi-agent-teams-hold-experts-back  
**Area**: Multi-Agent / LLM Collaboration / Organizational Psychology  
**Keywords**: Multi-agent, self-organized teams, expert utilization, consensus bias, alignment side-effects

## TL;DR
This paper systematically evaluates self-organized heterogeneous LLM teams using the organizational psychology standard of "strong synergy" (team $\ge$ strongest individual). It finds that even when explicitly informed of expert identities, teams underperform experts by 6.3%–41.1% on frontier ML benchmarks. The root cause is not the inability to recognize experts, but a reluctance to let them lead—LLMs favor "middle-ground integration" over "epistemic deference." This consensus mechanism dilutes expertise as team size grows but, conversely, makes teams exceptionally robust against adversarial members.

## Background & Motivation

**Background**: Multi-agent LLM systems are being deployed for open-ended tasks such as coding, research, and decision support. Existing work primarily ensures team performance through external coordination mechanisms, such as pre-designed roles (MetaGPT), fixed communication graphs (GPTSwarm/AFlow), or learned routing (AgentNet), effectively hard-coding answers into protocols.

**Limitations of Prior Work**: In real-world scenarios, coordination often **cannot be pre-specified** and must emerge through interaction. When different frontier models possess true "differential advantages"—e.g., GPT-5 excelling at MMLU while Claude excels at GPQA—can a role-less, protocol-free LLM team identify and utilize these differences like human teams? Prior work has either used homogeneous model copies (pseudo-heterogeneity) or pre-assigned roles (external imposition), never combining "self-organization + true heterogeneity + strong synergy."

**Key Challenge**: Organizational psychology shows that human teams achieve strong synergy when expert identities are revealed. However, LLMs aligned via RLHF are trained to be "agreeable and consensus-seeking," which naturally conflicts with "deferring judgment to an identified expert." Systems need to be both robust against noise and capable of letting experts make the final call; these two requirements may be two sides of the same mechanism.

**Goal**: To answer three questions in a role-less, self-organized setting—RQ1: Can teams achieve strong synergy (match the strongest individual)? RQ2: If not, is the bottleneck "identifying the expert" or "adopting the expert's view"? RQ3: Which team dynamics correlate with failure?

**Key Insight**: Pair classic organizational psychology experiments (NASA Moon Survival, Lost at Sea, Student Body President) with modern ML benchmarks (MMLU Pro/GPQA/SimpleQA/HLE/MATH-500). Use GEPA to optimize "Reveal Expert" prompts to eliminate "poor prompting" as a confounding factor, then decompose team failure into an "identification gap" and a "leveraging gap."

**Core Idea**: Use "Reveal Expert" ablations to maximize identification capability and measure the remaining performance gap. Perform dialogue labeling (Epistemic Deference ED / Integrative Compromise IC / Strategic Persistence SP / Epistemic Flexibility EF) to analyze the correlation between dialogue-level mechanisms and the strong synergy gap. Finally, use adversarial experiments to test if the consensus bias provides robustness.

## Method

### Overall Architecture

This is an evaluative study designed to determine if self-organized LLM teams without pre-set roles or processes can utilize true capability differences among members. The experimental design revolves around two axes: "Expert Distribution $\times$ Information Condition." Expert distribution includes "Centralized" (one member holds ground-truth information) and "Distributed" (information is split mutually exclusively). Information conditions include No Information (control), Expert Not Mentioned (expert exists but identity is hidden), Reveal Expert (explicitly stated expert), and Best Individual (expert acting alone). All experiments default to 4 agents $\times$ 4 rounds of discussion with a final majority vote. Results are standardized using a relative synergy gap formula: $(\max_t f(\{a_t\}) - f(\{a_1,\dots,a_T\})) / \max_t f(\{a_t\})$.

### Key Designs

**1. Decomposing the synergy gap into "Identification" and "Leveraging": Locating the bottleneck**

Prior multi-agent work often compared "team vs. average member" (weak synergy), failing to detect if teams underperformed the strongest member or why. This paper explicitly splits the gap: Identification Gap = $f(\text{Expert Not Mentioned}) - f(\text{Reveal Expert})$, measuring the improvement gained from naming the expert; Leveraging Gap = $f(\text{Reveal Expert}) - f(\text{Best Individual})$, measuring how far the team remains from the expert's solo performance after the expert is named. If teams fail to improve after disclosure yet lag behind the solo expert, the bottleneck is leveraging, not identification. To prevent "poor prompting" from polluting conclusions, "Reveal Expert" prompts were automatically optimized using GEPA.

**2. Labeling four categories of dialogue behavior for frequency-gap correlation analysis: Explaining failure via mechanism**

To explain "why" teams lose, each round of speech is coded based on the preemption thesis from the philosophy of authority and negotiation theory. Non-expert speeches are categorized into "Epistemic Deference (ED)" (accepting the expert's view) and "Integrative Compromise (IC)" (proposing a middle ground). Expert speeches include "Strategic Persistence (SP)" (standing firm) and "Epistemic Flexibility (EF)" (catering to non-experts). Labeling was automated via Gemini 3.0 Pro and verified by two humans on 50 dialogues with 94% agreement. Pearson correlations between behavior frequency and synergy gaps quantitatively identify "excessive compromise" as an endogenous failure mechanism.

**3. Team Size $\times$ Adversarial Member Ablation: Verifying that consensus bias dilutes experts and filters adversaries simultaneously**

If failure stems from a consensus bias toward the mean, it should have two observable consequences. One experiment fixed discussion to 4 rounds while varying team size (2/4/8) to see if the synergy gap increased—the result was a significant positive correlation ($p < 0.05$) across all tasks, even under the Reveal Expert condition; more members further diluted the expert's voice. Another experiment introduced an adversarial member instructed to "rank the worst and disrupt the team." The team remained nearly unaffected. These phenomena are unified under one mechanism: consensus averaging dilutes both experts and adversaries.

### Evaluation Metrics

Two sets of metrics are used across tasks. Human psychology tasks use L1 ranking error to measure the distance between team rankings and the ground truth. ML benchmarks introduce the At Least One Correct (ALOC) upper bound—the accuracy achievable if the team deferred to whichever member answered correctly for each question. The gap between team performance and ALOC serves as a "hard ceiling" metric for potential expert utilization.

## Key Experimental Results

### Main Results: Relative Synergy Gap on Frontier ML Benchmarks

| Benchmark | CoT+MV | Debate (Reveal) | Opt-Out (Reveal) | Team (No Mention) | **Team (Reveal)** | Best Ind. | ALOC | Relative Synergy Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MMLU Pro | 83.0% | 86.0% | 88.0% | 86.0% | **86.0%** | 86.5% | 91.8% | 6.3% |
| GPQA Diamond | 73.0% | 83.0% | 81.0% | 76.0% | **83.0%** | 78.0% | 88.8% | 14.4% |
| SimpleQA | 44.0% | 53.0% | 56.0% | 51.0% | **60.0%** | 52.0% | 62.3% | 18.1% |
| HLE Text-Only | 14.0% | 23.0% | 31.0% | 28.0% | **36.0%** | 29.0% | 47.5% | 41.1% |
| MATH-500 | 61.0% | 75.0% | 73.0% | 63.0% | **75.0%** | 73.5% | 79.0% | 20.3% |

Note: All Reveal Expert prompts were optimized via GEPA; no coordination protocol (CoT+MV, Debate, Opt-Out, Team Discussion) reached the ALOC bound.

### Ablation Study: Relative Synergy Gap on Human Psychology Tasks

| Task | Central-Not Mentioned | Central-Reveal | Distributed-Not Mentioned | Distributed-Reveal |
| :--- | :--- | :--- | :--- | :--- |
| NASA Moon Survival | 78.7% | 81.8% | 113.4% | 110.1% |
| Lost at Sea | 55.6% | 58.6% | 50.1% | 42.1% |
| Student Body President | 98.7% | 73.5% | 66.0% | 17.3% |

For most tasks, the gap remained nearly unchanged after revelation, indicating the bottleneck lies in leveraging rather than identification.

### Key Findings

- **41.1% Synergy Gap on HLE Text-Only**: Even with optimal prompts revealing the expert, the team only achieved 36% accuracy despite a potential 47.5% ALOC accuracy—failure to utilize experts can consume 1/4 of a model's effective capability.
- **Dialogue-level Causality**: Integrative Compromise (IC) strongly correlates with synergy gaps (NASA $r=0.55$, $p<0.001$; SBP $r=0.69$, $p<0.001$), while Epistemic Deference (ED) correlates negatively (NASA $r=-0.44$; SBP $r=-0.68$). More compromise leads to worse performance relative to the expert.
- **Scale Dilution**: As team size increased from 2 to 8, the synergy gap rose significantly ($p<0.05$) across all tasks, even with expert revelation.
- **Adversarial Robustness Byproduct**: Introducing a disruptor who intentionally provides the worst answers barely degraded team performance—the same mechanism that dilutes experts also filters out adversarial signals.

## Highlights & Insights

- **Distinction between Strong and Weak Synergy**: Using the "team $\ge$ best member" standard from organizational psychology exposes real performance gaps that "team $>$ single agent" metrics hide.
- **Identification $\neq$ Utilization**: This is a key insight; while previous research blamed prompt engineering, Ours shows the gap persists after optimization, attributing the failure to "alignment-induced reluctance to defer."
- **Common Roots of Robustness and Failure**: Explaining "excessive consensus" and "adversarial robustness" as the same alignment side-effect highlights a dilemma for future training: how to maintain robustness while learning situational deference?
- **Transferable Evaluation Framework**: The open-sourced teamwork harness allows any team combination to measure the "strong synergy gap," providing a standardized "thermometer" for alignment failure in multi-agent communities.

## Limitations & Future Work

- **Correlational Alignment Attribution**: The claim that "consensus bias stems from RLHF" is a hypothesis. Base vs. aligned models were not compared; future work needs to ablate the training stages.
- **Narrow Task Set**: 5 ML benchmarks and 3 psychology tasks do not yet cover real-world multi-agent scenarios like code collaboration, long-term planning, or tool invocation.
- **Lack of Structured Protocol Design**: This study only tested the "self-organized" extreme. It does not provide a solution for the sweet spot between self-organization and pre-set roles.
- **Authenticity of Expert Identity**: In ML benchmarks, "experts" are defined post-hoc as the best model for a specific question, but real-time identification of situational experts remains a challenge.

## Related Work & Insights

- **vs. MetaGPT / Virtual Lab / SiriuS**: These use pre-set roles to ensure collaboration; Ours proves that without roles, the same LLMs perform significantly worse, suggesting the success of SOTA multi-agent systems stems from frameworks rather than innate "collaborative ability."
- **vs. Mixture-of-Agents / GPTSwarm / AgentNet**: These focus on non-deliberative aggregation; Ours specifically studies "negotiated" collaboration, incorporating dialogue dynamics into the causal chain.
- **vs. Davidson et al. 2025 "Collaboration Gap"**: Independent concurrent work with similar conclusions—multi-agent does not necessarily outperform the strongest individual. Ours further locates the failure mechanism in dialogue behaviors (IC vs. ED).
- **Insight**: There is a structural conflict between alignment goals (helpful, agreeable) and decision effectiveness (defer to expert). Training objectives may need to introduce "context-aware epistemic authority."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to use strong synergy standards + "identification/leveraging" decomposition + dialogue coding to analyze why multi-agent collaboration fails.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual-track tasks, four information conditions, three scales, adversarial perturbations, and dialogue labeling with 94% human verification.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mapping between research questions, experimental design, and conclusions.
- Value: ⭐⭐⭐⭐⭐ A wake-up call for multi-agent system design: teams naturally underperform their best members in specialized scenarios due to alignment issues, which must be addressed at the training level rather than just the orchestration level.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Discovering Novel LLM Experts via Task-Capability Coevolution](../../ICLR2026/llm_nlp/discovering_novel_llm_experts_via_task-capability_coevolution.md)
- [\[ICML 2025\] Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective](../../ICML2025/llm_nlp/product_of_experts_with_llms_boosting_performance_on_arc_is_a_matter_of_perspect.md)
- [\[ICML 2026\] Why Are Linear RNNs More Parallelizable?](why_are_linear_rnns_more_parallelizable.md)
- [\[ACL 2025\] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](../../ACL2025/llm_nlp/agentdropout-dynamic-agent-elimination-for-multi-agent-collaboration.md)
- [\[ACL 2025\] Red-Teaming LLM Multi-Agent Systems via Communication Attacks](../../ACL2025/llm_nlp/red-teaming_llm_multi-agent_systems_via_communication_attacks.md)

</div>

<!-- RELATED:END -->
