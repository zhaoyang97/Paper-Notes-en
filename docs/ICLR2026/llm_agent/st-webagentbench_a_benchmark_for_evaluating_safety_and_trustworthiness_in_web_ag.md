---
title: >-
  [Paper Note] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents
description: >-
  [ICLR 2026][LLM Agent][Web Agent] This paper introduces ST-WebAgentBench, the first benchmark specifically designed to evaluate the safety and trustworthiness of web agents. Through a policy hierarchy framework and the Completion under Policy (CuP) metric, it reveals that current SOTA agents exhibit serious policy violations in enterprise settings.
tags:
  - ICLR 2026
  - LLM Agent
  - Web Agent
  - Safety
  - Trustworthiness
  - benchmark
  - Policy Compliance
date: 2026-05-08
content_hash: 7afbe03a36297472
---

# ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents

**Conference**: ICLR 2026
**arXiv**: [2410.06703](https://arxiv.org/abs/2410.06703)
**Code**: [https://sites.google.com/view/st-webagentbench/home](https://sites.google.com/view/st-webagentbench/home)
**Area**: LLM Agent
**Keywords**: Web Agent, Safety, Trustworthiness, benchmark, Policy Compliance

## TL;DR

This paper introduces ST-WebAgentBench, the first benchmark specifically designed to evaluate the safety and trustworthiness of web agents. Through a policy hierarchy framework and the Completion under Policy (CuP) metric, it reveals that current SOTA agents exhibit serious policy violations in enterprise settings.

## Background & Motivation

LLM-based web agents have advanced rapidly in recent years, with frameworks such as AutoGPT, LangGraph, and AutoGen giving rise to a large number of autonomous web agents. However, existing benchmarks—including WebArena, WorkArena, and Mind2Web—**focus exclusively on task completion rate**, entirely neglecting safety, policy compliance, and trustworthiness, which are critical factors for enterprise deployment.

Specific problems include:

**Neglected safety risks**: Agents may inadvertently delete user accounts, perform unintended actions, or leak sensitive data.

**Hallucination behavior**: Agents may populate fictitious information (e.g., fabricated email addresses) during task execution and still receive full task completion scores.

**Absence of policy compliance evaluation**: Enterprise environments require agents to strictly adhere to a hierarchical set of organizational policies, user preferences, and task instructions.

**Lack of human-in-the-loop support**: Existing benchmarks do not support agents proactively seeking human confirmation under uncertainty.

These issues constitute a significant barrier to large-scale deployment of web agents in real-world enterprise environments.

## Method

### Overall Architecture

ST-WebAgentBench is built on the BrowserGym environment, integrating WebArena and SuiteCRM application environments. It comprises 235 policy-augmented tasks spanning multiple safety categories.

### Key Designs

1. **Policy Hierarchy for Safe and Trustworthy Behavior**

   - **Organizational Policy $P_{org}$** (highest priority): e.g., "Never delete any record from the system."
   - **User Preferences $P_{user}$** (medium priority): e.g., "Always ask for my permission before submitting a new form."
   - **Task Instructions $P_{task}$** (lowest priority): execution instructions for a specific task.
   - Agent behavior must satisfy: $\pi_H(S_t) = \arg\max_{a_t \in A(S_t)} [R_{task}(S_t, a_t)]$ subject to $a_t \in H_t$

2. **Safety and Trustworthiness Dimensions (10 evaluation dimensions)**

   - User Consent and Action Confirmation
   - Boundary and Scope Restriction
   - Strict Task Execution
   - Policy Adherence
   - Robustness Against Jailbreaking
   - Security of Sensitive Data
   - Error Handling and Safety Nets
   - Legal and Ethical Compliance
   - Transparency and Explainability
   - Observation Integrity and Manipulation Defense
   - Reflection and Task Verification

3. **CuP Metric (Completion under Policy)**

   - A policy violation matrix $V$ is defined, where $V_{source,category}$ denotes the number of violations for a given source and category.
   - Metric formulation: $CuP = C_{task} \cdot \mathbb{1}\{V_{total} = 0\}$
   - Task completion is credited **only when there are zero policy violations**, making this metric strictly more demanding than the raw task completion rate.

4. **Risk Ratio Assessment**

   - $\text{Risk Ratio}_{source,category} = \frac{\sum_i V_{source,category}(i)}{\#Policies_{source}}$
   - Three-tier risk classification: low risk (≤5%), medium risk (5–15%), high risk (>15%).

### Benchmark Implementation

- **Task distribution**: Core benchmark (indices 0–84) + cognitive load tests (indices 85–234).
- **Evaluation functions**: `element_action_match`, `is_sequence_match`, `is_url_match`, `is_ask_the_user`, `is_action_count`, `is_program_html`.
- **Human-in-the-loop support**: BrowserGym's observation space is extended to include the policy hierarchy, enabling asynchronous agent integration.

## Key Experimental Results

### Main Results

| Agent | Completion Rate | CuP | Partial Completion | Partial CuP | Consent Violations | Strict Execution Violations |
|---|---|---|---|---|---|---|
| AWM | 0.238 | 0.238 | 0.369 | 0.238 | 37.0 (High Risk) | 24.0 (High Risk) |
| WebVoyager | 0.128 | 0.113 | 0.169 | 0.155 | 12.0 (High Risk) | 21.0 (High Risk) |
| WorkArena Legacy | 0.129 | 0.114 | 0.171 | 0.157 | 4.0 (Medium Risk) | 16.0 (Medium Risk) |

### Cognitive Load Experiment

| Difficulty | Policies/Task | AWM Performance |
|---|---|---|
| Easy | 3 | 14.8 |
| Medium | 10 | — |
| Hard | 17 | 11.5 |

### Key Findings

1. **CuP is substantially lower than the nominal completion rate**: AWM's CuP (0.238) is significantly lower than its partial completion rate (0.369), exposing critical safety gaps.
2. **Consent dimension incurs the most violations**: AWM accumulates 37 consent violations, with a risk ratio as high as 0.44.
3. **Cognitive load has a pronounced effect**: As the number of policies increases from 3 to 17, agent performance drops from 14.8 to 11.5.
4. **Hallucination is pervasive**: Agents execute additional steps beyond the task instructions, such as mistakenly creating repositories or filling in fabricated information.
5. **Boundary dimension has limited impact**: Likely because agents fail before boundary checks are triggered.

## Highlights & Insights

- **Elegant design of the CuP metric**: The zero-tolerance policy violation condition ($\mathbb{1}\{V_{total}=0\}$) accurately reflects the real requirements of enterprise environments.
- **Policy-aware architecture proposal**: The paper proposes multi-agent architectural principles incorporating a Policy Agent and an Interceptor Pattern.
- **Enterprise perspective**: Unlike academic benchmarks that solely pursue task completion, this work re-examines agent evaluation from the standpoint of enterprise safety and compliance.
- **BrowserGym integration**: The benchmark is open-sourced with a plan to contribute the extensions back to the BrowserGym ecosystem.

## Limitations & Future Work

1. The dataset is relatively small (235 tasks) and the distribution across policy categories is uneven.
2. Task design for the boundary dimension requires improvement, as it currently has limited impact on agent performance.
3. Manually annotating policy ground truth is costly; automated approaches warrant further exploration.
4. Only three agents are evaluated; broader evaluation across more agents is needed.
5. In-depth testing of advanced safety dimensions such as jailbreak attacks and sensitive data leakage is lacking.

## Related Work & Insights

- **WebArena/WorkArena series**: Provides foundational infrastructure for online interactive benchmarks.
- **GuardAgent**: An agent framework that enforces safety measures via knowledge-based reasoning.
- **R-Judge**: A benchmark for evaluating agent capability in handling safety-critical tasks.
- The key contribution of this work to the agent safety research community lies in establishing a paradigm shift in evaluation: from "can the task be completed?" to "is the task completed safely?"

## Rating

- Novelty: ⭐⭐⭐⭐ (First benchmark for safety and trustworthiness evaluation, though the methodology essentially amounts to adding policy constraints)
- Experimental Thoroughness: ⭐⭐⭐ (Only 3 agents evaluated; dataset is relatively small)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; problem definition is precise)
- Value: ⭐⭐⭐⭐⭐ (Fills the gap in agent safety evaluation; offers important guidance for enterprise-domain agent deployment)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [\[ICLR 2026\] LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News](livenewsbench_evaluating_llm_web_search_capabilities_with_fresh_news.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[CVPR 2026\] Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos](../../CVPR2026/llm_agent/ego2web_a_web_agent_benchmark_grounded_in_egocentric_videos.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)

<!-- RELATED:END -->
