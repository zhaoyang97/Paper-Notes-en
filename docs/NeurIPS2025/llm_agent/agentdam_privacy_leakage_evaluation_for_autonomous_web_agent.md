---
title: >-
  [Paper Note] AgentDAM: Privacy Leakage Evaluation for Autonomous Web Agents
description: >-
  [NeurIPS 2025][LLM Agent][AI agent privacy] This paper proposes AgentDAM, the first benchmark for end-to-end evaluation of data minimization compliance by AI agents in real web environments. It comprises 246 tasks spanning Reddit, GitLab, and Shopping platforms, and finds that leading models such as GPT-4o exhibit privacy leakage rates of 36–46% without mitigation, while a CoT-based privacy prompt reduces leakage rates to 6–8%.
tags:
  - NeurIPS 2025
  - LLM Agent
  - AI agent privacy
  - data minimization
  - web navigation
  - privacy benchmark
  - inference-time leakage
date: 2026-05-08
content_hash: c608f8a1885425f9
---

# AgentDAM: Privacy Leakage Evaluation for Autonomous Web Agents

**Conference**: NeurIPS 2025
**arXiv**: [2503.09780](https://arxiv.org/abs/2503.09780)
**Code**: [https://github.com/facebookresearch/ai-agent-privacy](https://github.com/facebookresearch/ai-agent-privacy)
**Area**: AI Security / LLM Agent
**Keywords**: AI agent privacy, data minimization, web navigation, privacy benchmark, inference-time leakage

## TL;DR
This paper proposes AgentDAM, the first benchmark for end-to-end evaluation of data minimization compliance by AI agents in real web environments. It comprises 246 tasks spanning Reddit, GitLab, and Shopping platforms, and finds that leading models such as GPT-4o exhibit privacy leakage rates of 36–46% without mitigation, while a CoT-based privacy prompt reduces leakage rates to 6–8%.

## Background & Motivation
**State of the Field**: Autonomous AI agents (e.g., web navigation agents) are advancing rapidly, capable of completing complex tasks such as paying bills and managing schedules on behalf of users. These tasks inevitably require access to sensitive user information (e.g., credit card numbers, email contents).

**Limitations of Prior Work**: Existing privacy evaluation methods are predominantly **probing-based** — directly asking an LLM whether sharing a piece of information is appropriate in a given scenario. This only tests the LLM's privacy reasoning capability and does not reflect agent behavior during actual multi-step web task execution.

**Root Cause**: Agents require access to sensitive user data to complete tasks, yet must also adhere to the principle of data minimization — using only information strictly necessary for task completion without leaking irrelevant sensitive data. Existing evaluations are either not end-to-end (probing only) or conducted in simulated environments (lacking realism).

**Paper Goals**: How can one evaluate end-to-end whether AI agents comply with the data minimization principle in real web environments?

**Starting Point**: Fully controllable real web environments (Reddit/GitLab/Shopping) are constructed based on WebArena/VisualWebArena. Tasks are designed to contain both relevant and irrelevant sensitive information, and LLM-as-a-judge is used to automatically detect privacy leakage in agent trajectories.

**Core Idea**: Evaluate agents' actual data usage behavior during real web interactions, rather than merely testing their privacy-related reasoning and judgment.

## Method

### Overall Architecture
AgentDAM consists of three components: (1) **Task Design**: 246 tasks, each comprising a `user_instruction` (the task to be completed), `user_data` (synthetic data containing both relevant and irrelevant sensitive information), and annotated `sensitive_data` (sensitive fields that should not be used); (2) **End-to-End Evaluation**: agents execute tasks on real web servers, with every action step logged and analyzed; (3) **Dual-Axis Scoring**: Utility (task completion rate) and Privacy (privacy non-leakage rate).

### Key Designs

1. **Sensitive Data Taxonomy**:

    - Function: Defines 6 major categories of sensitive information: personal/contact information, religious/cultural/political identity, employer/employment data, financial information, educational history, and medical data.
    - Mechanism: Human annotators first create Data Seeds (comprising a plot and `sensitive_data`), which are then used to prompt an LLM to generate complete `user_data` (e.g., chat logs), ensuring that sensitive information is naturally embedded but task-irrelevant.
    - Design Motivation: Sensitive information must appear naturally in context yet be unrelated to the current task, simulating real-world scenarios in which agents may incidentally use superfluous information.

2. **LLM-as-a-Judge Privacy Evaluator**:

    - Function: Automatically detects whether each agent action step contains leakage of annotated `sensitive_data`.
    - Mechanism: GPT-4o performs CoT reasoning to determine whether agent output text contains the annotated sensitive information, considering not only exact matches but also paraphrasing and anonymization.
    - Design Motivation: Privacy leakage cannot be assessed by string matching alone — agents may reword information while still exposing its core content. Human evaluation and LLM judge agreement reaches 98%.

3. **Mitigation Strategies**:

    - Pre-filtering (LLM-based filtering of `user_data` before execution) and post-filtering (checking final outputs) yield limited effectiveness.
    - **CoT privacy system prompt** proves most effective: the system prompt informs the agent of the data minimization principle and provides few-shot reasoning examples.

## Key Experimental Results

### Main Results (Privacy and Utility Evaluation)

| Model | Utility ↑ | Privacy (no mitigation) ↑ | Privacy (CoT prompt) ↑ | Utility (CoT prompt) |
|------|-----------|-------------------|-------------------|-----------------|
| gpt-4o | 0.455 | 0.638 | **0.915** | 0.415 |
| gpt-4o-mini | 0.341 | 0.541 | 0.821 | 0.321 |
| gpt-4-turbo | 0.480 | 0.748 | 0.894 | 0.423 |
| llama-3.2-90b | 0.333 | 0.882 | 0.923 | 0.297 |
| llama-3.3-70b | 0.402 | 0.882 | **0.939** | 0.386 |
| claude-cua | 0.350 | 0.902 | 0.935 | 0.309 |

### Probing vs. End-to-End Evaluation Comparison

| Model | AgentDAM Privacy | Probing Privacy | Gap |
|------|-----------------|-----------------|------|
| gpt-4o | 0.646 | 0.915 | Probing overestimates by 27% |
| gpt-4o-mini | 0.557 | 0.890 | Probing overestimates by 33% |
| gpt-4-turbo | 0.732 | 0.846 | Probing overestimates by 11% |
| llama-3.2-90b | 0.882 | 0.748 | Probing underestimates by 13% |
| llama-3.3-70b | 0.882 | 0.817 | Probing underestimates by 7% |

### Key Findings
- **Probing-based methods severely overestimate the privacy capability of GPT-series models**: GPT-4o achieves a privacy score of 91.5% under probing, but only 63.8% during actual execution — demonstrating that "knowing what constitutes privacy" and "protecting privacy during execution" are distinct capabilities.
- Llama and Claude exhibit substantially better raw privacy performance (~88–90%) than GPT-series models (~54–75%), though GPT models achieve higher task completion rates.
- The CoT privacy prompt is the most effective mitigation strategy, improving GPT-4o's privacy score from 63.8% to 91.5%, at the cost of reducing task completion rate from 45.5% to 41.5% — indicating a clear utility–privacy trade-off.
- Multimodal input (adding screenshots) yields marginal improvements in task performance but has negligible effect on privacy.

## Highlights & Insights
- **Core finding — the knowing–doing gap**: LLMs can correctly reason about privacy rules, yet leak information during complex multi-step web task execution. This underscores the necessity of end-to-end evaluation in real environments for AI safety assessment, rather than relying solely on probing.
- **Elegant benchmark design**: The "task-irrelevant yet naturally embedded" sensitive information design faithfully simulates real-world scenarios — user-provided chat logs genuinely contain various unrelated personal details.
- Quantifying the utility–privacy trade-off offers direct guidance for practical deployment of AI agents.

## Limitations & Future Work
- Coverage is limited to benign (non-adversarial) scenarios; malicious prompt injection and adversarial attacks are not considered.
- The scale of 246 tasks is relatively limited, and coverage across environment types could be broadened.
- Sensitive information categorization and annotation involve subjectivity; definitions of "sensitive" may vary across cultural contexts.
- Although LLM-as-a-judge achieves 98% human agreement, it may still err on subtle privacy reasoning cases.

## Related Work & Insights
- **vs. ConfAIde/CI-Bench**: These works rely solely on probing-based evaluation without running agents; this paper demonstrates that probing substantially overestimates actual privacy capability.
- **vs. PrivacyLens**: Evaluation is conducted in simulated environments; this paper performs end-to-end testing in real web environments.
- **vs. AirGapAgent**: That work focuses on information leakage under adversarial attacks; this paper addresses inadvertent leakage in non-adversarial settings.

## Rating
- Novelty: ⭐⭐⭐⭐ First benchmark for end-to-end evaluation of agent data minimization in real web environments.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 6 models across GPT/Llama/Claude families; the probing vs. end-to-end comparison is compelling.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and experimental design is well-structured.
- Value: ⭐⭐⭐⭐ Offers direct guidance for the safety-oriented deployment of AI agents.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](web-shepherd_advancing_prms_for_reinforcing_web_agents.md)
- [\[NeurIPS 2025\] AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents](agentauditor_humanlevel_safety_and_security_evaluation_for_l.md)
- [\[NeurIPS 2025\] CORE: Full-Path Evaluation of LLM Agents Beyond Final State](core_full-path_evaluation_of_llm_agents_beyond_final_state.md)
- [\[NeurIPS 2025\] AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness](agentchangebench_a_multi-dimensional_evaluation_framework_for_goal-shift_robustn.md)
- [\[AAAI 2026\] AutoGLM: Autonomous Foundation Agents for GUIs](../../AAAI2026/llm_agent/autoglm_autonomous_foundation_agents_for_guis.md)

<!-- RELATED:END -->
