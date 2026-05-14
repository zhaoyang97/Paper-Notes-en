---
title: >-
  [Paper Note] Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis
description: >-
  [ICLR 2026][LLM Evaluation][Agent Evaluation] This paper proposes TED (Talk, Evaluate, Diagnose), a framework that achieves user-aware dynamic agent evaluation via general, reusable expert/non-expert persona templates…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "Agent Evaluation"
  - "User Awareness"
  - "LLM-as-Judge"
  - "Error Analysis"
  - "Efficiency Metrics"
date: 2026-05-08
content_hash: 71a246ab2fa32f96
---

# Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis

**Conference**: ICLR 2026
**arXiv**: [2603.15483](https://arxiv.org/abs/2603.15483)
**Code**: [GitHub](https://github.com/SAP-samples/agent-quality-inspect)
**Area**: LLM Evaluation
**Keywords**: Agent Evaluation, User Awareness, LLM-as-Judge, Error Analysis, Efficiency Metrics

## TL;DR

This paper proposes TED (Talk, Evaluate, Diagnose), a framework that achieves user-aware dynamic agent evaluation via general, reusable expert/non-expert persona templates; enables fine-grained efficiency assessment through grading notes, LLM-as-judge scoring, and novel metrics such as MaxProgressRate@k; and provides actionable improvement feedback via automated error discovery and clustering. Experiments on τ²-bench and ToolSandbox reveal new insights into agent performance.

## Background & Motivation

- **Background**: LLM agents are increasingly deployed to automate diverse workflows, yet evaluation frameworks remain fragmented—each domain relies on its own methodology (database queries, regex matching, etc.) to determine task success.
- **Limitations of Prior Work**: (1) No unified cross-domain evaluation methodology exists; (2) the effect of user personas on agent behavior is not systematically considered; (3) evaluation stops at metric reporting, lacking diagnosis and actionable improvement guidance.
- **Key Challenge**: Agent behavior is heavily shaped by user interaction, yet user personas are left uncontrolled during evaluation.
- **Goal**: Construct a unified, user-aware, and diagnosable agent evaluation framework.
- **Key Insight**: A three-stage unification of Talk (user simulation) + Evaluate (assessment) + Diagnose (diagnosis).
- **Core Idea**: Effective agent evaluation requires not only correctness, but also conversation quality, efficiency, and systematic error diagnosis.

## Method

### Overall Architecture

Talk → Simulate expert/non-expert user interactions with the agent via reusable persona templates. Evaluate → Convert sub-goals into grading notes, score with LLM-as-judge, and compute metrics such as MaxProgressRate@k. Diagnose → Analyze judge–agent inconsistencies, then automatically discover and cluster error patterns.

### Key Designs

**Design 1: General Reusable Persona Templates**
- **Function**: Decouple user persona from task instructions, providing general expert/non-expert templates that are independent of specific tasks and agents.
- **Mechanism**: $u = f(p, i)$, combining persona prompt $p$ with task instruction $i$. Swapping the persona on the same task isolates the effect of user behavior. A reflect-then-respond two-step process is included.
- **Design Motivation**: Existing methods tightly couple persona with task, making it impossible to isolate the independent effect of user behavior.

**Design 2: Grading Notes + Efficiency Metrics**
- **Function**: Unify all sub-goals (tool calls, response content, etc.) into natural-language checklist items; introduce metrics including MaxProgressRate@k, MaxAUC@k, and MaxPPT@k.
- **Mechanism**: $\text{progress}(i) = \text{fraction of grading notes achieved}$; MaxProgressRate@k is the expected maximum progress across $k$ trials. AUC measures early-stage efficiency, and PPT measures per-turn progress rate.
- **Design Motivation**: Success rate is too coarse-grained; partial progress and conversational turn efficiency must be captured.

**Design 3: Automated Error Discovery**
- **Function**: Two-stage error analysis — low-level error identification followed by semantic clustering.
- **Mechanism**: For sub-goals where judge and agent disagree, an LLM extracts specific low-level error descriptions; these are then semantically clustered into high-level error categories. Judge variance and agent variance reflect judge unreliability and agent instability, respectively.
- **Design Motivation**: Close the loop from metric reporting → error discovery → improvement recommendations.

### Loss & Training

No training is involved; TED is a purely evaluation framework. LLM-as-judge is run multiple times with majority voting. GPT-4.1 serves as both judge and user proxy.

## Key Experimental Results

### Main Results

**τ²-bench Airline Easy (Expert | Non-expert)**

| Agent Model | MeanProg@k | MaxProg@k | pass@k |
|-------------|------------|-----------|--------|
| gpt-4.1 | 0.95 \| 0.82 | 1.00 \| 1.00 | 1.00 \| 1.00 |
| gpt-4o | 0.79 \| 0.86 | 1.00 \| 1.00 | 1.00 \| 1.00 |
| gpt-4o-mini | 0.70 \| 0.61 | 0.90 \| 0.90 | 0.80 \| 0.80 |
| gpt-5 | 0.92 \| 0.92 | 1.00 \| 1.00 | 1.00 \| 1.00 |

### Ablation Study

| Finding | Description |
|---------|-------------|
| Expert vs. Non-expert | Non-expert users systematically reduce agent MeanProg across most models |
| Performance gain after error fixing | 8–10% improvement in MaxProgressRate |
| Judge variance analysis | High-variance sub-goals are predominantly associated with ambiguously described grading notes |

### Key Findings

1. User expertise systematically affects agent performance — non-expert users lead to more turns and lower average progress.
2. MaxProgressRate@k provides finer-grained evaluation than pass@k, distinguishing "near success" from "complete failure."
3. Common error patterns identified through automated error analysis can be directly used to improve agent prompts, yielding 8–10% gains.
4. GPT-5 underperforms GPT-4o on certain ToolSandbox baselines, demonstrating that model upgrades do not necessarily translate to improved agent capabilities.

## Highlights & Insights

1. The Talk–Evaluate–Diagnose three-stage closed-loop design is both comprehensive and practically oriented.
2. The persona decoupling idea is concise yet consequential — isolating the user factor is a prerequisite for fair evaluation.
3. The complete loop from evaluation to diagnosis to improvement goes beyond merely "reporting scores."

## Limitations & Future Work

1. Constructing grading notes still requires manual effort, limiting the degree of automation.
2. Only two persona types (expert/non-expert) are considered; finer-grained user modeling remains unexplored.
3. The reliability of the judge itself is a systemic risk that requires further validation.

## Related Work & Insights

- AgentBoard first introduced progress rate but in an environment-interaction setting; TED extends this to multi-turn dialogue.
- τ²-bench employs domain-specific personas that are not generalizable; TED achieves generalization.
- Insight: Agent evaluation should be an integral part of the engineering feedback loop, rather than a standalone academic exercise.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ★★★★☆ |
| Practicality | ★★★★★ |
| Experimental Thoroughness | ★★★★☆ |
| Writing Quality | ★★★★★ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)
- [\[ACL 2026\] Towards Self-Improving Error Diagnosis in Multi-Agent Systems](../../ACL2026/llm_evaluation/towards_self-improving_error_diagnosis_in_multi-agent_systems.md)
- [\[ICLR 2026\] Which LLM Multi-Agent Protocol to Choose?](which_llm_multi-agent_protocol_to_choose.md)
- [\[ICLR 2026\] Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](unpacking_human_preference_for_llms_demographically_aware_evaluation_of_long-fo.md)
- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](../../NeurIPS2025/llm_evaluation/generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)

</div>

<!-- RELATED:END -->
