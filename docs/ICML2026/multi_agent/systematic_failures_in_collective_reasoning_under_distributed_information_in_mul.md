---
title: >-
  [Paper Note] Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs
description: >-
  [ICML 2026][Multi-Agent][HiddenBench] This paper adapts the social psychology Hidden Profile paradigm to multi-agent LLM evaluation…
tags:
  - "ICML 2026"
  - "Multi-Agent"
  - "HiddenBench"
  - "Hidden Profile"
  - "Distributed Information"
  - "Information Asymmetry"
  - "Collective Reasoning Failure"
date: 2026-05-08
content_hash: 511eb3450a4387dc
---

# Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs

**Conference**: ICML 2026  
**arXiv**: [2505.11556](https://arxiv.org/abs/2505.11556)  
**Code**: HuggingFace + GitHub (available)  
**Area**: LLM Agent / Multi-Agent / Collective Reasoning Evaluation  
**Keywords**: HiddenBench, Hidden Profile, Distributed Information, Information Asymmetry, Collective Reasoning Failure

## TL;DR
This paper adapts the social psychology Hidden Profile paradigm to multi-agent LLM evaluation, constructing a 65-task HiddenBench. Systematic evaluation on 15 cutting-edge LLMs reveals: for tasks where a single agent achieves 80.7% accuracy under Full Profile, multi-agent setups under distributed information achieve only 30.1%. The fundamental failure mode is the **inability to proactively elicit information not disclosed by others**. However, lightweight structured communication protocols can significantly mitigate this across model families.

## Background & Motivation

**Background**: Multi-agent LLM systems are increasingly deployed in software development, scientific discovery, and social simulation. The **core promise is that groups can integrate more information than single agents**. This assumption underpins the perceived superiority of the multi-agent paradigm over single models.

**Limitations of Prior Work**: In practice, many replication studies show multi-agent systems often underperform single agents. However, **no clean evaluation can disentangle "collective reasoning failure" from "individual reasoning deficiency"**—when a group fails, is it due to model limitations or poor information integration? Existing benchmarks conflate the two, making attribution impossible.

**Key Challenge**: To measure "collective reasoning" itself, it is essential to ensure: (i) tasks are **unsolvable by individuals** (necessitating group work); (ii) providing all information to a single agent must make the task **solvable** (excluding "task too hard" confounds). Ground truth must also be verifiable.

**Goal**: To engineer the social psychology Hidden Profile paradigm into a scalable multi-agent benchmark, and systematically characterize failure modes of state-of-the-art LLMs under distributed information, as well as whether simple protocols can rescue performance.

**Key Insight**: Hidden Profile is a classic paradigm in social psychology for studying human group decision failures—each member holds different key information, and only pooling can yield the correct answer; otherwise, shared information leads to wrong answers. Formalizing this for LLM evaluation naturally satisfies "unsolvable individually, solvable collectively, with ground truth".

**Core Idea**: Construct HiddenBench (65 tasks: 5 adapted from human studies, 3 hand-crafted, 57 auto-generated), evaluate 15 leading LLMs under both Hidden and Full Profile conditions, and use ablation to isolate the true bottleneck—agents **can** integrate disclosed information, but **do not** proactively elicit undisclosed information.

## Method

### Overall Architecture
Task structure: Each task consists of several decision options and several task-relevant facts. Under the **Hidden Profile condition**, some facts ($\mathcal{I}_s$) are shared among all agents, while the remaining unshared facts ($\mathcal{I}_u$) are uniquely assigned to each agent, i.e., agent $a_i$ receives $I_i=\mathcal{I}_s\cup\{u_i\}$. Shared information is constructed to **support incorrect options**, and only pooling all unshared facts points to the correct option. Under the **Full Profile condition**, all agents receive $\mathcal{I}_s\cup\mathcal{I}_u$. Agents are not told whether information asymmetry exists. Evaluation compares $Y^{\text{pre}}$ (before discussion), $Y^{\text{post}}$ (after discussion), and $Y^{\text{full}}$ (Full Profile upper bound).

### Key Designs

1. **HiddenBench Task Construction and Automated Generation Pipeline**:

    - Function: Extends the social psychology Hidden Profile paradigm to 65 cross-domain tasks (medical, organizational planning, cultural preservation, etc.), ensuring each task satisfies the formal constraint of "unsolvable individually, solvable collectively".
    - Mechanism: Three-stage pipeline—(i) **Generation**: GPT-4.1 generates candidate tasks (scenario + options + shared facts + unshared facts + ground truth) using structured templates; (ii) **Execution**: Each candidate is run 10 times under both Full and Hidden conditions to measure pre-discussion accuracy; (iii) **Selection**: Only tasks with Full Profile ≥ 80% and Hidden Profile ≤ 20% are retained. From 200 candidates, 57 are selected (28.5% pass rate), plus 5 adapted and 3 hand-written, totaling 65 tasks.
    - Design Motivation: Manual task creation is not scalable and introduces subjective bias; pure auto-generation cannot guarantee formal correctness. This "generate-execute-select" process turns the hard constraints of the social psychology paradigm into machine-verifiable thresholds, making the soft concept of "collective reasoning" a reproducible benchmark.

2. **Three-Condition Comparative Evaluation Protocol**:

    - Function: Uses $Y^{\text{full}}$ as the upper bound for individual reasoning, $Y^{\text{pre}}$ as the lower bound requiring group work, and $Y^{\text{post}}$ to assess collective reasoning. Comparing these yields two clean metrics: "collective gain" $Y^{\text{post}}-Y^{\text{pre}}$ and "gap to upper bound" $Y^{\text{post}}-Y^{\text{full}}$.
    - Mechanism: For each model and task, 10 sessions are run; 15 leading LLMs are evaluated (4 families: OpenAI GPT, Google Gemini, Alibaba Qwen, Meta Llama); communication depth $T\in\{5,10,15,20\}$ and group size are varied to test scaling.
    - Design Motivation: Traditional benchmarks report only one accuracy number, making attribution impossible; the three-condition comparison directly reveals whether failure is due to model limitations or poor coordination—this is the paper's key methodological contribution.

3. **Targeted Ablation of Failure Modes**:

    - Function: Further decomposes "collective failure" into aggregation failure, inference failure, and action selection failure, and experimentally localizes the bottleneck to action selection.
    - Mechanism: Varies communication rounds (5/10/15/20), prompting strategies (cooperative / conflictual / CoT / informing asymmetry / share-all), and enforces reveal-all intervention (mechanistically forcing all information to be disclosed in round 1). The key finding: reveal-all significantly narrows the gap, showing that once agents are forced to disclose, they can reason correctly—thus, the bottleneck is not reasoning but **failing to realize they should elicit others' information**.
    - Design Motivation: Simply stating "multi-agent fails" is uninformative; pinpointing the failure stage guides future improvements. This ablation elevates the conclusion from "phenomenon" to "mechanism diagnosis".

### Loss & Training
This is an evaluation paper; no training is performed. All models are evaluated zero-shot via API calls.

## Key Experimental Results

### Main Results
Performance of 15 leading LLMs on HiddenBench (65 tasks, 10 sessions, average rule, post-discussion accuracy under Hidden Profile):

| Model | $Y^{\text{full}}$ (Full) | $Y^{\text{pre}}$ (Hidden pre-discussion) | $Y^{\text{post}}$ (Hidden post-discussion) | Gain | Gap to Full |
|-------|--------------------------|------------------------------------------|--------------------------------------------|------|-------------|
| Gemini-2.5-Pro | 0.981 | 0.217 | **0.671** | +0.454 | -0.310 |
| Gemini-2.5-Flash | High | Medium | 0.550 | Substantial | Medium |
| Gemini-2.5-Flash-Lite | High | Medium | 0.394 | Moderate | Large |
| GPT-5 (minimal reasoning) | High | Medium | Medium | Small | **-0.750** |
| GPT-5-Nano | High | Medium | Low | **-0.004** (almost no improvement) | Very large |
| **Overall Mean (15 models)** | **0.807** | 0.082~0.217 | **0.301** | Moderate | -0.5 level |

Key cross-sectional findings: (i) Single agents achieve 80.7% on average under Full Profile, while multi-agent setups under Hidden Profile achieve only 30.1%, a 50-point gap; (ii) Model size/individual reasoning ability **does not** reliably predict collective performance (GPT-5 is strong individually but weak collectively); (iii) The Gemini family significantly outperforms others in collective settings.

### Ablation Study

| Intervention | Key Phenomenon | Interpretation |
|--------------|---------------|---------------|
| Communication depth $T=5/10/15/20$ | Peak $Y^{\text{post}}=0.233$ at $T=15$, drops to 0.133 at $T=20$ | Longer discussions reinforce erroneous consensus rather than promote exploration |
| Cooperative / Constructive prompt | $Y^{\text{post}}=0.20\sim 0.24$ | Cooperative prompts show no significant improvement |
| Conflictual prompt | $Y^{\text{post}}=0.0\sim 0.26$, most cases lack majority consensus | Conflictual prompts fail to converge |
| Zero-shot CoT | 0.222 | Limited improvement |
| Informing asymmetry (telling agents "there may be information asymmetry") | 0.367 | Simply informing helps but is insufficient |
| Share All Information (prompting to disclose) | 0.467 | Only halves the gap, indicating disclosure alone is insufficient |
| **Reveal-All (mechanistically force all to disclose in round 1)** | Significantly narrows the gap | **Proves the bottleneck is action selection, not inference** |
| Increasing group size | $Y^{\text{post}}$ decreases | More agents make coordination harder |

### Key Findings
- **Failure Mode Localization**: Agents can integrate disclosed information but **do not proactively elicit unshared information**—this is the core finding attributing the 50-point gap to a specific capability deficit.
- Model scale/individual reasoning ≠ collective reasoning; reasoning-heavy models like GPT-5 show no significant advantage in collective settings, challenging the default "scale up will solve it" assumption.
- Excessive communication rounds **reinforce premature consensus**, consistent with groupthink in human social psychology.
- A **lightweight structured communication protocol** (requiring agents to explicitly list their unique evidence before debate) substantially improves $Y^{\text{post}}$ across model families, showing the bottleneck is actionable—improvement is possible without changing the model.

## Highlights & Insights
- Transforming the soft concept of "collective reasoning" into a hard formal constraint ("unsolvable individually, solvable collectively" via dual-threshold selection) is an elegant engineering move, applicable to any "multi-agent benchmark" effort.
- The three-condition protocol (Hidden-pre / Hidden-post / Full) essentially adds **causal counterfactual control** to evaluation—the same task under different information conditions directly reveals "failure attribution", a paradigm extendable to other collective reasoning/cooperation evaluations.
- The contrast between "Reveal-All intervention" and "Share-All prompting" is particularly instructive: prompting makes models "aware they should disclose" but they still do not fully do so; mechanistic intervention forces full disclosure and yields large improvements—showing that elicitation is not a knowledge issue but a **goal/incentive issue**, suggesting future work should focus on RL objective design.
- Reveals a counterintuitive fact: **more agents ≠ better**—contrary to the naive "wisdom of crowds" assumption, aligning with March's exploration-exploitation and Janis's groupthink theories.

## Limitations & Future Work
- Tasks are all multiple-choice decision-making; open-ended generation, tool use, and long-term collaboration scenarios are not covered.
- Communication protocol is synchronous, fully connected broadcast; partial observability, async messaging, and structured organizational hierarchies are not tested.
- Prompting interventions demonstrate the bottleneck is elicitation behavior, but **no systematic solution at the training objective level** is provided—future work needs RL/SFT dataset design to truly address this behavior.
- The 15 models include both closed and some open-source, with broad scale and family coverage, but no finer-grained attribution by training data or RLHF recipe.

## Related Work & Insights
- **vs Du et al. on multi-agent debate**: They assume debate automatically brings benefits; this paper directly refutes that—more debate can sometimes worsen performance, and the core bottleneck is unrelated to "debate quality".
- **vs Cemri et al. on LLM coordination failure**: They observe coordination issues but lack controlled variables; HiddenBench isolates information asymmetry as the sole variable, enabling cleaner attribution.
- **vs Social Psychology Hidden Profile studies**: This paper engineers the human psychology paradigm into an LLM evaluation template, showing many AI agent failure modes are isomorphic to human group failures—this "borrowing from social psychology for AI evaluation" opens many future directions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Engineering the Hidden Profile paradigm into a scalable LLM benchmark is a pioneering bridge from social psychology to AI evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15 models, 65 tasks, dual conditions, multi-dimensional ablation—rare breadth and attribution depth.
- Writing Quality: ⭐⭐⭐⭐⭐ The logic chain pinpointing "failure mode to action selection" is clear and reproducible.
- Value: ⭐⭐⭐⭐⭐ Provides the multi-agent LLM community with a clean evaluation tool and clear research direction (elicit-aware coordination), with lasting impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](../../ACL2026/multi_agent/collaborative_multi-agent_scripts_generation_for_enhancing_imperfect-information.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](../../ACL2026/multi_agent/silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)
- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](../../ACL2026/multi_agent/from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)
- [\[ACL 2026\] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](../../ACL2026/multi_agent/diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)
- [\[ICML 2026\] E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory](e-mem_multi-agent_based_episodic_context_reconstruction_for_llm_agent_memory.md)

</div>

<!-- RELATED:END -->
